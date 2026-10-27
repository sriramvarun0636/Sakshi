"""Build private, event-sharded Qwen2.5-VL caption notebooks for WP5 images."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "runs/transfer/kaggle_jobs"
DATASET = "varun0636/sakshi-caption-images-20261027"
EVENTS = {"hurricane_harvey": "harvey", "hurricane_maria": "maria", "mexico_earthquake": "mexico"}

SETUP = r"""
import hashlib, json, os, sys, tarfile
from pathlib import Path

roots = [p.parent for p in Path('/kaggle/input').rglob('inventory.json') if p.parent.joinpath('entries.json').is_file()]
if len(roots) != 1:
    raise RuntimeError(f'expected one private caption dataset, found {roots}')
dataset = roots[0]
inventory = json.loads((dataset / 'inventory.json').read_text())
entries_path = dataset / 'entries.json'
if hashlib.sha256(entries_path.read_bytes()).hexdigest() != inventory['entries_sha256']:
    raise RuntimeError('caption entry list hash mismatch')
entries = [e for e in json.loads(entries_path.read_text()) if e['event'] == EVENT]
if len(entries) != inventory['events'][EVENT] or len({e['sample_id'] for e in entries}) != len(entries):
    raise RuntimeError('event caption selection changed')
image_root = dataset / 'images'
if not image_root.is_dir():
    archive = dataset / 'images.tar'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != inventory['archive_sha256']:
        raise RuntimeError('caption image archive hash mismatch')
    image_root = Path('/kaggle/working/images')
    image_root.mkdir(parents=True, exist_ok=True)
    wanted = {e['image'] for e in entries}
    with tarfile.open(archive) as stream:
        for member in stream:
            if member.name not in wanted:
                continue
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile():
                raise RuntimeError(f'unsafe caption image member: {member.name}')
            target = image_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('wb') as sink:
                sink.write(source.read())
if any(not (image_root / e['image']).is_file() for e in entries):
    raise RuntimeError('one or more selected caption images are absent')
os.environ['HF_HOME'] = '/kaggle/temp/sakshi_caption_hf_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
work = Path('/kaggle/working/captions')
work.mkdir(parents=True, exist_ok=True)
print({'event': EVENT, 'images': len(entries), 'image_root': str(image_root)})
"""

INSTALL = r"""
import subprocess
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--quiet',
    'transformers==4.57.1', 'accelerate>=1.0,<2', 'bitsandbytes==0.50.2',
], check=True)
"""

INFER = r"""
import datetime, hashlib, json, zipfile

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
PROMPT = (
    'Describe this image in one factual sentence of at most 20 words. '
    'Mention visible damage, flooding, fire, debris, people, vehicles or rescue activity if present. '
    'Do not guess the location, date, event or cause. '
    'If the image is a map, chart, screenshot, logo, poster or mostly text, say which.'
)
if not torch.cuda.is_available():
    raise RuntimeError('Qwen captions require a Kaggle GPU')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, revision=REVISION, torch_dtype=torch.float16, device_map='auto',
    attn_implementation='sdpa',
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    ),
).eval()
processor = AutoProcessor.from_pretrained(
    MODEL, revision=REVISION, min_pixels=28*28*16, max_pixels=28*28*256,
)
prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
output = work / f'{EVENT}.jsonl'
count = 0
with output.open('w') as sink:
    for entry in entries:
        with Image.open(image_root / entry['image']) as source:
            image = source.convert('RGB')
        try:
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT},
            ]}]
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[rendered], images=[image], return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=48)
            caption = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
        finally:
            image.close()
        if not caption:
            raise RuntimeError(f'empty caption for {entry["sample_id"]}')
        row = {
            'sample_id': entry['sample_id'], 'caption': caption, 'model': MODEL,
            'prompt_sha256': prompt_hash,
            'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        sink.write(json.dumps(row, ensure_ascii=False) + '\n')
        sink.flush()
        count += 1
        if count % 25 == 0:
            print({'event': EVENT, 'completed': count, 'total': len(entries)}, flush=True)
if count != len(entries):
    raise RuntimeError('caption count mismatch')
completion = {
    'event': EVENT, 'rows': count, 'model': MODEL, 'revision': REVISION,
    'prompt_sha256': prompt_hash, 'entries_sha256': inventory['entries_sha256'],
    'captions_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'quantization': 'bitsandbytes-nf4',
}
done = work / f'{EVENT}-completion.json'
done.write_text(json.dumps(completion, indent=2) + '\n')
export = Path('/kaggle/working') / f'captions-{EVENT}.zip'
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(output, arcname=output.name)
    archive.write(done, arcname=done.name)
print({'export': str(export), **completion})
"""


def build() -> list[Path]:
    output = []
    for event, short in EVENTS.items():
        job = JOBS / f"caption-{short}"
        job.mkdir(parents=True, exist_ok=True)
        notebook = nbformat.v4.new_notebook(
            metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        )
        notebook.cells = [
            nbformat.v4.new_markdown_cell(
                f"# SAKSHI WP5 captions · {event}\n\n"
                "Private Qwen2.5-VL-7B captions for the frozen packet images."
            ),
            nbformat.v4.new_code_cell(f"EVENT = {event!r}\n" + SETUP),
            nbformat.v4.new_code_cell(INSTALL),
            nbformat.v4.new_code_cell(INFER),
        ]
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"caption-{short}-cell-{index}", "exec")
        nbformat.validate(notebook)
        path = job / f"sakshi-caption-{short}.ipynb"
        nbformat.write(notebook, path)
        metadata = {
            "id": f"varun0636/sakshi-caption-{short}",
            "title": f"sakshi-caption-{short}",
            "code_file": path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": ["gpu"],
            "dataset_sources": [DATASET],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "docker_image": (
                "gcr.io/kaggle-private-byod/python@sha256:"
                "9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
            ),
            "machine_shape": "NvidiaTeslaT4",
        }
        (job / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        output.append(path)
    return output


if __name__ == "__main__":
    for path in build():
        print(path)
