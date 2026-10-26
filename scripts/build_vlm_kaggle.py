"""Build the private, official-selector B6 Qwen2.5-VL inference notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "runs/transfer/kaggle_jobs/b6-vlm"
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

SETUP = r'''
import hashlib, json, os, sys
from pathlib import Path

import pandas as pd

roots = sorted(Path('/kaggle/input').rglob('bundle.json'))
if len(roots) != 1:
    raise RuntimeError(f'expected one prepared-data bundle, found {roots}')
bundle = roots[0].parent
manifest_path = bundle / 'prepared/manifest.parquet'
manifest = pd.read_parquet(manifest_path)
inventories = [p for p in Path('/kaggle/input').rglob('inventory.json') if p.parent.joinpath('official_images').is_dir()]
if len(inventories) != 1:
    raise RuntimeError(f'expected one official image dataset, found {inventories}')
image_root = inventories[0].parent / 'official_images'
image_inventory = json.loads(inventories[0].read_text())
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != image_inventory['manifest_sha256']:
    raise RuntimeError('manifest changed since official images were staged')
selectors = {
    'informative': ('crisismmd', 'split_informative', 'y_informative', 1534),
    'humanitarian': ('crisismmd', 'split_humanitarian', 'y_humanitarian', 955),
    'severity': ('crisismmd', 'split_damage', 'y_severity', 529),
    'severity_dmd': ('cib', 'split_cib_damage', 'y_severity', 1138),
    'disaster_type': ('cib', 'split_cib_disaster_types', 'y_disaster_type', 3213),
}
selected = set()
for head, (dataset, split, gold, expected) in selectors.items():
    rows = manifest.loc[(manifest.dataset == dataset) & (manifest[split] == 'test') & manifest[gold].notna()]
    if len(rows) != expected or not (rows['split'] == 'test').all():
        raise RuntimeError(f'{head} official selector changed: {len(rows)}')
    selected.update(rows.sample_id.astype(str))
if len(selected) != image_inventory['selected_samples']:
    raise RuntimeError('official image count changed')
work = Path('/kaggle/working/b6')
work.mkdir(parents=True, exist_ok=True)
rows = manifest.loc[manifest.sample_id.astype(str).isin(selected)].sort_values('sample_id')
print({'official_union': len(rows), 'image_root': str(image_root), 'model': 'Qwen/Qwen2.5-VL-3B-Instruct'})
'''

INSTALL = r'''
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'transformers==4.57.1', 'accelerate>=1.0,<2', 'ftfy==6.3.1'], check=True)
'''

INFER = r"""
import gc, html, json, re, time
from collections import Counter
from pathlib import Path

import ftfy
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
MODEL_REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'
CLASSES = {
    'informative': ['not_informative', 'informative'],
    'humanitarian': ['affected_individuals', 'infrastructure_and_utility_damage', 'not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort'],
    'damage_severity': ['little_or_no_damage', 'mild_damage', 'severe_damage'],
    'disaster_type': ['earthquake', 'fire', 'flood', 'hurricane', 'landslide', 'not_disaster', 'other_disaster'],
}
GOLD = {'informative': 'y_informative', 'humanitarian': 'y_humanitarian', 'damage_severity': 'y_severity', 'disaster_type': 'y_disaster_type'}
majority = {}
for head, gold in GOLD.items():
    counts = manifest.loc[(manifest['split'] == 'train') & manifest[gold].notna(), gold].astype(str).value_counts()
    majority[head] = str(counts.idxmax())
if not torch.cuda.is_available():
    raise RuntimeError('B6 requires a Kaggle GPU')
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, min_pixels=28*28*16, max_pixels=28*28*256)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
PROMPT = '''You are annotating social-media posts about disasters for humanitarian response.
Look at the image and read the post text, then answer with JSON only, in this form:
{"informative": ..., "humanitarian": ..., "damage_severity": ..., "disaster_type": ...}

Definitions:
informative: "informative" if the post is useful for humanitarian response (it reports damage,
casualties, needs, rescue, donations, warnings or other disaster information); otherwise
"not_informative".
humanitarian (choose one):
  affected_individuals — people affected, injured, dead, missing, found, trapped or displaced
  infrastructure_and_utility_damage — damage to buildings, roads, bridges, power lines, vehicles or utilities
  rescue_volunteering_or_donation_effort — rescue, evacuation, volunteering, donations or relief supplies
  other_relevant_information — other disaster information (forecasts, maps, warnings, general updates)
  not_humanitarian — use this if and only if the post is not_informative
damage_severity (judge the image): severe_damage (structures destroyed or unusable),
mild_damage (partially damaged but usable), little_or_no_damage (no visible damage)
disaster_type (judge the image): earthquake, fire, flood, hurricane, landslide,
other_disaster, not_disaster

Post text: {text_clean}'''
MENTION = re.compile(r'@\w+')
URL = re.compile(r'https?://\S+|www\.\S+', re.I)
SPACE = re.compile(r'\s+')
SYNONYMS = {'not informative': 'not_informative', 'not humanitarian': 'not_humanitarian', 'little or no damage': 'little_or_no_damage', 'mild damage': 'mild_damage', 'severe damage': 'severe_damage', 'other disaster': 'other_disaster', 'not disaster': 'not_disaster'}

def clean_text(row):
    if row.dataset == 'cib' or not isinstance(row.text, str) or not row.text.strip():
        return '(none)'
    value = ftfy.fix_text(html.unescape(row.text))
    value = URL.sub('[LINK]', MENTION.sub('[USER]', value))
    return SPACE.sub(' ', value).strip() or '(none)'

def parse(raw):
    value = raw.strip()
    if value.startswith('```'):
        value = re.sub(r'^```(?:json)?\s*|\s*```$', '', value, flags=re.I)
    obj = json.loads(value)
    if not isinstance(obj, dict) or set(obj) != set(CLASSES):
        raise ValueError('unexpected JSON keys')
    result = {}
    for key, choices in CLASSES.items():
        if not isinstance(obj[key], str):
            raise ValueError(f'{key} is not a string')
        label = SYNONYMS.get(obj[key].strip().lower(), obj[key].strip().lower())
        if label not in choices:
            raise ValueError(f'unknown {key}: {label}')
        result[key] = label
    if (result['informative'] == 'not_informative') != (result['humanitarian'] == 'not_humanitarian'):
        raise ValueError('informative/humanitarian contradiction')
    return result

def infer(image, prompt):
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[template], images=[image], padding=True, return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    del inputs, output
    return decoded

output = work / 'predictions.jsonl'
completed = {}
if output.exists():
    for line in output.read_text().splitlines():
        item = json.loads(line)
        completed[item['sample_id']] = item
    if len(completed) != len(output.read_text().splitlines()):
        raise RuntimeError('duplicate resumed sample ID')
for index, row in enumerate(rows.itertuples(index=False), 1):
    sample_id = str(row.sample_id)
    if sample_id in completed:
        continue
    subdir = 'CrisisMMD_v2.0' if row.dataset == 'crisismmd' else 'crisis_vision_benchmarks'
    image_path = image_root / subdir / row.image_path
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    start = time.perf_counter()
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    prompt = PROMPT.replace('{text_clean}', clean_text(row))
    raw = infer(image, prompt)
    retry_raw = None
    failed = False
    try:
        labels = parse(raw)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        repair = prompt + '\n\nYour previous answer was invalid: ' + str(exc) + '\nReturn only one valid JSON object with exactly the four requested enum-valued fields.'
        retry_raw = infer(image, repair)
        try:
            labels = parse(retry_raw)
        except (ValueError, TypeError, json.JSONDecodeError):
            labels = majority.copy()
            failed = True
    item = {'sample_id': sample_id, 'labels': labels, 'raw': raw, 'retry_raw': retry_raw, 'failed': failed, 'latency_seconds': time.perf_counter() - start}
    with output.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    if index % 100 == 0:
        print({'completed': index, 'total': len(rows), 'failures': sum(json.loads(line)['failed'] for line in output.read_text().splitlines())}, flush=True)
    del image
    if index % 500 == 0:
        gc.collect()
record = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'temperature': 0, 'max_new_tokens': 128, 'max_pixels': 28*28*256, 'prompt': PROMPT, 'majority': majority, 'rows': len(rows), 'manifest_sha256': image_inventory['manifest_sha256']}
(work / 'protocol.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
print({'complete': sum(1 for _ in output.open()), 'output': str(output)})
"""


def build() -> Path:
    JOB.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# SAKSHI B6 · zero-shot VLM\n\nPrivate official-test inference with the Appendix E.7 prompt. The 3B fallback fits a T4 without quantization. This notebook does not train or calibrate on test labels."),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_code_cell(INSTALL),
        nbformat.v4.new_code_cell(INFER),
    ]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"b6-cell-{index}", "exec")
    path = JOB / "sakshi-b6-vlm.ipynb"
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    metadata = {
        "id": "varun0636/sakshi-b6-vlm-zero-shot",
        "title": "SAKSHI B6 VLM zero shot",
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": ["varun0636/sakshi-gpu-inputs-20261025", "varun0636/sakshi-vlm-images-20261026"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435",
        "machine_shape": "NvidiaTeslaT4",
    }
    (JOB / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(build())
