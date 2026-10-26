"""Build a private GPU timing pilot for the frozen WP3 test audits."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "runs/transfer/kaggle_jobs/formal-xai-pilot"

SETUP = r"""
import hashlib, json, os, shutil, sys
from pathlib import Path

inputs = [
    p for p in Path('/kaggle/input').rglob('inventory.json')
    if p.parent.joinpath('source').is_dir() and p.parent.joinpath('run').is_dir()
]
images = [p for p in Path('/kaggle/input').rglob('inventory.json') if p.parent.joinpath('official_images').is_dir()]
bases = list(Path('/kaggle/input').rglob('bundle.json'))
if len(inputs) != 1 or len(images) != 1 or len(bases) != 1:
    raise RuntimeError(f'expected one audit input, official image and base cache dataset: {inputs}, {images}, {bases}')
inventory = json.loads(inputs[0].read_text())
image_inventory = json.loads(images[0].read_text())
ROOT = Path('/kaggle/working/sakshi')
PROCESSED = ROOT / 'data/processed'
BASE = bases[0].parent
def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
for group, files in (('source', inventory['source_files']), ('run', {
    f"runs/classifier/{inventory['run_id']}/{name}": digest for name, digest in inventory['run_files'].items()
})):
    for relative, digest in files.items():
        source = inputs[0].parent / group / relative
        if digest_file(source) != digest:
            raise RuntimeError(f'audit staged file changed: {group}/{relative}')
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith('checkpoint.pt'):
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)
PROCESSED.mkdir(parents=True, exist_ok=True)
(PROCESSED / 'features').mkdir(exist_ok=True)
for relative in (
    'manifest.parquet','features/features_index.parquet',
    'features/clip_vitb16_tokens.f16.npy','features/clip_vitb16_embed.f32.npy'
):
    source = BASE / 'prepared' / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    (PROCESSED / relative).symlink_to(source)
run_inputs = json.loads((ROOT / 'runs/classifier' / inventory['run_id'] / 'inputs.json').read_text())
if (
    digest_file(PROCESSED / 'manifest.parquet') != image_inventory['manifest_sha256']
    or run_inputs['manifest'] != image_inventory['manifest_sha256']
):
    raise RuntimeError('official image manifest differs from the frozen classifier')
raw = ROOT / 'data/raw'
raw.symlink_to(images[0].parent / 'official_images', target_is_directory=True)
os.environ['SAKSHI_PROCESSED_DIR'] = str(PROCESSED)
os.environ['HF_HOME'] = '/kaggle/temp/sakshi_formal_xai_hf_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.chdir(ROOT)
print({
    'source_commit': inventory['source_commit'], 'run': inventory['run_id'],
    'official_images': image_inventory['selected_samples']
})
"""

INSTALL = r"""
import importlib.metadata, subprocess

torch_version = importlib.metadata.version('torch')
constraints = ROOT / 'host-torch.txt'
constraints.write_text(f'torch=={torch_version}\n')
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--quiet', '-e', f'{ROOT}[ml,xai]',
    '--constraint', str(constraints), '--upgrade-strategy', 'only-if-needed',
], check=True)
sys.path.insert(0, str(ROOT / 'src'))
import torch
if not torch.cuda.is_available():
    raise RuntimeError('formal WP3 audit pilot requires CUDA')
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
"""

PILOT = r"""
import logging, time, zipfile

from sakshi.explain.common import MAIN_HEADS
from sakshi.explain.workflow import audit
from sakshi.compute.files import hash_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
run = ROOT / 'runs/classifier' / inventory['run_id']
output_root = ROOT / 'runs/explain/formal-pilot'
timings = {}
for kind, count in (('faithfulness', 2), ('sanity', 1)):
    start = time.perf_counter()
    report = audit(
        run, kind=kind, output=output_root / kind, split='test',
        count=count, device='cuda', heads=MAIN_HEADS, bootstrap=100,
    )
    timings[kind] = {'seconds': time.perf_counter()-start, 'count_per_head': count, 'report': report.name}
    print(kind, timings[kind], flush=True)
timings['source_commit'] = inventory['source_commit']
timings['run_id'] = inventory['run_id']
(output_root / 'timings.json').write_text(json.dumps(timings, indent=2) + '\n')
export = Path('/kaggle/working/formal-xai-pilot.zip')
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output_root.rglob('*')):
        if path.is_file() and 'heatmaps' not in path.parts:
            archive.write(path, arcname=path.relative_to(output_root))
    archive.write(run / 'explanation_protocol.json', arcname='explanation_protocol.json')
print({'export': str(export), 'sha256': hash_file(export), 'bytes': export.stat().st_size})
"""


def build() -> Path:
    JOB.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# SAKSHI WP3 · formal audit timing pilot\n\n"
                "Private official-test images, frozen full seed 0, 2 faithfulness and 1 sanity post per head. "
                "This pilot is not the formal 500/100-per-head result."
            ),
            nbformat.v4.new_code_cell(SETUP),
            nbformat.v4.new_code_cell(INSTALL),
            nbformat.v4.new_code_cell(PILOT),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"formal-xai-pilot-cell-{index}", "exec")
    nbformat.validate(notebook)
    path = JOB / "sakshi-formal-xai-pilot.ipynb"
    nbformat.write(notebook, path)
    image_digest = "9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
    metadata = {
        "id": "varun0636/sakshi-formal-xai-timing-pilot",
        "title": "SAKSHI Formal XAI Timing Pilot",
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [
            "varun0636/sakshi-formal-xai-inputs-20261026",
            "varun0636/sakshi-gpu-inputs-20261025",
            "varun0636/sakshi-vlm-images-20261026",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": f"gcr.io/kaggle-private-byod/python@sha256:{image_digest}",
        "machine_shape": "NvidiaTeslaT4",
    }
    (JOB / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(build())
