"""Build the private Kaggle notebook for Maria and Mexico evidence packets."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

DOCKER_IMAGE_PREFIX = "gcr.io/kaggle-private-byod/python@sha256:"


def build(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    cells = [
        nbformat.v4.new_markdown_cell(
            "# SAKSHI · Maria and Mexico evidence packets\n\n"
            "Private, selected-image input only. Hashes are verified before model loading. "
            "Each completed event is exported separately for recovery."
        ),
        nbformat.v4.new_code_cell(
            """import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

BASE = Path('/kaggle/input/sakshi-gpu-inputs-20261025')
DELTA = Path('/kaggle/input/sakshi-xai-inputs-20261025')
ROOT = Path('/kaggle/working/sakshi/source')
PROCESSED = ROOT / 'data/processed'
inventory = json.loads((DELTA / 'inventory.json').read_text())
for member in inventory['members']:
    path = DELTA / member['path']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != member['bytes'] or digest.hexdigest() != member['sha256']:
        raise RuntimeError(f'XAI input integrity failed: {member["path"]}')
shutil.copytree(DELTA / 'source', ROOT)
for name, destination in (
    ('runs', ROOT / 'runs'),
    ('plans', ROOT / 'runs/aggregate/plans'),
    ('provenance', PROCESSED / 'provenance'),
    ('raw', ROOT / 'data/raw'),
):
    shutil.copytree(DELTA / name, destination, dirs_exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
(PROCESSED / 'features').mkdir(exist_ok=True)
for relative in (
    'manifest.parquet',
    'features/features_index.parquet',
    'features/clip_vitb16_tokens.f16.npy',
    'features/clip_vitb16_embed.f32.npy',
):
    source = BASE / 'prepared' / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    (PROCESSED / relative).symlink_to(source)
os.environ['SAKSHI_PROCESSED_DIR'] = str(PROCESSED)
os.environ['HF_HOME'] = '/kaggle/working/hf-cache'
print({'selected_samples': inventory['selected_samples'], 'members': len(inventory['members']), 'root': str(ROOT)})"""
        ),
        nbformat.v4.new_code_cell(
            """import importlib.metadata
import subprocess

torch_version = importlib.metadata.version('torch')
constraints = Path('/kaggle/working/host-torch.txt')
constraints.write_text(f'torch=={torch_version}\\n')
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-e', f'{ROOT}[ml,xai]',
     '--constraint', str(constraints), '--upgrade-strategy', 'only-if-needed'],
    check=True,
)
sys.path.insert(0, str(ROOT / 'src'))
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU is required for this packet job')
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})"""
        ),
        nbformat.v4.new_code_cell(
            """import logging
import tarfile

from sakshi.compute.files import hash_file
from sakshi.explain.workflow import packets

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
RUNS = {
    'hurricane_maria': 'full-s0-loeo_hurricane_maria-20261025-145019',
    'mexico_earthquake': 'full-s0-loeo_mexico_earthquake-20261025-150334',
}
exports = {}
for event, run_id in RUNS.items():
    model_id = f'full-s0-loeo_{event}'
    output = PROCESSED / 'packets' / f'{model_id}.jsonl'
    plan = ROOT / 'runs/aggregate/plans' / f'{model_id}.parquet'
    run = ROOT / 'runs/classifier' / run_id
    if not output.exists():
        packets(run, selector=str(plan), output=output, device='cuda')
    artifacts = output.parent / f'{output.stem}_artifacts'
    marker = json.loads((artifacts / 'complete.json').read_text())
    if marker['packet_sha256'] != hash_file(output):
        raise RuntimeError(f'packet hash mismatch: {event}')
    archive = Path('/kaggle/working') / f'{model_id}.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(output, arcname=output.relative_to(ROOT))
        stream.add(artifacts, arcname=artifacts.relative_to(ROOT))
        stream.add(run / 'explanation_protocol.json', arcname=f'protocols/{event}.json')
    exports[event] = {'archive': archive.name, 'sha256': hash_file(archive), 'packets': marker['packets']}
    Path('/kaggle/working/export.json').write_text(json.dumps(exports, indent=2) + '\\n')
    print(event, exports[event])"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    name = "sakshi-xai-packets-maria-mexico"
    notebook_path = output / f"{name}.ipynb"
    nbformat.write(notebook, notebook_path)
    metadata = {
        "id": f"varun0636/{name}",
        "title": "SAKSHI Maria and Mexico same-model evidence packets",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": ["varun0636/sakshi-gpu-inputs-20261025", "varun0636/sakshi-xai-inputs-20261025"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": DOCKER_IMAGE_PREFIX + "9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435",
        "machine_shape": "NvidiaTeslaT4",
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return notebook_path


if __name__ == "__main__":
    print(build(Path("runs/transfer/kaggle_jobs/xai-packets")))
