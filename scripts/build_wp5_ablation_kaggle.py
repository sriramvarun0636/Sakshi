"""Build a private GPU notebook for the fixed-sentence WP5 NLI ablations."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "runs/transfer/kaggle_jobs/wp5-ablation-captioned"
DATASET = "varun0636/sakshi-wp5-ablation-inputs-20261027"

SETUP = r"""
import hashlib, json, os, sys
from pathlib import Path

matches = [
    p for p in Path('/kaggle/input').rglob('inventory.json')
    if p.parent.joinpath('source').is_dir() and p.parent.joinpath('inputs').is_dir()
]
if len(matches) != 1:
    raise RuntimeError(f'expected one private WP5 ablation dataset, found {matches}')
dataset = matches[0].parent
inventory = json.loads(matches[0].read_text())
SOURCE_ROOT = Path('/kaggle/working/sakshi')
for group, names in (('source', inventory['source_files']), ('inputs', inventory['input_files'])):
    for relative, digest in names.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError(f'unsafe staged path: {relative}')
        origin = dataset / group / path
        content = origin.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f'staged file hash mismatch: {relative}')
        target = SOURCE_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
if inventory['windows'] != 53 or len(inventory['events']) != 3:
    raise RuntimeError('captioned study scope changed')
os.environ['HF_HOME'] = '/kaggle/temp/sakshi_nli_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.chdir(SOURCE_ROOT)
print({'source_commit': inventory['git_commit'], 'windows': inventory['windows']})
"""

INSTALL = r"""
import importlib.metadata, subprocess

torch_version = importlib.metadata.version('torch')
constraints = SOURCE_ROOT / 'host-torch.txt'
constraints.write_text(f'torch=={torch_version}\n')
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--quiet', '-e', str(SOURCE_ROOT),
    'transformers==4.57.1', 'accelerate>=1.0,<2', 'pyyaml>=6',
    'jsonschema>=4.21', 'emoji>=2.8', 'sentencepiece>=0.2',
    '--upgrade-strategy', 'only-if-needed', '--constraint', str(constraints),
], check=True)
sys.path.insert(0, str(SOURCE_ROOT / 'src'))
import torch
if not torch.cuda.is_available():
    raise RuntimeError('WP5 NLI ablation requires a Kaggle GPU')
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
"""

EVALUATE = r"""
import json, zipfile

from huggingface_hub import snapshot_download
from sakshi.compute.files import hash_file
from scripts.run_wp5_ablations import REVISION, run

NLI_ID = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
NLI_REVISION = 'b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'
if REVISION != NLI_REVISION:
    raise RuntimeError('NLI revision differs from the ablation source')
nli = snapshot_download(
    NLI_ID, revision=NLI_REVISION,
    allow_patterns=['*.safetensors', '*.json', '*.model', 'merges.txt'],
)
result = SOURCE_ROOT / 'runs/ablation_captioned/results.json'
report = SOURCE_ROOT / 'reports/generation/ablations_captioned.md'
run(str(nli), result, report)
provenance = {
    'source_commit': inventory['git_commit'],
    'nli_repository': NLI_ID,
    'nli_revision': NLI_REVISION,
    'windows': 53,
    'result_sha256': hash_file(result),
    'report_sha256': hash_file(report),
}
marker = SOURCE_ROOT / 'runs/ablation_captioned/completion.json'
marker.write_text(json.dumps(provenance, indent=2) + '\n')
export = Path('/kaggle/working/wp5-ablations-captioned.zip')
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(result, arcname='results.json')
    archive.write(report, arcname='ablations_captioned.md')
    archive.write(marker, arcname='completion.json')
print({'export': str(export), **provenance})
"""


def build() -> Path:
    JOB.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# SAKSHI WP5 fixed-sentence verifier ablations"),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_code_cell(INSTALL),
        nbformat.v4.new_code_cell(EVALUATE),
    ]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"wp5-ablation-cell-{index}", "exec")
    nbformat.validate(notebook)
    path = JOB / "sakshi-wp5-ablation-captioned.ipynb"
    nbformat.write(notebook, path)
    metadata = {
        "id": "varun0636/sakshi-wp5-ablation-captioned",
        "title": "sakshi-wp5-ablation-captioned",
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
    (JOB / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(build())
