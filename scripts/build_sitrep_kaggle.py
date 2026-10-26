"""Build three private, pinned GPU notebooks for the frozen WP5 windows."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "runs/transfer/kaggle_jobs"
DATASET = "varun0636/sakshi-sitrep-inputs-20261026"
EVENTS = {
    "hurricane_harvey": "harvey",
    "hurricane_maria": "maria",
    "mexico_earthquake": "mexico",
}

SETUP = r"""
import hashlib, json, os, shutil, sys
from pathlib import Path

matches = [
    p for p in Path('/kaggle/input').rglob('inventory.json')
    if p.parent.joinpath('source').is_dir() and p.parent.joinpath('inputs').is_dir()
]
if len(matches) != 1:
    raise RuntimeError(f'expected one expanded private sitrep dataset, found {matches}')
dataset = matches[0].parent
inventory = json.loads(matches[0].read_text())
SOURCE_ROOT = Path('/kaggle/working/sakshi')
for group, names in (('source', inventory['source_files']), ('inputs', inventory['input_files'])):
    for relative, digest in names.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'unsafe staged path: {relative}')
        origin = dataset / group / path
        content = origin.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f'staged file hash mismatch: {relative}')
        target = SOURCE_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
if len(inventory['events']) != 3 or inventory['plan_windows'] != 53:
    raise RuntimeError('frozen WP5 event/window plan changed')
os.environ['HF_HOME'] = '/kaggle/temp/sakshi_hf_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.chdir(SOURCE_ROOT)
print({'git_commit': inventory['git_commit'], 'event': EVENT, 'plan_windows': inventory['plan_windows']})
"""

INSTALL = r"""
import importlib.metadata, subprocess

torch_version = importlib.metadata.version('torch')
constraints = SOURCE_ROOT / 'host-torch.txt'
constraints.write_text(f'torch=={torch_version}\n')
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--quiet', '-e', str(SOURCE_ROOT),
    'transformers==4.57.1', 'accelerate>=1.0,<2', 'bitsandbytes==0.50.2',
    'pyyaml>=6', 'jsonschema>=4.21', 'emoji>=2.8', 'sentencepiece>=0.2',
    '--upgrade-strategy', 'only-if-needed', '--constraint', str(constraints),
], check=True)
sys.path.insert(0, str(SOURCE_ROOT / 'src'))
import torch
if not torch.cuda.is_available():
    raise RuntimeError('WP5 HF batch generation requires Kaggle GPU')
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
"""

GENERATE = r"""
import json
from dataclasses import replace

from huggingface_hub import snapshot_download

QWEN_ID = 'Qwen/Qwen2.5-7B-Instruct'
QWEN_REV = 'a09a35458c702b33eeacc393d103063234e8bc28'
NLI_ID = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
NLI_REV = 'b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'
patterns = ['*.safetensors', '*.json', '*.model', 'merges.txt']
qwen = snapshot_download(QWEN_ID, revision=QWEN_REV, allow_patterns=patterns)
nli = snapshot_download(NLI_ID, revision=NLI_REV, allow_patterns=patterns)

from sakshi.config import load
from sakshi.generate.workflow import GenerateConfig, run_sitreps

base = load(SOURCE_ROOT / 'configs/generation/sitrep.yaml', GenerateConfig)
config = replace(
    base,
    aggregate_dir=f'runs/aggregate/full-s0-loeo_{EVENT}/{EVENT}',
    packets=f'data/processed/packets/full-s0-loeo_{EVENT}.jsonl',
    output_root='runs/sitreps_hf',
    report_path='reports/generation/summary_hf.md',
    evaluation_plan='runs/generation/evaluation_plan.json',
    llm=replace(base.llm, backend='hf', model=str(qwen), tokenizer=str(qwen)),
    nli=replace(base.nli, model=str(nli)),
)
provenance = {
    'source_commit': inventory['git_commit'],
    'llm_repository': QWEN_ID, 'llm_revision': QWEN_REV,
    'nli_repository': NLI_ID, 'nli_revision': NLI_REV,
    'backend': 'hf', 'llm_quantization': 'bitsandbytes-4bit-on-T4',
    'nli_tau_entail': config.nli.tau_entail,
    'nli_tau_contra': config.nli.tau_contra,
    'threshold_status': 'uncalibrated_default_no_human_review',
}
target = SOURCE_ROOT / 'runs/sitreps_hf' / f'full-s0-loeo_{EVENT}' / EVENT
completion = run_sitreps(config)
(target / 'model_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print({'completion': str(completion), 'event': EVENT})

import zipfile
from sakshi.compute.files import hash_file

export = Path('/kaggle/working') / f'sitrep-hf-{EVENT}.zip'
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(target.glob('*.json')):
        archive.write(path, arcname=f'{EVENT}/{path.name}')
    archive.write(target / 'config.yaml', arcname=f'{EVENT}/config.yaml')
    for path in sorted((SOURCE_ROOT / 'runs/llm_cache').glob('*.json')):
        archive.write(path, arcname=f'llm_cache/{path.name}')
print({'export': str(export), 'sha256': hash_file(export), 'bytes': export.stat().st_size})
"""


def build() -> list[Path]:
    output = []
    for event, short in EVENTS.items():
        job = JOBS / f"sitrep-hf-{short}"
        job.mkdir(parents=True, exist_ok=True)
        notebook = nbformat.v4.new_notebook(
            metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        )
        notebook.cells = [
            nbformat.v4.new_markdown_cell(
                f"# SAKSHI WP5 · {event}\n\nPrivate, pinned Qwen2.5-7B and DeBERTa generation on the frozen 53-window plan. "
                "Default NLI thresholds are uncalibrated until human review."
            ),
            nbformat.v4.new_code_cell(f"EVENT = {event!r}\n" + SETUP),
            nbformat.v4.new_code_cell(INSTALL),
            nbformat.v4.new_code_cell(GENERATE),
        ]
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"sitrep-{short}-cell-{index}", "exec")
        nbformat.validate(notebook)
        path = job / f"sakshi-sitrep-hf-{short}.ipynb"
        nbformat.write(notebook, path)
        metadata = {
            "id": f"varun0636/sakshi-sitrep-hf-{short}",
            "title": f"SAKSHI Sitrep HF {short}",
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
                "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
            ),
            "machine_shape": "NvidiaTeslaT4",
        }
        (job / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        output.append(path)
    return output


if __name__ == "__main__":
    for path in build():
        print(path)
