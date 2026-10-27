"""Build one private Kaggle audit notebook per formal WP3 kind and head."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from build_formal_xai_kaggle import INSTALL, SETUP

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "runs/transfer/kaggle_jobs"
HEADS = ("informative", "humanitarian", "severity", "disaster_type")
AUDITS = {"faithfulness": 500, "sanity": 100}
DATASETS = [
    "varun0636/sakshi-formal-xai-inputs-20261026",
    "varun0636/sakshi-gpu-inputs-20261025",
    "varun0636/sakshi-vlm-images-20261026",
]
IMAGE_DIGEST = "9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"

RUN = r"""
import json, logging, time, zipfile

from sakshi.compute.files import hash_file
from sakshi.explain.workflow import audit

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
run = ROOT / 'runs/classifier' / inventory['run_id']
output = ROOT / 'runs/explain/formal' / KIND / HEAD
start = time.perf_counter()
report = audit(
    run, kind=KIND, output=output, split='test', count=COUNT,
    device='cuda', heads=(HEAD,), bootstrap=1000,
)
timing = {
    'kind': KIND, 'head': HEAD, 'count': COUNT,
    'seconds': time.perf_counter() - start,
    'source_commit': inventory['source_commit'],
    'run_id': inventory['run_id'],
    'report': report.name,
}
(output / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
export = Path('/kaggle/working') / f'formal-xai-{KIND}-{HEAD}.zip'
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output.rglob('*')):
        if path.is_file() and 'heatmaps' not in path.parts:
            archive.write(path, arcname=path.relative_to(output))
    archive.write(run / 'explanation_protocol.json', arcname='explanation_protocol.json')
print({'timing': timing, 'export': str(export), 'sha256': hash_file(export), 'bytes': export.stat().st_size})
"""


def build() -> list[Path]:
    built = []
    for kind, count in AUDITS.items():
        for head in HEADS:
            name = f"formal-xai-{kind}-{head.replace('_', '-')}"
            job = JOBS / name
            job.mkdir(parents=True, exist_ok=True)
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell(
                        f"# SAKSHI WP3 · {kind} · {head}\n\n"
                        f"Frozen official-test selector, seed 0, {count} posts, private GPU audit."
                    ),
                    nbformat.v4.new_code_cell(f"KIND = {kind!r}\nHEAD = {head!r}\nCOUNT = {count}\n" + SETUP),
                    nbformat.v4.new_code_cell(INSTALL),
                    nbformat.v4.new_code_cell(RUN),
                ],
                metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            )
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(cell.source, f"{name}-cell-{index}", "exec")
            nbformat.validate(notebook)
            path = job / f"sakshi-{name}.ipynb"
            nbformat.write(notebook, path)
            metadata = {
                "id": f"varun0636/sakshi-{name}",
                "title": f"SAKSHI {name.replace('-', ' ').title()}",
                "code_file": path.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "keywords": ["gpu"],
                "dataset_sources": DATASETS,
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "docker_image": f"gcr.io/kaggle-private-byod/python@sha256:{IMAGE_DIGEST}",
                "machine_shape": "NvidiaTeslaT4",
            }
            (job / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            built.append(path)
    return built


if __name__ == "__main__":
    for notebook in build():
        print(notebook)
