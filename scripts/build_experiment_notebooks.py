"""Build private Kaggle A4/A5 and B4 notebooks using the audited code snapshot."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "runs/transfer/kaggle_jobs/seed2/sakshi-classifier-seed-2.ipynb"
JOBS = ROOT / "runs/transfer/kaggle_jobs"
CODE_DATASET = "varun0636/sakshi-experiment-code-20261026"

OVERLAY = '''
import shutil
from pathlib import PurePosixPath

code_manifests = sorted(Path('/kaggle/input').rglob('inventory.json'))
if len(code_manifests) != 1:
    raise RuntimeError(f'expected one experiment-code inventory, found {code_manifests}')
code_root = code_manifests[0].parent
code_inventory = json.loads(code_manifests[0].read_text())
for member in code_inventory['members']:
    relative = PurePosixPath(member['path'])
    if relative.parts[0] != 'source' or '..' in relative.parts:
        raise ValueError(f'unsafe code member: {relative}')
    source = code_root / member['path']
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if source.stat().st_size != member['bytes'] or digest.hexdigest() != member['sha256']:
        raise RuntimeError(f'experiment code hash mismatch: {relative}')
    destination = SOURCE_ROOT.joinpath(*relative.parts[1:])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
print({'experiment_code_commit': code_inventory['git_commit'], 'files': len(code_inventory['members'])})
'''

VARIANTS = {
    "a4": ("a4_no_hierarchy", "model=replace(base.model, flat_humanitarian=True)"),
    "a5": (
        "a5_no_cib",
        "data=replace(base.data, use_cib=False), model=replace(base.model, disaster_type_head=False)",
    ),
    **{
        f"b4-{head.replace('_', '-')}": (
            f"b4_{head}",
            "model=replace(base.model, auxiliary_heads=False, "
            f"single_task_head='{head}'"
            + (", flat_humanitarian=True)" if head == "humanitarian" else ")"),
        )
        for head in ("informative", "humanitarian", "severity", "disaster_type")
    },
}


def build() -> list[Path]:
    template = nbformat.read(TEMPLATE, as_version=4)
    output = []
    for variant, (run_name, changes) in VARIANTS.items():
        for seed in (0, 1, 2):
            notebook = copy.deepcopy(template)
            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    cell.pop("outputs", None)
                    cell.pop("execution_count", None)
            slug = f"sakshi-{variant}-seed-{seed}"
            notebook.cells[0].source = f"# SAKSHI {variant.upper()} · seed {seed}\n\nFrozen standard-split experiment."
            params = notebook.cells[2].source.replace("SEED = 2", f"SEED = {seed}")
            params = params.replace("FINAL_EVALUATION = False", "FINAL_EVALUATION = True")
            if f"SEED = {seed}" not in params or "FINAL_EVALUATION = True" not in params:
                raise ValueError("template parameter cell changed")
            notebook.cells[2].source = params
            notebook.cells[4].source += "\n" + OVERLAY
            original = "config = replace(load(SOURCE_ROOT / CONFIG_PATH, TrainConfig), seed=SEED)"
            preflight = notebook.cells[8].source
            if preflight.count(original) != 1:
                raise ValueError("template preflight cell changed")
            notebook.cells[8].source = preflight.replace(
                original,
                "base = load(SOURCE_ROOT / CONFIG_PATH, TrainConfig)\n"
                f"config = replace(base, name='{run_name}', seed=SEED, {changes})",
            )
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(cell.source, f"{slug}:cell-{index}", "exec")
            job = JOBS / f"{variant}-s{seed}"
            job.mkdir(parents=True, exist_ok=True)
            path = job / f"{slug}.ipynb"
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            metadata = json.loads((TEMPLATE.parent / "kernel-metadata.json").read_text())
            metadata.update(
                {
                    "id": f"varun0636/{slug}",
                    "title": f"SAKSHI {variant.upper()} Seed {seed}",
                    "code_file": path.name,
                    "dataset_sources": ["varun0636/sakshi-gpu-inputs-20261025", CODE_DATASET],
                }
            )
            (job / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            output.append(path)
    return output


if __name__ == "__main__":
    for path in build():
        print(path)
