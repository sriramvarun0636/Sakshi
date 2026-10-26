"""Prepare the config-only A1–A3 Kaggle experiments from the frozen full notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "runs/transfer/kaggle_jobs/seed2/sakshi-classifier-seed-2.ipynb"
JOBS = ROOT / "runs/transfer/kaggle_jobs"
VARIANTS = {
    "a1": ("a1_no_cross_attention", "model", "cross_attention_layers", 0),
    "a2": ("a2_no_gate", "model", "gate", False),
    "a3": ("a3_no_modality_dropout", "data", "modality_dropout", 0.0),
}


def build() -> list[Path]:
    template = nbformat.read(TEMPLATE, as_version=4)
    output = []
    for name, (run_name, section, key, value) in VARIANTS.items():
        for seed in (0, 1, 2):
            notebook = copy.deepcopy(template)
            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    cell.pop("outputs", None)
                    cell.pop("execution_count", None)
            slug = f"sakshi-{name}-seed-{seed}"
            notebook.cells[0].source = f"# SAKSHI {name.upper()} · seed {seed}\n\nFrozen standard-split ablation."
            parameter = notebook.cells[2].source
            parameter = parameter.replace("SEED = 2", f"SEED = {seed}")
            parameter = parameter.replace("FINAL_EVALUATION = False", "FINAL_EVALUATION = True")
            if f"SEED = {seed}" not in parameter or "FINAL_EVALUATION = True" not in parameter:
                raise ValueError("template parameter cell changed")
            notebook.cells[2].source = parameter
            preflight = notebook.cells[8].source
            original = "config = replace(load(SOURCE_ROOT / CONFIG_PATH, TrainConfig), seed=SEED)"
            replacement = (
                "base = load(SOURCE_ROOT / CONFIG_PATH, TrainConfig)\n"
                f"config = replace(base, name='{run_name}', seed=SEED, "
                f"{section}=replace(base.{section}, {key}={value!r}))"
            )
            if preflight.count(original) != 1:
                raise ValueError("template preflight cell changed")
            notebook.cells[8].source = preflight.replace(original, replacement)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(cell.source, f"{slug}:cell-{index}", "exec")
            job = JOBS / f"{name}-s{seed}"
            job.mkdir(parents=True, exist_ok=True)
            path = job / f"{slug}.ipynb"
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            metadata = json.loads((TEMPLATE.parent / "kernel-metadata.json").read_text())
            metadata.update(
                {
                    "id": f"varun0636/{slug}",
                    "title": f"SAKSHI {name.upper()} Seed {seed}",
                    "code_file": path.name,
                    "dataset_sources": ["varun0636/sakshi-gpu-inputs-20261025"],
                }
            )
            (job / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            output.append(path)
    return output


if __name__ == "__main__":
    for notebook_path in build():
        print(notebook_path)
