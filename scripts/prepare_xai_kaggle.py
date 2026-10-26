"""Stage only the private inputs needed for the three research packet runs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from sakshi import paths
from sakshi.compute.bundle import source_files
from sakshi.compute.files import hash_file

RUNS = {
    "hurricane_harvey": "full-s0-loeo_hurricane_harvey-20261025-063127",
    "hurricane_maria": "full-s0-loeo_hurricane_maria-20261025-145019",
    "mexico_earthquake": "full-s0-loeo_mexico_earthquake-20261025-150334",
}
RUN_FILES = (
    "checkpoint.pt",
    "config.yaml",
    "env.json",
    "inputs.json",
    "label_maps.json",
    "temperatures.json",
    "thresholds.json",
    "evaluation_loeo_request.json",
    "evaluation_loeo_complete.json",
    "predictions_loeo.parquet",
    "metrics_loeo.json",
)


def stage_file(source: Path, destination: Path, *, link: bool = False) -> None:
    if not source.is_file() or source.is_symlink() or destination.exists():
        raise ValueError(f"unsafe or missing staging input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def prepare(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root = paths.REPO
    for source in source_files(root):
        stage_file(source, output / "source" / source.relative_to(root))

    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet").set_index("sample_id", drop=False)
    selected: set[str] = set()
    for event, run_id in RUNS.items():
        run = root / "runs" / "classifier" / run_id
        marker = json.loads((run / "evaluation_loeo_complete.json").read_text())
        for name, expected in marker.items():
            if hash_file(run / name) != expected:
                raise ValueError(f"unfrozen LOEO evaluation: {run_id}/{name}")
        for name in RUN_FILES:
            stage_file(run / name, output / "runs" / "classifier" / run_id / name, link=name == "checkpoint.pt")
        plan = root / "runs" / "aggregate" / "plans" / f"full-s0-loeo_{event}.parquet"
        ids = pd.read_parquet(plan).sample_id.astype(str).tolist()
        if not ids or len(ids) != len(set(ids)) or set(ids) - set(manifest.index):
            raise ValueError(f"invalid packet plan: {event}")
        if any(str(manifest.loc[sample_id, "event"]) != event for sample_id in ids):
            raise ValueError(f"packet plan includes a different event: {event}")
        selected.update(ids)
        stage_file(plan, output / "plans" / plan.name)

    for name in ("index.parquet", "onsets.json"):
        stage_file(paths.PROCESSED / "provenance" / name, output / "provenance" / name)
    for sample_id in sorted(selected):
        row = manifest.loc[sample_id]
        if row.dataset != "crisismmd":
            raise ValueError(f"unexpected image dataset: {sample_id}")
        image = paths.image_file(row)
        stage_file(image, output / "raw" / image.relative_to(paths.RAW), link=True)

    members = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            members.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": hash_file(path)})
    (output / "inventory.json").write_text(json.dumps({"selected_samples": len(selected), "members": members}, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SAKSHI private XAI inputs 20261025",
                "id": "varun0636/sakshi-xai-inputs-20261025",
                "licenses": [{"name": "unknown"}],
            },
            indent=2,
        )
        + "\n"
    )
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output))
