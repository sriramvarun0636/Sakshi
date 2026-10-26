"""Stage the frozen full-model run and current source for private WP3 GPU audits."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from hashlib import sha256
from pathlib import Path

from sakshi import paths
from sakshi.compute.files import hash_file

RUN_ID = "full-s0-20261024-062724"
RUN_FILES = (
    "checkpoint.pt",
    "config.yaml",
    "inputs.json",
    "label_maps.json",
    "temperatures.json",
    "thresholds.json",
    "evaluation_request.json",
    "evaluation_complete.json",
    "metrics_test.json",
    "predictions_official.parquet",
)


def prepare(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    source = output / "source.zip"
    with source.open("wb") as stream:
        subprocess.run(["git", "archive", "--format=zip", "HEAD"], cwd=paths.REPO, stdout=stream, check=True)
    with zipfile.ZipFile(source) as archive:
        source_files = {name: sha256(archive.read(name)).hexdigest() for name in archive.namelist() if not name.endswith("/")}
    if not source_files or "src/sakshi/explain/workflow.py" not in source_files:
        raise ValueError("source archive lacks the WP3 implementation")
    run = paths.REPO / "runs/classifier" / RUN_ID
    marker = json.loads((run / "evaluation_complete.json").read_text())
    if marker != {name: hash_file(run / name) for name in marker}:
        raise ValueError("frozen full-model official evaluation changed")
    run_hashes = {name: hash_file(run / name) for name in RUN_FILES}
    bundle = output / "run.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in RUN_FILES:
            archive.write(run / name, arcname=f"runs/classifier/{RUN_ID}/{name}")
    inventory = {
        "source_commit": commit,
        "source_files": source_files,
        "run_id": RUN_ID,
        "run_files": run_hashes,
        "source_zip_sha256": hash_file(source),
        "run_zip_sha256": hash_file(bundle),
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SAKSHI private formal XAI inputs 20261026",
                "id": "varun0636/sakshi-formal-xai-inputs-20261026",
                "licenses": [{"name": "unknown"}],
            },
            indent=2,
        )
        + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output))
