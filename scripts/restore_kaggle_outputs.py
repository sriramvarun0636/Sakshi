"""Recover completed private Kaggle training outputs with bounded download retries."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "runs/transfer/kaggle_jobs"
OUTPUTS = ROOT / "runs/transfer/kaggle_outputs"
RUNS = ROOT / "runs/classifier"
KAGGLE = ROOT / ".venv/bin/kaggle"
GPU = ROOT / ".venv/bin/sakshi-gpu"


def command(args: list[object], *, timeout: int = 300) -> str:
    result = subprocess.run([str(value) for value in args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode or "Kernel push error:" in result.stdout:
        raise RuntimeError((result.stderr + result.stdout)[-800:])
    return result.stdout


def restore(job: str) -> Path:
    metadata = json.loads((JOBS / job / "kernel-metadata.json").read_text())
    target = OUTPUTS / job
    target.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            command(
                [
                    KAGGLE,
                    "kernels",
                    "output",
                    metadata["id"],
                    "-p",
                    target,
                    "--file-pattern",
                    r"zip\.part[0-9]+$|export\.json$",
                    "--page-size",
                    "200",
                    "-q",
                ],
                timeout=300,
            )
            break
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise
            time.sleep(10 * attempt)
    exports = list(target.rglob("export.json"))
    if len(exports) != 1:
        raise ValueError(f"{job}: expected one export manifest, found {len(exports)}")
    manifest = json.loads(exports[0].read_text())
    if not {"checkpoint.pt", "metrics_dev.json"} <= set(manifest["members"]):
        raise ValueError(f"{job}: training export is incomplete")
    run = RUNS / manifest["run_id"]
    if not run.exists():
        command([GPU, "restore", "--export", exports[0].parent, "--output", run], timeout=600)
    if not (run / "checkpoint.pt").is_file():
        raise ValueError(f"{job}: restored checkpoint is absent")
    for part in manifest["parts"]:
        (exports[0].parent / part["path"]).unlink(missing_ok=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    for name in args.jobs:
        print(name, restore(name), flush=True)
