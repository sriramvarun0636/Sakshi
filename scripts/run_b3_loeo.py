"""Derive B3 late-fusion LOEO folds from frozen B1/B2 sources, one at a time."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from sakshi import paths
from sakshi.compute.files import hash_file

EVENTS = (
    "srilanka_floods",
    "hurricane_harvey",
    "hurricane_irma",
    "hurricane_maria",
    "mexico_earthquake",
    "california_wildfires",
    "iraq_iran_earthquake",
)


def source(prefix: str, event: str) -> Path:
    matches = sorted((paths.REPO / "runs/classifier").glob(f"{prefix}-s0-loeo_{event}-*"))
    if len(matches) != 1 or not (matches[0] / "evaluation_loeo_complete.json").is_file():
        raise ValueError(f"expected one frozen {prefix} fold for {event}: {matches}")
    return matches[0]


def completed(output: Path) -> bool:
    marker = output / "complete.json"
    if not marker.is_file():
        return False
    checks = json.loads(marker.read_text())
    if checks != {name: hash_file(output / name) for name in checks}:
        raise ValueError(f"B3 LOEO marker changed: {output}")
    return True


def main() -> None:
    executable = paths.REPO / ".venv/bin/sakshi-baselines"
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2"}
    for event in EVENTS:
        text = source("b1-text", event)
        image = source("b2-image", event)
        output = paths.REPO / f"runs/baselines/b3-late-s0-loeo_{event}-20261026"
        if completed(output):
            print(f"SKIP {event}: {output}", flush=True)
            continue
        print(f"START {event}: {text.name} + {image.name}", flush=True)
        subprocess.run(
            [
                str(executable),
                "late-fusion",
                "--text-run",
                str(text),
                "--image-run",
                str(image),
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            cwd=paths.REPO,
            env=env,
            check=True,
        )
        if not completed(output):
            raise RuntimeError(f"B3 LOEO fold ended without a valid completion marker: {event}")
        print(f"DONE {event}: {output}", flush=True)


if __name__ == "__main__":
    main()
