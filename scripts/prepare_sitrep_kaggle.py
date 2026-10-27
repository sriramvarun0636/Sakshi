"""Stage the frozen three-event WP5 inputs and source for a private Kaggle job."""

from __future__ import annotations

import json
import subprocess
import zipfile
from hashlib import sha256
from pathlib import Path

from sakshi import paths
from sakshi.compute.files import hash_file

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=True).stdout.strip()


def prepare(output: Path, *, captioned: bool = False) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    commit = _git("rev-parse", "HEAD")
    source_bytes = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD", "src", "pyproject.toml", "README.md", "configs/generation/sitrep.yaml"],
        cwd=paths.REPO,
        capture_output=True,
        check=True,
    ).stdout
    source = output / "source.zip"
    source.write_bytes(source_bytes)
    provenance = {"git_commit": commit, "git_dirty": False}
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("source_provenance.json", json.dumps(provenance, indent=2) + "\n")
    with zipfile.ZipFile(source) as archive:
        source_files = {name: sha256(archive.read(name)).hexdigest() for name in archive.namelist() if not name.endswith("/")}

    inputs = output / "inputs.zip"
    input_files = {}
    with zipfile.ZipFile(inputs, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        plan = paths.REPO / (
            "runs/generation/evaluation_plan_captioned.json" if captioned else "runs/generation/evaluation_plan.json"
        )
        files = [plan]
        for event in EVENTS:
            root = "aggregate_captioned" if captioned else "aggregate"
            packet_root = "packets_captioned" if captioned else "packets"
            files += sorted((paths.REPO / f"runs/{root}/full-s0-loeo_{event}/{event}").glob("*"))
            files.append(paths.PROCESSED / f"{packet_root}/full-s0-loeo_{event}.jsonl")
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = path.relative_to(paths.REPO).as_posix()
            input_files[relative] = hash_file(path)
            archive.write(path, arcname=relative)
    plan_data = json.loads(plan.read_text())
    inventory = {
        "git_commit": commit,
        "source_archive_sha256": hash_file(source),
        "source_files": source_files,
        "input_archive_sha256": hash_file(inputs),
        "input_files": input_files,
        "plan_windows": len(plan_data["selected"]),
        "events": list(EVENTS),
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": (
                    "SAKSHI captioned WP5 inputs 20261027"
                    if captioned else "SAKSHI private WP5 sitrep inputs 20261026"
                ),
                "id": (
                    "varun0636/sakshi-sitrep-captioned-inputs-20261027"
                    if captioned else "varun0636/sakshi-sitrep-inputs-20261026"
                ),
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
    parser.add_argument("--captioned", action="store_true")
    args = parser.parse_args()
    print(prepare(args.output, captioned=args.captioned))
