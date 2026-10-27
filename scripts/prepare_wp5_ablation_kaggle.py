"""Stage captioned sitreps, packets and pinned source for a private NLI ablation job."""

from __future__ import annotations

import json
import subprocess
import zipfile
from hashlib import sha256
from pathlib import Path

from sakshi import paths
from sakshi.compute.files import hash_file

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")


def prepare(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    source_bytes = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD", "src", "scripts/run_wp5_ablations.py", "pyproject.toml"],
        cwd=paths.REPO,
        capture_output=True,
        check=True,
    ).stdout
    source = output / "source.zip"
    source.write_bytes(source_bytes)
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("source_provenance.json", json.dumps({"git_commit": commit}, indent=2) + "\n")
    with zipfile.ZipFile(source) as archive:
        source_files = {name: sha256(archive.read(name)).hexdigest() for name in archive.namelist() if not name.endswith("/")}

    plan = paths.REPO / "runs/generation/evaluation_plan_captioned.json"
    files = [plan]
    for event in EVENTS:
        model_id = f"full-s0-loeo_{event}"
        files.append(paths.PROCESSED / "packets_captioned" / f"{model_id}.jsonl")
        files.extend(sorted((paths.REPO / "runs/sitreps_hf_captioned" / model_id / event).glob("*")))
    inputs = output / "inputs.zip"
    hashes = {}
    with zipfile.ZipFile(inputs, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = path.relative_to(paths.REPO).as_posix()
            hashes[relative] = hash_file(path)
            archive.write(path, arcname=relative)
    plan_data = json.loads(plan.read_text())
    if len(plan_data["selected"]) != 53:
        raise ValueError("captioned plan differs from the frozen 53-window study")
    inventory = {
        "git_commit": commit,
        "source_archive_sha256": hash_file(source),
        "source_files": source_files,
        "input_archive_sha256": hash_file(inputs),
        "input_files": hashes,
        "windows": 53,
        "events": list(EVENTS),
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SAKSHI WP5 ablation inputs 20261027",
                "id": "varun0636/sakshi-wp5-ablation-inputs-20261027",
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
