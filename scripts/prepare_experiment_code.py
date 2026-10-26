"""Package a clean classifier source checkout for private Kaggle ablations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from sakshi.compute.bundle import source_files
from sakshi.compute.files import hash_file


def prepare(source_root: Path, output: Path) -> Path:
    source_root, output = source_root.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root, text=True).strip():
        raise ValueError("experiment source checkout must be clean")
    output.mkdir(parents=True, exist_ok=True)
    for source in source_files(source_root):
        destination = output / "source" / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output / "source/source_provenance.json").write_text(
        json.dumps({"git_commit": commit, "git_dirty": False}, indent=2) + "\n"
    )
    members = [
        {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": hash_file(path)}
        for path in sorted((output / "source").rglob("*"))
        if path.is_file()
    ]
    (output / "inventory.json").write_text(json.dumps({"git_commit": commit, "members": members}, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SAKSHI private experiment code 20261026",
                "id": "varun0636/sakshi-experiment-code-20261026",
                "licenses": [{"name": "unknown"}],
            },
            indent=2,
        )
        + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source_root, args.output))
