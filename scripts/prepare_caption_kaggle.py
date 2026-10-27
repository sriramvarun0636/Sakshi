"""Stage the frozen WP5 packet images for private Qwen caption jobs."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd

from sakshi import paths
from sakshi.compute.files import hash_file

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")


def prepare(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = paths.PROCESSED / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path).set_index("sample_id")
    if not manifest.index.is_unique:
        raise ValueError("manifest sample IDs are not unique")
    entries = []
    for event in EVENTS:
        plan = paths.REPO / f"runs/aggregate/plans/full-s0-loeo_{event}.parquet"
        for sample_id in pd.read_parquet(plan).sample_id.astype(str):
            row = manifest.loc[sample_id]
            image = paths.image_file(row)
            if not image.is_file() or image.is_symlink():
                raise FileNotFoundError(image)
            entries.append({"sample_id": sample_id, "event": event, "image": image.relative_to(paths.RAW).as_posix()})
    if len(entries) != len({item["sample_id"] for item in entries}) or len(entries) != 1449:
        raise ValueError("frozen WP5 packet selection changed")

    archive = output / "images.tar"
    with tarfile.open(archive, "w") as stream:
        for item in sorted(entries, key=lambda row: row["image"]):
            stream.add(paths.RAW / item["image"], arcname=item["image"], recursive=False)
    (output / "entries.json").write_text(json.dumps(entries, indent=2) + "\n")
    inventory = {
        "entries": len(entries),
        "events": {event: sum(item["event"] == event for item in entries) for event in EVENTS},
        "manifest_sha256": hash_file(manifest_path),
        "archive_sha256": hash_file(archive),
        "entries_sha256": hash_file(output / "entries.json"),
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SAKSHI private WP5 caption images 20261027",
                "id": "varun0636/sakshi-caption-images-20261027",
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
