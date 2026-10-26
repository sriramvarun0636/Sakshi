"""Stage only official-test images for the private B6 Kaggle baseline."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.eval import selectors


def prepare(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    selected = set().union(*(set(manifest.loc[selectors.official(manifest, head), "sample_id"].astype(str)) for head in selectors.HEAD_LABELS))
    rows = manifest.loc[manifest.sample_id.astype(str).isin(selected)].sort_values("sample_id")
    if len(rows) != len(selected) or not selected:
        raise ValueError("official B6 selector union is empty or not unique")
    files = []
    for _, row in rows.iterrows():
        image = paths.image_file(row)
        if not image.is_file() or image.is_symlink():
            raise FileNotFoundError(f"official image unavailable: {row.sample_id}")
        files.append((image, image.relative_to(paths.RAW)))
    archive = output / "official_images.tar"
    with tarfile.open(archive, "w") as stream:
        for source, relative in sorted(files, key=lambda item: item[1].as_posix()):
            stream.add(source, arcname=relative.as_posix(), recursive=False)
    inventory = {
        "selected_samples": len(selected),
        "manifest_sha256": hash_file(paths.PROCESSED / "manifest.parquet"),
        "archive_sha256": hash_file(archive),
        "archive_bytes": archive.stat().st_size,
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "SAKSHI private B6 images 20261026", "id": "varun0636/sakshi-vlm-images-20261026", "licenses": [{"name": "unknown"}]},
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
