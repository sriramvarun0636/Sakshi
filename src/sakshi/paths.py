"""Filesystem layout. Everything under data/ is git-ignored."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw"
# Set before importing sakshi when mounting cached inputs read-only on a GPU host.
PROCESSED = Path(os.environ.get("SAKSHI_PROCESSED_DIR", DATA / "processed")).expanduser().resolve()
REPORTS = REPO / "reports"

CRISISMMD_SPLITS_ALL = RAW / "splits_all" / "crisismmd_datasplit_all"
CRISISMMD_SPLITS_AGREED = RAW / "splits_agreed" / "crisismmd_datasplit_agreed_label"
# Image paths in the CrisisMMD split files are relative to this directory ("data_image/...").
CRISISMMD_ROOT = RAW / "CrisisMMD_v2.0"
# Crisis Image Benchmark: label files under tasks/, images under data/; the label
# files' image paths ("data/...") are relative to this directory.
CIB_ROOT = RAW / "crisis_vision_benchmarks"
# HumAID text-only tweets: events_set1/<event>/<event>_{train,dev,test}.tsv
HUMAID_ROOT = RAW / "humaid"


def image_file(row: Mapping[str, Any]) -> Path:
    """Return the absolute image path for a manifest row without allowing path escapes."""
    dataset = row["dataset"]
    if dataset == "crisismmd":
        root = CRISISMMD_ROOT
    elif dataset == "cib":
        root = CIB_ROOT
    else:
        raise ValueError(f"unknown image dataset: {dataset!r}")

    relative = Path(row["image_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"image_path must stay within {root}: {relative}")
    return root / relative
