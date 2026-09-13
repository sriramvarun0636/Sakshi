"""Filesystem layout. Everything under data/ is git-ignored."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
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
