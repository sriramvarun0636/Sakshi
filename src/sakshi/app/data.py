"""Small, read-only adapters for the dashboard's precomputed artefacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sakshi import paths

_WINDOW = re.compile(r"\d{8}T\d{6}Z\.json")


@dataclass(frozen=True)
class AggregateSource:
    directory: Path
    model_id: str
    event: str
    mode: str
    evidence_ready: bool
    windows: tuple[Path, ...]


@dataclass(frozen=True)
class SitrepSource:
    directory: Path
    model_id: str
    event: str
    mode: str
    windows: tuple[Path, ...]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classifier_run(root: Path = paths.REPO) -> Path | None:
    runs = sorted((root / "runs" / "classifier").glob("*/checkpoint.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].parent if runs else None


def aggregates(root: Path = paths.REPO) -> list[AggregateSource]:
    result = []
    for marker in (root / "runs" / "aggregate").rglob("complete.json"):
        try:
            done = read_json(marker)
            directory = marker.parent
            windows = tuple(sorted(path for path in directory.glob("*.json") if _WINDOW.fullmatch(path.name)))
            if len(windows) != done["windows"] or not (directory / "tweets.parquet").is_file():
                continue
            sample = read_json(windows[0]) if windows else None
            if sample is None:
                continue
            result.append(
                AggregateSource(
                    directory,
                    str(sample["model_id"]),
                    str(sample["event"]),
                    str(done["mode"]),
                    bool(done["evidence_ready"]),
                    windows,
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    # Prefer evidence-backed research, then a validated demo preview.
    return sorted(
        result,
        key=lambda item: (item.evidence_ready, item.mode == "loeo", "validated" in str(item.directory), str(item.directory)),
        reverse=True,
    )


def preferred_aggregates(root: Path = paths.REPO) -> dict[str, AggregateSource]:
    result: dict[str, AggregateSource] = {}
    for source in aggregates(root):
        result.setdefault(source.event, source)
    return result


def sitreps(root: Path = paths.REPO) -> dict[str, SitrepSource]:
    result: dict[str, SitrepSource] = {}
    for marker in sorted((root / "runs" / "sitreps").glob("*/*/complete.json"), reverse=True):
        try:
            done = read_json(marker)
            windows = tuple(sorted(path for path in marker.parent.glob("*.json") if _WINDOW.fullmatch(path.name)))
            if not windows or len(windows) != done["windows"]:
                continue
            sample = read_json(windows[0])
            event = str(sample["event"])
            result.setdefault(
                event, SitrepSource(marker.parent, str(sample["models"]["classifier"]), event, done["mode"], windows)
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def manifest(root: Path = paths.REPO) -> pd.DataFrame:
    path = root / "data" / "processed" / "manifest.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=["sample_id", "dataset", "image_path", "text", "event", "posted_at", "tweet_id"])


def replay(root: Path, event: str) -> pd.DataFrame:
    if not re.fullmatch(r"[a-z_]+", event):
        raise ValueError("invalid event")
    path = root / "data" / "processed" / "provenance" / f"replay_{event}.parquet"
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def image_path(root: Path, row: dict | pd.Series | None) -> Path | None:
    if row is None:
        return None
    relative = row.get("image_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        return None
    dataset = row.get("dataset")
    if dataset == "crisismmd":
        base = root / "data" / "raw" / "CrisisMMD_v2.0"
    elif dataset == "cib":
        base = root / "data" / "raw" / "crisis_vision_benchmarks"
    else:
        return None
    return base / relative


def classifier_predictions(root: Path = paths.REPO) -> tuple[pd.DataFrame, str]:
    run = classifier_run(root)
    if run is not None:
        for name in ("predictions_all.parquet", "predictions_extended.parquet", "predictions_official.parquet"):
            path = run / name
            if path.is_file():
                return pd.read_parquet(path), "trained classifier"
    # A tiny real dev sample can populate the abstention lane before a full run exists.
    candidates = sorted((root / "runs" / "explain").glob("*/packets_artifacts/predictions.parquet"))
    frames = [pd.read_parquet(path) for path in candidates]
    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates("sample_id"), "trained classifier · dev preview"
    return pd.DataFrame(), "unavailable"


def report_files(root: Path = paths.REPO) -> list[Path]:
    folder = root / "reports" / "results"
    return sorted(folder.glob("*.md")) if folder.is_dir() else []
