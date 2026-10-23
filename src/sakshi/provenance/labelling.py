"""Blind two-rater provenance review export and adjudicated study metrics (§8.10)."""

from __future__ import annotations

import base64
import html
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score

from sakshi import atomic, paths
from sakshi.images import load_rgb

LABELS = (
    "recycled_photo",
    "concurrent_share",
    "same_scene_different_photo",
    "template_or_placeholder",
    "stock_or_press",
    "false_match",
)
FLAGGED = {"recycled", "cross_event_concurrent"}
CORRECT_FLAG = {"recycled_photo", "concurrent_share", "stock_or_press"}


def _pair_id(a: str, b: str) -> str:
    return sha256("\0".join(sorted((a, b))).encode()).hexdigest()[:16]


def _pairs(candidates: pd.DataFrame, replays: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Union pHash candidates and replay CLIP flags without leaking status to raters."""
    required = {"a", "b"}
    if required - set(candidates):
        raise ValueError("provenance candidates lack pair IDs")
    candidate_pairs = candidates[["a", "b"]].copy()
    candidate_pairs["detector_status"] = "candidate_unflagged"
    candidate_pairs["source"] = "phash"
    replay_pairs = replays.loc[
        ((replays.status.isin(FLAGGED) & (replays.tier == "clip")) | (replays.status == "template_match"))
        & replays.origin_post_id.notna(),
        ["post_id", "origin_post_id", "status"],
    ].rename(columns={"post_id": "a", "origin_post_id": "b", "status": "detector_status"})
    replay_pairs["source"] = "replay"
    pairs = pd.concat((replay_pairs, candidate_pairs), ignore_index=True)
    pairs["a"] = pairs.a.astype(str)
    pairs["b"] = pairs.b.astype(str)
    pairs["pair_id"] = [_pair_id(a, b) for a, b in zip(pairs.a, pairs.b, strict=True)]
    pairs = pairs.drop_duplicates("pair_id", keep="first")
    lookup = index.set_index("sample_id")
    if not pairs.a.isin(lookup.index).all() or not pairs.b.isin(lookup.index).all():
        raise ValueError("candidate pairs refer to images outside the provenance index")
    pairs["component_a"] = lookup.loc[pairs.a, "component"].to_numpy()
    pairs["component_b"] = lookup.loc[pairs.b, "component"].to_numpy()
    pairs["graphic_a"] = lookup.loc[pairs.a, "is_graphic"].to_numpy(dtype=bool)
    pairs["graphic_b"] = lookup.loc[pairs.b, "is_graphic"].to_numpy(dtype=bool)
    pairs["both_graphic"] = pairs.graphic_a & pairs.graphic_b
    pairs["group_id"] = [":".join(sorted((str(a), str(b)))) for a, b in zip(pairs.component_a, pairs.component_b, strict=True)]
    return pairs.reset_index(drop=True)


def select_pairs(pairs: pd.DataFrame, *, graphic_sample: int = 50, seed: int = 0) -> pd.DataFrame:
    """One representative per non-graphic component group, plus 50 graphic pairs."""
    if graphic_sample < 0:
        raise ValueError("graphic_sample must be nonnegative")
    nongraphic = pairs.loc[~pairs.both_graphic].copy()
    nongraphic["priority"] = (nongraphic.source != "replay").astype(int)
    nongraphic = nongraphic.sort_values(["priority", "pair_id"]).drop_duplicates("group_id").drop(columns="priority")
    graphic = pairs.loc[pairs.both_graphic].sort_values("pair_id")
    count = min(graphic_sample, len(graphic))
    if count < graphic_sample:
        raise ValueError(f"only {len(graphic)} graphic pairs available; requested {graphic_sample}")
    rng = np.random.default_rng(seed)
    picked = graphic.iloc[np.sort(rng.choice(len(graphic), size=count, replace=False))]
    return pd.concat((nongraphic, picked), ignore_index=True).sort_values("pair_id").reset_index(drop=True)


def _thumbnail(row: pd.Series) -> str:
    image = load_rgb(paths.image_file(row))
    if image is None:
        return ""
    try:
        image.thumbnail((440, 440), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=72, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    finally:
        image.close()


def _viewer(sheet: pd.DataFrame, manifest: pd.DataFrame) -> str:
    lookup = manifest.set_index("sample_id")
    chunks = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>SAKSHI provenance review</title>",
        "<style>body{font:16px system-ui;margin:2rem;max-width:1100px}article{border-top:1px solid #bbb;padding:1rem 0}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.pair img{max-width:100%;max-height:440px;object-fit:contain}"
        "small{color:#555}</style>",
        "<h1>Blind provenance review</h1><p>Judge each pair from the images, events and dates only. "
        "Record labels in the CSV by pair ID. Detector decisions are hidden.</p>",
        "<p>Labels: recycled_photo; concurrent_share; same_scene_different_photo; "
        "template_or_placeholder; stock_or_press; false_match.</p>",
    ]
    for row in sheet.itertuples(index=False):
        sides = []
        for sample_id in (row.sample_id_a, row.sample_id_b):
            item = lookup.loc[sample_id]
            source = _thumbnail(item)
            event = item.event if pd.notna(item.event) else item.folder if pd.notna(item.folder) else "unknown"
            date = item.posted_at.isoformat() if pd.notna(item.posted_at) else "undated"
            sides.append(
                "<div><img loading='lazy' src='"
                + source
                + "' alt='review image'><br><small>"
                + html.escape(f"{event} · {date} · {sample_id}")
                + "</small></div>"
            )
        chunks.append(
            "<article><h2>" + html.escape(row.pair_id) + "</h2><div class='pair'>" + "".join(sides) + "</div></article>"
        )
    chunks.append("</html>")
    return "\n".join(chunks)


def export_labels(
    *,
    candidates_path: Path = paths.REPORTS / "data" / "provenance_candidates.csv",
    replay_dir: Path = paths.PROCESSED / "provenance",
    index_path: Path = paths.PROCESSED / "provenance" / "index.parquet",
    manifest_path: Path = paths.PROCESSED / "manifest.parquet",
    sheet_path: Path = paths.REPORTS / "provenance" / "labelling_sheet.csv",
    viewer_path: Path = paths.REPO / "runs" / "provenance" / "labelling.html",
    metadata_path: Path = paths.REPO / "runs" / "provenance" / "label_metadata.parquet",
    graphic_sample: int = 50,
    seed: int = 0,
) -> Path:
    """Export a blinded CSV and local thumbnail viewer; preserve existing human work."""
    if sheet_path.exists():
        existing = pd.read_csv(sheet_path, keep_default_na=False)
        if any(existing[column].astype(str).str.strip().ne("").any() for column in ("label_a", "label_b", "final")):
            raise FileExistsError(f"{sheet_path} contains human labels; refusing to overwrite")
    candidates = pd.read_csv(candidates_path, usecols=["a", "b"])
    replay_paths = sorted(replay_dir.glob("replay_*.parquet"))
    if len(replay_paths) != 7:
        raise FileNotFoundError("all seven event replay files are required for label export")
    replays = pd.concat((pd.read_parquet(path) for path in replay_paths), ignore_index=True)
    index = pd.read_parquet(index_path)
    manifest = pd.read_parquet(manifest_path)
    selected = select_pairs(_pairs(candidates, replays, index), graphic_sample=graphic_sample, seed=seed)
    lookup = manifest.set_index("sample_id")
    records = []
    for pair in selected.itertuples(index=False):
        a, b = lookup.loc[pair.a], lookup.loc[pair.b]
        records.append(
            {
                "pair_id": pair.pair_id,
                "sample_id_a": pair.a,
                "event_a": a.event if pd.notna(a.event) else a.folder,
                "posted_at_a": a.posted_at,
                "sample_id_b": pair.b,
                "event_b": b.event if pd.notna(b.event) else b.folder,
                "posted_at_b": b.posted_at,
                "label_a": "",
                "label_b": "",
                "final": "",
            }
        )
    sheet = pd.DataFrame(records)
    metadata = selected[["pair_id", "source", "detector_status", "both_graphic", "group_id"]]
    viewer = _viewer(sheet, manifest)
    atomic.parquet(metadata_path, metadata)
    atomic.text(viewer_path, viewer)
    atomic.csv(sheet_path, sheet)
    return sheet_path


def score_labels(
    sheet_path: Path = paths.REPORTS / "provenance" / "labelling_sheet.csv",
    metadata_path: Path = paths.REPO / "runs" / "provenance" / "label_metadata.parquet",
    output: Path = paths.REPORTS / "provenance" / "hand_label_study.md",
) -> Path:
    """Score only fully adjudicated labels, reporting rater agreement and flag precision."""
    sheet = pd.read_csv(sheet_path, keep_default_na=False, dtype={"pair_id": "string"})
    if sheet.pair_id.duplicated().any():
        raise ValueError("labelling sheet contains duplicate pair IDs")
    for column in ("label_a", "label_b", "final"):
        if column not in sheet or not sheet[column].isin(LABELS).all():
            raise ValueError(f"{column} needs one valid label for every pair before scoring")
    metadata = pd.read_parquet(metadata_path)
    frame = sheet.merge(metadata, on="pair_id", how="left", validate="one_to_one")
    if frame.detector_status.isna().any() or len(frame) != len(metadata):
        raise ValueError("labelling sheet and detector metadata are misaligned")
    kappa = float(cohen_kappa_score(frame.label_a, frame.label_b, labels=list(LABELS)))
    flagged = frame.loc[frame.detector_status.isin(FLAGGED)]
    precision = float(flagged.final.isin(CORRECT_FLAG).mean()) if len(flagged) else None
    predicted_template = frame.detector_status == "template_match"
    actual_template = frame.final == "template_or_placeholder"
    template_precision = float(actual_template[predicted_template].mean()) if predicted_template.any() else None
    template_recall = float(predicted_template[actual_template].mean()) if actual_template.any() else None
    distribution = pd.crosstab(frame.detector_status, frame.final).reindex(columns=LABELS, fill_value=0)
    lines = [
        "# Hand-verified provenance study",
        "",
        f"Pairs: {len(frame)}. Cohen's κ between the two independent raters: {kappa:.3f}.",
        f"Precision of recycled/concurrent flags: {precision if precision is not None else 'n/a'} "
        f"({len(flagged)} flagged pairs).",
        f"Template precision: {template_precision if template_precision is not None else 'n/a'}; "
        f"template recall: {template_recall if template_recall is not None else 'n/a'}.",
        "",
        "## Human-label distribution by detector status",
        "",
        "| Status | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|" * len(LABELS),
    ]
    for status, row in distribution.iterrows():
        lines.append("| " + str(status) + " | " + " | ".join(str(int(value)) for value in row) + " |")
    atomic.text(output, "\n".join(lines) + "\n")
    return output
