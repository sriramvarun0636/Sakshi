"""Frozen B1/B2 scores against the original per-modality CrisisMMD labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.data import labels
from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE
from sakshi.models.evaluate_run import _score
from sakshi.models.predict import Predictor

HEADS = {"informative": labels.INFORMATIVE, "humanitarian": labels.HUMANITARIAN}
EXPECTED_ROWS = 2237


def evaluate_modality_predictions(
    manifest: pd.DataFrame, predictions: pd.DataFrame, *, modality: str, bootstrap: int = 1000
) -> dict:
    """Score the fixed official-task test roles using text or image labels directly."""
    if modality not in ("text", "image") or bootstrap < 1:
        raise ValueError("modality evaluation needs text or image and a positive bootstrap count")
    if predictions.empty or predictions.sample_id.duplicated().any() or predictions.model_id.nunique() != 1:
        raise ValueError("modality predictions need one model and unique sample IDs")
    heads = {}
    counts = {}
    for head, classes in HEADS.items():
        column = f"{head}_{modality}"
        role = f"split_{head}"
        selected = manifest.loc[
            (manifest.dataset == "crisismmd") & (manifest[role] == "test") & manifest[column].notna()
        ]
        if len(selected) != EXPECTED_ROWS or not (selected.split == "test").all():
            raise ValueError(f"{head} {modality} official task role differs from the fixed release")
        aligned = selected[["sample_id", column]].merge(
            predictions, on="sample_id", how="left", validate="one_to_one", indicator=True
        )
        if not (aligned._merge == "both").all():
            raise ValueError(f"{head} {modality} predictions do not cover the fixed test rows")
        gold = aligned[column].astype(str).to_numpy(dtype=object)
        guessed = np.array(
            [None if pd.isna(value) else str(value) for value in aligned[f"{head}_label"]], dtype=object
        )
        probs = aligned[[f"p_{head}__{name}" for name in classes]].to_numpy(dtype=float)
        heads[head] = _score(gold, guessed, classes, probs, bootstrap)
        counts[head] = len(selected)
    return {
        "baseline_id": "B1" if modality == "text" else "B2",
        "model_id": str(predictions.model_id.iloc[0]),
        "modality": modality,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": bootstrap,
        "selectors": counts,
        "heads": heads,
    }


def evaluate_modality_run(run_dir: Path, *, bootstrap: int = 1000) -> Path:
    """Freeze a standard baseline run before predicting per-modality test labels."""
    from sakshi.config import load
    from sakshi.models.train import TrainConfig

    run_dir = Path(run_dir)
    config = load(run_dir / "config.yaml", TrainConfig)
    if config.fold is not None or config.model.modality not in ("text", "image") or bootstrap < 1:
        raise ValueError("per-modality official evaluation requires a standard B1/B2 run")
    names = ("checkpoint.pt", "config.yaml", "label_maps.json", "temperatures.json", "thresholds.json")
    signature = {name: hash_file(run_dir / name) for name in names}
    signature.update(
        manifest=hash_file(paths.PROCESSED / "manifest.parquet"),
        features={
            name: hash_file(paths.PROCESSED / "features" / name) for name in (INDEX_FILE, TOKENS_FILE, EMBED_FILE)
        },
        evaluator=hash_file(Path(__file__)),
        modality=config.model.modality,
        bootstrap=bootstrap,
    )
    request = run_dir / "evaluation_modality_request.json"
    marker = run_dir / "evaluation_modality_complete.json"
    prediction_path = run_dir / "predictions_modality.parquet"
    metrics_path = run_dir / "metrics_modality.json"
    if request.exists():
        if json.loads(request.read_text(encoding="utf-8")) != signature:
            raise FileExistsError("modality evaluation inputs were frozen already and have changed")
    elif metrics_path.exists():
        raise FileExistsError("modality metrics exist without a freeze record")
    else:
        atomic.text(request, json.dumps(signature, indent=2) + "\n")
    if marker.exists():
        expected = {path.name: hash_file(path) for path in (prediction_path, metrics_path)}
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("completed modality evaluation artifacts changed")
        return metrics_path
    if not metrics_path.exists():
        manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
        mask = pd.Series(False, index=manifest.index)
        for head in HEADS:
            mask |= (manifest.dataset == "crisismmd") & (manifest[f"split_{head}"] == "test") & manifest[
                f"{head}_{config.model.modality}"
            ].notna()
        predictions = Predictor.from_run(run_dir).predict_rows(manifest.loc[mask])
        result = evaluate_modality_predictions(manifest, predictions, modality=config.model.modality, bootstrap=bootstrap)
        atomic.parquet(prediction_path, predictions)
        atomic.text(metrics_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not prediction_path.exists():
        raise ValueError("modality metrics exist but predictions are absent")
    atomic.text(
        marker,
        json.dumps({path.name: hash_file(path) for path in (prediction_path, metrics_path)}, indent=2) + "\n",
    )
    return metrics_path
