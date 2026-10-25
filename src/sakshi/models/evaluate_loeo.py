"""Frozen evaluation on the event held out by one LOEO classifier run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.data import labels
from sakshi.data.events import EVENTS
from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE
from sakshi.models.predict import Predictor

HEADS = {
    "informative": ("y_informative", labels.INFORMATIVE),
    "humanitarian": ("y_humanitarian", labels.HUMANITARIAN),
    "severity": ("y_severity", labels.SEVERITY),
}


def _score(gold: np.ndarray, predicted: np.ndarray, classes: list[str], *, bootstrap: int) -> dict:
    """Use the fold's fixed present-gold classes for every bootstrap draw."""
    from sakshi.eval.metrics import classification

    present = [name for name in classes if name in set(gold)]
    if not present:
        raise ValueError("LOEO head has no present gold class")

    def point(y: np.ndarray, guess: np.ndarray) -> tuple[float, dict]:
        detail = classification(y, guess, classes)
        return float(np.mean([detail["per_class"][name]["f1"] for name in present])), detail

    value, detail = point(gold, predicted)
    rng = np.random.default_rng(0)
    draws = np.empty(bootstrap)
    for i in range(bootstrap):
        chosen = rng.integers(0, len(gold), len(gold))
        draws[i] = point(gold[chosen], predicted[chosen])[0]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "n": len(gold),
        "present_classes": present,
        "macro_f1": {"point": value, "lo": float(low), "hi": float(high)},
        "weighted_f1": detail["weighted_f1"],
        "accuracy": detail["accuracy"],
        "per_class": detail["per_class"],
        "confusion": detail["confusion"],
    }


def evaluate_loeo_predictions(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    event: str,
    bootstrap: int = 1000,
    modality: str = "both",
) -> dict:
    """Score one fold's held-out event without using any other event's official rows."""
    if event not in EVENTS or bootstrap < 1 or modality not in ("both", "text", "image"):
        raise ValueError("LOEO evaluation needs a known event and positive bootstrap count")
    column = f"loeo_{event}"
    if column not in manifest or predictions.sample_id.duplicated().any() or predictions.empty:
        raise ValueError("LOEO manifest or unique predictions are missing")
    test = manifest.loc[manifest[column] == "test"]
    if test.empty or not (test.event == event).all() or not (test.dataset == "crisismmd").all():
        raise ValueError("LOEO test role must contain only the held-out CrisisMMD event")
    if ((manifest.event == event) & manifest[column].isin(("train", "dev"))).any():
        raise ValueError("held-out event leaked into the LOEO training or development role")
    if set(predictions.sample_id) != set(test.sample_id) or predictions.model_id.nunique() != 1:
        raise ValueError("LOEO predictions must cover exactly one held-out event and model")
    aligned = test.merge(predictions, on="sample_id", how="left", validate="one_to_one", indicator=True)
    if not (aligned._merge == "both").all():
        raise ValueError("LOEO predictions failed one-to-one alignment")
    heads = {}
    for head, (gold_column, classes) in HEADS.items():
        if modality == "text" and head == "severity":
            continue
        frame = aligned.loc[aligned[gold_column].notna()]
        if frame.empty:
            raise ValueError(f"LOEO {event} has no gold {head} rows")
        gold = frame[gold_column].astype(str).to_numpy(dtype=object)
        guessed = np.array(
            [None if pd.isna(value) else str(value) for value in frame[f"{head}_label"]], dtype=object
        )
        if head == "severity":
            probabilities = frame[[f"p_severity__{name}" for name in classes]].to_numpy(dtype=float)
            if not np.isfinite(probabilities).all():
                raise ValueError("LOEO severity has missing gold-gated probabilities")
            gold_gated = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
            heads["severity"] = {
                "gold_gated": _score(gold, gold_gated, classes, bootstrap=bootstrap),
                "predicted_gated": {
                    **_score(gold, guessed, classes, bootstrap=bootstrap),
                    "coverage": float(np.mean([value is not None for value in guessed])),
                },
            }
        else:
            heads[head] = _score(gold, guessed, classes, bootstrap=bootstrap)
    inf = predictions.informative_label
    hum = predictions.humanitarian_label
    sev = predictions.severity_label
    contradiction = ((inf == "not_informative") & ((hum != "not_humanitarian") | sev.notna())) | (
        (hum == "not_humanitarian") & sev.notna()
    )
    rate = float(contradiction.mean())
    if rate:
        raise ValueError(f"LOEO hierarchy contradiction rate is {rate}")
    return {
        "model_id": str(predictions.model_id.iloc[0]),
        "event": event,
        "modality": modality,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_test_rows": len(test),
        "bootstrap": bootstrap,
        "heads": heads,
        "contradiction_rate": rate,
    }


def evaluate_loeo_run(run_dir: Path, *, bootstrap: int = 1000) -> Path:
    """Freeze inputs once, then predict and score only the model's held-out event."""
    from sakshi.config import load
    from sakshi.models.train import TrainConfig

    run_dir = Path(run_dir)
    config = load(run_dir / "config.yaml", TrainConfig)
    if config.fold is None:
        raise ValueError("LOEO evaluation requires a run trained with a held-out fold")
    event = config.fold.removeprefix("loeo_")
    if event not in EVENTS or bootstrap < 1:
        raise ValueError("LOEO evaluation needs a known event and positive bootstrap count")
    required = ("checkpoint.pt", "config.yaml", "label_maps.json", "temperatures.json", "thresholds.json")
    signature = {name: hash_file(run_dir / name) for name in required}
    signature["manifest"] = hash_file(paths.PROCESSED / "manifest.parquet")
    signature["features"] = {
        name: hash_file(paths.PROCESSED / "features" / name) for name in (INDEX_FILE, TOKENS_FILE, EMBED_FILE)
    }
    signature.update(event=event, bootstrap=bootstrap)
    request = run_dir / "evaluation_loeo_request.json"
    marker = run_dir / "evaluation_loeo_complete.json"
    prediction_path = run_dir / "predictions_loeo.parquet"
    metrics_path = run_dir / "metrics_loeo.json"
    if request.exists():
        if json.loads(request.read_text(encoding="utf-8")) != signature:
            raise FileExistsError("LOEO evaluation inputs were frozen already and have changed")
    elif metrics_path.exists():
        raise FileExistsError("LOEO metrics exist without a freeze record")
    else:
        atomic.text(request, json.dumps(signature, indent=2) + "\n")
    if marker.exists():
        expected = {path.name: hash_file(path) for path in (prediction_path, metrics_path)}
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("completed LOEO evaluation artifacts changed")
        return metrics_path
    if not metrics_path.exists():
        manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
        heldout = manifest.loc[manifest[f"loeo_{event}"] == "test"]
        predictions = Predictor.from_run(run_dir).predict_rows(heldout)
        result = evaluate_loeo_predictions(
            manifest, predictions, event=event, bootstrap=bootstrap, modality=config.model.modality
        )
        atomic.parquet(prediction_path, predictions)
        atomic.text(metrics_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not prediction_path.exists():
        raise ValueError("LOEO metrics exist but predictions are absent")
    atomic.text(
        marker,
        json.dumps({path.name: hash_file(path) for path in (prediction_path, metrics_path)}, indent=2) + "\n",
    )
    return metrics_path
