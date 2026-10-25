"""Official WP1 test selectors, calibrated metrics and hierarchy checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.data import labels
from sakshi.eval import metrics, selectors
from sakshi.models.predict import Predictor

GOLD_COLUMNS = {
    "informative": "y_informative",
    "humanitarian": "y_humanitarian",
    "severity": "y_severity",
    "severity_dmd": "y_severity",
    "disaster_type": "y_disaster_type",
}


def _score(gold: np.ndarray, predicted: np.ndarray, classes: list[str], probs: np.ndarray, bootstrap: int) -> dict:
    """One head's classification, calibration and item-bootstrap intervals."""
    if bootstrap < 1:
        raise ValueError("bootstrap count must be positive")
    point = metrics.classification(gold, predicted, classes)
    rng = np.random.default_rng(0)
    draws = {metric: np.empty(bootstrap) for metric in ("macro_f1", "weighted_f1", "accuracy")}
    for index in range(bootstrap):
        take = rng.integers(0, len(gold), len(gold))
        result = metrics.classification(gold[take], predicted[take], classes)
        for metric in draws:
            draws[metric][index] = result[metric]
    output = {
        metric: {
            "point": point[metric],
            "lo": float(np.quantile(draws[metric], 0.025)),
            "hi": float(np.quantile(draws[metric], 0.975)),
        }
        for metric in draws
    }
    output["per_class"] = point["per_class"]
    output["confusion"] = point["confusion"]
    if probs.shape != (len(gold), len(classes)) or not np.isfinite(probs).all():
        raise ValueError("all evaluated rows need finite probabilities in canonical class order")
    truth = np.array([classes.index(value) for value in gold])
    output["ece"] = metrics.ece(probs, truth)
    output["brier"] = metrics.brier(probs, truth)
    output["nll"] = float(-np.log(np.clip(probs[np.arange(len(gold)), truth], 1e-12, 1)).mean())
    return output


def _probabilities(frame: pd.DataFrame, head: str, classes: list[str]) -> np.ndarray:
    return frame[[f"p_{head}__{name}" for name in classes]].to_numpy(dtype=np.float64)


def _head_frame(manifest: pd.DataFrame, predictions: pd.DataFrame, head: str) -> pd.DataFrame:
    gold = manifest.loc[selectors.official(manifest, head), ["sample_id", "source", GOLD_COLUMNS[head]]]
    aligned = gold.merge(predictions, on="sample_id", how="left", validate="one_to_one", indicator=True)
    if not (aligned._merge == "both").all():
        raise ValueError(f"predictions are missing {int((aligned._merge != 'both').sum())} official {head} rows")
    return aligned.drop(columns="_merge")


def _classification_columns(frame: pd.DataFrame, head: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = "severity" if head == "severity_dmd" else head
    classes = selectors.HEAD_LABELS[head]
    gold = frame[GOLD_COLUMNS[head]].astype(str).to_numpy(dtype=object)
    predicted = np.array([None if pd.isna(value) else str(value) for value in frame[f"{base}_label"]], dtype=object)
    return gold, predicted, _probabilities(frame, base, classes)


def evaluate_predictions(
    manifest: pd.DataFrame, predictions: pd.DataFrame, *, bootstrap: int = 1000, modality: str = "both"
) -> dict:
    """Score one prediction table using the exact published-compatible selectors."""
    if predictions.sample_id.duplicated().any() or predictions.empty:
        raise ValueError("prediction table must have unique sample IDs and at least one row")
    if predictions.model_id.nunique() != 1:
        raise ValueError("prediction table must contain one model_id")
    if modality not in ("both", "text", "image"):
        raise ValueError("modality must be both, text, or image")
    enabled = ("informative", "humanitarian") if modality == "text" else tuple(selectors.HEAD_LABELS)
    frames = {head: _head_frame(manifest, predictions, head) for head in enabled}
    heads = {}
    for head in ("informative", "humanitarian", "severity_dmd"):
        if head not in frames:
            continue
        gold, pred, probs = _classification_columns(frames[head], head)
        heads[head] = _score(gold, pred, selectors.HEAD_LABELS[head], probs, bootstrap)

    if "severity" in frames:
        severity = frames["severity"]
        gold, pred, probs = _classification_columns(severity, "severity")
        gold_gated = np.array(labels.SEVERITY, dtype=object)[probs.argmax(1)]
        heads["severity"] = {
            "gold_gated": _score(gold, gold_gated, labels.SEVERITY, probs, bootstrap),
            "predicted_gated": {
                **_score(gold, pred, labels.SEVERITY, probs, bootstrap),
                "coverage": float(np.mean([value is not None for value in pred])),
            },
        }

        disaster = frames["disaster_type"]
        disaster_scores = {}
        for source, part in (
            ("all", disaster),
            ("aidr_dt", disaster[disaster.source == "aidr_dt"]),
            ("dmd", disaster[disaster.source == "dmd"]),
        ):
            if part.empty:
                raise ValueError(f"disaster type selector has no {source} rows")
            gold, pred, probs = _classification_columns(part, "disaster_type")
            disaster_scores[source] = _score(gold, pred, labels.DISASTER_TYPE, probs, bootstrap)
        heads["disaster_type"] = disaster_scores

    inf = predictions.informative_label
    hum = predictions.humanitarian_label
    sev = predictions.severity_label
    contradiction = ((inf == "not_informative") & ((hum != "not_humanitarian") | sev.notna())) | (
        (hum == "not_humanitarian") & sev.notna()
    )
    contradiction_rate = float(contradiction.mean())
    if contradiction_rate:
        raise ValueError(f"hierarchical prediction contradiction rate is {contradiction_rate}")

    humanitarian = frames["humanitarian"]
    gold, pred, _ = _classification_columns(humanitarian, "humanitarian")
    retained = ~humanitarian.abstain.to_numpy(dtype=bool)
    confidence_inf = _probabilities(humanitarian, "informative", labels.INFORMATIVE).max(1)
    q_hum = _probabilities(humanitarian, "humanitarian", labels.HUMANITARIAN)
    p_inf = humanitarian["p_informative__informative"].to_numpy(dtype=float)
    conditional_confidence = np.divide(q_hum[:, [0, 1, 3, 4]].max(1), p_inf, out=np.zeros(len(p_inf)), where=p_inf > 0)
    confidence = np.where(
        humanitarian.informative_label.to_numpy() == "informative",
        np.minimum(confidence_inf, conditional_confidence),
        confidence_inf,
    )
    _, _, aurc = metrics.risk_coverage(confidence, gold == pred)
    errors = gold != pred
    abstention = {
        "coverage": float(retained.mean()),
        "selective_macro_f1": metrics.classification(gold[retained], pred[retained], labels.HUMANITARIAN)["macro_f1"]
        if retained.any()
        else None,
        "selective_accuracy": float((gold[retained] == pred[retained]).mean()) if retained.any() else None,
        "errors_routed_to_review": float((errors & ~retained).sum() / errors.sum()) if errors.any() else 0.0,
        "aurc": aurc,
    }
    return {
        "model_id": str(predictions.model_id.iloc[0]),
        "modality": modality,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selectors": {head: len(frame) for head, frame in frames.items()},
        "heads": heads,
        "contradiction_rate": contradiction_rate,
        "abstention": abstention,
    }


def evaluate_run(run_dir: Path, *, bootstrap: int = 1000) -> Path:
    """Refresh calibrated official predictions, then write metrics_test.json."""
    run_dir = Path(run_dir)
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    from sakshi.config import load
    from sakshi.models.train import TrainConfig

    config = load(run_dir / "config.yaml", TrainConfig)
    prediction_file = run_dir / "predictions_official.parquet"
    mask = pd.Series(False, index=manifest.index)
    active = ("informative", "humanitarian") if config.model.modality == "text" else tuple(selectors.HEAD_LABELS)
    for head in active:
        mask |= selectors.official(manifest, head)
    predictions = Predictor.from_run(run_dir).predict_rows(manifest.loc[mask])
    atomic.parquet(prediction_file, predictions)
    result = evaluate_predictions(manifest, predictions, bootstrap=bootstrap, modality=config.model.modality)
    env_file = run_dir / "env.json"
    if env_file.exists():
        result["git_commit"] = json.loads(env_file.read_text(encoding="utf-8")).get("git_commit")
    output = run_dir / "metrics_test.json"
    atomic.text(output, json.dumps(result, indent=2) + "\n")
    return output
