"""Event isolation and fixed-present-class scoring for LOEO evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import pytest
import yaml

from sakshi import paths
from sakshi.data import labels
from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE
from sakshi.models.evaluate_loeo import evaluate_loeo_predictions, evaluate_loeo_run
from sakshi.models.predict import Predictor
from sakshi.models.train import TrainConfig


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    event = "hurricane_harvey"
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "training"],
            "event": [event, event, event, "hurricane_irma"],
            "dataset": ["crisismmd"] * 4,
            f"loeo_{event}": ["test", "test", "test", "train"],
            "y_informative": ["informative", "not_informative", "informative", "informative"],
            "y_humanitarian": ["affected_individuals", "not_humanitarian", "affected_individuals", pd.NA],
            "y_severity": ["severe_damage", pd.NA, "mild_damage", pd.NA],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["c", "a", "b"],
            "model_id": ["loeo-model"] * 3,
            "informative_label": ["informative", "informative", "not_informative"],
            "humanitarian_label": ["affected_individuals", "affected_individuals", "not_humanitarian"],
            "severity_label": ["mild_damage", "mild_damage", None],
        }
    )
    for name in labels.SEVERITY:
        predictions[f"p_severity__{name}"] = [
            0.05 if name != "mild_damage" else 0.9,
            0.05 if name != "severe_damage" else 0.9,
            0.0,
        ]
    return manifest, predictions


def test_loeo_scores_only_held_out_event_and_gold_gated_severity():
    manifest, predictions = _tables()
    result = evaluate_loeo_predictions(manifest, predictions, event="hurricane_harvey", bootstrap=10)
    assert result["n_test_rows"] == 3
    assert result["contradiction_rate"] == 0
    assert result["heads"]["informative"]["macro_f1"]["point"] == 1
    assert result["heads"]["severity"]["gold_gated"]["macro_f1"]["point"] == 1
    assert result["heads"]["severity"]["predicted_gated"]["macro_f1"]["point"] == pytest.approx(1 / 3)


def test_loeo_rejects_training_leakage_and_foreign_prediction():
    manifest, predictions = _tables()
    leaked = manifest.copy()
    leaked.loc[leaked.sample_id == "a", "loeo_hurricane_harvey"] = "train"
    with pytest.raises(ValueError, match="leaked"):
        evaluate_loeo_predictions(leaked, predictions.iloc[1:], event="hurricane_harvey", bootstrap=2)
    with pytest.raises(ValueError, match="exactly one held-out event"):
        evaluate_loeo_predictions(manifest, predictions.iloc[1:], event="hurricane_harvey", bootstrap=2)


def test_text_only_loeo_reports_only_supported_heads():
    manifest, predictions = _tables()
    predictions = predictions.drop(columns=[name for name in predictions if name.startswith("p_severity__")])
    predictions["severity_label"] = None
    result = evaluate_loeo_predictions(
        manifest, predictions, event="hurricane_harvey", bootstrap=2, modality="text"
    )
    assert result["modality"] == "text"
    assert set(result["heads"]) == {"informative", "humanitarian"}


def test_loeo_run_freezes_inputs_and_reuses_completed_metrics(tmp_path, monkeypatch):
    manifest, predictions = _tables()
    processed = tmp_path / "processed"
    features = processed / "features"
    features.mkdir(parents=True)
    manifest.to_parquet(processed / "manifest.parquet")
    for name in (INDEX_FILE, TOKENS_FILE, EMBED_FILE):
        (features / name).write_bytes(b"fixed fixture")
    monkeypatch.setattr(paths, "PROCESSED", processed)

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(asdict(TrainConfig(fold="loeo_hurricane_harvey"))))
    for name in ("checkpoint.pt", "label_maps.json", "temperatures.json", "thresholds.json"):
        (run / name).write_bytes(b"fixed fixture")

    calls = []

    class FakePredictor:
        def predict_rows(self, rows):
            calls.append(len(rows))
            return predictions

    monkeypatch.setattr(Predictor, "from_run", classmethod(lambda cls, directory: FakePredictor()))
    metrics = evaluate_loeo_run(run, bootstrap=2)
    assert calls == [3]
    assert json.loads(metrics.read_text())["event"] == "hurricane_harvey"
    assert evaluate_loeo_run(run, bootstrap=2) == metrics
    assert calls == [3]
    metrics.write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        evaluate_loeo_run(run, bootstrap=2)
