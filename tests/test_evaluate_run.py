import numpy as np
import pandas as pd

from sakshi.data import labels
from sakshi.models.evaluate_run import evaluate_predictions


def test_official_evaluation_schema_and_gold_gated_severity():
    groups = []
    for name, count, dataset, source, split_column, gold_column, gold_label in (
        ("inf", 1534, "crisismmd", "crisismmd", "split_informative", "y_informative", "informative"),
        ("hum", 955, "crisismmd", "crisismmd", "split_humanitarian", "y_humanitarian", "affected_individuals"),
        ("sev", 529, "crisismmd", "crisismmd", "split_damage", "y_severity", "mild_damage"),
        ("dmdsev", 1138, "cib", "dmd", "split_cib_damage", "y_severity", "mild_damage"),
        ("dt_aidr", 2091, "cib", "aidr_dt", "split_cib_disaster_types", "y_disaster_type", "fire"),
        ("dt_dmd", 1122, "cib", "dmd", "split_cib_disaster_types", "y_disaster_type", "fire"),
    ):
        groups.append(
            pd.DataFrame(
                {
                    "sample_id": [f"{name}:{i}" for i in range(count)],
                    "dataset": dataset,
                    "source": source,
                    "split": "test",
                    split_column: "test",
                    gold_column: gold_label,
                }
            )
        )
    manifest = pd.concat(groups, ignore_index=True)
    predictions = pd.DataFrame(
        {
            "sample_id": manifest.sample_id,
            "model_id": "perfect",
            "informative_label": "informative",
            "humanitarian_label": "affected_individuals",
            "severity_label": None,
            "disaster_type_label": "fire",
            "abstain": False,
        }
    )
    severity_rows = manifest.y_severity.notna()
    predictions.loc[severity_rows, "severity_label"] = "mild_damage"
    for head, classes, winner in (
        ("informative", labels.INFORMATIVE, "informative"),
        ("humanitarian", labels.HUMANITARIAN, "affected_individuals"),
        ("severity", labels.SEVERITY, "mild_damage"),
        ("disaster_type", labels.DISASTER_TYPE, "fire"),
    ):
        for label in classes:
            predictions[f"p_{head}__{label}"] = float(label == winner)
    report = evaluate_predictions(manifest, predictions, bootstrap=3)
    assert report["selectors"] == {
        "informative": 1534,
        "humanitarian": 955,
        "severity": 529,
        "severity_dmd": 1138,
        "disaster_type": 3213,
    }
    assert report["contradiction_rate"] == 0
    assert report["heads"]["severity"]["gold_gated"]["accuracy"]["point"] == 1
    assert report["heads"]["severity"]["predicted_gated"]["coverage"] == 1
    assert report["heads"]["disaster_type"]["aidr_dt"]["accuracy"]["point"] == 1
    assert report["abstention"]["coverage"] == 1
    assert np.isfinite(report["abstention"]["aurc"])
    predictions.loc[predictions.sample_id == "sev:0", "severity_label"] = None
    gated = evaluate_predictions(manifest, predictions, bootstrap=2)
    assert gated["heads"]["severity"]["gold_gated"]["accuracy"]["point"] == 1
    assert gated["heads"]["severity"]["predicted_gated"]["accuracy"]["point"] < 1
    assert gated["heads"]["severity"]["predicted_gated"]["coverage"] < 1

    text_predictions = predictions.loc[predictions.sample_id.str.startswith(("inf:", "hum:"))].copy()
    text_predictions = text_predictions.drop(
        columns=[name for name in text_predictions if name.startswith(("p_severity__", "p_disaster_type__"))]
    )
    text_predictions["severity_label"] = None
    text_report = evaluate_predictions(manifest, text_predictions, bootstrap=2, modality="text")
    assert set(text_report["heads"]) == {"informative", "humanitarian"}
    assert text_report["selectors"] == {"informative": 1534, "humanitarian": 955}
