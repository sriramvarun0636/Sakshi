"""B1/B2 official per-modality test labels are scored on the fixed task roles."""

import pandas as pd
import pytest

from sakshi.data import labels
from sakshi.models.evaluate_modality import EXPECTED_ROWS, evaluate_modality_predictions


def test_modality_scores_fixed_task_rows_and_rejects_incomplete_predictions():
    ids = [f"row-{index}" for index in range(EXPECTED_ROWS)]
    manifest = pd.DataFrame(
        {
            "sample_id": ids,
            "dataset": "crisismmd",
            "split": "test",
            "split_informative": "test",
            "split_humanitarian": "test",
            "informative_text": "not_informative",
            "humanitarian_text": "not_humanitarian",
            "informative_image": "not_informative",
            "humanitarian_image": "not_humanitarian",
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ids,
            "model_id": "b1-text-s0",
            "informative_label": "not_informative",
            "humanitarian_label": "not_humanitarian",
        }
    )
    for head, classes, winner in (
        ("informative", labels.INFORMATIVE, "not_informative"),
        ("humanitarian", labels.HUMANITARIAN, "not_humanitarian"),
    ):
        for name in classes:
            predictions[f"p_{head}__{name}"] = float(name == winner)
    report = evaluate_modality_predictions(manifest, predictions, modality="text", bootstrap=2)
    assert report["baseline_id"] == "B1"
    assert report["selectors"] == {"informative": EXPECTED_ROWS, "humanitarian": EXPECTED_ROWS}
    assert report["heads"]["informative"]["accuracy"]["point"] == 1
    assert evaluate_modality_predictions(manifest, predictions, modality="image", bootstrap=2)["baseline_id"] == "B2"
    with pytest.raises(ValueError, match="do not cover"):
        evaluate_modality_predictions(manifest, predictions.iloc[:-1], modality="text", bootstrap=2)
    bad_role = manifest.copy()
    bad_role.loc[0, "split"] = "train"
    with pytest.raises(ValueError, match="fixed release"):
        evaluate_modality_predictions(bad_role, predictions, modality="text", bootstrap=2)
