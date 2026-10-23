import pandas as pd
import pytest

from sakshi.eval import tables
from sakshi.models.oracle import OraclePredictor


def test_oracle_obeys_hierarchy_and_missing_image_contract():
    rows = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "text": "hello",
                "image_path": "a.jpg",
                "y_informative": "not_informative",
                "y_humanitarian": "not_humanitarian",
                "y_severity": None,
                "y_disaster_type": None,
            },
            {
                "sample_id": "b",
                "text": "help",
                "image_path": None,
                "y_informative": "informative",
                "y_humanitarian": "affected_individuals",
                "y_severity": None,
                "y_disaster_type": None,
            },
        ]
    )
    first, second = OraclePredictor().predict_rows(rows)
    assert first.humanitarian.label == "not_humanitarian" and first.humanitarian.status == "gated"
    assert first.severity.label is None and first.severity.status == "gated"
    assert second.severity.status == "unavailable" and second.disaster_type.status == "unavailable"
    assert second.modality_weights == {"text": 1.0, "image": 0.0}


def test_oracle_rejects_duplicate_ids():
    rows = pd.DataFrame({"sample_id": ["a", "a"], "text": ["x", "x"], "image_path": ["a", "a"]})
    with pytest.raises(ValueError, match="duplicate"):
        OraclePredictor().predict_rows(rows)


def test_results_table_is_ordered_and_escapes_content():
    spec = {
        "title": "Scores",
        "columns": [{"key": "name", "header": "Model"}, {"key": "scores.f1", "header": "F1", "decimals": 2}],
    }
    assert tables.render([{"name": "A|B", "scores": {"f1": 0.1234}}], spec) == (
        "## Scores\n\n| Model | F1 |\n| --- | --- |\n| A\\|B | 0.12 |\n"
    )
