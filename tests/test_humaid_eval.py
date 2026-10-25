import pandas as pd

from sakshi.data import labels
from sakshi.eval.humaid import FULL, STRICT, score_humaid


def test_humaid_strict_and_full_mapping_keep_distinct_denominators():
    assert "caution_and_advice" not in STRICT and FULL["caution_and_advice"] == "other_relevant_information"
    rows = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "group": ["unseen", "unseen", "shared", "shared"],
            "class_label": [
                "not_humanitarian",
                "caution_and_advice",
                "infrastructure_and_utility_damage",
                "requests_or_urgent_needs",
            ],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": rows.sample_id,
            "informative_label": ["not_informative", "informative", "informative", "informative"],
            "humanitarian_label": [
                "not_humanitarian",
                "other_relevant_information",
                "infrastructure_and_utility_damage",
                "rescue_volunteering_or_donation_effort",
            ],
        }
    )
    for head, classes in (("informative", labels.INFORMATIVE), ("humanitarian", labels.HUMANITARIAN)):
        for name in classes:
            predictions[f"p_{head}__{name}"] = [
                0.8 if value == name else 0.2 / (len(classes) - 1) for value in predictions[f"{head}_label"]
            ]
    result = score_humaid(rows, predictions, bootstrap=10)
    assert result["groups"]["unseen"]["strict"]["n"] == 1
    assert result["groups"]["unseen"]["full"]["n"] == 2
    assert result["groups"]["shared"]["strict"]["n"] == 1
    assert result["groups"]["shared"]["full"]["n"] == 2
    assert result["groups"]["unseen"]["full"]["heads"]["informative"]["accuracy"]["point"] == 1
