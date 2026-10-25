import numpy as np
import pandas as pd
import pytest

from sakshi.baselines.late_fusion import fit_weights, fuse_predictions
from sakshi.data import labels
from sakshi.models.fusion import HUMANITARIAN_CONDITIONAL


def predictions(modality, ids, informative, conditional):
    rows = []
    for sample_id, p, q in zip(ids, informative, conditional, strict=True):
        row = {"sample_id": sample_id, "model_id": f"b-{modality}"}
        row.update({f"p_informative__{name}": value for name, value in zip(labels.INFORMATIVE, [1 - p, p], strict=True)})
        row.update(
            {f"p_humanitarian__{name}": (1 - p if name == "not_humanitarian" else p * q[name]) for name in labels.HUMANITARIAN}
        )
        row.update({f"p_severity__{name}": 1 / len(labels.SEVERITY) for name in labels.SEVERITY})
        row.update({f"p_disaster_type__{name}": 1 / len(labels.DISASTER_TYPE) for name in labels.DISASTER_TYPE})
        rows.append(row)
    return pd.DataFrame(rows)


def test_late_fusion_tunes_on_dev_and_preserves_hierarchy_with_image_only_rows():
    ids = ["a", "b", "c", "d"]
    correct = [
        dict.fromkeys(HUMANITARIAN_CONDITIONAL, 0.05) | {"infrastructure_and_utility_damage": 0.85},
        dict.fromkeys(HUMANITARIAN_CONDITIONAL, 0.25),
        dict.fromkeys(HUMANITARIAN_CONDITIONAL, 0.05) | {"rescue_volunteering_or_donation_effort": 0.85},
        dict.fromkeys(HUMANITARIAN_CONDITIONAL, 0.05) | {"other_relevant_information": 0.85},
    ]
    wrong = [dict.fromkeys(HUMANITARIAN_CONDITIONAL, 0.05) | {"affected_individuals": 0.85}] * 4
    text = predictions("text", ids, [0.9, 0.1, 0.9, 0.9], correct)
    image = predictions("image", ids, [0.1, 0.9, 0.1, 0.1], wrong)
    dev = pd.DataFrame(
        {
            "sample_id": ids,
            "y_informative": ["informative", "not_informative", "informative", "informative"],
            "y_humanitarian": [
                "infrastructure_and_utility_damage",
                "not_humanitarian",
                "rescue_volunteering_or_donation_effort",
                "other_relevant_information",
            ],
        }
    )
    weights = fit_weights(dev, text, image)
    assert weights["weight_informative"] >= 0.5
    assert weights["weight_humanitarian_conditional"] >= 0.5
    fused = fuse_predictions(text.iloc[:3], image, weights, model_id="b3-fixture")
    assert set(fused.sample_id) == set(ids)
    assert fused.set_index("sample_id").loc["d", "informative_label"] == "not_informative"
    assert fused.set_index("sample_id").loc["d", "alpha_image"] == 1
    assert not ((fused.informative_label == "not_informative") & (fused.humanitarian_label != "not_humanitarian")).any()
    assert np.isfinite(fused[[f"p_severity__{name}" for name in labels.SEVERITY]].to_numpy()).all()
    with pytest.raises(ValueError, match="both baselines"):
        fit_weights(dev, text.iloc[:3], image)
