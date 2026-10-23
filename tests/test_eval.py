import numpy as np
import pandas as pd
import pytest

from sakshi.eval import bootstrap, metrics, selectors


def test_official_selector_asserts_size_and_unified_test_membership():
    frame = pd.DataFrame(
        {
            "dataset": ["crisismmd"] * 1534,
            "split_informative": ["test"] * 1534,
            "y_informative": ["informative"] * 1534,
            "split": ["test"] * 1534,
        }
    )
    assert selectors.official(frame, "informative").sum() == 1534
    with pytest.raises(ValueError, match="expected 1534"):
        selectors.official(frame.iloc[:-1], "informative")
    frame.loc[0, "split"] = "train"
    with pytest.raises(ValueError, match="unified test"):
        selectors.official(frame, "informative")


def test_pool_uses_unified_split_only():
    frame = pd.DataFrame(
        {
            "dataset": ["crisismmd"] * 3,
            "split": ["train", "dev", "test"],
            "split_informative": ["test", "train", "dev"],
            "y_informative": ["informative", "informative", None],
        }
    )
    assert selectors.pool(frame, "informative", "train").tolist() == [True, False, False]
    assert selectors.pool(frame, "informative", "dev").tolist() == [False, True, False]


def test_classification_counts_unavailable_predictions_as_errors():
    result = metrics.classification(["a", "a", "b", "b"], ["a", "b", "b", None], ["a", "b"])
    assert result["confusion"] == {"labels": ["a", "b", "n/a"], "matrix": [[1, 1, 0], [0, 1, 1], [0, 0, 0]]}
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert result["per_class"]["b"]["support"] == 2
    with pytest.raises(ValueError, match="unknown"):
        metrics.classification(["a"], ["c"], ["a", "b"])


def test_calibration_ranking_and_risk_on_hand_computed_values():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    assert metrics.ece(probs, [0, 1], n_bins=2) == pytest.approx(0.25)
    assert metrics.brier(probs, [0, 1]) == pytest.approx(0.13)
    coverage, risk, aurc = metrics.risk_coverage([0.9, 0.8, 0.1], [True, False, True])
    assert coverage.tolist() == pytest.approx([1 / 3, 2 / 3, 1])
    assert risk.tolist() == pytest.approx([0, 0.5, 1 / 3])
    assert aurc == pytest.approx(2 / 9)
    assert metrics.ndcg_at_k([3, 2, 1], 3) == pytest.approx(1)
    assert metrics.ndcg_at_k([1, 0, 0], 3) == pytest.approx(1)
    assert metrics.ndcg_at_k([0, 0], 2) == 0


def test_bootstrap_is_seeded_and_paired():
    def metric(y, pred):
        return float((y == pred).mean())

    y = np.array([0, 1, 0, 1])
    a = y.copy()
    b = 1 - y
    assert bootstrap.ci(metric, y, a, n=30, seed=0) == (1.0, 1.0, 1.0)
    assert bootstrap.paired(metric, y, a, b, n=30, seed=0) == (1.0, 1.0, 1.0, 0.0)

    # A paired result may average several seed columns while resampling test items once.
    multi_a = np.stack([a, a], axis=1)
    multi_b = np.stack([b, b], axis=1)

    def seed_mean(gold, predictions):
        return float((gold[:, None] == predictions).mean())

    assert bootstrap.paired(seed_mean, y, multi_a, multi_b, n=30, seed=0) == (1.0, 1.0, 1.0, 0.0)
