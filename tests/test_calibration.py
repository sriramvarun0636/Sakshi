import numpy as np
import pytest
import torch

from sakshi.models.calibration import apply_temperature, fit_abstention, fit_temperature
from sakshi.models.losses import IGNORE


def test_temperature_round_trip_improves_dev_nll():
    # Overconfident logits with some mistakes need a softer distribution.
    logits = torch.tensor([[8.0, 0.0]] * 15 + [[0.0, 8.0]] * 5)
    gold = torch.tensor([0] * 12 + [1] * 3 + [1] * 4 + [0])
    fit = fit_temperature(logits, gold, min_dev_rows=10)
    assert fit.reason is None
    assert 1 < fit.temperature <= 20
    assert fit.nll_after < fit.nll_before
    calibrated = apply_temperature(logits, fit.temperature).softmax(1)
    assert torch.allclose(calibrated.sum(1), torch.ones(len(logits)))
    assert fit_temperature(logits, gold, min_dev_rows=100).temperature == 1
    assert fit_temperature(logits, torch.full_like(gold, IGNORE)).reason == "no_labels"


def test_abstention_grid_respects_coverage_and_hierarchy():
    p_inf = np.array([[0.1, 0.9]] * 8 + [[0.6, 0.4]] * 2)
    q_hum = np.array([[0.8, 0.1, 0.05, 0.05]] * 8 + [[0.25] * 4] * 2)
    gold = np.array([0] * 8 + [2] * 2)
    fit = fit_abstention(p_inf, q_hum, gold, target_coverage=0.9)
    assert fit.coverage >= 0.9
    assert fit.macro_f1 == pytest.approx(0.4)  # two correct classes among five canonical classes
    with pytest.raises(ValueError, match="probabilities must sum"):
        fit_abstention(p_inf * 0.5, q_hum, gold)
