"""Dev-only temperature scaling and abstention threshold selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor
from torch.nn import functional as F

from sakshi.models.losses import IGNORE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float | None
    nll_after: float | None
    labelled_rows: int
    reason: str | None = None


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return logits.float() / temperature


def fit_temperature(
    logits: Tensor,
    targets: Tensor,
    *,
    min_dev_rows: int = 100,
    t_min: float = 0.05,
    t_max: float = 20.0,
) -> TemperatureFit:
    """Fit log temperature with LBFGS; use identity scaling for small or degraded fits."""
    if logits.ndim != 2 or targets.shape != logits.shape[:1]:
        raise ValueError("logits and targets have inconsistent shapes")
    if min_dev_rows < 1 or not 0 < t_min < 1 < t_max:
        raise ValueError("invalid calibration bounds")
    present = targets != IGNORE
    n = int(present.sum())
    if n == 0:
        log.warning("temperature scaling skipped: no labelled dev rows")
        return TemperatureFit(1.0, None, None, 0, "no_labels")
    values = logits[present].detach().float().cpu()
    gold = targets[present].detach().long().cpu()
    if not torch.isfinite(values).all() or torch.any(gold < 0) or torch.any(gold >= logits.shape[1]):
        raise ValueError("calibration inputs contain invalid logits or labels")
    before = float(F.cross_entropy(values, gold))
    if n < min_dev_rows:
        log.warning("temperature scaling skipped: %d dev rows < %d", n, min_dev_rows)
        return TemperatureFit(1.0, before, before, n, "too_few_rows")
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_t.clamp(np.log(t_min), np.log(t_max)).exp()
        loss = F.cross_entropy(values / temperature, gold)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        temperature = float(log_t.detach().clamp(np.log(t_min), np.log(t_max)).exp())
        after = float(F.cross_entropy(values / temperature, gold))
    except (RuntimeError, ValueError) as exc:
        log.warning("temperature scaling failed; retaining T=1: %s", exc)
        return TemperatureFit(1.0, before, before, n, "optimisation_failed")
    if not np.isfinite(after) or after > before + 1e-7:
        log.warning("temperature scaling worsened NLL; retaining T=1")
        return TemperatureFit(1.0, before, before, n, "nll_worsened")
    return TemperatureFit(temperature, before, after, n)


@dataclass(frozen=True)
class AbstentionFit:
    tau_inf: float
    tau_hum: float
    coverage: float
    macro_f1: float


def fit_abstention(
    p_inf: np.ndarray,
    q_hum: np.ndarray,
    gold_humanitarian: np.ndarray,
    *,
    target_coverage: float = 0.9,
    grid: tuple[float, ...] = tuple(float(x) for x in np.arange(0.5, 0.951, 0.05)),
) -> AbstentionFit:
    """Choose dev thresholds with coverage >= target, then maximum selective macro-F1.

    Humanitarian gold indices follow ``sakshi.data.labels.HUMANITARIAN``. Inputs must
    already be temperature-scaled probabilities and use canonical class order.
    """
    if p_inf.ndim != 2 or p_inf.shape[1] != 2 or q_hum.shape != (len(p_inf), 4):
        raise ValueError("expected N×2 informativeness and N×4 humanitarian probabilities")
    if gold_humanitarian.shape != (len(p_inf),) or not 0 < target_coverage <= 1:
        raise ValueError("gold labels or target coverage are invalid")
    if len(p_inf) == 0 or not grid or any(not 0.5 <= t <= 1 for t in grid):
        raise ValueError("nonempty dev probabilities and thresholds in [0.5, 1] are required")
    if not np.isfinite(p_inf).all() or not np.isfinite(q_hum).all():
        raise ValueError("probabilities must be finite")
    if not np.allclose(p_inf.sum(1), 1, atol=1e-5) or not np.allclose(q_hum.sum(1), 1, atol=1e-5):
        raise ValueError("probabilities must sum to one")
    labelled = gold_humanitarian != IGNORE
    if not labelled.any():
        raise ValueError("at least one humanitarian dev label is required")
    p_inf, q_hum, gold = p_inf[labelled], q_hum[labelled], gold_humanitarian[labelled]
    informative = p_inf[:, 1] >= 0.5
    # Conditional order omits not_humanitarian (index 2 in the five-class list).
    conditional_indices = np.array([0, 1, 3, 4])
    predicted = np.where(informative, conditional_indices[q_hum.argmax(1)], 2)
    confidence_inf = p_inf.max(1)
    confidence_hum = q_hum.max(1)
    best: AbstentionFit | None = None
    for tau_inf in grid:
        for tau_hum in grid:
            retained = (confidence_inf >= tau_inf) & (~informative | (confidence_hum >= tau_hum))
            coverage = float(retained.mean())
            if coverage + 1e-12 < target_coverage:
                continue
            score = float(f1_score(gold[retained], predicted[retained], labels=list(range(5)), average="macro", zero_division=0))
            candidate = AbstentionFit(tau_inf, tau_hum, coverage, score)
            if best is None or (score, coverage, -tau_inf, -tau_hum) > (
                best.macro_f1,
                best.coverage,
                -best.tau_inf,
                -best.tau_hum,
            ):
                best = candidate
    if best is None:
        raise ValueError("no threshold pair meets target coverage")
    return best
