"""Item-level percentile and paired bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _arrays(*arrays: object) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(a) for a in arrays)
    if not converted or not len(converted[0]) or any(a.ndim < 1 or len(a) != len(converted[0]) for a in converted):
        raise ValueError("bootstrap inputs must be nonempty arrays aligned on their first dimension")
    return converted


def ci(
    metric_fn: Callable[..., float], *arrays: object, n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Point estimate and two-sided percentile interval from item resampling."""
    data = _arrays(*arrays)
    if n < 1 or not 0 < alpha < 1:
        raise ValueError("n must be positive and alpha between zero and one")
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*data))
    draws = np.empty(n, dtype=np.float64)
    for i in range(n):
        take = rng.integers(0, len(data[0]), len(data[0]))
        draws[i] = metric_fn(*(a[take] for a in data))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y: object,
    pred_a: object,
    pred_b: object,
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired A−B effect, percentile interval, and two-sided bootstrap p-value."""
    gold, a, b = _arrays(y, pred_a, pred_b)
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    point = float(metric_fn(gold, a) - metric_fn(gold, b))
    draws = np.empty(n, dtype=np.float64)
    for i in range(n):
        take = rng.integers(0, len(gold), len(gold))
        draws[i] = metric_fn(gold[take], a[take]) - metric_fn(gold[take], b[take])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    p = min(1.0, 2 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
    return point, float(lo), float(hi), p
