"""Deterministic metrics with canonical class order and explicit input checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def classification(y_true: Sequence[str], y_pred: Sequence[str | None], labels: list[str]) -> dict:
    """Classification metrics. A null prediction counts as wrong and appears as n/a."""
    gold, pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    if gold.ndim != 1 or pred.ndim != 1 or len(gold) != len(pred) or not len(gold):
        raise ValueError("gold and prediction arrays must be nonempty aligned vectors")
    if len(labels) != len(set(labels)):
        raise ValueError("labels must be unique")
    if set(gold) - set(labels):
        raise ValueError(f"gold contains unknown labels: {sorted(set(gold) - set(labels))}")
    unknown_pred = set(pred) - set(labels) - {None}
    if unknown_pred:
        raise ValueError(f"predictions contain unknown labels: {sorted(unknown_pred)}")

    all_labels = [*labels, "n/a"] if any(x is None for x in pred) else labels
    matrix = np.zeros((len(all_labels), len(all_labels)), dtype=np.int64)
    indices = {label: i for i, label in enumerate(all_labels)}
    for truth, guess in zip(gold, pred, strict=True):
        matrix[indices[truth], indices[guess if guess is not None else "n/a"]] += 1

    per_class = {}
    for label in labels:
        i = indices[label]
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    n = len(gold)
    return {
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "weighted_f1": float(sum(v["f1"] * v["support"] for v in per_class.values()) / n),
        "accuracy": float(np.trace(matrix) / n),
        "per_class": per_class,
        "confusion": {"labels": all_labels, "matrix": matrix.tolist()},
    }


def _probabilities(probs: np.ndarray, y_true: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if p.ndim != 2 or y.ndim != 1 or not len(y) or len(p) != len(y):
        raise ValueError("probs must be N×C and y_true a nonempty aligned vector of class indices")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("each probability row must be finite, nonnegative and sum to one")
    if ((y < 0) | (y >= p.shape[1])).any():
        raise ValueError("gold class index outside the probability columns")
    return p, y


def ece(probs: np.ndarray, y_true: Sequence[int], n_bins: int = 15) -> float:
    """Expected calibration error using equal-width confidence bins."""
    p, y = _probabilities(probs, y_true)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return float(
        sum(
            abs(float(correct[bins == b].mean()) - float(confidence[bins == b].mean())) * int((bins == b).sum()) / len(y)
            for b in range(n_bins)
            if (bins == b).any()
        )
    )


def brier(probs: np.ndarray, y_true: Sequence[int]) -> float:
    """Mean multiclass squared probability error (sum over classes per row)."""
    p, y = _probabilities(probs, y_true)
    target = np.eye(p.shape[1], dtype=np.float64)[y]
    return float(np.square(p - target).sum(axis=1).mean())


def risk_coverage(confidence: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray, float]:
    """Risk as the least confident posts are removed; AURC integrates from 0 to 1."""
    conf, ok = np.asarray(confidence, dtype=np.float64), np.asarray(correct, dtype=bool)
    if conf.ndim != 1 or ok.ndim != 1 or not len(conf) or len(conf) != len(ok) or not np.isfinite(conf).all():
        raise ValueError("confidence and correct must be nonempty aligned finite vectors")
    order = np.argsort(-conf, kind="stable")
    coverage = np.arange(1, len(conf) + 1, dtype=np.float64) / len(conf)
    risk = 1 - np.cumsum(ok[order]) / np.arange(1, len(conf) + 1)
    aurc = float(np.trapezoid(np.r_[0.0, risk], np.r_[0.0, coverage]))
    return coverage, risk, aurc


def ndcg_at_k(relevance_in_ranked_order: Sequence[float], k: int) -> float:
    """Normalised discounted cumulative gain, returning zero when ideal gain is zero."""
    values = np.asarray(relevance_in_ranked_order, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any() or k < 1:
        raise ValueError("relevance must be a finite nonnegative vector and k positive")
    top = values[:k]
    discounts = np.log2(np.arange(2, len(top) + 2))
    dcg = float(((2**top - 1) / discounts).sum())
    ideal = np.sort(values)[::-1][:k]
    idcg = float(((2**ideal - 1) / discounts).sum())
    return dcg / idcg if idcg else 0.0
