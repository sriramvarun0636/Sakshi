"""Masked, class-balanced focal losses for the classifier heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from sakshi.models.fusion import HEAD_CLASSES

IGNORE = -100
MAIN_HEADS = frozenset(("informative", "humanitarian", "severity", "disaster_type"))
IMAGE_HEADS = frozenset(("severity", "disaster_type", "informative_image", "humanitarian_image"))
TEXT_HEADS = frozenset(("informative_text", "humanitarian_text"))


def class_weights(counts: Tensor, power: float = 0.5) -> Tensor:
    """Inverse-frequency weights with a sample-weighted mean of one."""
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() < 2 or not torch.isfinite(counts).all() or torch.any(counts < 0):
        raise ValueError("counts must be a nonnegative, finite class vector")
    if not 0 <= power <= 1 or not torch.any(counts > 0):
        raise ValueError("power must lie in [0, 1], with at least one observed class")
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = (counts.max() / counts[present]).pow(power)
    weights /= (weights * counts).sum() / counts.sum()
    return weights


def mask_targets_for_dropout(targets: Mapping[str, Tensor], *, dropped_text: Tensor, dropped_image: Tensor) -> dict[str, Tensor]:
    """Remove supervision whose modality was dropped while retaining fused labels."""
    if dropped_text.shape != dropped_image.shape or dropped_text.ndim != 1:
        raise ValueError("dropout masks must have matching B shapes")
    if torch.any(dropped_text & dropped_image):
        raise ValueError("both modalities cannot be dropped from the same row")
    masked = {}
    for name, values in targets.items():
        if values.shape != dropped_text.shape:
            raise ValueError(f"{name} targets must have shape B")
        if name in IMAGE_HEADS:
            values = values.masked_fill(dropped_image, IGNORE)
        elif name in TEXT_HEADS:
            values = values.masked_fill(dropped_text, IGNORE)
        masked[name] = values
    return masked


def focal_loss(logits: Tensor, targets: Tensor, weights: Tensor, gamma: float = 2.0) -> Tensor | None:
    """Compute fp32 focal loss over labelled rows; return None for an empty head."""
    if logits.ndim != 2 or targets.shape != logits.shape[:1] or weights.shape != logits.shape[1:]:
        raise ValueError("logits, targets and weights have inconsistent shapes")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    present = targets != IGNORE
    if not bool(present.any()):
        return None
    chosen = targets[present]
    if torch.any(chosen < 0) or torch.any(chosen >= logits.shape[1]):
        raise ValueError("target class index is out of range")
    log_probs = F.log_softmax(logits[present].float(), dim=-1)
    log_p = log_probs.gather(1, chosen[:, None]).squeeze(1)
    p = log_p.exp()
    sample_weights = weights.float().to(logits.device)[chosen]
    return (-sample_weights * (1 - p).pow(gamma) * log_p).mean()


def multitask_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    weights: Mapping[str, Tensor],
    *,
    gamma: float = 2.0,
    main_weight: float = 1.0,
    aux_weight: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum enabled head losses, skipping heads with no labels in this batch."""
    if main_weight < 0 or aux_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    parts: dict[str, Tensor] = {}
    for name, logits in outputs.items():
        if name not in HEAD_CLASSES:
            continue  # e.g. the modality weights
        if name not in targets or name not in weights:
            raise KeyError(f"missing targets or class weights for enabled head {name}")
        part = focal_loss(logits, targets[name], weights[name], gamma=gamma)
        if part is not None:
            parts[name] = part
    if not parts:
        raise ValueError("batch has no supervised labels for enabled heads")
    total = sum((main_weight if name in MAIN_HEADS else aux_weight) * value for name, value in parts.items())
    return total, parts
