"""ERASER-style word/patch removal and sufficiency with fixed predicted targets."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from sakshi.explain.common import Example
from sakshi.models.fusion import HEAD_CLASSES

PERCENTAGES = (1, 5, 10, 20, 50)


def mask_text(input_ids, groups: list[list[int]], selected: set[int], *, keep_only: bool, mask_token_id: int):
    result = input_ids.clone()
    for index, positions in enumerate(groups):
        if (index not in selected) if keep_only else (index in selected):
            result[:, positions] = mask_token_id
    return result


def mask_patches(pixels, selected: set[int], *, keep_only: bool):
    if pixels.shape[-2:] != (224, 224) or any(not 0 <= index < 196 for index in selected):
        raise ValueError("patch perturbation requires a 224×224 crop and valid 14×14 indices")
    result = pixels.clone()
    for index in range(196):
        if (index not in selected) if keep_only else (index in selected):
            y, x = divmod(index, 14)
            result[:, :, y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16] = 0
    return result


def ranking_curve(scores: np.ndarray, probability: Callable[[set[int], bool], float], base: float) -> list[dict]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("faithfulness requires finite, nonempty unit scores")
    order = np.argsort(-scores, kind="stable")
    result = []
    for percentage in PERCENTAGES:
        count = min(len(order), max(1, math.ceil(percentage * len(order) / 100)))
        selected = set(int(index) for index in order[:count])
        removed, sufficient = probability(selected, False), probability(selected, True)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (base, removed, sufficient)):
            raise ValueError("perturbed class probability must be finite and in [0, 1]")
        result.append({"k": percentage, "units": count, "comprehensiveness": base - removed, "sufficiency": base - sufficient})
    return result


def audit_modality(model, tokenizer, live, example: Example, head: str, *, text=None, image=None, modality: str) -> dict:
    target = HEAD_CLASSES[head].index(getattr(example.prediction, head).label)
    model.eval()

    @torch.no_grad()
    def score(selected: set[int] | None, keep_only: bool) -> float:
        inputs = dict(example.inputs)
        if selected is not None:
            if modality == "text":
                inputs["input_ids"] = mask_text(
                    inputs["input_ids"], text.groups, selected, keep_only=keep_only, mask_token_id=tokenizer.mask_token_id
                )
            else:
                inputs["image_tokens"] = live(mask_patches(example.pixels, selected, keep_only=keep_only))
        return float(model(**inputs)[head].float().softmax(-1)[0, target].cpu().item())

    if modality == "text" and text is not None:
        scores = np.array([value for _, value in text.words])
    elif modality == "image" and image is not None:
        scores = image.heatmap.reshape(-1)
    else:
        raise ValueError("requested modality has no attribution")
    base = score(None, False)
    observed = ranking_curve(scores, score, base)
    random = [ranking_curve(np.random.default_rng(seed).random(len(scores)), score, base) for seed in range(5)]
    random_mean = [
        {
            "k": percentage,
            **{metric: float(np.mean([curve[i][metric] for curve in random])) for metric in ("comprehensiveness", "sufficiency")},
        }
        for i, percentage in enumerate(PERCENTAGES)
    ]
    return {
        "post_id": example.post_id,
        "head": head,
        "target": getattr(example.prediction, head).label,
        "modality": modality,
        "base_probability": base,
        "explainer": observed,
        "random": random_mean,
        "random_seeds": list(range(5)),
        "aopc": {
            method: {metric: float(np.mean([row[metric] for row in curve])) for metric in ("comprehensiveness", "sufficiency")}
            for method, curve in (("explainer", observed), ("random", random_mean))
        },
    }
