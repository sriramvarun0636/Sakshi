"""Shared single-post inputs and gradient lifecycle management."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sakshi.types import Prediction

MAIN_HEADS = ("informative", "humanitarian", "severity", "disaster_type")
TEXT_HEADS = MAIN_HEADS[:2]


@dataclass
class Example:
    post_id: str
    model_text: str | None
    text_clean: str | None
    inputs: dict[str, Tensor]
    prediction: Prediction
    pixels: Tensor | None = None
    image_size: tuple[int, int] | None = None


def expand_inputs(inputs: dict[str, Tensor], batch_size: int) -> dict[str, Tensor]:
    if any(value.shape[0] != 1 for value in inputs.values()):
        raise ValueError("explanations require exactly one original post")
    return {name: value.expand(batch_size, *value.shape[1:]) for name, value in inputs.items()}


@contextmanager
def gradients(model: nn.Module):
    """Freeze parameter gradients temporarily; preserve modes, flags and existing gradients."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("explanations cannot run inside torch.inference_mode")
    modes = [(module, module.training) for module in model.modules()]
    parameters = [(parameter, parameter.requires_grad, parameter.grad) for parameter in model.parameters()]
    try:
        model.eval()
        for parameter, _, _ in parameters:
            parameter.requires_grad_(False)
        with torch.enable_grad():
            yield
    finally:
        for parameter, required, gradient in parameters:
            parameter.requires_grad_(required)
            parameter.grad = gradient
        for module, training in modes:
            module.training = training
