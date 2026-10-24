"""Cascading parameter randomisation that restores the original model and RNG state."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


def correlation(original, randomized) -> float | None:
    from scipy.stats import spearmanr

    left, right = np.asarray(original).reshape(-1), np.asarray(randomized).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("sanity attribution vectors must match and be finite")
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(spearmanr(left, right).statistic)


def _reset(module):
    for child in module.modules():
        if isinstance(child, nn.MultiheadAttention):
            nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)
            for bias in (child.bias_k, child.bias_v):
                if bias is not None:
                    nn.init.xavier_normal_(bias)
        elif hasattr(child, "reset_parameters"):
            child.reset_parameters()


@contextmanager
def cascade(model, head: str, *, seed: int = 0):
    modules = [
        ("head", model.heads[head]),
        ("cross_attention", model.blocks[-1] if len(model.blocks) else None),
        ("gate", model.gate_network),
    ]
    saved = [
        (module, {name: tensor.detach().clone() for name, tensor in module.state_dict().items()})
        for _, module in modules
        if module is not None
    ]
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None

    def steps():
        for name, module in modules:
            if module is not None:
                _reset(module)
            yield name

    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed)
            yield steps()
    finally:
        for module, state in saved:
            module.load_state_dict(state, strict=True)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
