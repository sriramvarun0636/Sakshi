"""Epoch-boundary random state needed to resume the sampler and modality dropout."""

from __future__ import annotations

import random

import numpy as np
import torch


def capture_rng(generator: torch.Generator, *, device: torch.device | None = None) -> dict:
    device = device or torch.device("cpu")
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "sampler": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "mps": torch.mps.get_rng_state() if device.type == "mps" else None,
    }


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume requires the same CUDA device count as the saved random state")
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("resume requires the MPS backend used by the saved run")
        torch.mps.set_rng_state(state["mps"])
