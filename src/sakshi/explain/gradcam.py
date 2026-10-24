"""Differentiable frozen CLIP features and ViT Grad-CAM in original-image coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sakshi.explain.common import Example, expand_inputs, gradients
from sakshi.models.fusion import HEAD_CLASSES
from sakshi.types import ImageRegion


def pool_live_tokens(tokens: Tensor) -> Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != 197:
        raise ValueError("CLIP must supply CLS plus a 14×14 patch grid")
    batch, _, dim = tokens.shape
    patches = tokens[:, 1:].reshape(batch, 7, 2, 7, 2, dim).mean(dim=(2, 4))
    return torch.cat((tokens[:, :1], patches.reshape(batch, 49, dim)), dim=1)


class LiveClipFeaturizer(nn.Module):
    """Share the cache encoder's pinned tower/processor, without inference mode or NumPy detaches."""

    def __init__(self, featurizer):
        super().__init__()
        self.tower = featurizer.model.vision_model
        self.processor = featurizer.processor
        self.device = featurizer.device
        self.tower.eval().requires_grad_(False)

    @property
    def inner_tower(self):
        return self.tower if hasattr(self.tower, "post_layernorm") else self.tower.vision_model

    @property
    def target_layer(self):
        return self.inner_tower.encoder.layers[-1].layer_norm1

    def pixels(self, image) -> Tensor:
        return self.processor(images=[image], return_tensors="pt")["pixel_values"].to(self.device)

    def forward(self, pixel_values: Tensor) -> Tensor:
        states = self.tower(pixel_values=pixel_values).last_hidden_state
        return pool_live_tokens(self.inner_tower.post_layernorm(states))

    @torch.no_grad()
    def assert_cached(self, pixels: Tensor, cached: Tensor, *, tolerance: float = 1e-2) -> float:
        live = self(pixels).float()
        if live.shape != cached.shape or not torch.isfinite(live).all() or not torch.isfinite(cached).all():
            raise ValueError("live/cached CLIP features have invalid shapes or values")
        error = float((live - cached.to(live)).abs().max().cpu().item())
        if error > tolerance:
            raise ValueError(f"preprocessing drift: live/cached CLIP maximum error {error:.6f} > {tolerance}")
        return error


def reshape_transform(states: Tensor) -> Tensor:
    if states.ndim != 3 or states.shape[1] != 197:
        raise ValueError("Grad-CAM target layer must contain 197 ViT tokens")
    return states[:, 1:].reshape(states.shape[0], 14, 14, states.shape[2]).permute(0, 3, 1, 2)


class ImageHead(nn.Module):
    """Keep text fixed, recomputing exactly the classifier's live image features."""

    def __init__(self, classifier, live: LiveClipFeaturizer, example: Example, head: str):
        super().__init__()
        self.classifier, self.live = classifier, live
        self.inputs, self.head = example.inputs, head

    def forward(self, pixels: Tensor) -> Tensor:
        data = expand_inputs(self.inputs, len(pixels))
        data["image_tokens"] = self.live(pixels)
        return self.classifier(**data)[self.head]


def crop_to_original(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    if min(width, height) <= 0:
        raise ValueError("image dimensions must be positive")
    scale = 224 / min(width, height)
    off_x, off_y = (width * scale - 224) / 2, (height * scale - 224) / 2
    left, top, right, bottom = box
    return (
        max(0, min(width, math.floor((left + off_x) / scale))),
        max(0, min(height, math.floor((top + off_y) / scale))),
        max(0, min(width, math.ceil((right + off_x) / scale))),
        max(0, min(height, math.ceil((bottom + off_y) / scale))),
    )


def crop_heatmap(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    if values.shape != (14, 14):
        raise ValueError("heatmap must be 14×14")
    return F.interpolate(torch.from_numpy(values.copy())[None, None], size=(224, 224), mode="bilinear", align_corners=False)[
        0, 0
    ].numpy()


def extract_regions(heatmap: np.ndarray, size: tuple[int, int], head: str) -> list[ImageRegion]:
    from scipy.ndimage import label

    if np.asarray(heatmap).shape != (14, 14):
        raise ValueError("heatmap must be 14×14")
    if not np.isfinite(heatmap).all() or np.max(heatmap) <= 0:
        return []
    crop = crop_heatmap(heatmap)
    components, count = label((crop >= np.percentile(crop, 80)) & (crop > 0), structure=np.ones((3, 3)))
    regions = []
    total = float(crop.sum())
    for index in range(1, count + 1):
        mask = components == index
        if int(mask.sum()) < 0.02 * crop.size:
            continue
        y, x = np.where(mask)
        box = crop_to_original((int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1), size)
        regions.append((box, float(crop[mask].sum()) / total))
    return [
        ImageRegion(f"I{i}", head, box, score)
        for i, (box, score) in enumerate(sorted(regions, key=lambda r: (-r[1], r[0]))[:2], 1)
    ]


@dataclass
class ImageAttribution:
    heatmap: np.ndarray
    regions: list[ImageRegion]
    flags: list[str]


def grad_cam(classifier, live: LiveClipFeaturizer, example: Example, head: str) -> ImageAttribution | None:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if head not in HEAD_CLASSES:
        raise ValueError("unknown explanation head")
    if example.pixels is None or getattr(example.prediction, head).status != "predicted":
        return None
    if example.image_size is None:
        raise ValueError("image size is required for Grad-CAM")
    live.assert_cached(example.pixels, example.inputs["image_tokens"])
    target = HEAD_CLASSES[head].index(getattr(example.prediction, head).label)
    wrapper = ImageHead(classifier, live, example, head)

    class PatchGradCAM(GradCAM):
        def get_target_width_height(self, input_tensor):
            return (14, 14)

    with (
        gradients(wrapper),
        PatchGradCAM(model=wrapper, target_layers=[live.target_layer], reshape_transform=reshape_transform) as cam,
    ):
        heatmap = cam(
            input_tensor=example.pixels.detach().clone().requires_grad_(True), targets=[ClassifierOutputTarget(target)]
        )[0]
    flags = []
    if not np.isfinite(heatmap).all() or float(np.max(heatmap)) <= 0:
        heatmap = np.zeros((14, 14), dtype=np.float32)
        flags.append("no_salient_region")
    regions = extract_regions(heatmap, example.image_size, head)
    if not regions and "no_salient_region" not in flags:
        flags.append("no_salient_region")
    return ImageAttribution(np.asarray(heatmap, dtype=np.float32), regions, flags)
