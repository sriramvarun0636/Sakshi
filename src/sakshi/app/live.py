"""Interactive single-post inference without manifest rows or persistent upload files."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch
from PIL import Image

from sakshi.data.features import TOKEN_SHAPE, ClipFeaturizer
from sakshi.explain.common import Example
from sakshi.explain.gradcam import LiveClipFeaturizer, crop_heatmap, grad_cam
from sakshi.explain.ig import integrated_gradients
from sakshi.models.predict import Predictor
from sakshi.text import normalise_for_model
from sakshi.types import Prediction


@dataclass
class LiveResult:
    prediction: Prediction
    model_text: str | None
    text_clean: str | None
    tokens: np.ndarray
    image: Image.Image | None


def classify(predictor: Predictor, text: str | None, image: Image.Image | None) -> LiveResult:
    from sakshi.app.privacy import display_text

    model_text = normalise_for_model(text)
    if model_text is None and image is None:
        raise ValueError("Enter text or upload an image to classify a post.")
    tokens = np.zeros(TOKEN_SHAPE, dtype=np.float32)
    if image is not None:
        if predictor._live_clip is None:
            predictor._live_clip = ClipFeaturizer(device="cpu")
        batch, _ = predictor._live_clip.encode([image])
        tokens = np.asarray(batch[0], dtype=np.float32)
    identity = sha256((model_text or "").encode() + (image.tobytes() if image is not None else b"")).hexdigest()[:16]
    item = {"id": f"input:{identity}", "text": model_text, "tokens": tokens, "has_image": image is not None}
    prediction = predictor._predict_batch([item])[0][0]
    return LiveResult(prediction, model_text, display_text(text) or None, tokens, image)


def example_for(predictor: Predictor, result: LiveResult) -> tuple[Example, LiveClipFeaturizer | None]:
    encoded = predictor.tokenizer([result.model_text or ""], truncation=True, max_length=128, return_tensors="pt")
    inputs = {name: encoded[name].to(predictor.device) for name in ("input_ids", "attention_mask")}
    inputs.update(
        image_tokens=torch.from_numpy(result.tokens.copy())[None].to(predictor.device),
        has_text=torch.tensor([result.model_text is not None], device=predictor.device),
        has_image=torch.tensor([result.image is not None], device=predictor.device),
    )
    live = LiveClipFeaturizer(predictor._live_clip) if result.image is not None and predictor._live_clip else None
    pixels = live.pixels(result.image) if live is not None else None
    example = Example(
        result.prediction.post_id,
        result.model_text,
        result.text_clean,
        inputs,
        result.prediction,
        pixels,
        result.image.size if result.image is not None else None,
    )
    return example, live


def explain(predictor: Predictor, result: LiveResult, head: str) -> tuple[object | None, object | None]:
    example, live = example_for(predictor, result)
    text = (
        integrated_gradients(predictor.model, predictor.tokenizer, example, head)
        if head in {"informative", "humanitarian"}
        else None
    )
    image = grad_cam(predictor.model, live, example, head) if live is not None else None
    return text, image


def overlay(image: Image.Image, heatmap: np.ndarray, *, presentation: bool) -> Image.Image:
    """Overlay a 14×14 Grad-CAM in the exact CLIP centre-crop coordinates."""
    import cv2

    from sakshi.app.privacy import blur_faces

    base = blur_faces(image) if presentation else image.convert("RGB")
    rgb = np.asarray(base).copy()
    width, height = base.size
    scale = 224 / min(width, height)
    crop_width, crop_height = 224 / scale, 224 / scale
    left, top = round((width - crop_width) / 2), round((height - crop_height) / 2)
    right, bottom = min(width, left + round(crop_width)), min(height, top + round(crop_height))
    crop = cv2.resize(crop_heatmap(heatmap), (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(crop, 0, 1) * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    region = rgb[top:bottom, left:right]
    region[crop > 0] = np.uint8(np.clip(0.55 * region[crop > 0] + 0.45 * colour[crop > 0], 0, 255))
    return Image.fromarray(rgb)
