"""Prepare a post once and compute only explanations allowed by its final decisions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from sakshi import atomic, paths
from sakshi.data.features import ClipFeaturizer
from sakshi.explain.common import MAIN_HEADS, Example
from sakshi.explain.gradcam import LiveClipFeaturizer, grad_cam
from sakshi.explain.ig import extract_spans, integrated_gradients
from sakshi.images import load_rgb
from sakshi.models.data import FeatureStore
from sakshi.text import clean_for_display, normalise_for_model
from sakshi.types import Explanation


class Explainer:
    def __init__(self, predictor, *, features=None, live=None):
        self.predictor = predictor
        self.features = features if features is not None else FeatureStore()
        self._live = live

    @property
    def live(self):
        if self._live is None:
            self._live = LiveClipFeaturizer(ClipFeaturizer(device=str(self.predictor.device)))
        return self._live

    def prepare(self, row) -> Example:
        sample_id = str(row["sample_id"])
        text = row["text"] if isinstance(row["text"], str) else None
        model_text = normalise_for_model(text)
        has_image = self.features.has_image(sample_id)
        if model_text is None and not has_image:
            raise ValueError("post has neither usable text nor image")
        tokens = self.features.tokens(sample_id)
        item = {"id": sample_id, "text": model_text, "tokens": tokens, "has_image": has_image}
        prediction = self.predictor._predict_batch([item])[0][0]
        encoded = self.predictor.tokenizer([model_text or ""], truncation=True, max_length=128, return_tensors="pt")
        inputs = {name: encoded[name].to(self.predictor.device) for name in ("input_ids", "attention_mask")}
        inputs.update(image_tokens=torch.from_numpy(tokens.copy())[None].to(self.predictor.device),
                      has_text=torch.tensor([model_text is not None], device=self.predictor.device),
                      has_image=torch.tensor([has_image], device=self.predictor.device))
        pixels, size = None, None
        if has_image:
            image = load_rgb(paths.image_file(row))
            if image is None:
                raise FileNotFoundError(f"cached image is unavailable for live explanation: {sample_id}")
            try:
                size = image.size
                pixels = self.live.pixels(image)
                self.live.assert_cached(pixels, inputs["image_tokens"])
            finally:
                image.close()
        return Example(sample_id, model_text, clean_for_display(text) if model_text else None,
                       inputs, prediction, pixels, size)

    def explain(self, example: Example, head: str, output_dir: Path):
        if head not in MAIN_HEADS:
            raise ValueError("only the four main heads have explanations")
        decision = getattr(example.prediction, head)
        if decision.status != "predicted":
            return None, None, None
        text = integrated_gradients(self.predictor.model, self.predictor.tokenizer, example, head)
        image = grad_cam(self.predictor.model, self.live, example, head) if example.pixels is not None else None
        heatmap_path = None
        if image is not None:
            name = sha256(example.post_id.encode()).hexdigest()[:24]
            heatmap_path = Path(output_dir) / f"{name}-{head}.npy"
            atomic.write(heatmap_path, lambda temporary: np.save(temporary, image.heatmap, allow_pickle=False))
        result = Explanation(
            example.post_id, head, decision.label,
            text.words if text else [], extract_spans(text.words, head) if text else [],
            heatmap_path, image.regions if image else [], text.delta if text else None,
            [*(text.flags if text else []), *(image.flags if image else [])],
            self.predictor.model_id,
        )
        return result, text, image
