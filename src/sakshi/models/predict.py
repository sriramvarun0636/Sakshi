"""Hierarchical decisions and calibrated probability composition for WP1."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sakshi import atomic
from sakshi.data import labels
from sakshi.data.features import TOKEN_SHAPE, ClipFeaturizer
from sakshi.images import load_rgb
from sakshi.models.data import FeatureStore, load_tokenizer
from sakshi.models.fusion import HEAD_CLASSES, HUMANITARIAN_CONDITIONAL, FusionClassifier
from sakshi.types import HeadOutput, Post, Prediction

OUTPUT_CLASSES = {
    "informative": labels.INFORMATIVE,
    "humanitarian": labels.HUMANITARIAN,
    "severity": labels.SEVERITY,
    "disaster_type": labels.DISASTER_TYPE,
}


def _validate(probs: Mapping[str, float], classes: list[str]) -> dict[str, float]:
    if set(probs) != set(classes):
        raise ValueError(f"probabilities must have exactly these classes: {classes}")
    values = {key: float(probs[key]) for key in classes}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values.values()):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    return values


def composite_humanitarian(p_inf: Mapping[str, float], q_hum: Mapping[str, float]) -> dict[str, float]:
    """Compose an informative probability and a conditional four-class distribution."""
    p_inf = _validate(p_inf, labels.INFORMATIVE)
    q_hum = _validate(q_hum, HUMANITARIAN_CONDITIONAL)
    return {
        name: p_inf["not_informative"] if name == "not_humanitarian" else p_inf["informative"] * q_hum[name]
        for name in labels.HUMANITARIAN
    }


def finalize(
    *,
    post_id: str,
    model_id: str,
    p_inf: Mapping[str, float],
    q_hum: Mapping[str, float],
    p_sev: Mapping[str, float],
    p_dt: Mapping[str, float],
    has_text: bool,
    has_image: bool,
    alpha_image: float,
    tau_inf: float = 0.5,
    tau_hum: float = 0.5,
) -> Prediction:
    """Apply the invariant-preserving decision rule and review thresholds."""
    if not has_text and not has_image:
        raise ValueError("a prediction needs text or image")
    if not 0.5 <= tau_inf <= 1 or not 0.5 <= tau_hum <= 1:
        raise ValueError("review thresholds must lie in [0.5, 1]")
    if not math.isfinite(alpha_image) or not 0 <= alpha_image <= 1:
        raise ValueError("alpha_image must lie in [0, 1]")
    if not has_image and alpha_image > 1e-5 or not has_text and alpha_image < 1 - 1e-5:
        raise ValueError("modality weight contradicts the available inputs")
    p_inf = _validate(p_inf, labels.INFORMATIVE)
    q_hum = _validate(q_hum, HUMANITARIAN_CONDITIONAL)
    informative = p_inf["informative"] >= 0.5
    hum_probs = composite_humanitarian(p_inf, q_hum)
    inf_output = HeadOutput("informative" if informative else "not_informative", p_inf, "predicted")
    hum_output = HeadOutput(
        max(q_hum, key=q_hum.get) if informative else "not_humanitarian",
        hum_probs,
        "predicted" if informative else "gated",
    )
    if has_image:
        sev_probs = _validate(p_sev, labels.SEVERITY)
        dt_probs = _validate(p_dt, labels.DISASTER_TYPE)
        sev_output = (
            HeadOutput(max(sev_probs, key=sev_probs.get), sev_probs, "predicted")
            if informative
            else HeadOutput(None, {}, "gated")
        )
        dt_output = HeadOutput(max(dt_probs, key=dt_probs.get), dt_probs, "predicted")
    else:
        sev_output = HeadOutput(None, {}, "unavailable")
        dt_output = HeadOutput(None, {}, "unavailable")
    confidence_inf = max(p_inf.values())
    if confidence_inf < tau_inf:
        reason = "low_confidence:informative"
    elif informative and max(q_hum.values()) < tau_hum:
        reason = "low_confidence:humanitarian"
    else:
        reason = None
    return Prediction(
        post_id=post_id,
        informative=inf_output,
        humanitarian=hum_output,
        severity=sev_output,
        disaster_type=dt_output,
        modality_weights={"text": 1 - alpha_image, "image": alpha_image},
        abstain=reason is not None,
        abstain_reason=reason,
        model_id=model_id,
    )


class Predictor:
    """Load a trained run, apply calibration and produce invariant-safe predictions."""

    def __init__(
        self,
        model: FusionClassifier,
        tokenizer: object,
        *,
        model_id: str,
        device: torch.device,
        temperatures: Mapping[str, float] | None = None,
        tau_inf: float = 0.5,
        tau_hum: float = 0.5,
        features: FeatureStore | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.temperatures = dict(temperatures or {})
        self.tau_inf = tau_inf
        self.tau_hum = tau_hum
        self.features = features
        self._live_clip: ClipFeaturizer | None = None

    @classmethod
    def from_run(cls, run_dir: Path, device: str = "auto") -> Predictor:
        """Reject incompatible label maps before loading weights."""
        import json

        from sakshi.config import load
        from sakshi.models.train import TrainConfig, select_device

        run_dir = Path(run_dir)
        config = load(run_dir / "config.yaml", TrainConfig)
        saved_maps = json.loads((run_dir / "label_maps.json").read_text(encoding="utf-8"))
        expected = (
            HEAD_CLASSES
            if config.model.auxiliary_heads
            else {head: HEAD_CLASSES[head] for head in ("informative", "humanitarian", "severity", "disaster_type")}
        )
        if saved_maps != expected:
            raise ValueError("checkpoint label maps differ from canonical sakshi.data.labels")
        selected_device = select_device(device)
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = FusionClassifier.from_pretrained(config.model)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        temperatures = (
            json.loads((run_dir / "temperatures.json").read_text(encoding="utf-8"))
            if (run_dir / "temperatures.json").exists()
            else {}
        )
        thresholds = (
            json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
            if (run_dir / "thresholds.json").exists()
            else {}
        )
        return cls(
            model,
            load_tokenizer(config.model.text_encoder),
            model_id=checkpoint["model_id"],
            device=selected_device,
            temperatures=temperatures,
            tau_inf=thresholds.get("tau_inf", 0.5),
            tau_hum=thresholds.get("tau_hum", 0.5),
        )

    @torch.inference_mode()
    def _predict_batch(self, items: list[dict]) -> tuple[list[Prediction], list[dict[str, dict[str, float]]]]:
        texts = [item["text"] or "" for item in items]
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        output = self.model(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            image_tokens=torch.from_numpy(np.stack([item["tokens"] for item in items])).to(self.device),
            has_text=torch.tensor([item["text"] is not None for item in items], device=self.device),
            has_image=torch.tensor([item["has_image"] for item in items], device=self.device),
        )
        probabilities = {
            head: (output[head].float() / self.temperatures.get(head, 1.0)).softmax(-1).cpu().numpy()
            for head in ("informative", "humanitarian", "severity", "disaster_type")
        }
        result = []
        raw_probabilities = []
        for index, item in enumerate(items):
            mapped = {
                head: dict(zip(HEAD_CLASSES[head], probabilities[head][index].tolist(), strict=True)) for head in probabilities
            }
            raw_probabilities.append(mapped)
            result.append(
                finalize(
                    post_id=item["id"],
                    model_id=self.model_id,
                    p_inf=mapped["informative"],
                    q_hum=mapped["humanitarian"],
                    p_sev=mapped["severity"],
                    p_dt=mapped["disaster_type"],
                    has_text=item["text"] is not None,
                    has_image=item["has_image"],
                    alpha_image=float(output["alpha"][index, 1]),
                    tau_inf=self.tau_inf,
                    tau_hum=self.tau_hum,
                )
            )
        return result, raw_probabilities

    def predict(self, posts: list[Post], batch_size: int = 32) -> list[Prediction]:
        """Use live CLIP for user-supplied images, retaining input order."""
        from sakshi.text import normalise_for_model

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len({post.post_id for post in posts}) != len(posts):
            raise ValueError("duplicate post IDs")
        result = []
        for start in range(0, len(posts), batch_size):
            group = posts[start : start + batch_size]
            items = [
                {
                    "id": post.post_id,
                    "text": normalise_for_model(post.text),
                    "has_image": False,
                    "tokens": np.zeros(TOKEN_SHAPE, dtype=np.float32),
                }
                for post in group
            ]
            images, positions = [], []
            try:
                for index, post in enumerate(group):
                    image = load_rgb(post.image_path) if post.image_path is not None else None
                    if image is not None:
                        images.append(image)
                        positions.append(index)
                if images:
                    if self._live_clip is None:
                        self._live_clip = ClipFeaturizer(device=str(self.device))
                    tokens, _ = self._live_clip.encode(images)
                    for position, value in zip(positions, tokens, strict=True):
                        items[position]["has_image"] = True
                        items[position]["tokens"] = value
            finally:
                for image in images:
                    image.close()
            if any(item["text"] is None and not item["has_image"] for item in items):
                raise ValueError("post has neither usable text nor image")
            result.extend(self._predict_batch(items)[0])
        return result

    def predict_rows(self, rows: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Use the sample-ID-indexed feature cache for manifest rows."""
        from sakshi.text import normalise_for_model

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not {"sample_id", "text"}.issubset(rows):
            raise ValueError("rows need sample_id and text columns")
        if rows.sample_id.duplicated().any():
            raise ValueError("duplicate sample IDs")
        if self.features is None:
            self.features = FeatureStore()
        records = []
        for start in range(0, len(rows), batch_size):
            group = rows.iloc[start : start + batch_size]
            items = []
            for _, row in group.iterrows():
                sample_id = str(row.sample_id)
                items.append(
                    {
                        "id": sample_id,
                        "text": normalise_for_model(row.text) if isinstance(row.text, str) else None,
                        "has_image": self.features.has_image(sample_id),
                        "tokens": self.features.tokens(sample_id),
                    }
                )
            if any(item["text"] is None and not item["has_image"] for item in items):
                raise ValueError("manifest row has neither usable text nor cached image")
            predictions, raw_probabilities = self._predict_batch(items)
            for item, prediction, raw in zip(items, predictions, raw_probabilities, strict=True):
                record = {
                    "sample_id": item["id"],
                    "model_id": self.model_id,
                    "has_text": item["text"] is not None,
                    "has_image": item["has_image"],
                    "informative_label": prediction.informative.label,
                    "humanitarian_label": prediction.humanitarian.label,
                    "humanitarian_status": prediction.humanitarian.status,
                    "severity_label": prediction.severity.label,
                    "severity_status": prediction.severity.status,
                    "disaster_type_label": prediction.disaster_type.label,
                    "disaster_type_status": prediction.disaster_type.status,
                    "alpha_text": prediction.modality_weights["text"],
                    "alpha_image": prediction.modality_weights["image"],
                    "abstain": prediction.abstain,
                    "abstain_reason": prediction.abstain_reason,
                }
                for head, classes in OUTPUT_CLASSES.items():
                    for class_name in classes:
                        # Keep image-head probabilities on gated rows for gold-gated evaluation.
                        probability = (
                            raw[head].get(class_name, np.nan)
                            if head in ("severity", "disaster_type") and item["has_image"]
                            else getattr(prediction, head).probs.get(class_name, np.nan)
                        )
                        record[f"p_{head}__{class_name}"] = probability
                records.append(record)
        return pd.DataFrame(records)

    def write_rows(self, rows: pd.DataFrame, path: Path, batch_size: int = 32) -> Path:
        """Atomically publish a prediction table with one row per sample ID."""
        atomic.parquet(path, self.predict_rows(rows, batch_size=batch_size))
        return path
