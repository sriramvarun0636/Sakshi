"""B5: fixed-prompt CLIP ViT-B/16 image-only zero-shot classifier (§7.11, E.8)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.config import create_run_dir
from sakshi.data import labels
from sakshi.data.features import EMBED_FILE, MODEL_NAME, MODEL_REVISION, load_index
from sakshi.models.evaluate_run import evaluate_predictions

PROMPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "informative": {
        "informative": (
            "a photo of damage caused by a natural disaster",
            "a photo of people affected by a disaster",
            "a photo of rescue workers helping disaster victims",
            "a weather forecast map of a storm",
        ),
        "not_informative": (
            "a selfie or a photo of people unrelated to any disaster",
            "a meme, cartoon or advertisement",
            "a photo of food, pets or everyday life",
        ),
    },
    "humanitarian": {
        "affected_individuals": (
            "a photo of injured or displaced people after a disaster",
            "a photo of people in an evacuation shelter",
            "a photo of disaster victims",
        ),
        "infrastructure_and_utility_damage": (
            "a photo of collapsed buildings",
            "a photo of a damaged road or bridge",
            "a photo of fallen power lines and destroyed houses",
        ),
        "not_humanitarian": (
            "a selfie or a photo of people unrelated to any disaster",
            "a meme, cartoon or advertisement",
            "a photo of food, pets or everyday life",
        ),
        "other_relevant_information": (
            "a weather forecast map of a hurricane",
            "a satellite image of a storm",
            "an aerial photo of a flooded area",
        ),
        "rescue_volunteering_or_donation_effort": (
            "a photo of rescue workers saving people",
            "a photo of volunteers distributing relief supplies",
            "a photo of donations for disaster victims",
        ),
    },
    "severity": {
        "little_or_no_damage": (
            "a photo with no visible damage",
            "a photo of an intact building",
            "a photo of an undamaged street",
        ),
        "mild_damage": (
            "a photo of a partially damaged building",
            "a photo of minor damage to houses",
            "a photo of a damaged roof",
        ),
        "severe_damage": (
            "a photo of completely destroyed buildings",
            "a photo of severe damage after a disaster",
            "a photo of collapsed houses and debris",
        ),
    },
    "disaster_type": {
        "earthquake": (
            "a photo of earthquake damage",
            "a photo of buildings collapsed by an earthquake",
            "a photo of rubble after an earthquake",
        ),
        "fire": ("a photo of a wildfire", "a photo of burning houses", "a photo of a forest fire"),
        "flood": ("a photo of a flooded street", "a photo of flood water in houses", "a photo of a flooded town"),
        "hurricane": (
            "a photo of hurricane damage",
            "a photo of a hurricane with strong wind and rain",
            "a photo of trees and houses destroyed by a hurricane",
        ),
        "landslide": ("a photo of a landslide", "a photo of a mudslide covering a road", "a photo of a collapsed hillside"),
        "not_disaster": (
            "a photo of everyday life with no disaster",
            "a selfie",
            "a photo of a city on a normal day",
        ),
        "other_disaster": (
            "a photo of a tornado, snowstorm or drought",
            "a photo of a volcanic eruption",
            "a photo of an explosion or industrial accident",
        ),
    },
}
CLASSES = {
    "informative": labels.INFORMATIVE,
    "humanitarian": labels.HUMANITARIAN,
    "severity": labels.SEVERITY,
    "disaster_type": labels.DISASTER_TYPE,
}


@dataclass(frozen=True)
class ZeroShotConfig:
    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    logit_scale: float = 100.0
    prompt_set: str = "design_spec_v3_appendix_e8"


def encode_class_prompts(config: ZeroShotConfig | None = None, *, device: str = "auto") -> dict[str, np.ndarray]:
    """Mean the unit CLIP text vectors for each fixed class prompt set."""
    import torch
    from transformers import AutoTokenizer, CLIPModel

    config = config or ZeroShotConfig()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    if len(tokenizer) < 1000:
        raise RuntimeError("CLIP tokenizer vocabulary is incomplete")
    model = CLIPModel.from_pretrained(config.model_name, revision=config.model_revision).to(device).eval()
    model.requires_grad_(False)
    result = {}
    for head, classes in CLASSES.items():
        if set(PROMPTS[head]) != set(classes):
            raise ValueError(f"CLIP prompts do not cover canonical {head} classes")
        texts = [prompt for class_name in classes for prompt in PROMPTS[head][class_name]]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.inference_mode():
            vectors = model.get_text_features(**encoded)
            if not isinstance(vectors, torch.Tensor):
                vectors = vectors.pooler_output
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy()
        centroids = []
        offset = 0
        for class_name in classes:
            count = len(PROMPTS[head][class_name])
            centroids.append(vectors[offset : offset + count].mean(axis=0))
            offset += count
        result[head] = np.stack(centroids).astype(np.float32)
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - logits.max(axis=1, keepdims=True)
    scores = np.exp(stable)
    return scores / scores.sum(axis=1, keepdims=True)


def classify_embeddings(
    sample_ids: list[str],
    embeddings: np.ndarray,
    class_vectors: dict[str, np.ndarray],
    *,
    model_id: str = "clip_zeroshot-s0",
    logit_scale: float = 100.0,
) -> pd.DataFrame:
    """Score cached unit image vectors and apply the shared hierarchy for comparison."""
    image = np.asarray(embeddings, dtype=np.float32)
    if image.shape != (len(sample_ids), 512) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("image embeddings need unique sample IDs and shape N×512")
    if not np.isfinite(image).all() or not np.allclose(np.linalg.norm(image, axis=1), 1, atol=1e-3):
        raise ValueError("CLIP image embeddings must be finite unit vectors")
    if not np.isfinite(logit_scale) or logit_scale <= 0:
        raise ValueError("logit_scale must be positive")
    probabilities = {}
    for head, classes in CLASSES.items():
        vectors = np.asarray(class_vectors[head], dtype=np.float32)
        if vectors.shape != (len(classes), 512) or not np.isfinite(vectors).all():
            raise ValueError(f"invalid CLIP {head} class vectors")
        probabilities[head] = _softmax(logit_scale * (image @ vectors.T))
    p_inf = probabilities["informative"]
    informative = p_inf[:, labels.INFORMATIVE.index("informative")] >= 0.5
    # Keep the four humanitarian categories conditional on informative, matching the
    # classifier's final-decision protocol. The fifth zero-shot prompt set remains
    # defined, but non-informative probability supplies the final not_humanitarian mass.
    conditional_indices = [i for i, name in enumerate(labels.HUMANITARIAN) if name != "not_humanitarian"]
    conditional = probabilities["humanitarian"][:, conditional_indices]
    conditional /= conditional.sum(axis=1, keepdims=True)
    humanitarian = np.zeros((len(sample_ids), len(labels.HUMANITARIAN)), dtype=np.float32)
    humanitarian[:, conditional_indices] = p_inf[:, 1, None] * conditional
    humanitarian[:, labels.HUMANITARIAN.index("not_humanitarian")] = p_inf[:, 0]
    conditional_classes = np.asarray([labels.HUMANITARIAN[i] for i in conditional_indices], dtype=object)
    hum_labels = np.where(informative, conditional_classes[conditional.argmax(1)], "not_humanitarian")
    sev_labels = np.asarray(labels.SEVERITY, dtype=object)[probabilities["severity"].argmax(1)]
    sev_labels[~informative] = None
    result = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "model_id": model_id,
            "has_text": False,
            "has_image": True,
            "informative_label": np.where(informative, "informative", "not_informative"),
            "humanitarian_label": hum_labels,
            "humanitarian_status": np.where(informative, "predicted", "gated"),
            "severity_label": sev_labels,
            "severity_status": np.where(informative, "predicted", "gated"),
            "disaster_type_label": np.asarray(labels.DISASTER_TYPE, dtype=object)[probabilities["disaster_type"].argmax(1)],
            "disaster_type_status": "predicted",
            "alpha_text": 0.0,
            "alpha_image": 1.0,
            "abstain": False,
            "abstain_reason": None,
        }
    )
    final = {**probabilities, "humanitarian": humanitarian}
    for head, classes in CLASSES.items():
        for column, class_name in enumerate(classes):
            result[f"p_{head}__{class_name}"] = final[head][:, column]
    return result


def run_clip_zeroshot(
    *,
    config: ZeroShotConfig | None = None,
    output_root: Path = paths.REPO / "runs" / "baselines",
    bootstrap: int = 1000,
) -> Path:
    """Create a B5 run with official selector metrics and reproducible prompt metadata."""
    config = config or ZeroShotConfig()
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    features = load_index()
    aligned = manifest[["sample_id"]].merge(features, on="sample_id", how="left", validate="one_to_one")
    if aligned.ok.isna().any() or not aligned.ok.all():
        raise ValueError("B5 requires usable cached images for every manifest row")
    embeddings = np.load(paths.PROCESSED / "features" / EMBED_FILE, mmap_mode="r")
    vectors = encode_class_prompts(config)
    model_id = "clip_zeroshot-s0"
    predictions = classify_embeddings(
        aligned.sample_id.astype(str).tolist(),
        embeddings[aligned.row.to_numpy(dtype=int)],
        vectors,
        model_id=model_id,
        logit_scale=config.logit_scale,
    )
    scores = evaluate_predictions(manifest, predictions, bootstrap=bootstrap)
    scores["baseline_id"] = "B5"
    scores["prompt_set"] = config.prompt_set
    scores["model_revision"] = config.model_revision
    run_dir, _ = create_run_dir(config, name="clip_zeroshot", seed=0, root=output_root)
    scores["git_commit"] = json.loads((run_dir / "env.json").read_text(encoding="utf-8")).get("git_commit")
    atomic.text(run_dir / "prompts.json", json.dumps(PROMPTS, indent=2) + "\n")
    atomic.parquet(run_dir / "predictions_all.parquet", predictions)
    atomic.text(run_dir / "metrics_test.json", json.dumps(scores, indent=2) + "\n")
    atomic.text(paths.REPORTS / "results" / "baselines_clip_zeroshot.md", format_report(scores, run_dir))
    return run_dir


def format_report(scores: dict, run_dir: Path) -> str:
    """Short, fixed-selector B5 result table with bootstrap uncertainty."""
    rows = []
    for label, result in (
        ("Informativeness", scores["heads"]["informative"]),
        ("Humanitarian", scores["heads"]["humanitarian"]),
        ("Severity, gold-gated", scores["heads"]["severity"]["gold_gated"]),
        ("Severity, predicted-gated", scores["heads"]["severity"]["predicted_gated"]),
        ("Severity, DMD", scores["heads"]["severity_dmd"]),
        ("Disaster type", scores["heads"]["disaster_type"]["all"]),
    ):
        metric = result["macro_f1"]
        rows.append(f"| {label} | {metric['point']:.3f} | [{metric['lo']:.3f}, {metric['hi']:.3f}] | {result['ece']:.3f} |")
    counts = scores["selectors"]
    return "\n".join(
        [
            "# B5 — CLIP zero-shot, image only",
            "",
            "The frozen CLIP ViT-B/16 checkpoint and fixed Appendix E.8 prompts were applied to the cached images.",
            "Each class vector is the mean of its unit-normalised prompt embeddings; logits are 100×cosine.",
            "The final humanitarian and severity decisions use the same hierarchy as the full classifier.",
            "No labels from the official test sets were used for prompts, tuning or model selection.",
            "",
            f"Run: `{run_dir}`. Checkpoint revision: `{scores['model_revision']}`.",
            "",
            "| Official task | Macro-F1 | 95% bootstrap CI | ECE |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "Official selector sizes: " + ", ".join(f"{head} {n:,}" for head, n in counts.items()) + ".",
            f"Hierarchy contradiction rate: {scores['contradiction_rate']:.3f}.",
            f"Predicted-gated severity coverage: {scores['heads']['severity']['predicted_gated']['coverage']:.3f}.",
            "This is a zero-shot image baseline; it is not a trained SAKSHI classifier result.",
            "",
        ]
    )
