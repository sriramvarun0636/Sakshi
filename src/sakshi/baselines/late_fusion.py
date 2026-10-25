"""B3: dev-tuned late fusion of frozen B1 text and B2 image predictions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.config import load
from sakshi.data import labels
from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE
from sakshi.eval.metrics import classification
from sakshi.models.calibration import fit_abstention
from sakshi.models.evaluate_loeo import evaluate_loeo_predictions
from sakshi.models.evaluate_run import evaluate_predictions
from sakshi.models.fusion import HUMANITARIAN_CONDITIONAL
from sakshi.models.predict import Predictor, finalize
from sakshi.models.train import TrainConfig
from sakshi.text import normalise_for_model

GRID = tuple(round(i / 10, 1) for i in range(11))


def _probabilities(frame: pd.DataFrame, head: str, classes: list[str], suffix: str) -> np.ndarray:
    values = frame[[f"p_{head}__{name}_{suffix}" for name in classes]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1, atol=1e-5):
        raise ValueError(f"{suffix} {head} probabilities are incomplete or uncalibrated")
    return values


def _conditional(frame: pd.DataFrame, suffix: str, informative: np.ndarray) -> np.ndarray:
    composite = _probabilities(frame, "humanitarian", labels.HUMANITARIAN, suffix)
    expected_nonhumanitarian = 1 - informative[:, 1]
    if not np.allclose(composite[:, labels.HUMANITARIAN.index("not_humanitarian")], expected_nonhumanitarian, atol=1e-5):
        raise ValueError("baseline humanitarian probabilities violate hierarchical composition")
    indices = [labels.HUMANITARIAN.index(name) for name in HUMANITARIAN_CONDITIONAL]
    if np.any(informative[:, 1] <= 0):
        raise ValueError("cannot recover a conditional humanitarian distribution from zero informative probability")
    conditional = composite[:, indices] / informative[:, 1, None]
    if not np.allclose(conditional.sum(1), 1, atol=1e-4):
        raise ValueError("baseline conditional humanitarian probabilities do not sum to one")
    return conditional / conditional.sum(1, keepdims=True)


def _aligned(text: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    if text.sample_id.duplicated().any() or image.sample_id.duplicated().any():
        raise ValueError("B1/B2 predictions need unique sample IDs")
    paired = text.merge(image, on="sample_id", how="outer", validate="one_to_one", suffixes=("_text", "_image"))
    if paired.model_id_image.isna().any():
        raise ValueError("B2 predictions must cover every fused row")
    return paired


def fit_weights(manifest: pd.DataFrame, text: pd.DataFrame, image: pd.DataFrame) -> dict:
    """Tune each head on agreed development labels, never on an evaluation role."""
    paired = _aligned(text, image)
    if paired.model_id_text.isna().any() or set(paired.sample_id) != set(manifest.sample_id):
        raise ValueError("both baselines must cover exactly the chosen development rows")
    aligned = manifest[["sample_id", "y_informative", "y_humanitarian"]].merge(
        paired, on="sample_id", how="left", validate="one_to_one", indicator=True
    )
    if not (aligned._merge == "both").all():
        raise ValueError("development labels and predictions are misaligned")
    p_text = _probabilities(aligned, "informative", labels.INFORMATIVE, "text")
    p_image = _probabilities(aligned, "informative", labels.INFORMATIVE, "image")
    q_text = _conditional(aligned, "text", p_text)
    q_image = _conditional(aligned, "image", p_image)
    informative = aligned.y_informative.notna().to_numpy()
    humanitarian = aligned.y_humanitarian.notna().to_numpy()
    if not informative.any() or not humanitarian.any():
        raise ValueError("late fusion needs agreed informative and humanitarian dev labels")
    gold_inf = aligned.loc[informative, "y_informative"].astype(str).to_numpy(dtype=object)
    gold_hum = aligned.loc[humanitarian, "y_humanitarian"].astype(str).to_numpy(dtype=object)

    def choose(scores: dict[float, float]) -> float:
        return max(GRID, key=lambda weight: (scores[weight], -abs(weight - 0.5), -weight))

    inf_scores = {}
    for weight in GRID:
        mixed = weight * p_text + (1 - weight) * p_image
        predicted = np.asarray(labels.INFORMATIVE, dtype=object)[(mixed[:, 1] >= 0.5).astype(int)]
        inf_scores[weight] = classification(gold_inf, predicted[informative], labels.INFORMATIVE)["macro_f1"]
    weight_inf = choose(inf_scores)
    p_mix = weight_inf * p_text + (1 - weight_inf) * p_image
    hum_scores = {}
    for weight in GRID:
        conditional = weight * q_text + (1 - weight) * q_image
        predicted = np.where(
            p_mix[:, 1] >= 0.5,
            np.asarray(HUMANITARIAN_CONDITIONAL, dtype=object)[conditional.argmax(1)],
            "not_humanitarian",
        )
        hum_scores[weight] = classification(gold_hum, predicted[humanitarian], labels.HUMANITARIAN)["macro_f1"]
    weight_hum = choose(hum_scores)
    q_mix = weight_hum * q_text + (1 - weight_hum) * q_image
    gold_indices = np.asarray([labels.HUMANITARIAN.index(value) for value in gold_hum], dtype=int)
    thresholds = fit_abstention(p_mix[humanitarian], q_mix[humanitarian], gold_indices)
    return {
        "weight_informative": weight_inf,
        "weight_humanitarian_conditional": weight_hum,
        "dev_macro_f1_informative": inf_scores[weight_inf],
        "dev_macro_f1_humanitarian": hum_scores[weight_hum],
        "grid": list(GRID),
        "dev_rows": len(aligned),
        "dev_labelled_informative": int(informative.sum()),
        "dev_labelled_humanitarian": int(humanitarian.sum()),
        "abstention": asdict(thresholds),
    }


def fuse_predictions(text: pd.DataFrame, image: pd.DataFrame, weights: dict, *, model_id: str) -> pd.DataFrame:
    """Preserve B2 image heads and hierarchical decisions on mixed main-head scores."""
    paired = _aligned(text, image)
    records = []
    for _, row in paired.iterrows():
        has_text = pd.notna(row.model_id_text)
        p_image = {name: float(row[f"p_informative__{name}_image"]) for name in labels.INFORMATIVE}
        p_text = {name: float(row[f"p_informative__{name}_text"]) for name in labels.INFORMATIVE} if has_text else p_image
        q_values = {}
        for suffix, informative in (("text", p_text), ("image", p_image)):
            q_values[suffix] = (
                {
                    name: float(row[f"p_humanitarian__{name}_{suffix}"]) / informative["informative"]
                    for name in HUMANITARIAN_CONDITIONAL
                }
                if suffix == "image" or has_text
                else None
            )
        if not has_text:
            q_values["text"] = q_values["image"]
        w_inf = weights["weight_informative"] if has_text else 0.0
        w_hum = weights["weight_humanitarian_conditional"] if has_text else 0.0
        p_inf = {name: w_inf * p_text[name] + (1 - w_inf) * p_image[name] for name in labels.INFORMATIVE}
        q_hum = {
            name: w_hum * q_values["text"][name] + (1 - w_hum) * q_values["image"][name] for name in HUMANITARIAN_CONDITIONAL
        }
        prediction = finalize(
            post_id=str(row.sample_id),
            model_id=model_id,
            p_inf=p_inf,
            q_hum=q_hum,
            p_sev={name: float(row[f"p_severity__{name}_image"]) for name in labels.SEVERITY},
            p_dt={name: float(row[f"p_disaster_type__{name}_image"]) for name in labels.DISASTER_TYPE},
            has_text=bool(has_text),
            has_image=True,
            alpha_image=1 - w_inf,
            tau_inf=weights["abstention"]["tau_inf"],
            tau_hum=weights["abstention"]["tau_hum"],
        )
        record = {
            "sample_id": str(row.sample_id),
            "model_id": model_id,
            "has_text": bool(has_text),
            "has_image": True,
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
        for head, classes in (("informative", labels.INFORMATIVE), ("humanitarian", labels.HUMANITARIAN)):
            record.update({f"p_{head}__{name}": getattr(prediction, head).probs[name] for name in classes})
        for head, classes in (("severity", labels.SEVERITY), ("disaster_type", labels.DISASTER_TYPE)):
            record.update({f"p_{head}__{name}": float(row[f"p_{head}__{name}_image"]) for name in classes})
        records.append(record)
    return pd.DataFrame(records)


def run_late_fusion(text_run: Path, image_run: Path, output: Path, *, bootstrap: int = 1000, device: str = "auto") -> Path:
    """Freeze source runs and dev choices before reading their held-out predictions."""
    text_run, image_run, output = Path(text_run), Path(image_run), Path(output)
    text_config = load(text_run / "config.yaml", TrainConfig)
    image_config = load(image_run / "config.yaml", TrainConfig)
    if text_config.model.modality != "text" or image_config.model.modality != "image":
        raise ValueError("B3 requires B1 text and B2 image runs")
    if text_config.seed != image_config.seed or text_config.fold != image_config.fold or bootstrap < 1:
        raise ValueError("B3 source runs need the same seed and fold and a positive bootstrap count")
    if device not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("B3 device must be auto, cpu, mps or cuda")
    fold = text_config.fold.removeprefix("loeo_") if text_config.fold else None
    marker_name = "evaluation_loeo_complete.json" if fold else "evaluation_complete.json"
    prediction_name = "predictions_loeo.parquet" if fold else "predictions_official.parquet"
    for run in (text_run, image_run):
        marker = json.loads((run / marker_name).read_text(encoding="utf-8"))
        if marker != {name: hash_file(run / name) for name in marker}:
            raise ValueError("B3 source evaluation marker does not match frozen outputs")
    manifest_path = paths.PROCESSED / "manifest.parquet"
    feature_hashes = {name: hash_file(paths.PROCESSED / "features" / name) for name in (INDEX_FILE, TOKENS_FILE, EMBED_FILE)}
    for run in (text_run, image_run):
        inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
        if inputs["manifest"] != hash_file(manifest_path) or any(
            inputs[name] != digest for name, digest in feature_hashes.items()
        ):
            raise ValueError("B3 local manifest or feature cache differs from a frozen source run")
    signature = {
        "text": {name: hash_file(text_run / name) for name in ("checkpoint.pt", "config.yaml", marker_name, prediction_name)},
        "image": {name: hash_file(image_run / name) for name in ("checkpoint.pt", "config.yaml", marker_name, prediction_name)},
        "manifest": hash_file(manifest_path),
        "features": feature_hashes,
        "implementation": hash_file(Path(__file__)),
        "bootstrap": bootstrap,
        "device": device,
    }
    request = output / "request.json"
    if request.exists():
        if json.loads(request.read_text(encoding="utf-8")) != signature:
            raise FileExistsError("B3 frozen sources changed; choose a new output path")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("B3 output already contains a different run")
    else:
        atomic.text(request, json.dumps(signature, indent=2) + "\n")
    completion = output / "complete.json"
    if completion.exists():
        hashes = json.loads(completion.read_text(encoding="utf-8"))
        if hashes != {name: hash_file(output / name) for name in hashes}:
            raise ValueError("B3 completed artifacts changed")
        return completion
    manifest = pd.read_parquet(manifest_path)
    role = manifest[f"loeo_{fold}"] == "dev" if fold else manifest.split == "dev"
    dev = manifest.loc[(manifest.dataset == "crisismmd") & role & manifest.y_informative.notna()].copy()
    if fold and (dev.event == fold).any():
        raise ValueError("held-out event leaked into B3 development tuning")
    if dev.empty or dev.text.map(normalise_for_model).isna().any():
        raise ValueError("B3 development pool needs agreed labels and usable text")
    dev_files = (output / "predictions_b1_dev.parquet", output / "predictions_b2_dev.parquet")
    for run, path in zip((text_run, image_run), dev_files, strict=True):
        if not path.exists():
            atomic.parquet(path, Predictor.from_run(run, device=device).predict_rows(dev))
    weights_path = output / "weights.json"
    if not weights_path.exists():
        weights = fit_weights(manifest.loc[dev.index], pd.read_parquet(dev_files[0]), pd.read_parquet(dev_files[1]))
        atomic.text(weights_path, json.dumps(weights, indent=2, allow_nan=False) + "\n")
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    prediction_path = output / prediction_name
    metrics_path = output / ("metrics_loeo.json" if fold else "metrics_test.json")
    model_id = f"b3-late-s{text_config.seed}" + (f"-loeo_{fold}" if fold else "")
    if not prediction_path.exists():
        fused = fuse_predictions(
            pd.read_parquet(text_run / prediction_name),
            pd.read_parquet(image_run / prediction_name),
            weights,
            model_id=model_id,
        )
        atomic.parquet(prediction_path, fused)
    if not metrics_path.exists():
        fused = pd.read_parquet(prediction_path)
        metrics = (
            evaluate_loeo_predictions(manifest, fused, event=fold, bootstrap=bootstrap)
            if fold
            else evaluate_predictions(manifest, fused, bootstrap=bootstrap)
        )
        metrics["baseline_id"] = "B3"
        metrics["source_models"] = [text_run.name, image_run.name]
        atomic.text(metrics_path, json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    artifacts = [*dev_files, weights_path, prediction_path, metrics_path]
    atomic.text(completion, json.dumps({path.name: hash_file(path) for path in artifacts}, indent=2) + "\n")
    return completion
