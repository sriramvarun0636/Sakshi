"""Reproducible packet production and resumable dev/official explanation audits."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.explain.captions import CaptionCache
from sakshi.explain.common import MAIN_HEADS
from sakshi.explain.faithfulness import audit_modality
from sakshi.explain.packets import make_packet, validate_packet
from sakshi.explain.sanity import cascade, correlation
from sakshi.explain.service import Explainer
from sakshi.models.predict import Predictor
from sakshi.provenance.index import ProvenanceIndex

log = logging.getLogger(__name__)


def select_rows(manifest: pd.DataFrame, selector: str) -> pd.DataFrame:
    from sakshi.eval import selectors

    if manifest.sample_id.duplicated().any():
        raise ValueError("manifest sample IDs must be unique")
    if selector in ("train", "dev", "test"):
        return manifest.loc[manifest.split == selector].copy()
    if selector.startswith("official:"):
        return manifest.loc[selectors.official(manifest, selector.removeprefix("official:"))].copy()
    ids = pd.read_parquet(selector)
    if "sample_id" not in ids or ids.sample_id.duplicated().any() or not ids.sample_id.isin(manifest.sample_id).all():
        raise ValueError("row selector must contain unique, known sample IDs")
    # Take content only from the canonical manifest, never an edited selector file.
    return manifest.set_index("sample_id", drop=False).loc[ids.sample_id].reset_index(drop=True)


def protocol(run: Path, manifest: pd.DataFrame, selected: pd.DataFrame) -> dict:
    if selected.empty:
        raise ValueError("row selector is empty")
    inputs = json.loads((run / "inputs.json").read_text())
    if hash_file(paths.PROCESSED / "manifest.parquet") != inputs["manifest"]:
        raise ValueError("manifest differs from the classifier's recorded inputs")
    feature_hashes = {}
    for name in ("features_index.parquet", "clip_vitb16_tokens.f16.npy", "clip_vitb16_embed.f32.npy"):
        feature_hashes[name] = hash_file(paths.PROCESSED / "features" / name)
        if feature_hashes[name] != inputs[name]:
            raise ValueError("feature cache differs from the classifier's recorded inputs")
    signature = {
        "classifier": {
            name: hash_file(run / name)
            for name in ("checkpoint.pt", "config.yaml", "temperatures.json", "thresholds.json", "label_maps.json")
        },
        "manifest": inputs["manifest"],
        "features": feature_hashes,
        "implementation": {
            str(file.relative_to(paths.REPO)): hash_file(file)
            for file in sorted(
                [
                    *(paths.REPO / "src/sakshi/explain").glob("*.py"),
                    *(
                        paths.REPO / "src/sakshi" / name
                        for name in (
                            "data/features.py",
                            "models/fusion.py",
                            "models/predict.py",
                            "types.py",
                            "text.py",
                            "images.py",
                            "schemas/evidence_packet.json",
                        )
                    ),
                ]
            )
        },
        "packages": {
            name: version(name)
            for name in ("torch", "transformers", "captum", "grad-cam", "numpy", "scipy", "opencv-python-headless", "pillow")
        },
        "ig_steps": [50, 150],
        "ig_internal_batch_size": 16,
        "ig_delta_fraction": 0.05,
        "cam_layer": "last_vision_layer.layer_norm1",
        "feature_tolerance": 0.01,
        "percentages": [1, 5, 10, 20, 50],
        "random_seeds": [0, 1, 2, 3, 4],
    }
    run_config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    fold = run_config.get("fold")
    if fold is not None:
        from sakshi.models.evaluate_loeo import evaluate_loeo_run

        event = str(fold).removeprefix("loeo_")
        column = f"loeo_{event}"
        if column not in selected or not (selected[column] == "test").all():
            raise ValueError("LOEO explanations require only frozen held-out-event rows")
        if not (run / "evaluation_loeo_complete.json").is_file():
            raise ValueError("LOEO classifier evaluation must be frozen/completed before explaining held-out rows")
        evaluate_loeo_run(run)
    elif (selected.split == "test").any():
        from sakshi.compute.runner import final_evaluate

        if not (run / "evaluation_complete.json").is_file():
            raise ValueError("official classifier evaluation must be frozen/completed before explaining test rows; use dev first")
        final_evaluate(run)
    if fold is not None or (selected.split == "test").any():
        freeze = run / "explanation_protocol.json"
        if freeze.exists() and json.loads(freeze.read_text()) != signature:
            raise ValueError("explanation protocol was frozen before test inspection and has changed")
        if not freeze.exists():
            atomic.text(freeze, json.dumps(signature, indent=2) + "\n")
    return signature


def _request(output: Path, request: dict) -> None:
    path = output / "request.json"
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise ValueError("output belongs to a different explanation request")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("explanation output is not empty")
    else:
        atomic.text(path, json.dumps(request, indent=2, allow_nan=False) + "\n")


def prediction_record(example) -> dict:
    prediction = example.prediction
    result = {
        "sample_id": example.post_id,
        "model_id": prediction.model_id,
        "has_text": example.model_text is not None,
        "has_image": example.pixels is not None,
        "alpha_text": prediction.modality_weights["text"],
        "alpha_image": prediction.modality_weights["image"],
        "abstain": prediction.abstain,
        "abstain_reason": prediction.abstain_reason,
    }
    for head in MAIN_HEADS:
        decision = getattr(prediction, head)
        result[f"{head}_label"], result[f"{head}_status"] = decision.label, decision.status
        result.update({f"p_{head}__{name}": value for name, value in decision.probs.items()})
    return result


def explain(
    run: Path,
    *,
    kind: str,
    selector: str,
    head: str,
    output: Path,
    device: str = "auto",
    limit: int | None = None,
    overlays: bool = False,
) -> Path:
    from sakshi.explain.render import export_overlay
    from sakshi.images import load_rgb

    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    rows = select_rows(manifest, selector)
    if kind not in ("ig", "gradcam") or head not in MAIN_HEADS or kind == "ig" and head not in MAIN_HEADS[:2]:
        raise ValueError("IG explains fused text heads; Grad-CAM explains the four main heads")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows.iloc[:limit]
    signature = protocol(run, manifest, rows)
    if output.exists():
        raise FileExistsError("explanation output already exists")
    artifacts = output.parent / (output.stem + "_artifacts")
    _request(
        artifacts,
        {
            "protocol": signature,
            "kind": kind,
            "head": head,
            "sample_ids": rows.sample_id.tolist(),
            "device": device,
            "overlays": overlays,
        },
    )
    engine = Explainer(Predictor.from_run(run, device=device))
    records, skipped = [], 0
    for _, row in rows.iterrows():
        example = engine.prepare(row)
        result, text, image = engine.explain(
            example, head, artifacts / "heatmaps", modalities=("text",) if kind == "ig" else ("image",)
        )
        if result is None:
            skipped += 1
            continue
        record = asdict(result)
        record["image_heatmap_path"] = str(result.image_heatmap_path) if result.image_heatmap_path else None
        if text is not None:
            record.update(ig_steps=text.steps, ig_output_difference=text.output_difference)
        if overlays and image is not None:
            original = load_rgb(paths.image_file(row))
            if original is None:
                raise FileNotFoundError("source image became unavailable before figure export")
            try:
                figure = result.image_heatmap_path.with_suffix(".png")
                export_overlay(original, image.heatmap, figure)
                record["overlay_path"] = str(figure)
            finally:
                original.close()
        records.append(record)
    atomic.text(output, "".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    atomic.text(
        artifacts / "complete.json",
        json.dumps({"explained": len(records), "skipped": skipped, "sha256": hash_file(output)}, indent=2) + "\n",
    )
    return output


def packets(run: Path, *, selector: str, output: Path, device: str = "auto", limit: int | None = None) -> Path:
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    rows = select_rows(manifest, selector)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows.iloc[:limit]
    signature = protocol(run, manifest, rows)
    if output.exists():
        raise FileExistsError("packet output already exists; preserve it or choose a new path")
    artifacts = output.parent / (output.stem + "_artifacts")
    provenance_dir = paths.PROCESSED / "provenance"
    caption_path = paths.PROCESSED / "captions.parquet"
    model_id = json.loads((run / "env.json").read_text(encoding="utf-8"))["model_id"]
    request = {
        "protocol": signature,
        "model_id": model_id,
        "selector": selector,
        "sample_ids": rows.sample_id.tolist(),
        "device": device,
        "provenance": {name: hash_file(provenance_dir / name) for name in ("index.parquet", "onsets.json")},
        "captions": hash_file(caption_path) if caption_path.is_file() else None,
    }
    _request(artifacts, request)
    engine = None
    provenance = None
    captions = None
    records, predictions, diagnostics = [], [], []
    for number, (_, row) in enumerate(rows.iterrows(), 1):
        checkpoint = artifacts / "posts" / f"{sha256(str(row.sample_id).encode()).hexdigest()}.json"
        if checkpoint.is_file():
            item = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                item["sample_id"] != str(row.sample_id)
                or item["packet"]["post_id"] != str(row.sample_id)
                or item["packet"]["model_id"] != model_id
            ):
                raise ValueError("packet checkpoint differs from the frozen request")
            validate_packet(item["packet"])
            if item["prediction"]["sample_id"] != str(row.sample_id):
                raise ValueError("packet checkpoint prediction differs from the row")
        else:
            log.info("packet %d/%d: %s", number, len(rows), row.sample_id)
            if engine is None:
                engine = Explainer(Predictor.from_run(run, device=device))
                provenance = ProvenanceIndex.load()
                captions = CaptionCache(caption_path)
            example = engine.prepare(row)
            explanations = []
            row_diagnostics = []
            for head in MAIN_HEADS:
                explanation, text, _ = engine.explain(example, head, artifacts / "heatmaps")
                if explanation is not None:
                    explanations.append(explanation)
                    details = asdict(explanation)
                    details["image_heatmap_path"] = (
                        str(explanation.image_heatmap_path) if explanation.image_heatmap_path else None
                    )
                    if text is not None:
                        details.update(ig_steps=text.steps, ig_output_difference=text.output_difference)
                    row_diagnostics.append(details)
            item = {
                "sample_id": str(row.sample_id),
                "packet": make_packet(
                    example,
                    event=None if pd.isna(row.event) else str(row.event),
                    posted_at=None if pd.isna(row.posted_at) else row.posted_at,
                    provenance=provenance.check(str(row.sample_id)),
                    explanations=explanations,
                    image_caption=captions.get(str(row.sample_id)),
                ),
                "prediction": prediction_record(example),
                "diagnostics": row_diagnostics,
            }
            atomic.text(checkpoint, json.dumps(item, allow_nan=False) + "\n")
        records.append(item["packet"])
        predictions.append(item["prediction"])
        diagnostics.extend(item["diagnostics"])
    atomic.parquet(artifacts / "predictions.parquet", pd.DataFrame(predictions))
    atomic.text(artifacts / "explanations.jsonl", "".join(json.dumps(value, allow_nan=False) + "\n" for value in diagnostics))
    atomic.text(output, "".join(json.dumps(value, allow_nan=False) + "\n" for value in records))
    atomic.text(
        artifacts / "complete.json", json.dumps({"packets": len(records), "packet_sha256": hash_file(output)}, indent=2) + "\n"
    )
    return output


def stratified_ids(predictions: pd.DataFrame, head: str, count: int, *, seed: int = 0) -> list[str]:
    """Round-robin classes then sample remaining members, all without replacement."""
    if count < 1 or predictions.sample_id.duplicated().any():
        raise ValueError("positive sample size and unique sample IDs are required")
    rng = np.random.default_rng(seed)
    groups = []
    for _, frame in predictions.groupby(f"{head}_label", dropna=False, sort=True):
        groups.append(list(rng.permutation(sorted(frame.sample_id.astype(str)))))
    selected = []
    while groups and len(selected) < count:
        groups = [group for group in groups if group]
        for group in groups:
            if len(selected) == count:
                break
            selected.append(group.pop())
    return selected


def audit(
    run: Path,
    *,
    kind: str,
    output: Path,
    split: str = "dev",
    count: int | None = None,
    device: str = "auto",
    heads: tuple[str, ...] = MAIN_HEADS,
    bootstrap: int = 1000,
) -> Path:
    from sakshi.eval.selectors import official
    from sakshi.explain.reports import write_report

    count = count if count is not None else (500 if kind == "faithfulness" else 100)
    if kind not in ("faithfulness", "sanity") or split not in ("dev", "test"):
        raise ValueError("choose faithfulness/sanity and dev/test")
    if bootstrap < 1 or count < 1 or not heads or set(heads) - set(MAIN_HEADS) or len(set(heads)) != len(heads):
        raise ValueError("audit needs positive counts and unique main heads")
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    if split == "dev":
        pools = {head: manifest.loc[(manifest.split == "dev") & manifest[f"y_{head}"].notna()] for head in heads}
    else:
        pools = {head: manifest.loc[official(manifest, head)] for head in heads}
    union = manifest.loc[manifest.sample_id.isin(set().union(*(set(frame.sample_id) for frame in pools.values())))]
    signature = protocol(run, manifest, union)
    request = {
        "protocol": signature,
        "kind": kind,
        "split": split,
        "count": count,
        "heads": list(heads),
        "bootstrap": bootstrap,
        "sample_seed": 0,
        "pool_ids": {head: frame.sample_id.tolist() for head, frame in pools.items()},
        "device": device,
    }
    _request(output, request)
    completed = output / "complete.json"
    if completed.exists():
        completion = json.loads(completed.read_text())
        if any(hash_file(output / name) != digest for name, digest in completion["artifacts"].items()):
            raise ValueError("completed audit artifacts changed")
        return output / completion["report"]
    engine = Explainer(Predictor.from_run(run, device=device))
    # Cache the frozen model's sampling predictions, including gated classes.
    prediction_path = output / "sampling_predictions.parquet"
    if prediction_path.exists():
        checksum = prediction_path.with_suffix(".sha256")
        if not checksum.exists() or checksum.read_text().strip() != hash_file(prediction_path):
            raise ValueError("sampling prediction cache is incomplete or changed")
        predictions = pd.read_parquet(prediction_path)
    else:
        log.info("Computing sampling predictions for %d %s rows", len(union), split)
        predictions = engine.predictor.predict_rows(union)
        atomic.parquet(prediction_path, predictions)
        atomic.text(prediction_path.with_suffix(".sha256"), hash_file(prediction_path) + "\n")
    if (
        set(predictions.sample_id) != set(union.sample_id)
        or predictions.sample_id.duplicated().any()
        or set(predictions.model_id) != {engine.predictor.model_id}
    ):
        raise ValueError("sampling predictions do not match this model and row pool")
    rows_by_id = manifest.set_index("sample_id", drop=False)
    results = []
    for head in heads:
        ids = stratified_ids(predictions.loc[predictions.sample_id.isin(pools[head].sample_id)], head, count)
        for number, sample_id in enumerate(ids, 1):
            log.info("%s %s: %d/%d", kind, head, number, len(ids))
            key = sha256(f"{sample_id}:{head}".encode()).hexdigest()[:24]
            record_path = output / "records" / f"{key}.json"
            if record_path.exists():
                saved = json.loads(record_path.read_text())
                if (saved["post_id"], saved["head"]) != (sample_id, head):
                    raise ValueError("audit record identity differs from the request")
                results.append(saved)
                continue
            example = engine.prepare(rows_by_id.loc[sample_id])
            result = {
                "post_id": sample_id,
                "head": head,
                "target": getattr(example.prediction, head).label,
                "status": getattr(example.prediction, head).status,
                "flags": [],
                "measurements": [],
            }
            if result["status"] == "predicted":
                explanation, text, image = engine.explain(example, head, output / "heatmaps")
                result["flags"] = explanation.flags
                if kind == "faithfulness":
                    for modality, attribution in (("text", text), ("image", image)):
                        if modality == "text" and text is not None and not text.words:
                            result["flags"].append("no_text_units")
                            continue
                        if attribution is not None:
                            result["measurements"].append(
                                audit_modality(
                                    engine.predictor.model,
                                    engine.predictor.tokenizer,
                                    engine.live if image is not None else None,
                                    example,
                                    head,
                                    text=text,
                                    image=image,
                                    modality=modality,
                                )
                            )
                else:
                    with cascade(engine.predictor.model, head) as stages:
                        for stage in stages:
                            changed_explanation, changed_text, changed_image = engine.explain(
                                example, head, output / "sanity_heatmaps" / stage
                            )
                            result.setdefault("stage_flags", {})[stage] = changed_explanation.flags
                            for modality, original, changed in (("text", text, changed_text), ("image", image, changed_image)):
                                if original is not None:
                                    left = [score for _, score in original.words] if modality == "text" else original.heatmap
                                    right = [score for _, score in changed.words] if modality == "text" else changed.heatmap
                                    result["measurements"].append(
                                        {"stage": stage, "modality": modality, "rho": correlation(left, right)}
                                    )
            atomic.text(record_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
            results.append(result)
    atomic.text(output / "results.json", json.dumps(results, indent=2, allow_nan=False) + "\n")
    report = write_report(kind, results, output, split=split, count=count, bootstrap=bootstrap, heads=heads)
    exported = [report, output / "results.json"]
    if kind == "faithfulness":
        exported.append(output / f"faithfulness_curves{'_dev' if split == 'dev' else ''}.png")
    if set(heads) == set(MAIN_HEADS) and (split == "dev" or count == (500 if kind == "faithfulness" else 100)):
        atomic.text(paths.REPORTS / "results" / report.name, report.read_text())
        if kind == "faithfulness":
            source = exported[-1]
            atomic.write(paths.REPORTS / "figures" / source.name, lambda temporary: temporary.write_bytes(source.read_bytes()))
    atomic.text(
        completed,
        json.dumps({"report": report.name, "artifacts": {file.name: hash_file(file) for file in exported}}, indent=2) + "\n",
    )
    return report
