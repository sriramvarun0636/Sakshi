"""Reproducible packet production and resumable dev/official explanation audits."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.explain.captions import CaptionCache
from sakshi.explain.common import MAIN_HEADS, TEXT_HEADS
from sakshi.explain.faithfulness import audit_modality
from sakshi.explain.packets import make_packet
from sakshi.explain.sanity import cascade, correlation
from sakshi.explain.service import Explainer
from sakshi.models.predict import Predictor
from sakshi.provenance.index import ProvenanceIndex


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
    signature = {"classifier": {name: hash_file(run / name) for name in
                                ("checkpoint.pt", "config.yaml", "temperatures.json", "thresholds.json", "label_maps.json")},
                 "manifest": inputs["manifest"],
                 "implementation": {str(file.relative_to(paths.REPO)): hash_file(file)
                                    for file in sorted((paths.REPO / "src/sakshi/explain").glob("*.py"))},
                 "packages": {name: version(name) for name in ("torch", "transformers", "captum", "grad-cam", "numpy", "scipy")},
                 "ig_steps": [50, 150], "ig_internal_batch_size": 16, "ig_delta_fraction": 0.05,
                 "cam_layer": "last_vision_layer.layer_norm1", "feature_tolerance": 0.01,
                 "percentages": [1, 5, 10, 20, 50], "random_seeds": [0, 1, 2, 3, 4]}
    if (selected.split == "test").any():
        from sakshi.compute.runner import final_evaluate

        if not (run / "evaluation_complete.json").is_file():
            raise ValueError("official classifier evaluation must be frozen/completed before explaining test rows; use dev first")
        # This call verifies/reuses completed scores; it cannot launch fresh scoring here.
        final_evaluate(run)
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
    result = {"sample_id": example.post_id, "model_id": prediction.model_id,
              "has_text": example.model_text is not None, "has_image": example.pixels is not None,
              "alpha_text": prediction.modality_weights["text"], "alpha_image": prediction.modality_weights["image"],
              "abstain": prediction.abstain, "abstain_reason": prediction.abstain_reason}
    for head in MAIN_HEADS:
        decision = getattr(prediction, head)
        result[f"{head}_label"], result[f"{head}_status"] = decision.label, decision.status
        result.update({f"p_{head}__{name}": value for name, value in decision.probs.items()})
    return result


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
    request = {"protocol": signature, "selector": selector, "sample_ids": rows.sample_id.tolist()}
    _request(artifacts, request)
    engine = Explainer(Predictor.from_run(run, device=device))
    provenance = ProvenanceIndex.load()
    captions = CaptionCache(paths.PROCESSED / "captions.parquet")
    records, predictions, diagnostics = [], [], []
    for _, row in rows.iterrows():
        example = engine.prepare(row)
        explanations = []
        for head in MAIN_HEADS:
            explanation, text, _ = engine.explain(example, head, artifacts / "heatmaps")
            if explanation is not None:
                explanations.append(explanation)
                details = asdict(explanation)
                details["image_heatmap_path"] = str(explanation.image_heatmap_path) if explanation.image_heatmap_path else None
                if text is not None:
                    details.update(ig_steps=text.steps, ig_output_difference=text.output_difference)
                diagnostics.append(details)
        records.append(make_packet(example, event=None if pd.isna(row.event) else str(row.event),
                                   posted_at=None if pd.isna(row.posted_at) else row.posted_at,
                                   provenance=provenance.check(str(row.sample_id)), explanations=explanations,
                                   image_caption=captions.get(str(row.sample_id))))
        predictions.append(prediction_record(example))
    atomic.parquet(artifacts / "predictions.parquet", pd.DataFrame(predictions))
    atomic.text(artifacts / "explanations.jsonl", "".join(json.dumps(value, allow_nan=False) + "\n" for value in diagnostics))
    atomic.text(output, "".join(json.dumps(value, allow_nan=False) + "\n" for value in records))
    atomic.text(artifacts / "complete.json", json.dumps({"packets": len(records), "packet_sha256": hash_file(output)}, indent=2) + "\n")
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


def audit(run: Path, *, kind: str, output: Path, split: str = "dev", count: int | None = None,
          device: str = "auto", heads: tuple[str, ...] = MAIN_HEADS, bootstrap: int = 1000) -> Path:
    from sakshi.explain.reports import write_report
    from sakshi.eval.selectors import official

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
    request = {"protocol": signature, "kind": kind, "split": split, "count": count, "heads": list(heads),
               "bootstrap": bootstrap, "sample_seed": 0, "pool_ids": {head: frame.sample_id.tolist() for head, frame in pools.items()}}
    _request(output, request)
    engine = Explainer(Predictor.from_run(run, device=device))
    # Cache the frozen model's sampling predictions, including gated classes.
    prediction_path = output / "sampling_predictions.parquet"
    if prediction_path.exists():
        predictions = pd.read_parquet(prediction_path)
    else:
        predictions = engine.predictor.predict_rows(union)
        atomic.parquet(prediction_path, predictions)
    rows_by_id = manifest.set_index("sample_id", drop=False)
    results = []
    for head in heads:
        ids = stratified_ids(predictions.loc[predictions.sample_id.isin(pools[head].sample_id)], head, count)
        for sample_id in ids:
            key = sha256(f"{sample_id}:{head}".encode()).hexdigest()[:24]
            record_path = output / "records" / f"{key}.json"
            if record_path.exists():
                results.append(json.loads(record_path.read_text()))
                continue
            example = engine.prepare(rows_by_id.loc[sample_id])
            result = {"post_id": sample_id, "head": head, "target": getattr(example.prediction, head).label,
                      "status": getattr(example.prediction, head).status, "flags": [], "measurements": []}
            if result["status"] == "predicted":
                explanation, text, image = engine.explain(example, head, output / "heatmaps")
                result["flags"] = explanation.flags
                if kind == "faithfulness":
                    for modality, attribution in (("text", text), ("image", image)):
                        if attribution is not None:
                            result["measurements"].append(audit_modality(engine.predictor.model, engine.predictor.tokenizer,
                                                                         engine.live, example, head, text=text, image=image,
                                                                         modality=modality))
                else:
                    with cascade(engine.predictor.model, head) as stages:
                        for stage in stages:
                            _, changed_text, changed_image = engine.explain(example, head, output / "sanity_heatmaps")
                            for modality, original, changed in (("text", text, changed_text), ("image", image, changed_image)):
                                if original is not None:
                                    left = [score for _, score in original.words] if modality == "text" else original.heatmap
                                    right = [score for _, score in changed.words] if modality == "text" else changed.heatmap
                                    result["measurements"].append({"stage": stage, "modality": modality,
                                                                   "rho": correlation(left, right)})
            atomic.text(record_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
            results.append(result)
    atomic.text(output / "results.json", json.dumps(results, indent=2, allow_nan=False) + "\n")
    return write_report(kind, results, output, split=split, count=count, bootstrap=bootstrap, heads=heads)
