"""Blind human audit export and scored calibration/evaluation sheets."""

from __future__ import annotations

import html
import json
import random
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pandas as pd
from PIL import Image

from sakshi import atomic, paths
from sakshi.explain.render import blur_faces
from sakshi.generate.workflow import _hash, _json

LABELS = {"supported", "partially_supported", "unsupported"}


def _pool(root: Path) -> list[dict]:
    rows = []
    for marker in sorted(root.glob("*/*/complete.json")):
        complete = json.loads(marker.read_text(encoding="utf-8"))
        if complete["mode"] != "loeo":
            continue
        if any(_hash(marker.parent / name) != digest for name, digest in complete["artifacts"].items()):
            raise ValueError("completed sitrep artifacts changed before audit")
        request = json.loads((marker.parent / "request.json").read_text(encoding="utf-8"))
        packet_path = Path(request["config"]["packets"])
        if not packet_path.is_absolute():
            packet_path = paths.REPO / packet_path
        for path in sorted(marker.parent.glob("[0-9]*Z.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            for sentence in report["sentences"]:
                rows.append(
                    {
                        "event": report["event"],
                        "window": report["window"]["start"],
                        "sentence": sentence,
                        "pids": report["input"]["pids"],
                        "sample_ids": report["input"]["sample_ids"],
                        "packet_path": packet_path,
                        "model_id": complete["model_id"],
                        "tau_entail": request["config"]["nli"]["tau_entail"],
                        "tau_contra": request["config"]["nli"]["tau_contra"],
                    }
                )
    return rows


def _images(sample_ids: set[str], manifest: Path, output: Path) -> dict[str, str]:
    rows = pd.read_parquet(manifest).set_index("sample_id")
    output.mkdir(parents=True, exist_ok=True)
    links = {}
    for sample_id in sorted(sample_ids):
        if sample_id not in rows.index:
            raise ValueError("audit sample is absent from manifest")
        name = sha256(sample_id.encode()).hexdigest()[:20] + ".jpg"
        target = output / name
        if not target.exists():
            with Image.open(paths.image_file(rows.loc[sample_id])) as source:
                blurred = blur_faces(source.convert("RGB"))
                try:
                    blurred.save(target, quality=85)
                finally:
                    blurred.close()
        links[sample_id] = name
    return links


def export_audit(
    sitrep_root: Path,
    sheet: Path,
    viewer: Path,
    *,
    manifest: Path = paths.PROCESSED / "manifest.parquet",
    calibration_count: int = 50,
    accepted_count: int = 100,
    rejected_count: int = 50,
    seed: int = 0,
    include_images: bool = True,
    phase: str = "evaluation",
    calibration_path: Path = paths.REPO / "reports/generation/audit_calibration.json",
) -> tuple[Path, Path]:
    """Sample fixed-role sentences; keep outcomes in ignored metadata, away from reviewers."""
    if phase not in {"calibration", "evaluation"}:
        raise ValueError("audit export needs a single fixed role")
    if min(calibration_count, accepted_count, rejected_count) < 0 or sheet.exists() or viewer.exists():
        raise FileExistsError("audit outputs already exist or counts are invalid")
    pool = _pool(sitrep_root)
    calibration = [item for item in pool if item["event"] == "hurricane_harvey" and item["sentence"]["round"] == 0]
    evaluation = [item for item in pool if item["event"] in {"hurricane_maria", "mexico_earthquake"}]
    accepted = [item for item in evaluation if item["sentence"]["accepted"]]
    rejected = [item for item in evaluation if not item["sentence"]["accepted"]]
    needed_calibration = calibration_count if phase == "calibration" else 0
    needed_accepted = accepted_count if phase == "evaluation" else 0
    needed_rejected = rejected_count if phase == "evaluation" else 0
    if len(calibration) < needed_calibration or len(accepted) < needed_accepted or len(rejected) < needed_rejected:
        raise ValueError("insufficient frozen LOEO sitrep sentences for the requested audit sample")
    rng = random.Random(seed)
    chosen = (
        [("calibration", item) for item in rng.sample(calibration, needed_calibration)]
        + [("evaluation", item) for item in rng.sample(accepted, needed_accepted)]
        + [("evaluation", item) for item in rng.sample(rejected, needed_rejected)]
    )
    if phase == "evaluation":
        calibrated = json.loads(calibration_path.read_text(encoding="utf-8"))["chosen"]
        if any(
            item["tau_entail"] != calibrated["tau_entail"] or item["tau_contra"] != calibrated["tau_contra"] for _, item in chosen
        ):
            raise ValueError("evaluation sitreps were not generated with the Harvey-calibrated NLI thresholds")
    rng.shuffle(chosen)
    packet_cache = {}
    records, metadata = [], []
    all_ids = set()
    for index, (phase, item) in enumerate(chosen, 1):
        pids = item["sentence"]["citations"]
        sample_ids = [sample_id for pid in pids if pid in item["sample_ids"] for sample_id in item["sample_ids"][pid]]
        all_ids.update(sample_ids)
        cache_path = item["packet_path"]
        if cache_path not in packet_cache:
            from sakshi.aggregate.packets import load_packets

            packet_cache[cache_path] = load_packets(cache_path, item["model_id"])
        packets = packet_cache[cache_path]
        texts = [f"{pid}: {packets[item['pids'][pid]].get('text_clean') or '(no text)'}" for pid in pids if pid in item["pids"]]
        identifier = f"G{index:04d}"
        records.append(
            {
                "item_id": identifier,
                "phase": phase,
                "event": item["event"],
                "window": item["window"],
                "sentence": item["sentence"]["text"],
                "citations": ",".join(pids),
                "cited_post_texts": "\n".join(texts),
                "label_a": "",
                "label_b": "",
                "label_final": "",
            }
        )
        metadata.append(
            {
                "item_id": identifier,
                "phase": phase,
                "accepted": item["sentence"]["accepted"],
                "gates": item["sentence"]["gates"],
                "sample_ids": sample_ids,
                "immutable_sha256": sha256(
                    json.dumps(
                        {key: value for key, value in records[-1].items() if not key.startswith("label_")},
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
            }
        )
    links = _images(all_ids, manifest, viewer.parent / "audit_images") if include_images else {}
    body = [
        "<!doctype html><meta charset='utf-8'><title>SAKSHI blind generation audit</title>",
        "<h1>Generation audit</h1><p>Judge each claim only against its cited posts and images. "
        "Use supported, partially_supported or unsupported. Gate outcomes are hidden.</p>",
    ]
    for record, meta in zip(records, metadata, strict=True):
        body.append(f"<section><h2>{record['item_id']} — {html.escape(record['event'])}</h2>")
        body.append(f"<p>{html.escape(record['sentence'])}</p><pre>{html.escape(record['cited_post_texts'])}</pre>")
        for sample_id in meta["sample_ids"]:
            if sample_id in links:
                body.append(f"<img style='max-width:400px' src='audit_images/{links[sample_id]}' alt='face-blurred cited image'>")
        body.append("</section><hr>")
    atomic.csv(sheet, pd.DataFrame(records))
    atomic.text(viewer, "\n".join(body) + "\n")
    _json(
        viewer.parent / f"{phase}_sampling.json",
        {
            "seed": seed,
            "pools": {"calibration": len(calibration), "accepted": len(accepted), "rejected": len(rejected)},
            "items": metadata,
            "sheet_sha256": _hash(sheet),
        },
    )
    return sheet, viewer


def _kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    a_counts, b_counts = Counter(left), Counter(right)
    expected = sum(a_counts[label] * b_counts[label] for label in LABELS) / len(left) ** 2
    return (observed - expected) / (1 - expected) if expected < 1 else None


def score_audit(sheet: Path, metadata: Path, output: Path, *, phase: str) -> Path:
    if phase not in {"calibration", "evaluation"}:
        raise ValueError("audit phase must be calibration or evaluation")
    labels = pd.read_csv(sheet, keep_default_na=False)
    stored = json.loads(metadata.read_text(encoding="utf-8"))
    by_id = {item["item_id"]: item for item in stored["items"]}
    selected = labels.loc[labels.phase == phase]
    if len(selected) != sum(item["phase"] == phase for item in stored["items"]):
        raise ValueError("audit sheet size differs from frozen sampling metadata")
    for row in selected.itertuples():
        if row.item_id not in by_id or any(getattr(row, key) not in LABELS for key in ("label_a", "label_b", "label_final")):
            raise ValueError("audit labels need two independent reviews and a final adjudicated label")
        expected_hash = by_id[row.item_id].get("immutable_sha256")
        if expected_hash:
            immutable = {key: getattr(row, key) for key in labels.columns if not key.startswith("label_")}
            actual_hash = sha256(json.dumps(immutable, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError("blind audit sample text or citations changed after export")
    result = {
        "phase": phase,
        "n": len(selected),
        "kappa": _kappa(selected.label_a.tolist(), selected.label_b.tolist()),
        "labels": selected.label_final.value_counts().to_dict(),
    }
    if phase == "calibration":
        scored = []
        for step in range(30, 91, 5):
            threshold = step / 100
            tp = fp = fn = 0
            for row in selected.itertuples():
                gate = by_id[row.item_id]["gates"]["entailment"]
                predicted = gate.get("score", 0) >= threshold and gate.get("p_contra", 1) < 0.3
                positive = row.label_final == "supported"
                tp += predicted and positive
                fp += predicted and not positive
                fn += not predicted and positive
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
            scored.append({"tau_entail": threshold, "tau_contra": 0.3, "f1": f1})
        result["grid"] = scored
        result["chosen"] = max(scored, key=lambda item: (item["f1"], -item["tau_entail"]))
    else:
        approved = selected.loc[selected.item_id.map(lambda identifier: by_id[identifier]["accepted"])]
        result["estimated_precision"] = float((approved.label_final == "supported").mean()) if len(approved) else None
        pool = stored["pools"]
        sampled = {
            True: sum(by_id[row.item_id]["accepted"] for row in selected.itertuples()),
            False: sum(not by_id[row.item_id]["accepted"] for row in selected.itertuples()),
        }
        weighted_rejected = weighted_unsupported = 0.0
        for row in selected.itertuples():
            if row.label_final == "supported":
                continue
            accepted = by_id[row.item_id]["accepted"]
            weight = pool["accepted" if accepted else "rejected"] / sampled[accepted]
            weighted_unsupported += weight
            if not accepted:
                weighted_rejected += weight
        result["estimated_unsupported_recall"] = weighted_rejected / weighted_unsupported if weighted_unsupported else None
        result["per_gate_unsupported"] = {
            gate: sum(
                row.label_final != "supported" and not by_id[row.item_id]["gates"][gate]["passed"]
                for row in selected.itertuples()
            )
            for gate in ("citation", "provenance", "entailment", "label", "number")
        }
    _json(output, result)
    return output
