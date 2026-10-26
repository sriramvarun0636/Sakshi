"""Score the private B6 VLM output on the frozen official selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.eval import metrics, selectors

FIELDS = {
    "informative": ("informative", "y_informative"),
    "humanitarian": ("humanitarian", "y_humanitarian"),
    "severity": ("damage_severity", "y_severity"),
    "severity_dmd": ("damage_severity", "y_severity"),
    "disaster_type": ("disaster_type", "y_disaster_type"),
}


def score(gold: np.ndarray, predicted: np.ndarray, classes: list[str], *, draws: int = 1000) -> dict:
    point = metrics.classification(gold, predicted, classes)
    rng = np.random.default_rng(0)
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        take = rng.integers(0, len(gold), len(gold))
        samples[index] = metrics.classification(gold[take], predicted[take], classes)["macro_f1"]
    return {
        "n": len(gold),
        "macro_f1": {
            "point": point["macro_f1"],
            "lo": float(np.quantile(samples, 0.025)),
            "hi": float(np.quantile(samples, 0.975)),
        },
        "weighted_f1": point["weighted_f1"],
        "accuracy": point["accuracy"],
        "per_class": point["per_class"],
        "confusion": point["confusion"],
    }


def evaluate(predictions: Path, output: Path) -> dict:
    predictions = predictions.resolve()
    protocol = predictions.with_name("protocol.json")
    meta = json.loads(protocol.read_text())
    manifest_path = paths.PROCESSED / "manifest.parquet"
    if meta["manifest_sha256"] != hash_file(manifest_path):
        raise ValueError("B6 source manifest differs from local frozen manifest")
    manifest = pd.read_parquet(manifest_path)
    expected = set().union(*(set(manifest.loc[selectors.official(manifest, head), "sample_id"].astype(str)) for head in FIELDS))
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if len(rows) != len(expected) or len(rows) != meta["rows"]:
        raise ValueError(f"B6 output count {len(rows)} differs from official selector union {len(expected)}")
    samples = [str(row["sample_id"]) for row in rows]
    if len(set(samples)) != len(samples) or set(samples) != expected:
        raise ValueError("B6 output sample IDs do not match the official selector union")
    for row in rows:
        if set(row["labels"]) != {"informative", "humanitarian", "damage_severity", "disaster_type"}:
            raise ValueError(f"B6 output is missing a head: {row['sample_id']}")
        if row["failed"] and row["labels"] != meta["majority"]:
            raise ValueError("B6 failure did not use the frozen training majority")
        if not np.isfinite(row["latency_seconds"]) or row["latency_seconds"] <= 0:
            raise ValueError("B6 latency is invalid")
    by_id = {str(row["sample_id"]): row for row in rows}
    heads = {}
    for head, (field, gold_column) in FIELDS.items():
        part = manifest.loc[selectors.official(manifest, head), ["sample_id", gold_column]]
        gold = part[gold_column].astype(str).to_numpy(dtype=object)
        pred = np.asarray([by_id[str(sample)]["labels"][field] for sample in part.sample_id], dtype=object)
        heads[head] = score(gold, pred, selectors.HEAD_LABELS[head])
        heads[head]["failure_rate"] = float(np.mean([by_id[str(sample)]["failed"] for sample in part.sample_id]))
    latency = np.asarray([row["latency_seconds"] for row in rows], dtype=float)
    contradiction = np.mean(
        [
            (row["labels"]["informative"] == "not_informative") != (row["labels"]["humanitarian"] == "not_humanitarian")
            for row in rows
        ]
    )
    result = {
        "baseline_id": "B6",
        "model": meta["model"],
        "revision": meta["revision"],
        "quantization": meta["quantization"],
        "fallback_reason": meta["fallback_reason"],
        "temperature": meta["temperature"],
        "max_new_tokens": meta["max_new_tokens"],
        "max_pixels": meta["max_pixels"],
        "protocol_sha256": hash_file(protocol),
        "predictions_sha256": hash_file(predictions),
        "n_unique_posts": len(rows),
        "failed": sum(bool(row["failed"]) for row in rows),
        "failure_rate": float(np.mean([row["failed"] for row in rows])),
        "latency_seconds": {
            "mean": float(latency.mean()),
            "median": float(np.median(latency)),
            "p95": float(np.quantile(latency, 0.95)),
        },
        "hierarchy_contradiction_rate": float(contradiction),
        "heads": heads,
        "calibration": "not applicable: zero-shot JSON labels have no class probabilities",
    }
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise FileExistsError("frozen B6 evaluation differs; choose a new output path")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.output)["heads"], indent=2))
