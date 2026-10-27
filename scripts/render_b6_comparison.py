"""Paired official-selector comparison of full seed 0 and the frozen B6 VLM."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.eval import metrics, selectors

FULL = paths.REPO / "runs/classifier/full-s0-20261024-062724"
B6 = paths.REPO / "runs/transfer/kaggle_outputs/b6-results/b6"
METRICS = paths.REPO / "runs/baselines/b6-vlm-20261026/metrics_test.json"
TASKS = (
    ("Informativeness", "informative", "informative", "informative", "y_informative", False),
    ("Humanitarian", "humanitarian", "humanitarian", "humanitarian", "y_humanitarian", False),
    ("Severity, gold gated", "severity", "severity", "damage_severity", "y_severity", True),
    ("Severity, predicted gated", "severity", "severity", "damage_severity", "y_severity", False),
    ("Severity, DMD", "severity_dmd", "severity", "damage_severity", "y_severity", False),
    ("Severity, DMD, equal coverage", "severity_dmd", "severity", "damage_severity", "y_severity", True),
    ("Disaster type", "disaster_type", "disaster_type", "disaster_type", "y_disaster_type", False),
)


def _indexed(values: np.ndarray, classes: list[str]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(classes)}
    return np.asarray([mapping.get(value, len(classes)) for value in values], dtype=np.int64)


def _macro(gold: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    confusion = np.bincount(gold * (n_classes + 1) + predicted, minlength=n_classes * (n_classes + 1)).reshape(
        n_classes, n_classes + 1
    )
    tp = np.diag(confusion[:, :n_classes])
    denominator = confusion.sum(axis=1) + confusion[:, :n_classes].sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(n_classes), where=denominator > 0)
    return float(f1.mean())


def render(*, draws: int = 1000) -> str:
    full_marker = json.loads((FULL / "evaluation_complete.json").read_text())
    if full_marker != {name: hash_file(FULL / name) for name in full_marker}:
        raise ValueError("full-model official evaluation changed")
    full_metrics = json.loads((FULL / "metrics_test.json").read_text())
    b6_metrics = json.loads(METRICS.read_text())
    predictions = B6 / "predictions.jsonl"
    protocol = B6 / "protocol.json"
    if b6_metrics["predictions_sha256"] != hash_file(predictions) or b6_metrics["protocol_sha256"] != hash_file(protocol):
        raise ValueError("B6 predictions or protocol changed after frozen scoring")
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    full = pd.read_parquet(FULL / "predictions_official.parquet")
    b6_rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    b6 = {str(row["sample_id"]): row for row in b6_rows}
    if len(b6) != b6_metrics["n_unique_posts"]:
        raise ValueError("B6 prediction IDs are duplicated or incomplete")
    lines = [
        "# Frozen official test · full seed 0 versus B6 zero-shot VLM",
        "",
        "The same official posts are compared for each task. Δ is full minus B6 macro-F1; its 95% interval",
        f"resamples posts together ({draws:,} seeded paired draws). The interval reflects test-post sampling,",
        "not variability across training seeds. B6 uses text and image, but emits JSON labels rather than probabilities.",
        f"Its frozen majority fallback was used on {b6_metrics['failed']:,}/{b6_metrics['n_unique_posts']:,} posts",
        f"({b6_metrics['failure_rate']:.1%}), so the comparison includes generation reliability.",
        "",
        "| Task | n | Full F1 | B6 F1 | Δ [95% CI] | B6 failure rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for title, selector, head, field, gold_column, force_argmax in TASKS:
        selected = manifest.loc[selectors.official(manifest, selector), ["sample_id", gold_column]]
        aligned = selected.merge(full, on="sample_id", how="left", validate="one_to_one", indicator=True)
        if not (aligned._merge == "both").all():
            raise ValueError(f"full-model predictions are missing {selector} posts")
        classes = selectors.HEAD_LABELS[selector]
        gold = aligned[gold_column].astype(str).to_numpy(dtype=object)
        if force_argmax:
            probabilities = aligned[[f"p_{head}__{name}" for name in classes]].to_numpy(dtype=float)
            if not np.isfinite(probabilities).all():
                raise ValueError(f"full-model {selector} probabilities are incomplete")
            full_labels = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
        else:
            full_labels = np.asarray([None if pd.isna(value) else str(value) for value in aligned[f"{head}_label"]], dtype=object)
        if any(str(sample) not in b6 for sample in aligned.sample_id):
            raise ValueError(f"B6 is missing {selector} posts")
        baseline = np.asarray([b6[str(sample)]["labels"][field] for sample in aligned.sample_id], dtype=object)
        failures = np.asarray([b6[str(sample)]["failed"] for sample in aligned.sample_id], dtype=bool)
        full_point = metrics.classification(gold, full_labels, classes)["macro_f1"]
        baseline_point = metrics.classification(gold, baseline, classes)["macro_f1"]
        if abs(baseline_point - b6_metrics["heads"][selector]["macro_f1"]["point"]) > 1e-10:
            raise ValueError(f"B6 stored {selector} score differs from predictions")
        saved = full_metrics["heads"][selector]
        if selector == "severity" and force_argmax:
            saved = saved["gold_gated"]
        elif selector == "severity":
            saved = saved["predicted_gated"]
        elif selector == "disaster_type":
            saved = saved["all"]
        elif selector == "severity_dmd" and force_argmax:
            saved = None
        if saved is not None and abs(full_point - saved["macro_f1"]["point"]) > 1e-10:
            raise ValueError(f"full stored {selector} score differs from predictions")
        truth_index = _indexed(gold, classes)
        full_index = _indexed(full_labels, classes)
        b6_index = _indexed(baseline, classes)
        rng = np.random.default_rng(0)
        samples = np.empty(draws)
        for index in range(draws):
            take = rng.integers(0, len(gold), len(gold))
            samples[index] = _macro(truth_index[take], full_index[take], len(classes)) - _macro(
                truth_index[take], b6_index[take], len(classes)
            )
        lines.append(
            f"| {title} | {len(gold):,} | {full_point:.3f} | {baseline_point:.3f} | "
            f"{full_point - baseline_point:+.3f} [{np.quantile(samples, 0.025):+.3f}, "
            f"{np.quantile(samples, 0.975):+.3f}] | {failures.mean():.1%} |"
        )
    lines += [
        "",
        "The ordinary DMD severity row compares the full model's gated output with B6's always-present label.",
        "The equal-coverage row forces the full model's severity argmax on every DMD post.",
        "These exploratory paired intervals have no multiple-comparison correction.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    output = paths.REPORTS / "results/comparison_seed0_b6.md"
    output.write_text(render())
    print(output)
