"""Render three-seed A1–A5 and B4 effects from frozen official predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.eval import metrics, selectors
from sakshi.models.comparison import TASKS, _prediction

VARIANTS = {
    "A1 no cross-attention": "a1_no_cross_attention",
    "A2 no modality gate": "a2_no_gate",
    "A3 no modality dropout": "a3_no_modality_dropout",
    "A4 flat humanitarian head": "a4_no_hierarchy",
    "A5 no CIB training": "a5_no_cib",
    "B4 informative only": "b4_informative",
    "B4 humanitarian only": "b4_humanitarian",
    "B4 severity only": "b4_severity",
    "B4 disaster type only": "b4_disaster_type",
}


def frozen_run(prefix: str, seed: int) -> tuple[Path, dict] | None:
    matches = sorted((paths.REPO / "runs/classifier").glob(f"{prefix}-s{seed}-20*"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous experiment {prefix} seed {seed}: {matches}")
    run = matches[0]
    marker = run / "evaluation_complete.json"
    if not marker.is_file():
        return None
    checks = json.loads(marker.read_text())
    if checks != {name: hash_file(run / name) for name in checks}:
        raise ValueError(f"frozen evaluation changed: {run}")
    return run, json.loads((run / "metrics_test.json").read_text())


def effect(
    manifest: pd.DataFrame,
    full: list[tuple[Path, dict]],
    variants: list[tuple[Path, dict]],
    task: tuple,
    *,
    draws: int = 1000,
) -> tuple[int, float, float, float, float, float, float]:
    title, selector, head, classes, gold_column, metric_variant = task
    gold_frame = manifest.loc[selectors.official(manifest, selector), ["sample_id", gold_column]]
    arrays = []
    scores = []
    for (full_run, full_metrics), (variant_run, variant_metrics) in zip(full, variants, strict=True):
        if selector not in full_metrics["heads"] or selector not in variant_metrics["heads"]:
            raise ValueError(f"{title} missing from a completed seed: {variant_run}")
        full_pred = pd.read_parquet(full_run / "predictions_official.parquet")
        variant_pred = pd.read_parquet(variant_run / "predictions_official.parquet")
        aligned = gold_frame.merge(full_pred, on="sample_id", how="left", validate="one_to_one", indicator=True)
        if not (aligned._merge == "both").all():
            raise ValueError(f"full seed is missing {selector} rows")
        aligned = aligned.drop(columns="_merge").merge(
            variant_pred, on="sample_id", how="left", validate="one_to_one", indicator=True, suffixes=("_full", "_variant")
        )
        if not (aligned._merge == "both").all():
            raise ValueError(f"{variant_run} is missing {selector} rows")
        truth = aligned[gold_column].astype(str).to_numpy(dtype=object)
        full_labels = _prediction(aligned, head, classes, "full", gold_gated=metric_variant == "gold_gated")
        variant_labels = _prediction(aligned, head, classes, "variant", gold_gated=metric_variant == "gold_gated")
        full_score = metrics.classification(truth, full_labels, classes)["macro_f1"]
        variant_score = metrics.classification(truth, variant_labels, classes)["macro_f1"]
        f_saved = full_metrics["heads"][selector]
        v_saved = variant_metrics["heads"][selector]
        for saved, actual in ((f_saved, full_score), (v_saved, variant_score)):
            if metric_variant is not None:
                saved = saved[metric_variant]
            if abs(saved["macro_f1"]["point"] - actual) > 1e-10:
                raise ValueError(f"stored {selector} metric differs from predictions")
        arrays.append((full_labels, variant_labels))
        scores.append((full_score, variant_score))
    if any(len(left) != len(gold_frame) for left, _ in arrays):
        raise ValueError("seed predictions are not aligned to the same official selector")
    truth = gold_frame[gold_column].astype(str).to_numpy(dtype=object)
    rng = np.random.default_rng(0)
    boot = np.empty(draws)
    for i in range(draws):
        take = rng.integers(0, len(truth), len(truth))
        boot[i] = np.mean(
            [
                metrics.classification(truth[take], variant[take], classes)["macro_f1"]
                - metrics.classification(truth[take], baseline[take], classes)["macro_f1"]
                for baseline, variant in arrays
            ]
        )
    scores = np.asarray(scores)
    delta = scores[:, 1] - scores[:, 0]
    return (
        len(truth),
        float(scores[:, 0].mean()),
        float(scores[:, 1].mean()),
        float(delta.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        float(np.std(delta, ddof=1)),
    )


def render() -> str:
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    full = [frozen_run("full", seed) for seed in (0, 1, 2)]
    if any(item is None for item in full):
        raise ValueError("all three full-model official evaluations must be frozen")
    lines = [
        "# Ablations and single-task baselines · three frozen seeds",
        "",
        "Every complete variant is compared with the corresponding full-model seed on identical official test posts.",
        "Δ is variant minus full macro-F1. Its 95% interval resamples test posts together across all three seeds",
        "(1,000 paired draws); seed SD describes variation among the three trained runs.",
        "The interval captures test-post sampling, not uncertainty over all possible training seeds.",
    ]
    for title, prefix in VARIANTS.items():
        runs = [frozen_run(prefix, seed) for seed in (0, 1, 2)]
        count = sum(item is not None for item in runs)
        lines += ["", f"## {title}", "", f"Frozen seeds: **{count}/3**."]
        if count != 3:
            continue
        lines += ["", "| Selector | n | Full F1 | Variant F1 | Δ [95% CI] | Seed SD of Δ |", "|---|---:|---:|---:|---:|---:|"]
        available = set.intersection(*(set(item[1]["heads"]) for item in runs))
        for task in TASKS:
            name, selector = task[:2]
            if selector not in available:
                continue
            n, baseline, variant, delta, low, high, seed_sd = effect(manifest, full, runs, task)
            lines.append(
                f"| {name} | {n:,} | {baseline:.3f} | {variant:.3f} | {delta:+.3f} [{low:+.3f}, {high:+.3f}] | {seed_sd:.3f} |"
            )
    lines += [
        "",
        "The experiments share fixed official selectors, but the many reported effects have no multiple-comparison correction.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    destination = paths.REPORTS / "results/experiments_three_seed.md"
    destination.write_text(render())
    print(destination)
