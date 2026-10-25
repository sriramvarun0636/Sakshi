"""Paired official-selector comparison of the first full model with B5 CLIP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.data import labels
from sakshi.eval import bootstrap, metrics, selectors

TASKS = (
    ("Informativeness", "informative", "informative", labels.INFORMATIVE, "y_informative", None),
    ("Humanitarian", "humanitarian", "humanitarian", labels.HUMANITARIAN, "y_humanitarian", None),
    ("Severity, gold gated", "severity", "severity", labels.SEVERITY, "y_severity", "gold_gated"),
    ("Severity, predicted gated", "severity", "severity", labels.SEVERITY, "y_severity", "predicted_gated"),
    ("Severity, DMD", "severity_dmd", "severity", labels.SEVERITY, "y_severity", None),
    ("Disaster type", "disaster_type", "disaster_type", labels.DISASTER_TYPE, "y_disaster_type", "all"),
)


def _prediction(frame: pd.DataFrame, head: str, classes: list[str], side: str, *, gold_gated: bool) -> np.ndarray:
    if gold_gated:
        probabilities = frame[[f"p_{head}__{name}_{side}" for name in classes]].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{side} has missing gold-gated probabilities")
        return np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
    return np.array([None if pd.isna(value) else str(value) for value in frame[f"{head}_label_{side}"]], dtype=object)


def render_b5_comparison(
    run_dir: Path,
    baseline_dir: Path,
    *,
    output: Path = paths.REPORTS / "results" / "comparison_seed0_b5.md",
    draws: int = 1000,
) -> Path:
    """Compute paired bootstrap effects without altering either frozen prediction table."""
    run_dir, baseline_dir = Path(run_dir), Path(baseline_dir)
    marker = json.loads((run_dir / "evaluation_complete.json").read_text(encoding="utf-8"))
    if marker != {name: hash_file(run_dir / name) for name in ("metrics_test.json", "predictions_official.parquet")}:
        raise ValueError("full-model official predictions do not match the evaluation marker")
    request = json.loads((run_dir / "evaluation_request.json").read_text(encoding="utf-8"))
    manifest_path = paths.PROCESSED / "manifest.parquet"
    if hash_file(manifest_path) != request["manifest"]:
        raise ValueError("manifest differs from the frozen evaluation input")
    baseline_metrics = json.loads((baseline_dir / "metrics_test.json").read_text(encoding="utf-8"))
    if baseline_metrics.get("baseline_id") != "B5":
        raise ValueError("the comparison baseline must be the completed B5 CLIP run")
    full_metrics = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
    manifest = pd.read_parquet(manifest_path)
    full = pd.read_parquet(run_dir / "predictions_official.parquet")
    baseline = pd.read_parquet(baseline_dir / "predictions_all.parquet")
    if full.sample_id.duplicated().any() or baseline.sample_id.duplicated().any():
        raise ValueError("comparison predictions need unique sample IDs")
    lines = [
        "# Frozen official test: full seed 0 versus B5 CLIP zero-shot",
        "",
        f"Full run: `{run_dir.name}`; B5 run: `{baseline_dir.name}`.",
        f"Paired item bootstrap: {draws:,} draws, seed 0; two-sided p-values.",
        "Each row uses the exact same official samples for both models. Positive Δ favours",
        "the full model. B5 uses only images; the full model can use text and images.",
        "",
        "| Task | Rows | Full macro-F1 | B5 macro-F1 | Δ [95% CI] | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for title, selector, head, classes, gold_column, variant in TASKS:
        selected = manifest.loc[selectors.official(manifest, selector), ["sample_id", gold_column]]
        aligned = selected.merge(full, on="sample_id", how="left", validate="one_to_one", indicator=True)
        if not (aligned._merge == "both").all():
            raise ValueError(f"full model is missing {selector} official predictions")
        aligned = aligned.drop(columns="_merge").merge(
            baseline, on="sample_id", how="left", validate="one_to_one", indicator=True, suffixes=("_full", "_b5")
        )
        if not (aligned._merge == "both").all():
            raise ValueError(f"B5 is missing {selector} official predictions")
        gold = aligned[gold_column].astype(str).to_numpy(dtype=object)
        gold_gated = variant == "gold_gated"
        prediction_full = _prediction(aligned, head, classes, "full", gold_gated=gold_gated)
        prediction_b5 = _prediction(aligned, head, classes, "b5", gold_gated=gold_gated)

        def macro(y: np.ndarray, predicted: np.ndarray, class_names: list[str] = classes) -> float:
            return metrics.classification(y, predicted, class_names)["macro_f1"]

        point, low, high, p = bootstrap.paired(macro, gold, prediction_full, prediction_b5, n=draws, seed=0)
        full_score = macro(gold, prediction_full)
        b5_score = macro(gold, prediction_b5)
        saved_full = full_metrics["heads"][selector]
        saved_b5 = baseline_metrics["heads"][selector]
        if variant is not None:
            saved_full, saved_b5 = saved_full[variant], saved_b5[variant]
        if abs(saved_full["macro_f1"]["point"] - full_score) > 1e-10 or abs(saved_b5["macro_f1"]["point"] - b5_score) > 1e-10:
            raise ValueError(f"paired {selector} predictions disagree with published metrics")
        p_display = f"<{min(1.0, 2 / draws):.3f}" if p == 0 else f"{p:.3f}"
        lines.append(
            f"| {title} | {len(gold):,} | {full_score:.3f} | {b5_score:.3f} | "
            f"{point:+.3f} [{low:+.3f}, {high:+.3f}] | {p_display} |"
        )
    lines += [
        "",
        "This is a one-seed comparison, not the planned three-seed E2 result. The six",
        f"reported comparisons have no multiple-comparison correction. With {draws:,} draws,",
        f"zero empirical tail counts are reported as p < {min(1.0, 2 / draws):.3f} rather than p = 0.",
        "The paired bootstrap captures test-item sampling for these fixed runs; it does not measure training-seed",
        "variation or make the models' input modalities equivalent.",
        "",
        "## Artifact identities",
        "",
        f"- Full predictions SHA256: `{hash_file(run_dir / 'predictions_official.parquet')}`",
        f"- B5 predictions SHA256: `{hash_file(baseline_dir / 'predictions_all.parquet')}`",
        f"- B5 metrics SHA256: `{hash_file(baseline_dir / 'metrics_test.json')}`",
        "",
    ]
    atomic.text(Path(output), "\n".join(lines))
    return Path(output)
