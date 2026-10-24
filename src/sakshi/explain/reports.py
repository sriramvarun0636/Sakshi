"""Generate explanation audit tables and curves directly from recorded measurements."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from sakshi import atomic
from sakshi.explain.common import MAIN_HEADS
from sakshi.explain.faithfulness import PERCENTAGES


def interval(values, replicates: int = 1000) -> tuple[float, float, float] | None:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all() or replicates < 1:
        raise ValueError("bootstrap requires finite values and positive replicates")
    rng = np.random.default_rng(0)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(replicates)])
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _format(values, bootstrap):
    result = interval(values, bootstrap)
    return "n/a" if result is None else f"{result[0]:.4f} [{result[1]:.4f}, {result[2]:.4f}]"


def write_report(
    kind: str, results: list[dict], output: Path, *, split: str, count: int, bootstrap: int, heads: tuple[str, ...] = MAIN_HEADS
) -> Path:
    suffix = "_dev" if split == "dev" else ""
    report = output / f"{kind}{suffix}.md"
    text = f"# {kind.title()} audit — {split}\n\n"
    if split == "dev":
        text += "Development check only. These are not the official 500-post faithfulness or 100-post sanity results.\n\n"
    text += (
        f"Requested {count} posts per head, stratified by the frozen model's predicted class (seed 0). "
        f"Mean estimates use {bootstrap} post-level bootstrap replicates and percentile 95% intervals. "
        "Small dev samples are smoke checks, not evidence of explainer quality.\n\n"
    )
    text += "| Head | Sampled | Explained | Gated/unavailable | Flags |\n|---|---:|---:|---:|---|\n"
    for head in heads:
        rows = [row for row in results if row["head"] == head]
        flags = dict(Counter(flag for row in rows for flag in row["flags"]))
        stage_flags = Counter(
            f"{stage}:{flag}" for row in rows for stage, values in row.get("stage_flags", {}).items() for flag in values
        )
        flags.update(stage_flags)
        explained = sum(bool(row["measurements"]) for row in rows)
        text += f"| {head} | {len(rows)} | {explained} | {sum(row['status'] != 'predicted' for row in rows)} | {flags} |\n"
    if kind == "faithfulness":
        text += (
            "\nUncalibrated probabilities; other modality held fixed. Text uses `<mask>` and preserves special tokens; "
            "images zero normalized 16×16 patches and rerun CLIP. Random baselines average seeds 0–4 per post. "
            "Comprehensiveness: higher is better; sufficiency: lower is better. Negative differences are retained.\n\n"
            "| Head | Modality | Method | Posts | AOPC comprehensiveness [95% CI] | AOPC sufficiency [95% CI] |\n"
            "|---|---|---|---:|---|---|\n"
        )
        for head in heads:
            for modality in ("text", "image"):
                measurements = [
                    m for row in results if row["head"] == head for m in row["measurements"] if m["modality"] == modality
                ]
                for method in ("explainer", "random"):
                    if measurements:
                        comp = _format([m["aopc"][method]["comprehensiveness"] for m in measurements], bootstrap)
                        suff = _format([m["aopc"][method]["sufficiency"] for m in measurements], bootstrap)
                        text += f"| {head} | {modality} | {method} | {len(measurements)} | {comp} | {suff} |\n"
        text += "\nComparisons that fail to beat random are retained. Do not retune explainers from test audit results.\n"
        _plot(results, output / f"faithfulness_curves{suffix}.png", heads)
    else:
        text += (
            "\nCascade: head MLP → last cross-attention block → gate. The original predicted target is held fixed. "
            "Constant or length-one vectors have undefined Spearman correlation and are excluded from the mean, "
            "with their counts retained. Gate randomisation does not affect the image-only head readout.\n\n"
            "| Head | Modality | Stage | Defined | Undefined | Mean Spearman rho [95% CI] |\n"
            "|---|---|---|---:|---:|---|\n"
        )
        for head in heads:
            for modality in ("text", "image"):
                for stage in ("head", "cross_attention", "gate"):
                    values = [
                        m["rho"]
                        for row in results
                        if row["head"] == head
                        for m in row["measurements"]
                        if m["modality"] == modality and m["stage"] == stage
                    ]
                    if values:
                        defined = [value for value in values if value is not None]
                        text += (
                            f"| {head} | {modality} | {stage} | {len(defined)} | {len(values) - len(defined)} | "
                            f"{_format(defined, bootstrap)} |\n"
                        )
    atomic.text(report, text)
    return report


def _plot(results, output, heads):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(heads), figsize=(4 * len(heads), 7), squeeze=False)
    for column, head in enumerate(heads):
        for row, metric in enumerate(("comprehensiveness", "sufficiency")):
            axis = axes[row, column]
            for modality, colour in (("text", "#2563eb"), ("image", "#d97706")):
                measurements = [
                    m for result in results if result["head"] == head for m in result["measurements"] if m["modality"] == modality
                ]
                for method, style in (("explainer", "-"), ("random", "--")):
                    if measurements:
                        means = np.mean([[point[metric] for point in m[method]] for m in measurements], axis=0)
                        axis.plot(PERCENTAGES, means, style, color=colour, marker="o", label=f"{modality}: {method}")
            axis.set(title=head.replace("_", " "), xlabel="Top units (%)", ylabel=metric)
            axis.axhline(0, color="#94a3b8", linewidth=0.7)
            if axis.lines and axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
    figure.tight_layout()
    atomic.write(output, lambda temporary: figure.savefig(temporary, dpi=160))
    plt.close(figure)
