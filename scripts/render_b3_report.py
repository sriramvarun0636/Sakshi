"""Summarise frozen B3 late-fusion runs across the three standard seeds."""

from __future__ import annotations

import json

import pandas as pd
from render_experiments_report import effect, frozen_run

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.models.comparison import TASKS


def b3_run(seed: int) -> tuple | None:
    matches = sorted((paths.REPO / "runs/baselines").glob(f"b3-late-s{seed}-*"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous B3 seed {seed}: {matches}")
    run = matches[0]
    marker = run / "complete.json"
    if not marker.is_file():
        return None
    checks = json.loads(marker.read_text())
    if checks != {name: hash_file(run / name) for name in checks}:
        raise ValueError(f"B3 artifacts changed after freeze: {run}")
    return run, json.loads((run / "metrics_test.json").read_text())


def render() -> str:
    full = [frozen_run("full", seed) for seed in (0, 1, 2)]
    b3 = [b3_run(seed) for seed in (0, 1, 2)]
    if any(item is None for item in full):
        raise ValueError("three frozen full-model seeds are required")
    count = sum(item is not None for item in b3)
    lines = [
        "# B3 late fusion · standard split",
        "",
        "B3 tunes informative and conditional-humanitarian mixing weights separately on each seed's",
        "agreed development rows, then scores the frozen official test selectors. Severity and disaster",
        "type predictions come from that seed's B2 image model.",
        "",
        f"Frozen seeds: **{count}/3**.",
    ]
    if count != 3:
        return "\n".join(lines + [""])
    lines += [
        "",
        "Δ is B3 minus full macro-F1. The 95% interval resamples the same test posts across",
        "all three seeds (1,000 paired draws); seed SD describes the three run effects.",
        "",
        "| Selector | n | Full F1 | B3 F1 | Δ [95% CI] | Seed SD of Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    for task in TASKS:
        name = task[0]
        n, baseline, variant, delta, low, high, sd = effect(manifest, full, b3, task)
        lines.append(f"| {name} | {n:,} | {baseline:.3f} | {variant:.3f} | {delta:+.3f} [{low:+.3f}, {high:+.3f}] | {sd:.3f} |")
    weights = [json.loads((run / "weights.json").read_text()) for run, _ in b3]
    lines += [
        "",
        "## Development choices",
        "",
        "| Seed | Informative text weight | Conditional-humanitarian text weight | Dev rows |",
        "|---:|---:|---:|---:|",
    ]
    for seed, choice in enumerate(weights):
        lines.append(
            f"| {seed} | {choice['weight_informative']:.1f} | "
            f"{choice['weight_humanitarian_conditional']:.1f} | {choice['dev_rows']:,} |"
        )
    lines += [
        "",
        "The intervals capture test-post sampling for these fixed runs, not every possible",
        "training seed. No multiple-comparison correction was applied.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    destination = paths.REPORTS / "results/b3_three_seed.md"
    destination.write_text(render())
    print(destination)
