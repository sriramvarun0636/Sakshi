"""Render the frozen full-model LOEO folds without treating partial coverage as final."""

from __future__ import annotations

import json

import numpy as np

from sakshi import paths
from sakshi.compute.files import hash_file

EVENTS = (
    "srilanka_floods",
    "hurricane_harvey",
    "hurricane_irma",
    "hurricane_maria",
    "mexico_earthquake",
    "california_wildfires",
    "iraq_iran_earthquake",
)


def metric(heads: dict, name: str) -> tuple[int, float]:
    score = heads[name]["gold_gated"] if name == "severity" else heads[name]
    return score["n"], score["macro_f1"]["point"]


def render() -> str:
    found = {}
    for run in sorted((paths.REPO / "runs/classifier").glob("full-s0-loeo_*")):
        marker = run / "evaluation_loeo_complete.json"
        if not marker.is_file():
            continue
        expected = json.loads(marker.read_text())
        if expected != {name: hash_file(run / name) for name in expected}:
            raise ValueError(f"LOEO evaluation changed after freeze: {run}")
        result = json.loads((run / "metrics_loeo.json").read_text())
        event = result["event"]
        if event in found:
            raise ValueError(f"multiple frozen full runs for {event}")
        found[event] = (run, result)
    if not found:
        raise ValueError("no frozen full-model LOEO folds")
    unknown = set(found) - set(EVENTS)
    if unknown:
        raise ValueError(f"unexpected LOEO events: {sorted(unknown)}")
    missing = [event for event in EVENTS if event not in found]
    lines = [
        "# Full-model leave-one-event-out evaluation",
        "",
        "Each row is seed 0 trained without the named CrisisMMD event in train or dev,",
        "calibrated on that fold's dev rows, then evaluated once on its frozen held-out",
        "event predictions. Macro-F1 uses only classes present in that event's gold labels.",
        "Severity is gold-gated. Disaster type has no CrisisMMD LOEO gold labels.",
        "",
        f"Frozen folds: **{len(found)}/7**."
        + (f" Still missing: {', '.join(missing)}." if missing else " The seven-fold estimate is complete."),
        "",
        "| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 | Severity n / F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    values = {name: [] for name in ("informative", "humanitarian", "severity")}
    for event in EVENTS:
        if event not in found:
            continue
        _, result = found[event]
        if result["contradiction_rate"] != 0:
            raise ValueError(f"hierarchy contradiction in {event}")
        scores = [metric(result["heads"], name) for name in values]
        for name, (_, score) in zip(values, scores, strict=True):
            values[name].append(score)
        lines.append(
            f"| {event.replace('_', ' ').title()} | {result['n_test_rows']:,} | "
            + " | ".join(f"{n:,} / {value:.3f}" for n, value in scores)
            + " |"
        )
    if not missing:
        lines += [
            "",
            "Seven-fold macro-F1 (mean ± sample SD across events): "
            + "; ".join(
                f"{name.replace('_', ' ')} {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}"
                for name, scores in values.items()
            )
            + ".",
        ]
    lines += [
        "",
        "The event folds vary in sample count and class mix. Their unweighted mean estimates",
        "transfer to a new event, not the pooled performance on all held-out samples.",
        "Individual fold JSON files preserve per-class scores, present classes, 1,000-draw",
        "bootstrap intervals and both severity gates. B1/B2/B3 fold comparisons remain separate.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    output = paths.REPO / "reports/results/loeo.md"
    output.write_text(render())
    print(output)
