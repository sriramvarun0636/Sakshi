"""Summarise seven frozen B3 late-fusion leave-one-event-out folds."""

from __future__ import annotations

import json

import numpy as np
from render_loeo_report import EVENTS, metric

from sakshi import paths
from sakshi.compute.files import hash_file


def render() -> str:
    root = paths.REPO / "runs/baselines"
    found = {}
    for event in EVENTS:
        matches = sorted(root.glob(f"b3-late-s0-loeo_{event}-*"))
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"ambiguous B3 fold for {event}: {matches}")
        run = matches[0]
        marker = run / "complete.json"
        if not marker.is_file():
            continue
        checks = json.loads(marker.read_text())
        if checks != {name: hash_file(run / name) for name in checks}:
            raise ValueError(f"B3 LOEO fold changed after freeze: {run}")
        result = json.loads((run / "metrics_loeo.json").read_text())
        if result["event"] != event or result["baseline_id"] != "B3" or result["contradiction_rate"] != 0:
            raise ValueError(f"B3 LOEO evaluation is inconsistent: {run}")
        weights = json.loads((run / "weights.json").read_text())
        found[event] = (result, weights)
    missing = [event for event in EVENTS if event not in found]
    lines = [
        "# B3 late fusion · leave one event out",
        "",
        "Each fold tunes informative and conditional-humanitarian mixing weights on its own",
        "development events, then scores the held-out event. Severity comes unchanged from B2.",
        "Macro-F1 uses the same agreed-gold fold selectors as the full, B1 and B2 models.",
        "",
        f"Frozen folds: **{len(found)}/7**."
        + (f" Missing: {', '.join(missing)}." if missing else " The seven-fold estimate is complete."),
        "",
        "| Held-out event | Rows | Informative n / F1 | Humanitarian n / F1 | Severity n / F1 | Text weight I / H |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    values = {head: [] for head in ("informative", "humanitarian", "severity")}
    for event in EVENTS:
        if event not in found:
            continue
        result, weights = found[event]
        scores = {head: metric(result["heads"], head) for head in values}
        for head, (_, f1) in scores.items():
            values[head].append(f1)
        lines.append(
            f"| {event.replace('_', ' ').title()} | {result['n_test_rows']:,} | "
            + " | ".join(f"{n:,} / {f1:.3f}" for n, f1 in scores.values())
            + f" | {weights['weight_informative']:.1f} / {weights['weight_humanitarian_conditional']:.1f} |"
        )
    if not missing:
        lines += [
            "",
            "Seven-fold macro-F1 (unweighted mean ± sample SD): "
            + "; ".join(f"{head} {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}" for head, scores in values.items())
            + ".",
        ]
    lines += [
        "",
        "Fold sizes and class mixes differ; these are unweighted event-transfer estimates,",
        "not pooled post-level scores. B3 severity is inherited from B2 rather than fused.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    destination = paths.REPORTS / "results/b3_loeo.md"
    destination.write_text(render())
    print(destination)
