"""Summarise frozen B1/B2 held-out-event runs without completing partial means."""

from __future__ import annotations

import json

import numpy as np

from sakshi import paths
from sakshi.compute.files import hash_file
from render_loeo_report import EVENTS, metric

BASELINES = {"b1-text": "B1 text only", "b2-image": "B2 image only"}


def render() -> str:
    lines = [
        "# B1/B2 leave-one-event-out evaluation",
        "",
        "Each seed-zero baseline was trained without the named CrisisMMD event,",
        "calibrated on the fold's development data, and scored on the same frozen",
        "held-out event and agreed gold labels as the full model. Severity is gold-gated.",
        "Macro-F1 averages only classes present in that event's gold labels.",
    ]
    for prefix, title in BASELINES.items():
        found = {}
        for run in sorted((paths.REPO / "runs/classifier").glob(f"{prefix}-s0-loeo_*")):
            marker = run / "evaluation_loeo_complete.json"
            if not marker.is_file():
                continue
            expected = json.loads(marker.read_text())
            if expected != {name: hash_file(run / name) for name in expected}:
                raise ValueError(f"LOEO evaluation changed after freeze: {run}")
            result = json.loads((run / "metrics_loeo.json").read_text())
            event = result["event"]
            if event not in EVENTS or event in found:
                raise ValueError(f"unexpected or duplicate {title} event: {event}")
            if result["contradiction_rate"] != 0:
                raise ValueError(f"hierarchy contradiction in {run}")
            found[event] = result
        missing = [event for event in EVENTS if event not in found]
        names = ("informative", "humanitarian") if prefix == "b1-text" else (
            "informative", "humanitarian", "severity"
        )
        lines += [
            "",
            f"## {title}",
            "",
            f"Frozen folds: **{len(found)}/7**."
            + (f" Missing: {', '.join(missing)}." if missing else " The seven-fold estimate is complete."),
            "",
            "| Held-out event | Rows | " + " | ".join(name.title() + " n / F1" for name in names) + " |",
            "|---|---:|" + "---:|" * len(names),
        ]
        values = {name: [] for name in names}
        for event in EVENTS:
            if event not in found:
                continue
            result = found[event]
            scores = [metric(result["heads"], name) for name in names]
            for name, (_, score) in zip(names, scores, strict=True):
                values[name].append(score)
            lines.append(
                f"| {event.replace('_', ' ').title()} | {result['n_test_rows']:,} | "
                + " | ".join(f"{n:,} / {score:.3f}" for n, score in scores)
                + " |"
            )
        if not missing:
            lines += [
                "",
                "Seven-fold macro-F1 (unweighted mean ± sample SD): "
                + "; ".join(
                    f"{name.replace('_', ' ')} {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}"
                    for name, scores in values.items()
                )
                + ".",
            ]
    lines += [
        "",
        "The folds vary in size and class mix. These unweighted means describe transfer",
        "across events; they are not pooled sample scores. B3 late fusion is reported separately.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    output = paths.REPO / "reports/results/loeo_baselines.md"
    output.write_text(render())
    print(output)
