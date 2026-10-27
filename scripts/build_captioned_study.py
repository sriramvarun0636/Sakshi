"""Rebind frozen LOEO aggregation and the 53-window plan to captioned packets."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sakshi import paths
from sakshi.aggregate.workflow import AggregateConfig, run_aggregation
from sakshi.generate.workflow import build_evaluation_plan

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")


def build() -> Path:
    root = paths.REPO / "runs/aggregate_captioned"
    for event in EVENTS:
        model_id = f"full-s0-loeo_{event}"
        original = paths.REPO / "runs/aggregate" / model_id / event / "config.json"
        base = AggregateConfig(**json.loads(original.read_text()))
        config = replace(
            base,
            packets=str(paths.PROCESSED / "packets_captioned" / f"{model_id}.jsonl"),
            output_root=str(root),
            report_dir="reports/results_captioned",
        )
        print(run_aggregation(config), flush=True)
    target = paths.REPO / "runs/generation/evaluation_plan_captioned.json"
    build_evaluation_plan(root, target)
    frozen = json.loads((paths.REPO / "runs/generation/evaluation_plan.json").read_text())
    captioned = json.loads(target.read_text())
    def key(item: dict) -> tuple[str, str, str]:
        return item["event"], item["window"], item["model_id"]
    if [key(item) for item in frozen["selected"]] != [key(item) for item in captioned["selected"]]:
        raise ValueError("captioned study changed the 53-window selection")
    print({"windows": len(captioned["selected"]), "plan": str(target)})
    return target


if __name__ == "__main__":
    build()
