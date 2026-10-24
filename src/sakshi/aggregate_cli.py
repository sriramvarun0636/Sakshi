"""Run model-scoped UTC aggregation and gold-label ranking audits."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from sakshi.aggregate.workflow import AggregateConfig, run_aggregation
from sakshi.config import load
from sakshi.data.events import EVENTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-aggregate", description="Aggregate event tweets and evaluate urgency ranking.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--event", choices=tuple(EVENTS))
    parser.add_argument("--window", choices=("6h", "12h", "24h", "48h"))
    parser.add_argument("--mode", choices=("loeo", "heldout", "demo"))
    parser.add_argument("--predictions")
    parser.add_argument("--run", dest="classifier_run")
    parser.add_argument("--packets")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    config = load(args.config, AggregateConfig) if args.config else AggregateConfig()
    changes = {
        name: getattr(args, name)
        for name in ("model_id", "event", "mode", "predictions", "classifier_run", "packets", "seed", "output_root")
        if getattr(args, name) is not None
    }
    if args.window:
        changes["window_hours"] = int(args.window.removesuffix("h"))
    config = replace(config, **changes)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    print(run_aggregation(config))


if __name__ == "__main__":
    main()
