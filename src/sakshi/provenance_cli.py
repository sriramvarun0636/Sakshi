"""CLI for building and replaying the temporal provenance index."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from sakshi import paths
from sakshi.config import load
from sakshi.data.events import EVENTS
from sakshi.provenance.benchmark import BenchmarkConfig, run_benchmark
from sakshi.provenance.index import ProvenanceIndex, build_index
from sakshi.provenance.labelling import export_labels, score_labels
from sakshi.provenance.replay import replay_all, replay_event


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-provenance", description="Temporal image provenance detection.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-index", help="join the manifest with CLIP embeddings and flag graphics")
    replay = sub.add_parser("replay", help="check dated CrisisMMD images chronologically")
    replay.add_argument("--event", choices=(*EVENTS, "all"), required=True)
    benchmark = sub.add_parser("benchmark", help="run the closed-world injection benchmark")
    benchmark.add_argument("--config", default="configs/provenance/benchmark.yaml")
    sub.add_parser("label-export", help="export blinded pairs and local image viewer for two raters")
    sub.add_parser("label-score", help="score a fully adjudicated provenance labelling sheet")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.command == "build-index":
        build_index(pd.read_parquet(paths.PROCESSED / "manifest.parquet"))
    elif args.command == "benchmark":
        run_benchmark(load(args.config, BenchmarkConfig))
    elif args.command == "label-export":
        export_labels()
    elif args.command == "label-score":
        score_labels()
    else:
        index = ProvenanceIndex.load()
        if args.event == "all":
            replay_all(index)
        else:
            replay_event(index, args.event)


if __name__ == "__main__":
    main()
