"""Classifier command line entry point."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from sakshi.config import load
from sakshi.models.train import TrainConfig, run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-model", description="Train the SAKSHI multimodal classifier.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train one classifier run")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--fold")
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--resume", type=Path, help="resume an unchanged run from its last completed epoch")
    calibrate = sub.add_parser("calibrate", help="fit dev temperatures and review thresholds")
    calibrate.add_argument("--run", type=Path, required=True)
    predict = sub.add_parser("predict", help="predict manifest rows from a trained run")
    predict.add_argument("--run", type=Path, required=True)
    predict.add_argument("--rows", choices=("official", "extended", "loeo", "all"), default="official")
    predict.add_argument("--output", type=Path)
    predict.add_argument("--batch-size", type=int, default=32)
    evaluate = sub.add_parser("evaluate", help="score the official held-out test selectors")
    evaluate.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.command == "calibrate":
        from sakshi.models.calibrate_run import calibrate_run

        calibrate_run(args.run)
        return
    if args.command == "predict":
        import pandas as pd

        from sakshi import paths
        from sakshi.eval.selectors import HEAD_LABELS, official
        from sakshi.models.predict import Predictor

        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        config = load(args.run / "config.yaml", TrainConfig)
        manifest = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
        if args.rows == "official":
            mask = pd.Series(False, index=manifest.index)
            for head in HEAD_LABELS:
                mask |= official(manifest, head)
            manifest = manifest.loc[mask]
        elif args.rows == "extended":
            manifest = manifest.loc[manifest.split == "test"]
        elif args.rows == "loeo":
            if config.fold is None:
                parser.error("--rows loeo requires a run with a LOEO fold")
            column = config.fold if config.fold.startswith("loeo_") else f"loeo_{config.fold}"
            manifest = manifest.loc[manifest[column] == "test"]
        output = args.output or args.run / f"predictions_{args.rows}.parquet"
        Predictor.from_run(args.run).write_rows(manifest, output, batch_size=args.batch_size)
        return
    if args.command == "evaluate":
        from sakshi.compute.runner import final_evaluate

        final_evaluate(args.run)
        return
    config = load(args.config, TrainConfig)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.fold is not None:
        config = replace(config, fold=args.fold)
    kwargs = {"output_dir": args.output_dir} if args.output_dir else {}
    if args.resume:
        if args.output_dir:
            parser.error("--resume uses its existing run directory; omit --output-dir")
        kwargs["resume_run"] = args.resume
    run_training(config, **kwargs)


if __name__ == "__main__":
    main()
