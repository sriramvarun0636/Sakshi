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
    loeo = sub.add_parser("evaluate-loeo", help="freeze and score one run's held-out event")
    loeo.add_argument("--run", type=Path, required=True)
    report = sub.add_parser("report", help="render a hash-verified official test report")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path)
    compare = sub.add_parser("compare", help="paired official-test comparison with the completed B5 baseline")
    compare.add_argument("--run", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--output", type=Path)
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
            heads = ("informative", "humanitarian") if config.model.modality == "text" else HEAD_LABELS
            for head in heads:
                mask |= official(manifest, head)
            manifest = manifest.loc[mask]
        elif args.rows == "extended":
            manifest = manifest.loc[manifest.split == "test"]
        elif args.rows == "loeo":
            if config.fold is None:
                parser.error("--rows loeo requires a run with a LOEO fold")
            column = config.fold if config.fold.startswith("loeo_") else f"loeo_{config.fold}"
            manifest = manifest.loc[manifest[column] == "test"]
        if config.model.modality == "text":
            from sakshi.text import normalise_for_model

            manifest = manifest.loc[
                manifest.text.map(lambda value: isinstance(value, str) and normalise_for_model(value) is not None)
            ]
        elif config.model.modality == "image":
            from sakshi.models.data import FeatureStore

            features = FeatureStore()
            manifest = manifest.loc[manifest.sample_id.map(features.has_image)]
        output = args.output or args.run / f"predictions_{args.rows}.parquet"
        Predictor.from_run(args.run).write_rows(manifest, output, batch_size=args.batch_size)
        return
    if args.command == "evaluate":
        from sakshi.compute.runner import final_evaluate

        final_evaluate(args.run)
        return
    if args.command == "evaluate-loeo":
        from sakshi.models.evaluate_loeo import evaluate_loeo_run

        evaluate_loeo_run(args.run)
        return
    if args.command == "report":
        from sakshi.models.figures import render_classifier_figures
        from sakshi.models.report import render_classifier_report

        kwargs = {"output": args.output} if args.output else {}
        render_classifier_report(args.run, **kwargs)
        render_classifier_figures(args.run)
        return
    if args.command == "compare":
        from sakshi.models.comparison import render_b5_comparison

        kwargs = {"output": args.output} if args.output else {}
        render_b5_comparison(args.run, args.baseline, **kwargs)
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
