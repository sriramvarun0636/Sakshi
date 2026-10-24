"""Prepare private GPU inputs, inspect caches, smoke training and export a run."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sakshi import atomic, paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-gpu")
    sub = parser.add_subparsers(dest="command", required=True)
    bundle = sub.add_parser("bundle", help="prepare a local private-upload folder; no network upload")
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--processed", type=Path, default=paths.PROCESSED)
    check = sub.add_parser("check", help="inspect mounted prepared inputs and split isolation")
    smoke = sub.add_parser("smoke", help="forward/backward/update all modality patterns on CUDA")
    for command in (check, smoke):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--allow-cpu", action="store_true", help="developer smoke only; never claims GPU validation")
    evaluate = sub.add_parser("final-evaluate", help="freeze calibrated inputs and score the official tests once")
    evaluate.add_argument("--run", type=Path, required=True)
    export = sub.add_parser("export", help="archive a run into checksummed download parts")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-resume", action="store_true")
    restore = sub.add_parser("restore", help="verify downloaded archive parts and restore one run")
    restore.add_argument("--export", dest="export_dir", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.command == "bundle":
        from sakshi.compute.bundle import prepare_bundle

        logging.info("bundle ready: %s", prepare_bundle(args.output, processed=args.processed))
    elif args.command in ("check", "smoke"):
        from sakshi.config import load
        from sakshi.models.train import TrainConfig

        config = load(args.config, TrainConfig)
        if args.command == "check":
            from sakshi.compute.preflight import inspect_inputs

            atomic.text(args.output, json.dumps(inspect_inputs(config), indent=2) + "\n")
        else:
            from sakshi.compute.runner import training_smoke

            training_smoke(config, args.output, require_cuda=not args.allow_cpu)
    elif args.command == "final-evaluate":
        from sakshi.compute.runner import final_evaluate

        logging.info("final metrics: %s", final_evaluate(args.run))
    elif args.command == "export":
        from sakshi.compute.runner import export_run

        logging.info("export ready: %s", export_run(args.run, args.output, include_resume=args.include_resume))
    else:
        from sakshi.compute.runner import restore_export

        logging.info("run restored: %s", restore_export(args.export_dir, args.output))


if __name__ == "__main__":
    main()
