"""Command-line entry point for published-comparable baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sakshi.baselines.clip_zeroshot import run_clip_zeroshot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-baselines", description="Official-set classifier baselines.")
    sub = parser.add_subparsers(dest="command", required=True)
    clip = sub.add_parser("clip-zeroshot", help="run fixed Appendix E.8 CLIP image prompts")
    clip.add_argument("--bootstrap", type=int, default=1000)
    late = sub.add_parser("late-fusion", help="derive B3 from frozen B1/B2 runs with dev-tuned weights")
    late.add_argument("--text-run", type=Path, required=True)
    late.add_argument("--image-run", type=Path, required=True)
    late.add_argument("--output", type=Path, required=True)
    late.add_argument("--bootstrap", type=int, default=1000)
    late.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.command == "clip-zeroshot":
        print(run_clip_zeroshot(bootstrap=args.bootstrap))
    elif args.command == "late-fusion":
        from sakshi.baselines.late_fusion import run_late_fusion

        print(run_late_fusion(args.text_run, args.image_run, args.output, bootstrap=args.bootstrap, device=args.device))


if __name__ == "__main__":
    main()
