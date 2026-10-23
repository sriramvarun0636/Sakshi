"""Command-line entry point for published-comparable baselines."""

from __future__ import annotations

import argparse
import logging

from sakshi.baselines.clip_zeroshot import run_clip_zeroshot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-baselines", description="Official-set classifier baselines.")
    sub = parser.add_subparsers(dest="command", required=True)
    clip = sub.add_parser("clip-zeroshot", help="run fixed Appendix E.8 CLIP image prompts")
    clip.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.command == "clip-zeroshot":
        print(run_clip_zeroshot(bootstrap=args.bootstrap))


if __name__ == "__main__":
    main()
