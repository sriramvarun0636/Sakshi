"""Command-line interface: `sakshi-data download` and `sakshi-data prepare`."""

from __future__ import annotations

import argparse
import logging

from sakshi.data import download, pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-data", description="Download and prepare the SAKSHI datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download the source datasets (resumable)")
    dl.add_argument("sources", nargs="*", help=f"any of: {', '.join(download.SOURCES)} (default: all)")
    dl.add_argument("--parts", type=int, default=8, help="parallel byte ranges per file (default: 8)")

    prep = sub.add_parser("prepare", help="run the data preparation pipeline")
    prep.add_argument("steps", nargs="*", help=f"any of: {', '.join(pipeline.STEPS)} (default: all, in order)")
    prep.add_argument(
        "--max-distance", type=int, default=pipeline.Config.max_distance, help="pHash linking threshold (default: 8)"
    )
    prep.add_argument("--workers", type=int, default=pipeline.Config.workers, help="processes for image hashing (default: 8)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if args.command == "download":
        unknown = set(args.sources) - set(download.SOURCES)
        if unknown:
            parser.error(f"unknown sources: {', '.join(sorted(unknown))}")
        for name in args.sources or download.SOURCES:
            download.download(download.SOURCES[name], parts=args.parts)
    else:
        unknown = set(args.steps) - set(pipeline.STEPS)
        if unknown:
            parser.error(f"unknown steps: {', '.join(sorted(unknown))}")
        pipeline.run(args.steps, pipeline.Config(max_distance=args.max_distance, workers=args.workers))


if __name__ == "__main__":
    main()
