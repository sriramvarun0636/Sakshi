"""Generate same-model evidence packets and run explanation audits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sakshi import paths
from sakshi.explain.common import MAIN_HEADS
from sakshi.explain.workflow import audit, explain, packets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-explain")
    sub = parser.add_subparsers(dest="command", required=True)
    packet = sub.add_parser("packets", help="generate private packets and explanation diagnostics")
    packet.add_argument("--rows", required=True, help="train/dev/test, official:<head>, or Parquet with sample_id")
    packet.add_argument("--limit", type=int, help="explicit subset for development checks")
    for name in ("ig", "gradcam"):
        command = sub.add_parser(name, help="explain one modality; gated/unavailable heads are skipped")
        command.add_argument("--rows", required=True)
        command.add_argument("--head", choices=MAIN_HEADS[:2] if name == "ig" else MAIN_HEADS, required=True)
        command.add_argument("--limit", type=int)
        if name == "gradcam":
            command.add_argument("--overlays", action="store_true", help="also export face-blurred overlays")
    for name in ("faithfulness", "sanity"):
        command = sub.add_parser(name, help="resumable per-post audit; test requires a frozen classifier")
        command.add_argument("--split", choices=("dev", "test"), default="dev")
        command.add_argument("--count", type=int, help="posts per head; defaults to 500/100")
        command.add_argument("--heads", nargs="+", choices=MAIN_HEADS, default=list(MAIN_HEADS))
        command.add_argument("--bootstrap", type=int, default=1000)
    for command in sub.choices.values():
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--output", type=Path)
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_id = json.loads((args.run / "env.json").read_text())["model_id"]
    if args.command == "packets":
        output = args.output or paths.PROCESSED / "packets" / f"{model_id}.jsonl"
        print(packets(args.run, selector=args.rows, output=output, device=args.device, limit=args.limit))
    elif args.command in ("ig", "gradcam"):
        output = args.output or paths.REPO / "runs/explain" / model_id / f"{args.command}-{args.head}.jsonl"
        print(
            explain(
                args.run,
                kind=args.command,
                selector=args.rows,
                head=args.head,
                output=output,
                device=args.device,
                limit=args.limit,
                overlays=getattr(args, "overlays", False),
            )
        )
    else:
        output = args.output or paths.REPO / "runs/explain" / model_id / f"{args.command}-{args.split}"
        print(
            audit(
                args.run,
                kind=args.command,
                output=output,
                split=args.split,
                count=args.count,
                device=args.device,
                heads=tuple(args.heads),
                bootstrap=args.bootstrap,
            )
        )


if __name__ == "__main__":
    main()
