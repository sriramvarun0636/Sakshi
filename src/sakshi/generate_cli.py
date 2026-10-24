"""WP5 generation, caption and human-audit commands."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from sakshi import paths
from sakshi.config import load
from sakshi.generate.audit import export_audit, score_audit
from sakshi.generate.captions import generate_captions
from sakshi.generate.llm import CachedLLM, OllamaClient
from sakshi.generate.rationale import explain_post
from sakshi.generate.verify import DebertaNLI
from sakshi.generate.workflow import GenerateConfig, build_evaluation_plan, run_sitreps


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sakshi-generate")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--aggregate-root", type=Path, default=paths.REPO / "runs/aggregate")
    plan.add_argument("--output", type=Path, default=paths.REPO / "runs/generation/evaluation_plan.json")
    sitreps = sub.add_parser("sitreps")
    sitreps.add_argument("--config", type=Path, default=paths.REPO / "configs/generation/sitrep.yaml")
    sitreps.add_argument("--aggregate-dir", required=True)
    sitreps.add_argument("--packets", required=True)
    sitreps.add_argument("--mode", choices=("loeo", "heldout", "demo"), default="loeo")
    sitreps.add_argument("--output-root")
    sitreps.add_argument("--plan")
    audit_export = sub.add_parser("audit-export")
    audit_export.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    audit_export.add_argument("--sitrep-root", type=Path, default=paths.REPO / "runs/sitreps")
    audit_export.add_argument("--sheet", type=Path)
    audit_export.add_argument("--viewer", type=Path)
    audit_export.add_argument("--manifest", type=Path, default=paths.PROCESSED / "manifest.parquet")
    audit_export.add_argument("--calibration", type=Path, default=paths.REPO / "reports/generation/audit_calibration.json")
    audit_score = sub.add_parser("audit-score")
    audit_score.add_argument("--sheet", type=Path)
    audit_score.add_argument("--metadata", type=Path)
    audit_score.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    audit_score.add_argument("--output", type=Path)
    captions = sub.add_parser("captions")
    captions.add_argument("--ids", type=Path, required=True, help="JSON array of sample IDs selected for LLM inputs")
    captions.add_argument("--cache", type=Path, default=paths.PROCESSED / "captions.parquet")
    captions.add_argument("--manifest", type=Path, default=paths.PROCESSED / "manifest.parquet")
    rationale = sub.add_parser("explanations")
    rationale.add_argument("--packet", type=Path, required=True, help="one JSON evidence packet")
    rationale.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.command == "plan":
        try:
            print(build_evaluation_plan(args.aggregate_root, args.output))
        except ValueError as exc:
            parser.error(str(exc))
    elif args.command == "sitreps":
        config = load(args.config, GenerateConfig)
        config = replace(
            config,
            aggregate_dir=args.aggregate_dir,
            packets=args.packets,
            mode=args.mode,
            output_root=args.output_root or config.output_root,
            evaluation_plan=args.plan or config.evaluation_plan,
        )
        print(run_sitreps(config))
    elif args.command == "audit-export":
        name = "calibration_sheet" if args.phase == "calibration" else "audit_sheet"
        sheet = args.sheet or paths.REPO / f"reports/generation/{name}.csv"
        viewer = args.viewer or paths.REPO / f"runs/generation/{args.phase}_viewer.html"
        print(
            export_audit(
                args.sitrep_root,
                sheet,
                viewer,
                manifest=args.manifest,
                phase=args.phase,
                calibration_path=args.calibration,
            )
        )
    elif args.command == "audit-score":
        output = args.output or paths.REPO / f"reports/generation/audit_{args.phase}.json"
        name = "calibration_sheet" if args.phase == "calibration" else "audit_sheet"
        sheet = args.sheet or paths.REPO / f"reports/generation/{name}.csv"
        metadata = args.metadata or paths.REPO / f"runs/generation/{args.phase}_sampling.json"
        print(score_audit(sheet, metadata, output, phase=args.phase))
    elif args.command == "captions":
        print(generate_captions(json.loads(args.ids.read_text(encoding="utf-8")), args.cache, manifest=args.manifest))
    else:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        from sakshi.explain.packets import validate_packet

        validate_packet(packet)
        from hashlib import sha256

        packet_sha = sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if args.output.exists():
            saved = json.loads(args.output.read_text(encoding="utf-8"))
            if saved.get("packet_sha256") != packet_sha:
                raise FileExistsError("cached per-post explanation belongs to a different packet")
            print(args.output)
            return
        result = explain_post(packet, CachedLLM(OllamaClient()), DebertaNLI())
        from sakshi import atomic

        atomic.text(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
