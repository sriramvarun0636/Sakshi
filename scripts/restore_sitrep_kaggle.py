"""Import one completed private WP5 Kaggle export after checking frozen inputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.generate.reports import summary

EVENTS = {"hurricane_harvey", "hurricane_maria", "mexico_earthquake"}
WINDOW = re.compile(r"\d{8}T\d{6}Z\.json")


def restore(archive: Path, event: str, *, captioned: bool = False) -> Path:
    if event not in EVENTS:
        raise ValueError(f"unsupported research event: {event}")
    suffix = "_captioned" if captioned else ""
    aggregate_root = "aggregate_captioned" if captioned else "aggregate"
    packet_root = "packets_captioned" if captioned else "packets"
    plan_path = paths.REPO / f"runs/generation/evaluation_plan{suffix}.json"
    plan = json.loads(plan_path.read_text())
    model_id = f"full-s0-loeo_{event}"
    chosen = {item["window"]: item["aggregate_sha256"] for item in plan["selected"] if item["event"] == event}
    if not chosen or any(item["model_id"] != model_id for item in plan["selected"] if item["event"] == event):
        raise ValueError("frozen plan has unexpected event model or no windows")
    target = paths.REPO / f"runs/sitreps_hf{suffix}" / model_id / event
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{event}-restore-", dir=target.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            if source.testzip() is not None:
                raise ValueError("sitrep export ZIP checksum failed")
            names = source.namelist()
            event_prefix = f"{event}/"
            event_names = [name for name in names if name.startswith(event_prefix)]
            if len(event_names) != len(set(event_names)) or not event_names:
                raise ValueError("sitrep export has missing or duplicate event entries")
            for name in event_names:
                relative = name.removeprefix(event_prefix)
                if not relative or "/" in relative or relative.startswith("."):
                    raise ValueError(f"unsafe sitrep export entry: {name}")
                if not (relative.endswith(".json") or relative == "config.yaml"):
                    raise ValueError(f"unexpected sitrep export entry: {name}")
                (temporary / relative).write_bytes(source.read(name))
        complete = json.loads((temporary / "complete.json").read_text())
        if (complete["model_id"], complete["event"], complete["mode"]) != (model_id, event, "loeo"):
            raise ValueError("sitrep export identity differs from the frozen research event")
        if complete["windows"] != len(chosen):
            raise ValueError("sitrep export has the wrong number of planned windows")
        if {name for name in complete["artifacts"] if WINDOW.fullmatch(name)} != set(chosen):
            raise ValueError("sitrep export windows differ from the frozen plan")
        for name, digest in complete["artifacts"].items():
            if "/" in name or not (temporary / name).is_file() or hash_file(temporary / name) != digest:
                raise ValueError(f"sitrep completion hash mismatch: {name}")
        request = json.loads((temporary / "request.json").read_text())
        config = request["config"]
        if config["mode"] != "loeo" or config["llm"]["backend"] != "hf":
            raise ValueError("sitrep export is not the frozen HF LOEO protocol")
        inputs = request["inputs"]
        expected_inputs = {
            "aggregate_complete": paths.REPO / f"runs/{aggregate_root}" / model_id / event / "complete.json",
            "tweets": paths.REPO / f"runs/{aggregate_root}" / model_id / event / "tweets.parquet",
            "packets": paths.PROCESSED / packet_root / f"{model_id}.jsonl",
            "evaluation_plan": plan_path,
        }
        for key, path in expected_inputs.items():
            if not path.is_file() or inputs[key] != hash_file(path):
                raise ValueError(f"sitrep export input differs locally: {key}")
        for name, digest in chosen.items():
            source_window = paths.REPO / f"runs/{aggregate_root}" / model_id / event / name
            if inputs["windows"].get(name) != digest or not source_window.is_file() or hash_file(source_window) != digest:
                raise ValueError(f"sitrep export aggregate window changed: {name}")
        provenance = json.loads((temporary / "model_provenance.json").read_text())
        inventory_root = "sitrep-captioned-inputs-20261027" if captioned else "sitrep-inputs-20261026"
        inventory_path = paths.REPO / "runs/transfer" / inventory_root / "inventory.json"
        inventory = json.loads(inventory_path.read_text())
        if provenance["source_commit"] != inventory["git_commit"]:
            raise ValueError("sitrep export source commit differs from staged code")
        if provenance["backend"] != "hf" or provenance["threshold_status"] != "uncalibrated_default_no_human_review":
            raise ValueError("sitrep export has unexpected model or threshold status")
        if bool(provenance.get("captioned", False)) != captioned:
            raise ValueError("sitrep export caption variant differs")
        if (provenance["nli_tau_entail"], provenance["nli_tau_contra"]) != (0.5, 0.3):
            raise ValueError("sitrep export thresholds differ from the recorded defaults")
        if target.exists():
            if sorted(path.name for path in target.iterdir()) != sorted(path.name for path in temporary.iterdir()) or any(
                hash_file(target / path.name) != hash_file(path) for path in temporary.iterdir()
            ):
                raise FileExistsError(f"existing sitrep output differs: {target}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, target)
        summary(paths.REPO / f"runs/sitreps_hf{suffix}", paths.REPORTS / f"generation/summary_hf{suffix}.md")
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("event", choices=sorted(EVENTS))
    parser.add_argument("--captioned", action="store_true")
    args = parser.parse_args()
    print(restore(args.archive, args.event, captioned=args.captioned))
