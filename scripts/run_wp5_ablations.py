"""Score WP5 fixed-sentence gate ablations on the frozen captioned sitreps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sakshi import atomic, paths
from sakshi.aggregate.packets import load_packets
from sakshi.compute.files import hash_file
from sakshi.generate.ablations import evaluate_window, summarize
from sakshi.generate.verify import DebertaNLI, Verifier

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")
REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"


def run(model_path: str, output: Path, report_path: Path) -> Path:
    plan_path = paths.REPO / "runs/generation/evaluation_plan_captioned.json"
    plan = json.loads(plan_path.read_text())
    if len(plan["selected"]) != 53:
        raise ValueError("captioned 53-window plan differs")
    verifier = Verifier(DebertaNLI(model_path), tau_entail=0.5, tau_contra=0.3)
    measurements = []
    inputs = {"plan": hash_file(plan_path)}
    for event in EVENTS:
        model_id = f"full-s0-loeo_{event}"
        packet_path = paths.PROCESSED / "packets_captioned" / f"{model_id}.jsonl"
        packets = load_packets(packet_path, model_id)
        directory = paths.REPO / "runs/sitreps_hf_captioned" / model_id / event
        marker = directory / "complete.json"
        completed = json.loads(marker.read_text())
        planned = {item["window"] for item in plan["selected"] if item["event"] == event}
        if completed["mode"] != "loeo" or completed["windows"] != len(planned):
            raise ValueError(f"captioned sitrep completion differs for {event}")
        for name, digest in completed["artifacts"].items():
            if hash_file(directory / name) != digest:
                raise ValueError(f"captioned sitrep artifact changed: {event}/{name}")
        if {name for name in completed["artifacts"] if name.endswith("Z.json")} != planned:
            raise ValueError(f"captioned sitrep windows differ for {event}")
        inputs[event] = {"packets": hash_file(packet_path), "sitreps": hash_file(marker)}
        for name in sorted(planned):
            item = evaluate_window(json.loads((directory / name).read_text()), packets, verifier)
            measurements.append(item)
            print({"event": event, "window": name, "round0": item["round0"]}, flush=True)
    if len(measurements) != 53:
        raise ValueError("ablation output window count changed")
    result = {
        "scope": "captioned_loeo_fixed_round0",
        "nli_revision": REVISION,
        "tau_entail": 0.5,
        "tau_contra": 0.3,
        "inputs": inputs,
        "measurements": measurements,
    }
    atomic.text(output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    atomic.text(report_path, summarize(measurements))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local directory of the pinned DeBERTa NLI revision")
    parser.add_argument("--output", type=Path, default=paths.REPO / "runs/ablation_captioned/results.json")
    parser.add_argument("--report", type=Path, default=paths.REPORTS / "generation/ablations_captioned.md")
    args = parser.parse_args()
    print(run(args.model, args.output, args.report))
