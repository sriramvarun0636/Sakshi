"""Restore the private WP5 ablation export after checking inputs and report derivation."""

from __future__ import annotations

import argparse
import json
import zipfile
from hashlib import sha256
from pathlib import Path

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.generate.ablations import summarize

REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")


def restore(archive: Path, staged: Path) -> Path:
    inventory = json.loads((staged / "inventory.json").read_text())
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None or set(source.namelist()) != {
            "results.json", "ablations_captioned.md", "completion.json"
        }:
            raise ValueError("unexpected WP5 ablation export contents")
        result_bytes = source.read("results.json")
        report_bytes = source.read("ablations_captioned.md")
        marker_bytes = source.read("completion.json")
    result = json.loads(result_bytes)
    marker = json.loads(marker_bytes)
    if marker != {
        "source_commit": inventory["git_commit"],
        "nli_repository": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "nli_revision": REVISION,
        "windows": 53,
        "result_sha256": sha256(result_bytes).hexdigest(),
        "report_sha256": sha256(report_bytes).hexdigest(),
    }:
        raise ValueError("WP5 ablation completion marker differs")
    if result["scope"] != "captioned_loeo_fixed_round0" or result["nli_revision"] != REVISION:
        raise ValueError("WP5 ablation protocol differs")
    if (result["tau_entail"], result["tau_contra"]) != (0.5, 0.3):
        raise ValueError("WP5 ablation thresholds differ")
    measurements = result["measurements"]
    plan_path = paths.REPO / "runs/generation/evaluation_plan_captioned.json"
    plan = json.loads(plan_path.read_text())
    expected = {(item["event"], item["window"]) for item in plan["selected"]}
    actual = {(item["event"], item["window"].replace("-", "").replace(":", "") + ".json") for item in measurements}
    if len(measurements) != 53 or len(actual) != 53 or actual != expected:
        raise ValueError("WP5 ablation windows differ from the frozen plan")
    if result["inputs"]["plan"] != hash_file(plan_path):
        raise ValueError("WP5 ablation plan hash differs")
    for event in EVENTS:
        model_id = f"full-s0-loeo_{event}"
        expected_inputs = {
            "packets": paths.PROCESSED / "packets_captioned" / f"{model_id}.jsonl",
            "sitreps": paths.REPO / "runs/sitreps_hf_captioned" / model_id / event / "complete.json",
        }
        if result["inputs"][event] != {name: hash_file(path) for name, path in expected_inputs.items()}:
            raise ValueError(f"WP5 ablation inputs differ for {event}")
    if report_bytes.decode() != summarize(measurements):
        raise ValueError("WP5 ablation report differs from restored measurements")
    target = paths.REPO / "runs/ablation_captioned/results.json"
    report = paths.REPORTS / "generation/ablations_captioned.md"
    complete = target.parent / "completion.json"
    for path, content in ((target, result_bytes), (report, report_bytes), (complete, marker_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"WP5 ablation output already differs: {path}")
        if not path.exists():
            atomic.text(path, content.decode("utf-8"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--staged", type=Path, default=paths.REPO / "runs/transfer/wp5-ablation-inputs-20261027")
    args = parser.parse_args()
    print(restore(args.archive, args.staged))
