"""Computed sitrep verification and coverage reports."""

from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from sakshi import atomic

GATES = ("citation", "provenance", "entailment", "label", "number")


def measure(report: dict) -> dict:
    initial = [item for item in report["sentences"] if item["round"] == 0]
    accepted = [item for item in report["sentences"] if item["accepted"]]
    cited = {pid for item in accepted for pid in item["citations"]}
    pids = set(report["input"]["pids"])
    if not cited <= pids:
        raise ValueError("accepted sentence cites an unknown post")
    count = len(initial)
    return {
        "event": report["event"],
        "window": report["window"]["start"],
        "round0": count,
        "initial_accepted": sum(item["accepted"] for item in initial),
        "final_accepted": len(accepted),
        "gate_failures": {name: sum(not item["gates"][name]["passed"] for item in initial) for name in GATES},
        "selected": len(pids),
        "covered": len(cited),
        "citations": sum(len(item["citations"]) for item in accepted),
        "generation_failures": report["generation_failures"],
        "latency_s": report["latency_s"],
    }


def _percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.3f}" if denominator else "n/a"


def _row(label: str, measurements: list[dict]) -> str:
    n = len(measurements)
    round0 = sum(item["round0"] for item in measurements)
    initial = sum(item["initial_accepted"] for item in measurements)
    final = sum(item["final_accepted"] for item in measurements)
    selected = sum(item["selected"] for item in measurements)
    covered = sum(item["covered"] for item in measurements)
    citations = sum(item["citations"] for item in measurements)
    failures = sum(item["generation_failures"] for item in measurements)
    latency = sum(item["latency_s"] for item in measurements)
    rates = [_percent(sum(item["gate_failures"][name] for item in measurements), round0) for name in GATES]
    return (
        f"| {label} | {n} | {round0} | {_percent(initial, round0)} | {_percent(final, round0)} | "
        f"{_percent(covered, selected)} | {final / n:.2f} | {citations / final if final else 0:.2f} | "
        f"{failures} | {latency / n:.2f} | " + " | ".join(rates) + " |"
    )


def summary(output_root: Path, report_path: Path) -> Path:
    """Report only completed runs; demo and research scopes stay separate."""
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for marker in sorted(output_root.glob("*/*/complete.json")):
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if any(
            not (marker.parent / name).is_file() or sha256((marker.parent / name).read_bytes()).hexdigest() != digest
            for name, digest in completed["artifacts"].items()
        ):
            raise ValueError(f"incomplete or changed sitrep run at {marker.parent}")
        for path in sorted(marker.parent.glob("[0-9]*Z.json")):
            by_mode[completed["mode"]].append(measure(json.loads(path.read_text(encoding="utf-8"))))
    lines = [
        "# Situation report verification",
        "",
        "Rates use round-0 sentences as the denominator. "
        "Evidence coverage uses selected posts; latency is seconds per window. "
        "A sentence can fail multiple gates.",
        "",
    ]
    header = (
        "| Scope | Windows | Round 0 | VR initial | VR final | Evidence coverage | Final sentences/window | "
        "Citations/sentence | Generation failures | Latency/window | G1 | G2 | G3 | G4 | G5 |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for mode, items in sorted(by_mode.items()):
        lines.extend([f"## {mode}", "", header, divider, _row("Overall", items)])
        events = sorted({item["event"] for item in items})
        lines.extend(_row(event, [item for item in items if item["event"] == event]) for event in events)
        lines.extend(["", "### Windows", "", header, divider])
        lines.extend(_row(item["window"], [item]) for item in items)
        lines.append("")
    if not by_mode:
        lines.append("No completed sitrep runs.")
    atomic.text(report_path, "\n".join(lines).rstrip() + "\n")
    return report_path
