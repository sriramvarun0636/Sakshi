"""Fixed-sentence WP5 gate ablations with captions removed only from G3."""

from __future__ import annotations

from collections import defaultdict

from sakshi.generate.verify import Verifier


def evaluate_window(report: dict, packets: dict[str, dict], verifier: Verifier) -> dict:
    """Compare gates on the same round-0 sentences; keep the original rewrite outcome."""
    by_pid = {}
    for pid, sample_id in report["input"]["pids"].items():
        if sample_id not in packets:
            raise ValueError(f"sitrep cites a missing packet: {sample_id}")
        by_pid[pid] = {**packets[sample_id], "image_caption": None}
    initial = [sentence for sentence in report["sentences"] if sentence["round"] == 0]
    counts = {
        "round0": len(initial),
        "full_all_gates_initial": 0,
        "full_all_gates_final": sum(bool(sentence["accepted"]) for sentence in report["sentences"]),
        "entailment_only_initial": 0,
        "text_only_g3_all_gates_initial": 0,
        "g3_decisions_changed": 0,
    }
    for sentence in initial:
        gates = sentence["gates"]
        if set(gates) != {"citation", "provenance", "entailment", "label", "number"}:
            raise ValueError("sitrep sentence has a missing verification gate")
        if bool(sentence["accepted"]) != all(bool(gate["passed"]) for gate in gates.values()):
            raise ValueError("saved acceptance differs from the five gates")
        original_g3 = bool(gates["entailment"]["passed"])
        rescored = verifier.verify(sentence["text"], sentence["citations"], by_pid)["entailment"]
        text_only_g3 = bool(rescored["passed"])
        other_gates = all(bool(gate["passed"]) for name, gate in gates.items() if name != "entailment")
        counts["full_all_gates_initial"] += bool(sentence["accepted"])
        counts["entailment_only_initial"] += original_g3
        counts["text_only_g3_all_gates_initial"] += other_gates and text_only_g3
        counts["g3_decisions_changed"] += original_g3 != text_only_g3
    if counts["full_all_gates_final"] < counts["full_all_gates_initial"]:
        raise ValueError("final accepted count is below the round-0 count")
    return {"event": report["event"], "window": report["window"]["start"], **counts}


def summarize(measurements: list[dict]) -> str:
    """Produce a scope-level report without publishing any private sentence text."""
    if not measurements:
        raise ValueError("no ablation measurements")
    fields = (
        "round0",
        "full_all_gates_initial",
        "full_all_gates_final",
        "entailment_only_initial",
        "text_only_g3_all_gates_initial",
        "g3_decisions_changed",
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in measurements:
        groups[item["event"]].append(item)
    lines = [
        "# WP5 fixed-sentence verifier ablations",
        "",
        "All rates use the same round-0 sentences as their denominator. The no-regeneration",
        "condition is full five-gate acceptance before rewrite. The entailment-only condition",
        "uses the saved G3 decision and ignores G1, G2, G4 and G5. The text-only G3",
        "condition rescored G3 after removing image captions from its premise while holding",
        "the other four saved gates fixed. The final full condition includes the original rewrite",
        "round; the ablations do not generate new sentences.",
        "",
        "| Scope | Windows | Round 0 | Full initial | Full final | G3 only initial | "
        "Text-only G3 initial | G3 decisions changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, items in [("Overall", measurements), *sorted(groups.items())]:
        totals = {field: sum(int(item[field]) for item in items) for field in fields}
        denominator = totals["round0"]
        if denominator == 0:
            raise ValueError("ablation has no round-0 sentences")
        lines.append(
            f"| {name} | {len(items)} | {denominator} | "
            f"{totals['full_all_gates_initial'] / denominator:.3f} | "
            f"{totals['full_all_gates_final'] / denominator:.3f} | "
            f"{totals['entailment_only_initial'] / denominator:.3f} | "
            f"{totals['text_only_g3_all_gates_initial'] / denominator:.3f} | "
            f"{totals['g3_decisions_changed']} |"
        )
    lines += ["", "NLI thresholds remain the uncalibrated defaults because independent human labels are unavailable.", ""]
    return "\n".join(lines)
