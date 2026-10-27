"""Compare the frozen 53-window WP5 runs with and without Qwen image captions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.generate.reports import measure

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")


def _load(root: Path, plan: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for event in EVENTS:
        model_id = f"full-s0-loeo_{event}"
        directory = root / model_id / event
        marker = json.loads((directory / "complete.json").read_text())
        chosen = {item["window"] for item in plan["selected"] if item["event"] == event}
        if marker["windows"] != len(chosen) or marker["mode"] != "loeo":
            raise ValueError(f"sitrep completion differs for {event}")
        for name, digest in marker["artifacts"].items():
            if hash_file(directory / name) != digest:
                raise ValueError(f"sitrep artifact changed: {event}/{name}")
        if {name for name in marker["artifacts"] if name.endswith("Z.json")} != chosen:
            raise ValueError(f"sitrep windows changed for {event}")
        result[event] = [measure(json.loads((directory / name).read_text())) for name in sorted(chosen)]
    return result


def _stats(items: list[dict]) -> dict:
    initial = sum(item["initial_accepted"] for item in items)
    final = sum(item["final_accepted"] for item in items)
    round0 = sum(item["round0"] for item in items)
    selected = sum(item["selected"] for item in items)
    return {
        "windows": len(items),
        "round0": round0,
        "initial": initial / round0 if round0 else None,
        "final": final / round0 if round0 else None,
        "coverage": sum(item["covered"] for item in items) / selected if selected else None,
        "failures": sum(item["generation_failures"] for item in items),
    }


def compare(output: Path) -> Path:
    original = json.loads((paths.REPO / "runs/generation/evaluation_plan.json").read_text())
    captioned = json.loads((paths.REPO / "runs/generation/evaluation_plan_captioned.json").read_text())
    if original != captioned or len(original["selected"]) != 53:
        raise ValueError("captioned sitrep plan differs from the frozen 53-window plan")
    base = _load(paths.REPO / "runs/sitreps_hf", original)
    with_captions = _load(paths.REPO / "runs/sitreps_hf_captioned", captioned)
    lines = [
        "# WP5 captioned versus uncaptioned generation",
        "",
        "The two research runs use the same 53 windows, aggregate files, classifier folds,",
        "LLM family, NLI model and default thresholds. Captions change the LLM input and",
        "the verifier premise, so this is an end-to-end input comparison. The fixed-sentence",
        "G3 caption ablation is reported separately.",
        "",
        "| Scope | Variant | Windows | Round 0 | VR initial | VR final | Evidence coverage | Generation failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for event in ("Overall", *EVENTS):
        for name, source in (("No caption", base), ("Qwen caption", with_captions)):
            items = [item for group in source.values() for item in group] if event == "Overall" else source[event]
            stats = _stats(items)
            lines.append(
                f"| {event} | {name} | {stats['windows']} | {stats['round0']} | "
                f"{stats['initial']:.3f} | {stats['final']:.3f} | {stats['coverage']:.3f} | {stats['failures']} |"
            )
    lines += [
        "",
        "Rates are automatic verifier outcomes, not human-rated factual accuracy.",
        "NLI thresholds remain uncalibrated because independent reviewers are unavailable.",
        "",
    ]
    atomic.text(output, "\n".join(lines))
    return output


if __name__ == "__main__":
    print(compare(paths.REPORTS / "generation/caption_comparison.md"))
