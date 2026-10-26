"""Poll private B6/WP5 kernels and collect each completed result once."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from sakshi import paths

CLI = paths.REPO / ".venv/bin/kaggle"
OUTPUTS = paths.REPO / "runs/transfer/kaggle_outputs"
JOBS = {
    "b6-vlm": ("varun0636/sakshi-b6-vlm-zero-shot", None),
    "sitrep-hf-harvey": ("varun0636/sakshi-sitrep-hf-harvey", "hurricane_harvey"),
    "sitrep-hf-maria": ("varun0636/sakshi-sitrep-hf-maria", "hurricane_maria"),
    "sitrep-hf-mexico": ("varun0636/sakshi-sitrep-hf-mexico", "mexico_earthquake"),
}


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def command(args: list[str | Path], *, timeout: int = 300) -> str:
    result = subprocess.run([str(value) for value in args], cwd=paths.REPO, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout)[-1200:])
    return result.stdout


def status(slug: str) -> str:
    try:
        output = command([CLI, "kernels", "status", slug], timeout=30)
    except RuntimeError as exc:
        if "Permission 'kernels.get' was denied" in str(exc):
            return "NOT_SUBMITTED"
        raise
    for state in ("COMPLETE", "RUNNING", "ERROR", "CANCELLED", "QUEUED"):
        if f"KernelWorkerStatus.{state}" in output:
            return state
    raise ValueError(f"unrecognized Kaggle kernel status: {output.strip()}")


def single(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def render_b6(result: dict) -> Path:
    output = paths.REPORTS / "results/b6_vlm.md"
    if output.exists():
        return output
    latency = result["latency_seconds"]
    fallback = str(result["fallback_reason"]).replace("\n", " ")
    lines = [
        "# B6 zero-shot vision-language baseline",
        "",
        f"Model: `{result['model']}` at revision `{result['revision']}`; quantization: `{result['quantization']}`.",
        f"Fallback reason: `{fallback}`. Official selector union: **{result['n_unique_posts']:,}** posts.",
        f"Failures: **{result['failed']:,}** ({result['failure_rate']:.1%}); failed parses use the frozen training majority.",
        f"Latency per post: median **{latency['median']:.2f}s**, mean **{latency['mean']:.2f}s**, p95 **{latency['p95']:.2f}s**.",
        f"Hierarchy contradiction rate: **{result['hierarchy_contradiction_rate']:.3f}**.",
        "The model emits labels only; probability calibration is not applicable.",
        "",
        "| Official selector | n | Macro-F1 [95% bootstrap CI] | Weighted-F1 | Accuracy | Failure rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for head, score in result["heads"].items():
        f1 = score["macro_f1"]
        lines.append(
            f"| {head.replace('_', ' ')} | {score['n']:,} | {f1['point']:.3f} "
            f"[{f1['lo']:.3f}, {f1['hi']:.3f}] | {score['weighted_f1']:.3f} | "
            f"{score['accuracy']:.3f} | {score['failure_rate']:.1%} |"
        )
    lines += [
        "",
        "Scores use frozen official test selectors and 1,000 seeded item-bootstrap draws.",
        f"Protocol SHA256: `{result['protocol_sha256']}`. Prediction SHA256: `{result['predictions_sha256']}`.",
        "",
    ]
    output.write_text("\n".join(lines))
    return output


def collect_b6(slug: str) -> Path:
    from evaluate_vlm_outputs import evaluate

    target = OUTPUTS / "b6-results"
    target.mkdir(parents=True, exist_ok=True)
    command([CLI, "kernels", "output", slug, "-p", target, "--file-pattern", r"predictions\.jsonl$|protocol\.json$", "-q"])
    predictions = single(target, "predictions.jsonl")
    single(target, "protocol.json")
    result = evaluate(predictions, paths.REPO / "runs/baselines/b6-vlm-20261026/metrics_test.json")
    return render_b6(result)


def collect_sitrep(slug: str, event: str) -> Path:
    from restore_sitrep_kaggle import restore

    target = OUTPUTS / f"sitrep-hf-{event}"
    target.mkdir(parents=True, exist_ok=True)
    command([CLI, "kernels", "output", slug, "-p", target, "--file-pattern", rf"sitrep-hf-{event}\.zip$", "-q"])
    return restore(single(target, f"sitrep-hf-{event}.zip"), event)


def main() -> None:
    done = set()
    failed: dict[str, str] = {}
    attempts: dict[str, int] = {}
    states: dict[str, str] = {}
    markers = OUTPUTS / "remote-collect"
    markers.mkdir(parents=True, exist_ok=True)
    log("collector started; polling every 60 seconds")
    while len(done) + len(failed) < len(JOBS):
        for name, (slug, event) in JOBS.items():
            if name in done or name in failed:
                continue
            marker = markers / f"{name}.json"
            if marker.exists():
                done.add(name)
                continue
            try:
                state = status(slug)
                if states.get(name) != state:
                    states[name] = state
                    log(f"STATE {name}: {state}")
                if state == "COMPLETE":
                    output = collect_b6(slug) if event is None else collect_sitrep(slug, event)
                    marker.write_text(json.dumps({"slug": slug, "output": str(output)}, indent=2) + "\n")
                    done.add(name)
                    log(f"COLLECTED {name}: {output}")
                elif state in {"ERROR", "CANCELLED"}:
                    failed[name] = state
                    log(f"REMOTE FAILED {name}: {state}")
            except Exception as exc:
                attempts[name] = attempts.get(name, 0) + 1
                log(f"COLLECTION RETRY {name} ({attempts[name]}/3): {exc}")
                if attempts[name] >= 3:
                    failed[name] = str(exc)
        if len(done) + len(failed) < len(JOBS):
            time.sleep(60)
    log(f"collector ended; collected={sorted(done)}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
