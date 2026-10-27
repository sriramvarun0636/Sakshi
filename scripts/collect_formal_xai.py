"""Collect and hash-check the eight private formal WP3 Kaggle shards."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from sakshi import paths

CLI = paths.REPO / ".venv/bin/kaggle"
OUTPUTS = paths.REPO / "runs/transfer/kaggle_outputs"
KINDS = ("faithfulness", "sanity")
HEADS = ("informative", "humanitarian", "severity", "disaster_type")
JOBS = {(kind, head): f"sakshi-formal-xai-{kind}-{head.replace('_', '-')}" for kind in KINDS for head in HEADS}


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def command(args: list[str | Path], *, timeout: int = 300) -> str:
    result = subprocess.run([str(value) for value in args], cwd=paths.REPO, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout)[-1200:])
    return result.stdout


def status(slug: str) -> str:
    try:
        output = command([CLI, "kernels", "status", f"varun0636/{slug}"], timeout=30)
    except RuntimeError as exc:
        if "Permission 'kernels.get' was denied" in str(exc):
            return "NOT_SUBMITTED"
        raise
    for state in ("COMPLETE", "RUNNING", "ERROR", "CANCELLED", "QUEUED"):
        if f"KernelWorkerStatus.{state}" in output:
            return state
    raise ValueError(f"unrecognized Kaggle status: {output.strip()}")


def collect(kind: str, head: str, slug: str) -> Path:
    from restore_formal_xai_kaggle import restore

    target = OUTPUTS / slug
    target.mkdir(parents=True, exist_ok=True)
    filename = f"formal-xai-{kind}-{head}.zip"
    command([CLI, "kernels", "output", f"varun0636/{slug}", "-p", target, "--file-pattern", rf"{re.escape(filename)}$", "-q"])
    matches = list(target.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"expected one {filename} in Kaggle output, found {len(matches)}")
    return restore(matches[0], kind, head)


def main() -> None:
    done: set[tuple[str, str]] = set()
    failed: dict[str, str] = {}
    attempts: dict[tuple[str, str], int] = {}
    states: dict[tuple[str, str], str] = {}
    markers = OUTPUTS / "formal-xai-collect"
    markers.mkdir(parents=True, exist_ok=True)
    log("formal WP3 collector started; polling every 60 seconds")
    while len(done) + len(failed) < len(JOBS):
        for (kind, head), slug in JOBS.items():
            key = (kind, head)
            if key in done or slug in failed:
                continue
            marker = markers / f"{slug}.json"
            if marker.exists():
                done.add(key)
                continue
            try:
                state = status(slug)
                if states.get(key) != state:
                    states[key] = state
                    log(f"STATE {slug}: {state}")
                if state == "COMPLETE":
                    output = collect(kind, head, slug)
                    marker.write_text(json.dumps({"slug": slug, "output": str(output)}, indent=2) + "\n")
                    done.add(key)
                    log(f"COLLECTED {slug}: {output}")
                elif state in {"ERROR", "CANCELLED"}:
                    failed[slug] = state
                    log(f"REMOTE FAILED {slug}: {state}")
            except Exception as exc:
                attempts[key] = attempts.get(key, 0) + 1
                log(f"COLLECTION RETRY {slug} ({attempts[key]}/3): {exc}")
                if attempts[key] >= 3:
                    failed[slug] = str(exc)
        if len(done) + len(failed) < len(JOBS):
            time.sleep(60)
    log(f"formal collector ended: {len(done)} collected, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
