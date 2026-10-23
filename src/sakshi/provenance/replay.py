"""Chronological CrisisMMD provenance replay and status summaries."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sakshi import atomic, paths
from sakshi.data.events import EVENTS
from sakshi.provenance.index import ProvenanceIndex
from sakshi.types import ProvStatus


def replay_event(
    index: ProvenanceIndex,
    event: str,
    *,
    output_dir: Path = paths.PROCESSED / "provenance",
) -> Path:
    """Check every image for one event against only its time-visible index items."""
    if event not in EVENTS:
        raise ValueError(f"unknown CrisisMMD event {event!r}")
    rows = index.manifest
    subset = rows.loc[(rows.dataset == "crisismmd") & (rows.event == event)].reset_index(drop=True)
    subset = subset.sort_values(["posted_at", "sample_id"], kind="stable")
    records = []
    for row in subset.itertuples():
        result = index.check(sample_id=str(row.sample_id))
        records.append({**asdict(result), "status": result.status.value, "event": event, "posted_at": row.posted_at})
    output = output_dir / f"replay_{event}.parquet"
    atomic.parquet(output, pd.DataFrame(records))
    return output


def summary_table(results: dict[str, pd.DataFrame]) -> str:
    """Render all nine statuses for each replayed event, including zero counts."""
    statuses = [status.value for status in ProvStatus]
    lines = [
        "# Provenance replay",
        "",
        "| Event | Total | " + " | ".join(statuses) + " |",
        "|---|---:|" + "|".join("---:" for _ in statuses) + "|",
    ]
    for event, frame in results.items():
        counts = frame.status.value_counts()
        lines.append(f"| {event} | {len(frame)} | " + " | ".join(str(int(counts.get(status, 0))) for status in statuses) + " |")
    return "\n".join(lines) + "\n"


def replay_all(
    index: ProvenanceIndex,
    *,
    output_dir: Path = paths.PROCESSED / "provenance",
    report_path: Path = paths.REPO / "reports" / "provenance" / "replay_summary.md",
) -> dict[str, Path]:
    """Replay the seven events and publish their count table."""
    outputs = {event: replay_event(index, event, output_dir=output_dir) for event in EVENTS}
    atomic.text(report_path, summary_table({event: pd.read_parquet(path) for event, path in outputs.items()}))
    return outputs
