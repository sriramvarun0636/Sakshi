"""Verify private Kaggle Qwen caption shards before publishing the local cache."""

from __future__ import annotations

import argparse
import json
import zipfile
from hashlib import sha256
from pathlib import Path

import pandas as pd

from sakshi import atomic, paths
from sakshi.explain.captions import CaptionCache
from sakshi.generate.captions import MODEL, PROMPT

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def restore(source: Path, staged: Path, output: Path) -> Path:
    inventory = json.loads((staged / "inventory.json").read_text())
    entry_bytes = (staged / "entries.json").read_bytes()
    if sha256(entry_bytes).hexdigest() != inventory["entries_sha256"]:
        raise ValueError("staged caption entry list changed")
    entries = json.loads(entry_bytes)
    expected = {event: {item["sample_id"] for item in entries if item["event"] == event} for event in EVENTS}
    if len(entries) != inventory["entries"] or sum(map(len, expected.values())) != len(entries):
        raise ValueError("staged caption selection changed")
    prompt_hash = sha256(PROMPT.encode()).hexdigest()
    rows = []
    for event in EVENTS:
        path = source / f"captions-{event}.zip"
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(f"{event}.jsonl")
            completion = json.loads(archive.read(f"{event}-completion.json"))
        if completion != {
            "event": event,
            "rows": len(expected[event]),
            "model": MODEL,
            "revision": REVISION,
            "prompt_sha256": prompt_hash,
            "entries_sha256": inventory["entries_sha256"],
            "captions_sha256": sha256(payload).hexdigest(),
            "quantization": "bitsandbytes-nf4",
        }:
            raise ValueError(f"caption completion marker differs for {event}")
        shard = [json.loads(line) for line in payload.splitlines()]
        if len(shard) != len(expected[event]) or {item["sample_id"] for item in shard} != expected[event]:
            raise ValueError(f"caption IDs differ for {event}")
        for item in shard:
            if (
                set(item) != {"sample_id", "caption", "model", "prompt_sha256", "created_at"}
                or not isinstance(item["caption"], str)
                or not item["caption"].strip()
                or item["model"] != MODEL
                or item["prompt_sha256"] != prompt_hash
            ):
                raise ValueError(f"invalid caption row in {event}")
        rows.extend(shard)
    if len({item["sample_id"] for item in rows}) != len(rows):
        raise ValueError("caption shards overlap")
    frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    frame["created_at"] = pd.to_datetime(frame.created_at, utc=True)
    if output.exists():
        old = pd.read_parquet(output)
        if not old.equals(frame):
            raise FileExistsError("caption cache already exists with different rows")
    else:
        atomic.parquet(output, frame)
    CaptionCache(output)
    print({"captions": len(frame), "cache": str(output)})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--staged", type=Path, default=paths.REPO / "runs/transfer/caption-images-20261027")
    parser.add_argument("--output", type=Path, default=paths.PROCESSED / "captions.parquet")
    args = parser.parse_args()
    restore(args.source, args.staged, args.output)
