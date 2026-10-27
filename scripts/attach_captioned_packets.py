"""Derive captioned WP3 packets from frozen explanations without rerunning the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sakshi import atomic, paths
from sakshi.aggregate.packets import load_packets
from sakshi.compute.files import hash_file
from sakshi.explain.captions import CaptionCache
from sakshi.explain.packets import validate_packet

EVENTS = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")


def attach(cache_path: Path, output_dir: Path) -> list[Path]:
    captions = CaptionCache(cache_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for event in EVENTS:
        model_id = f"full-s0-loeo_{event}"
        source = paths.PROCESSED / "packets" / f"{model_id}.jsonl"
        output = output_dir / source.name
        metadata = output.with_suffix(".caption_provenance.json")
        plan = pd.read_parquet(paths.REPO / f"runs/aggregate/plans/{model_id}.parquet")
        expected = set(plan.sample_id.astype(str))
        if output.exists() or metadata.exists():
            raise FileExistsError(f"preserve existing captioned packet output: {output}")
        rows = []
        for line in source.read_text().splitlines():
            packet = json.loads(line)
            post_id = packet["post_id"]
            if post_id not in expected or packet["image_caption"] is not None or not packet["has_image"]:
                raise ValueError(f"source packet differs from frozen caption scope: {post_id}")
            caption = captions.get(post_id)
            if caption is None:
                raise ValueError(f"caption missing for frozen packet: {post_id}")
            packet["image_caption"] = caption
            validate_packet(packet)
            rows.append(packet)
        if {row["post_id"] for row in rows} != expected or len(rows) != len(expected):
            raise ValueError(f"packet selection changed for {event}")
        atomic.text(output, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        load_packets(output, model_id)
        atomic.text(
            metadata,
            json.dumps(
                {
                    "derivation": "frozen packet with image_caption replaced by validated Qwen cache value",
                    "model_id": model_id,
                    "rows": len(rows),
                    "source_packets_sha256": hash_file(source),
                    "source_complete_sha256": hash_file(source.parent / (source.stem + "_artifacts") / "complete.json"),
                    "caption_cache_sha256": hash_file(cache_path),
                    "output_packets_sha256": hash_file(output),
                },
                indent=2,
            )
            + "\n",
        )
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=paths.PROCESSED / "captions.parquet")
    parser.add_argument("--output-dir", type=Path, default=paths.PROCESSED / "packets_captioned")
    args = parser.parse_args()
    for path in attach(args.cache, args.output_dir):
        print(path)
