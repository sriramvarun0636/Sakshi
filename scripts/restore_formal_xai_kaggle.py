"""Validate formal WP3 GPU shards and merge all four heads into frozen reports."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.explain.common import MAIN_HEADS
from sakshi.explain.reports import write_report
from sakshi.explain.workflow import stratified_ids

COUNTS = {"faithfulness": 500, "sanity": 100}
ROOT = paths.REPO / "runs/explain/formal"
INVENTORY = paths.REPO / "runs/transfer/formal-xai-inputs-20261026/inventory.json"


def restore(archive: Path, kind: str, head: str) -> Path:
    if kind not in COUNTS or head not in MAIN_HEADS:
        raise ValueError("unknown formal audit kind or head")
    inventory = json.loads(INVENTORY.read_text())
    target = ROOT / kind / head
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{head}-restore-", dir=target.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            if source.testzip() is not None:
                raise ValueError("formal audit ZIP checksum failed")
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError("formal audit ZIP has duplicate paths")
            for name in names:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or name.endswith("/"):
                    raise ValueError(f"unsafe formal audit ZIP entry: {name}")
                destination = temporary / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read(name))
        timing = json.loads((temporary / "timing.json").read_text())
        if (timing["kind"], timing["head"], timing["count"], timing["source_commit"], timing["run_id"]) != (
            kind,
            head,
            COUNTS[kind],
            inventory["source_commit"],
            inventory["run_id"],
        ):
            raise ValueError("formal audit identity or source differs")
        request = json.loads((temporary / "request.json").read_text())
        if (
            request["kind"] != kind
            or request["split"] != "test"
            or request["count"] != COUNTS[kind]
            or request["heads"] != [head]
            or request["bootstrap"] != 1000
        ):
            raise ValueError("formal audit request differs from the design protocol")
        if json.loads((temporary / "explanation_protocol.json").read_text()) != request["protocol"]:
            raise ValueError("formal audit explanation protocol differs from its request")
        if request["protocol"]["classifier"]["checkpoint.pt"] != inventory["run_files"]["checkpoint.pt"]:
            raise ValueError("formal audit used another classifier checkpoint")
        complete = json.loads((temporary / "complete.json").read_text())
        for name, digest in complete["artifacts"].items():
            path = temporary / name
            if "/" in name or not path.is_file() or hash_file(path) != digest:
                raise ValueError(f"formal audit artifact hash mismatch: {name}")
        predictions = temporary / "sampling_predictions.parquet"
        if hash_file(predictions) != (temporary / "sampling_predictions.sha256").read_text().strip():
            raise ValueError("formal audit sampling predictions changed")
        frame = pd.read_parquet(predictions)
        pool = set(request["pool_ids"][head])
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != pool:
            raise ValueError("formal audit sampling pool differs from request")
        expected = stratified_ids(frame, head, COUNTS[kind])
        results = json.loads((temporary / "results.json").read_text())
        if [row["post_id"] for row in results] != expected or any(row["head"] != head for row in results):
            raise ValueError("formal audit post selection differs from frozen sampler")
        if target.exists():
            if sorted(path.relative_to(target) for path in target.rglob("*") if path.is_file()) != sorted(
                path.relative_to(temporary) for path in temporary.rglob("*") if path.is_file()
            ) or any(
                hash_file(target / path.relative_to(temporary)) != hash_file(path)
                for path in temporary.rglob("*")
                if path.is_file()
            ):
                raise FileExistsError(f"existing formal audit shard differs: {target}")
        else:
            os.replace(temporary, target)
        merge(kind)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def merge(kind: str) -> Path | None:
    if kind not in COUNTS:
        raise ValueError(kind)
    shards = [ROOT / kind / head for head in MAIN_HEADS]
    if any(not (shard / "complete.json").is_file() for shard in shards):
        return None
    protocols = [json.loads((shard / "explanation_protocol.json").read_text()) for shard in shards]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("formal audit head protocols differ")
    results = []
    inputs = {}
    for head, shard in zip(MAIN_HEADS, shards, strict=True):
        rows = json.loads((shard / "results.json").read_text())
        if len(rows) != COUNTS[kind] or any(row["head"] != head for row in rows):
            raise ValueError(f"formal audit shard is incomplete: {head}")
        results.extend(rows)
        inputs[head] = hash_file(shard / "results.json")
    output = ROOT / kind / "merged"
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "complete.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["inputs"] != inputs or any(hash_file(output / name) != digest for name, digest in old["artifacts"].items()):
            raise ValueError("completed formal audit report changed")
        return output / old["report"]
    report = write_report(kind, results, output, split="test", count=COUNTS[kind], bootstrap=1000, heads=MAIN_HEADS)
    artifacts = {report.name: hash_file(report)}
    if kind == "faithfulness":
        figure = output / "faithfulness_curves.png"
        artifacts[figure.name] = hash_file(figure)
        shutil.copy2(figure, paths.REPORTS / "figures/faithfulness_curves.png")
    shutil.copy2(report, paths.REPORTS / "results" / report.name)
    marker.write_text(json.dumps({"inputs": inputs, "artifacts": artifacts, "report": report.name}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("kind", choices=sorted(COUNTS))
    parser.add_argument("head", choices=MAIN_HEADS)
    args = parser.parse_args()
    print(restore(args.archive, args.kind, args.head))
