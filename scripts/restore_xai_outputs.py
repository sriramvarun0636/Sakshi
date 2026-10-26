"""Restore and validate the private Kaggle Maria/Mexico packet exports."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import pandas as pd

from sakshi import paths
from sakshi.aggregate.packets import load_packets, validate_selected
from sakshi.compute.files import hash_file

EVENTS = {
    "hurricane_maria": "full-s0-loeo_hurricane_maria-20261025-145019",
    "mexico_earthquake": "full-s0-loeo_mexico_earthquake-20261025-150334",
}
SLUG = "varun0636/sakshi-xai-packets-maria-mexico"
OUTPUTS = paths.REPO / "runs/transfer/kaggle_outputs/xai-packets-final"


def download() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    command = [
        str(paths.REPO / ".venv/bin/kaggle"), "kernels", "output", SLUG,
        "-p", str(OUTPUTS), "--file-pattern", r"\.tar\.gz$|export\.json$", "--page-size", "200", "-q",
    ]
    subprocess.run(command, cwd=paths.REPO, check=True, timeout=600)
    exports = list(OUTPUTS.rglob("export.json"))
    if len(exports) != 1:
        raise ValueError(f"expected one Kaggle XAI export manifest, found {len(exports)}")
    return exports[0]


def extract_checked(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"unsafe XAI archive path: {member.name}")
            output = destination.joinpath(*relative.parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                output.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError(f"archive member unavailable: {member.name}")
                with source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
            else:
                raise ValueError(f"unsupported XAI archive member: {member.name}")


def validate(event: str, stage: Path, expected_count: int) -> tuple[Path, Path, Path]:
    model_id = f"full-s0-loeo_{event}"
    packet = stage / f"data/processed/packets/{model_id}.jsonl"
    artifacts = stage / f"data/processed/packets/{model_id}_artifacts"
    protocol = stage / f"protocols/{event}.json"
    if not packet.is_file() or not (artifacts / "complete.json").is_file() or not protocol.is_file():
        raise ValueError(f"XAI archive is incomplete for {event}")
    marker = json.loads((artifacts / "complete.json").read_text())
    if marker != {"packets": expected_count, "packet_sha256": hash_file(packet)}:
        raise ValueError(f"XAI packet marker differs for {event}")
    run = paths.REPO / "runs/classifier" / EVENTS[event]
    signature = json.loads(protocol.read_text())
    frozen = json.loads((run / "inputs.json").read_text())
    if signature["manifest"] != frozen["manifest"]:
        raise ValueError(f"XAI manifest differs for {event}")
    for name, expected in signature["classifier"].items():
        if hash_file(run / name) != expected:
            raise ValueError(f"XAI classifier input differs: {event}/{name}")
    for name, expected in signature["features"].items():
        if frozen[name] != expected:
            raise ValueError(f"XAI feature input differs: {event}/{name}")
    for name, expected in signature["implementation"].items():
        if hash_file(paths.REPO / name) != expected:
            raise ValueError(f"XAI implementation differs: {event}/{name}")
    packets = load_packets(packet, model_id)
    selected = set(pd.read_parquet(paths.REPO / f"runs/aggregate/plans/{model_id}.parquet").sample_id.astype(str))
    if set(packets) != selected or len(packets) != expected_count:
        raise ValueError(f"XAI packet IDs differ from plan for {event}")
    rows = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    predictions = pd.read_parquet(run / "predictions_loeo.parquet")
    provenance = pd.read_parquet(paths.PROCESSED / f"provenance/replay_{event}.parquet")
    validate_selected(selected, packets, rows, predictions, provenance)
    return packet, artifacts, protocol


def restore() -> list[Path]:
    export_path = download()
    manifest = json.loads(export_path.read_text())
    if set(manifest) != set(EVENTS):
        raise ValueError(f"XAI export events differ: {sorted(manifest)}")
    restored = []
    with tempfile.TemporaryDirectory(prefix="sakshi-xai-", dir=paths.REPO / "runs/transfer") as temporary:
        for event, record in manifest.items():
            model_id = f"full-s0-loeo_{event}"
            if record["archive"] != f"{model_id}.tar.gz":
                raise ValueError(f"unexpected XAI archive name: {record['archive']}")
            archives = list(OUTPUTS.rglob(record["archive"]))
            if len(archives) != 1 or hash_file(archives[0]) != record["sha256"]:
                raise ValueError(f"XAI archive hash mismatch for {event}")
            stage = Path(temporary) / event
            stage.mkdir()
            extract_checked(archives[0], stage)
            packet, artifacts, protocol = validate(event, stage, int(record["packets"]))
            target = paths.PROCESSED / "packets" / packet.name
            target_artifacts = target.parent / artifacts.name
            target_protocol = paths.REPO / "runs/classifier" / EVENTS[event] / "explanation_protocol.json"
            if target.exists() or target_artifacts.exists() or target_protocol.exists():
                raise FileExistsError(f"XAI output already exists for {event}; inspect it before restoring")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(artifacts, target_artifacts)
            shutil.move(protocol, target_protocol)
            shutil.move(packet, target)
            restored.append(target)
    return restored


if __name__ == "__main__":
    for path in restore():
        print(path, flush=True)
