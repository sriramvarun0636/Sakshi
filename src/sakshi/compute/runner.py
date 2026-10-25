"""Bounded training smoke, frozen final evaluation and portable result exports."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import TYPE_CHECKING

from sakshi import atomic, paths
from sakshi.compute.files import hash_file

EXPORT_PART_BYTES = 768 * 1024**2

if TYPE_CHECKING:
    from sakshi.models.train import TrainConfig


def training_smoke(config: TrainConfig, output: Path, *, processed: Path = paths.PROCESSED, require_cuda: bool = True) -> dict:
    """Run a real padded forward/backward/update for all three modality patterns."""
    import pandas as pd
    import torch
    from torch.nn import functional as F

    from sakshi.config import seed_everything
    from sakshi.models.data import ClassifierCollator, ClassifierDataset, FeatureStore, load_tokenizer
    from sakshi.models.fusion import HEAD_CLASSES, FusionClassifier
    from sakshi.models.losses import class_weights, mask_targets_for_dropout
    from sakshi.models.train import make_scaler, optimizer_groups, precision_for, train_one_epoch

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("GPU training smoke requires CUDA; enable the notebook GPU accelerator first")
    device = torch.device("cuda:0" if require_cuda else "cpu")
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    features = FeatureStore(Path(processed) / "features")
    rows = pd.read_parquet(Path(processed) / "manifest.parquet")
    heads = tuple(HEAD_CLASSES if config.model.auxiliary_heads else list(HEAD_CLASSES)[:4])
    dataset = ClassifierDataset(
        rows,
        features,
        split="train",
        fold=config.fold.removeprefix("loeo_") if config.fold else None,
        enabled_heads=heads,
        use_cib=config.data.use_cib,
    )
    text_positions = [i for i, sample in enumerate(dataset.samples) if sample["text"] and features.has_image(sample["sample_id"])]
    image_only = [i for i, sample in enumerate(dataset.samples) if not sample["text"] and features.has_image(sample["sample_id"])]
    if not text_positions:
        raise ValueError("smoke needs train samples with both usable text and cached images")
    image_positions = [*text_positions[: config.optim.batch_size // 2], *image_only] or text_positions
    tokenizer = load_tokenizer(config.model.text_encoder, revision=config.model.text_revision)
    collator = ClassifierCollator(tokenizer, max_text_tokens=config.data.max_text_tokens)
    model = FusionClassifier.from_pretrained(config.model).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config.optim))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    weights = {
        head: class_weights(values, config.loss.class_weight_power).to(device) for head, values in dataset.class_counts().items()
    }
    scaler = make_scaler(device)
    results = {}
    try:
        for pattern in ("both", "text_only", "image_only"):
            positions = image_positions if pattern == "image_only" else text_positions
            items = [dataset[positions[i % len(positions)]] for i in range(config.optim.batch_size)]
            batch = collator(items)
            inputs = batch["inputs"]
            padding = config.data.max_text_tokens - inputs["input_ids"].shape[1]
            inputs["input_ids"] = F.pad(inputs["input_ids"], (0, padding), value=getattr(tokenizer, "pad_token_id", 0))
            inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, padding))
            dropped_image = inputs["has_image"].clone() if pattern == "text_only" else torch.zeros(len(items), dtype=torch.bool)
            dropped_text = inputs["has_text"].clone() if pattern == "image_only" else torch.zeros(len(items), dtype=torch.bool)
            inputs["has_image"] &= ~dropped_image
            inputs["has_text"] &= ~dropped_text
            batch["targets"] = mask_targets_for_dropout(batch["targets"], dropped_text=dropped_text, dropped_image=dropped_image)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=precision_for(device),
                ),
            ):
                logits = model(**{key: value.to(device) for key, value in inputs.items()})
                for head in heads:
                    if logits[head].shape != (len(items), len(HEAD_CLASSES[head])) or not torch.isfinite(logits[head]).all():
                        raise RuntimeError(f"smoke produced invalid {head} logits")
                alpha = logits["alpha"]
                if not torch.allclose(alpha.float().sum(1), torch.ones(len(items), device=device), atol=1e-3):
                    raise RuntimeError("modality weights do not sum to one")
                for column, name in enumerate(("has_text", "has_image")):
                    if not (alpha[~inputs[name].to(device), column] == 0).all():
                        raise RuntimeError("missing modality received nonzero gate weight")
            for attempt in range(1, 9):
                metrics = train_one_epoch(model, [batch], optimizer, scheduler, weights, config, device, scaler=scaler)
                results[pattern] = {
                    "metrics": metrics,
                    "attempts": attempt,
                    "batch_size": len(items),
                    "padded_text_tokens": inputs["input_ids"].shape[1],
                }
                if metrics["optimizer_updates"]:
                    break
            else:
                raise RuntimeError("AMP did not produce a finite parameter update within eight smoke attempts")
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "GPU smoke ran out of memory. Use configs/classifier/t4.yaml (batch 16, accumulation 2), then rerun smoke."
        ) from exc
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "kind": "training_smoke",
        "config": asdict(config),
        "device": str(device),
        "patterns": results,
        "wall_seconds": perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    atomic.text(Path(output), json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def final_evaluate(run_dir: Path, *, bootstrap: int = 1000) -> Path:
    """Freeze standard-run inputs once; repeated identical requests reuse stored scores."""
    from sakshi.config import load
    from sakshi.models.evaluate_run import evaluate_run
    from sakshi.models.train import TrainConfig

    run_dir = Path(run_dir)
    if bootstrap < 1:
        raise ValueError("bootstrap iterations must be positive")
    config = load(run_dir / "config.yaml", TrainConfig)
    if config.fold is not None:
        raise ValueError("official global test evaluation is for standard runs; LOEO needs its held-out-event evaluator")
    required = ("checkpoint.pt", "config.yaml", "label_maps.json", "temperatures.json", "thresholds.json")
    signature = {name: hash_file(run_dir / name) for name in required}
    signature["manifest"] = hash_file(paths.PROCESSED / "manifest.parquet")
    signature["bootstrap"] = bootstrap
    from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE

    signature["features"] = {
        name: hash_file(paths.PROCESSED / "features" / name) for name in (INDEX_FILE, TOKENS_FILE, EMBED_FILE)
    }
    request, marker = run_dir / "evaluation_request.json", run_dir / "evaluation_complete.json"
    result = run_dir / "metrics_test.json"
    prediction = run_dir / "predictions_official.parquet"
    if request.exists():
        if json.loads(request.read_text()) != signature:
            raise FileExistsError(
                "final evaluation inputs were frozen already and have changed; do not retune against test results"
            )
    elif result.exists():
        raise FileExistsError("test metrics already exist without a freeze record; refusing to rescore")
    else:
        atomic.text(request, json.dumps(signature, indent=2) + "\n")
    if marker.exists():
        hashes = json.loads(marker.read_text())
        if hashes != {path.name: hash_file(path) for path in (result, prediction)}:
            raise ValueError("stored final evaluation artifacts changed")
        return result
    # Recover an interruption after atomic metrics publication without scoring twice.
    if not result.exists():
        evaluate_run(run_dir, bootstrap=bootstrap)
    if not prediction.exists():
        raise ValueError("test metrics exist but their aligned predictions are absent")
    atomic.text(marker, json.dumps({path.name: hash_file(path) for path in (result, prediction)}, indent=2) + "\n")
    return result


def export_run(run_dir: Path, output: Path, *, include_resume: bool = False, part_bytes: int = EXPORT_PART_BYTES) -> Path:
    """Export allowlisted artifacts into checksummed archive parts smaller than 1 GB."""
    run_dir, output = Path(run_dir), Path(output)
    if not 1 <= part_bytes <= EXPORT_PART_BYTES:
        raise ValueError("export part size must be positive and at most 768 MiB")
    names = {
        "checkpoint.pt",
        "config.yaml",
        "env.json",
        "inputs.json",
        "label_maps.json",
        "class_weights.json",
        "tokenization.json",
        "train_log.jsonl",
        "training_summary.json",
        "metrics_dev.json",
        "temperatures.json",
        "thresholds.json",
        "calibration_metrics.json",
        "metrics_test.json",
        "predictions_official.parquet",
        "evaluation_request.json",
        "evaluation_complete.json",
        "metrics_loeo.json",
        "predictions_loeo.parquet",
        "evaluation_loeo_request.json",
        "evaluation_loeo_complete.json",
        "requirements.lock.txt",
        "gpu_preflight.json",
        "gpu_smoke.json",
    }
    if include_resume:
        names.add("last.pt")
    required = {
        "checkpoint.pt",
        "config.yaml",
        "env.json",
        "inputs.json",
        "label_maps.json",
        "train_log.jsonl",
        "training_summary.json",
    }
    if any(not (run_dir / name).is_file() for name in required):
        raise FileNotFoundError("export needs a completed training run with checkpoint and metadata")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("export output is not empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    members = sorted(name for name in names if (run_dir / name).is_file())
    records = []
    with tempfile.TemporaryDirectory(prefix="sakshi-export-") as temporary:
        archive = Path(temporary) / "run.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
            for name in members:
                zipped.write(run_dir / name, arcname=run_dir.name + "/" + name)
        with archive.open("rb") as stream:
            number = 0
            while stream.tell() < archive.stat().st_size:
                number += 1
                destination = output / f"{run_dir.name}.zip.part{number:03}"
                remaining = part_bytes
                with destination.open("xb") as target:
                    while remaining and (chunk := stream.read(min(1024**2, remaining))):
                        target.write(chunk)
                        remaining -= len(chunk)
                records.append({"path": destination.name, "bytes": destination.stat().st_size, "sha256": hash_file(destination)})
    manifest = {
        "format_version": 1,
        "run_id": run_dir.name,
        "includes_resume": include_resume,
        "members": members,
        "parts": records,
        "artifacts": {name: hash_file(run_dir / name) for name in members},
    }
    atomic.text(output / "SHA256SUMS", "".join(f"{record['sha256']}  {record['path']}\n" for record in records))
    atomic.text(output / "export.json", json.dumps(manifest, indent=2) + "\n")
    return output / "export.json"


def restore_export(export_dir: Path, output: Path) -> Path:
    """Verify downloaded parts, reconstruct the archive and restore one run safely."""
    export_dir, output = Path(export_dir), Path(output)
    manifest = json.loads((export_dir / "export.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported run export format")

    def safe_name(name: str) -> bool:
        return bool(name) and name not in (".", "..") and PurePosixPath(name).name == name and "\\" not in name

    run_id = manifest["run_id"]
    if not safe_name(run_id) or not manifest["parts"] or not all(safe_name(name) for name in manifest["members"]):
        raise ValueError("unsafe or empty export inventory")
    if len(set(manifest["members"])) != len(manifest["members"]):
        raise ValueError("duplicate export members")
    for number, part in enumerate(manifest["parts"], 1):
        if part["path"] != f"{run_id}.zip.part{number:03}":
            raise ValueError("export parts must be unique and consecutively ordered")
        path = export_dir / part["path"]
        if not path.is_file() or path.stat().st_size != part["bytes"] or hash_file(path) != part["sha256"]:
            raise ValueError(f"export integrity check failed: {part['path']}")
    if output.exists():
        raise FileExistsError("restore output already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sakshi-restore-", dir=output.parent) as temporary:
        staging = Path(temporary)
        archive = staging / "run.zip"
        with archive.open("wb") as target:
            for part in manifest["parts"]:
                with (export_dir / part["path"]).open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024**2)
        with zipfile.ZipFile(archive) as zipped:
            expected = {f"{run_id}/{name}" for name in manifest["members"]}
            if set(zipped.namelist()) != expected or len(zipped.namelist()) != len(expected):
                raise ValueError("archive members differ from the declared export inventory")
            for name in manifest["members"]:
                destination = staging / run_id / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(f"{run_id}/{name}") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024**2)
                if hash_file(destination) != manifest["artifacts"][name]:
                    raise ValueError(f"restored artifact integrity check failed: {name}")
        (staging / run_id).rename(output)
    return output
