"""WP1 training loop over frozen CLIP tokens and a fine-tuned BERTweet tower."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sakshi import atomic, paths
from sakshi.config import create_run_dir, seed_everything, worker_init_fn
from sakshi.models.data import ClassifierCollator, ClassifierDataset, FeatureStore, load_tokenizer
from sakshi.models.fusion import HEAD_CLASSES, FusionClassifier, FusionConfig
from sakshi.models.losses import IGNORE, class_weights, multitask_loss

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSettings:
    crisismmd_share: float = 0.6
    samples_per_epoch: int = 16000
    modality_dropout: float = 0.15
    max_text_tokens: int = 128
    use_cib: bool = True


@dataclass(frozen=True)
class LossSettings:
    focal_gamma: float = 2.0
    class_weight_power: float = 0.5
    main_weight: float = 1.0
    aux_weight: float = 0.5


@dataclass(frozen=True)
class OptimSettings:
    lr_text: float = 2e-5
    lr_other: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    batch_size: int = 32
    grad_accum: int = 1
    max_epochs: int = 10
    patience: int = 3
    min_delta: float = 0.002
    grad_clip: float = 1.0


@dataclass(frozen=True)
class CalibrationSettings:
    ece_bins: int = 15
    min_dev_rows: int = 100
    t_min: float = 0.05
    t_max: float = 20.0


@dataclass(frozen=True)
class AbstentionSettings:
    target_coverage: float = 0.90
    grid_start: float = 0.50
    grid_stop: float = 0.95
    grid_step: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    name: str = "full"
    seed: int = 0
    fold: str | None = None
    data: DataSettings = field(default_factory=DataSettings)
    model: FusionConfig = field(default_factory=FusionConfig)
    loss: LossSettings = field(default_factory=LossSettings)
    optim: OptimSettings = field(default_factory=OptimSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    abstention: AbstentionSettings = field(default_factory=AbstentionSettings)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def optimizer_groups(model: nn.Module, config: OptimSettings) -> list[dict]:
    """Separate text and fusion learning rates, excluding bias and LayerNorm from decay."""
    no_decay = {
        id(parameter) for module in model.modules() if isinstance(module, nn.LayerNorm) for parameter in module.parameters()
    }
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_text = name.startswith("text_encoder.")
        decay = not (name.endswith(".bias") or id(parameter) in no_decay)
        groups.setdefault((is_text, decay), []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": config.lr_text if is_text else config.lr_other,
            "weight_decay": config.weight_decay if decay else 0.0,
        }
        for (is_text, decay), parameters in groups.items()
    ]


def _move_batch(batch: dict, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
    targets = {key: value.to(device) for key, value in batch["targets"].items()}
    return inputs, targets


def train_one_epoch(
    model: FusionClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    weights: dict[str, Tensor],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch; skip nonfinite batches and abort after ten such batches."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    bad = 0
    pending = 0

    def apply_update(group_size: int) -> None:
        scaler.unscale_(optimizer)
        if group_size != config.optim.grad_accum:
            correction = config.optim.grad_accum / group_size
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip, error_if_nonfinite=True)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # GradScaler skips an optimizer update when it detects overflow. The learning-rate
        # schedule must advance only with a real parameter update.
        if not scaler.is_enabled() or scaler.get_scale() >= previous_scale:
            scheduler.step()

    for index, batch in enumerate(loader):
        inputs, targets = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(**inputs)
            loss, parts = multitask_loss(
                outputs,
                targets,
                weights,
                gamma=config.loss.focal_gamma,
                main_weight=config.loss.main_weight,
                aux_weight=config.loss.aux_weight,
            )
        if not torch.isfinite(loss):
            bad += 1
            log.warning("skipping nonfinite training loss at batch %d (%d total)", index, bad)
            if bad >= 10:
                raise FloatingPointError("ten nonfinite training batches")
            continue
        scaler.scale(loss / config.optim.grad_accum).backward()
        pending += 1
        for head, value in parts.items():
            labelled = int((targets[head] != IGNORE).sum())
            sums[head] = sums.get(head, 0.0) + float(value.detach()) * labelled
            counts[head] = counts.get(head, 0) + labelled
        if pending == config.optim.grad_accum:
            apply_update(pending)
            pending = 0
    if pending:
        apply_update(pending)
    return {
        **{head: sums[head] / counts[head] for head in sums},
        "nonfinite_batches": float(bad),
        **{f"n_{head}": float(count) for head, count in counts.items()},
    }


@torch.inference_mode()
def evaluate_dev(model: FusionClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Selection macro-F1 across the four main heads on labelled dev rows."""
    model.eval()
    actual: dict[str, list[int]] = {head: [] for head in ("informative", "humanitarian", "severity", "disaster_type")}
    predicted: dict[str, list[int]] = {head: [] for head in actual}
    for batch in loader:
        inputs, targets = _move_batch(batch, device)
        outputs = model(**inputs)
        inf_pred = outputs["informative"].argmax(1)
        hum_pred = outputs["humanitarian"].argmax(1)
        conditional_indices = torch.tensor([0, 1, 3, 4], device=device)
        hum_pred = torch.where(inf_pred == 1, conditional_indices[hum_pred], 2)
        for head in actual:
            gold = batch["humanitarian_gold"].to(device) if head == "humanitarian" else targets[head]
            mask = gold != IGNORE
            pred = hum_pred if head == "humanitarian" else outputs[head].argmax(1)
            actual[head].extend(gold[mask].cpu().tolist())
            predicted[head].extend(pred[mask].cpu().tolist())
    scores = {}
    for head in actual:
        if not actual[head]:
            log.warning("no dev labels for %s", head)
            continue
        n_classes = 5 if head == "humanitarian" else len(HEAD_CLASSES[head])
        scores[head] = float(
            f1_score(actual[head], predicted[head], labels=list(range(n_classes)), average="macro", zero_division=0)
        )
    if not scores:
        raise ValueError("no dev labels for model selection")
    scores["selection_score"] = sum(scores.values()) / len(scores)
    return scores


def run_training(config: TrainConfig, *, output_dir: Path = paths.REPO / "runs" / "classifier") -> Path:
    """Train and retain the best dev checkpoint, with reproducible run metadata."""
    if config.optim.batch_size < 1 or config.optim.grad_accum < 1 or config.optim.max_epochs < 1:
        raise ValueError("batch size, accumulation and epochs must be positive")
    if config.optim.patience < 1 or not 0 <= config.optim.warmup_ratio < 1:
        raise ValueError("invalid patience or warmup ratio")
    fold = config.fold.removeprefix("loeo_") if config.fold else None
    generator = seed_everything(config.seed)
    device = select_device()
    rows = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    feature_store = FeatureStore()
    enabled_heads = tuple(
        HEAD_CLASSES if config.model.auxiliary_heads else ("informative", "humanitarian", "severity", "disaster_type")
    )
    train_data = ClassifierDataset(
        rows, feature_store, split="train", fold=fold, enabled_heads=enabled_heads, use_cib=config.data.use_cib
    )
    dev_data = ClassifierDataset(
        rows, feature_store, split="dev", fold=fold, enabled_heads=enabled_heads, use_cib=config.data.use_cib
    )
    if not train_data or not dev_data:
        raise ValueError("training and dev splits must have supervised rows with usable inputs")
    tokenizer = load_tokenizer(config.model.text_encoder)
    train_loader = DataLoader(
        train_data,
        batch_size=config.optim.batch_size,
        sampler=train_data.sampler(
            samples_per_epoch=config.data.samples_per_epoch, crisismmd_share=config.data.crisismmd_share, generator=generator
        ),
        collate_fn=ClassifierCollator(
            tokenizer, max_text_tokens=config.data.max_text_tokens, modality_dropout=config.data.modality_dropout
        ),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=config.optim.batch_size,
        shuffle=False,
        collate_fn=ClassifierCollator(tokenizer, max_text_tokens=config.data.max_text_tokens),
    )
    model = FusionClassifier.from_pretrained(config.model).to(device)
    counts = train_data.class_counts()
    weights = {head: class_weights(counts[head], config.loss.class_weight_power).to(device) for head in enabled_heads}
    optimizer = torch.optim.AdamW(optimizer_groups(model, config.optim), betas=(0.9, 0.999), eps=1e-8)
    total_updates = math.ceil(len(train_loader) / config.optim.grad_accum) * config.optim.max_epochs
    warmup = round(total_updates * config.optim.warmup_ratio)

    def lr_multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return max(0.0, (total_updates - step) / max(total_updates - warmup, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    run_dir, model_id = create_run_dir(
        config, name=config.name, seed=config.seed, fold=config.fold, root=output_dir, device=str(device)
    )
    atomic.text(
        run_dir / "class_weights.json",
        json.dumps(
            {
                head: {label: float(weight) for label, weight in zip(HEAD_CLASSES[head], weights[head].cpu(), strict=True)}
                for head in enabled_heads
            },
            indent=2,
        )
        + "\n",
    )
    atomic.text(run_dir / "label_maps.json", json.dumps({head: HEAD_CLASSES[head] for head in enabled_heads}, indent=2) + "\n")
    best = float("-inf")
    stale = 0
    for epoch in range(1, config.optim.max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, weights, config, device)
        dev_metrics = evaluate_dev(model, dev_loader, device)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        with (run_dir / "train_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log.info("epoch %d: dev selection macro-F1 %.4f", epoch, dev_metrics["selection_score"])
        if dev_metrics["selection_score"] > best + config.optim.min_delta:
            best = dev_metrics["selection_score"]
            stale = 0
            atomic.write(
                run_dir / "checkpoint.pt",
                lambda temporary, epoch=epoch: torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch, "model_id": model_id}, temporary
                ),
            )
            atomic.text(run_dir / "metrics_dev.json", json.dumps(dev_metrics, indent=2) + "\n")
        else:
            stale += 1
            if stale >= config.optim.patience:
                break
    return run_dir
