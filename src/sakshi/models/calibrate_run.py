"""Fit and record all classifier calibration values on the dev split only."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sakshi import atomic, paths
from sakshi.config import load
from sakshi.eval.metrics import brier, ece
from sakshi.models.calibration import fit_abstention, fit_temperature
from sakshi.models.data import ClassifierCollator, ClassifierDataset, FeatureStore
from sakshi.models.losses import IGNORE
from sakshi.models.predict import Predictor
from sakshi.models.train import TrainConfig

log = logging.getLogger(__name__)
MAIN_HEADS = ("informative", "humanitarian", "severity", "disaster_type")


def _mce(probs: np.ndarray, gold: np.ndarray, bins: int) -> float:
    confidence = probs.max(1)
    correct = probs.argmax(1) == gold
    groups = np.minimum((confidence * bins).astype(int), bins - 1)
    return max(
        (
            abs(float(correct[groups == index].mean()) - float(confidence[groups == index].mean()))
            for index in range(bins)
            if (groups == index).any()
        ),
        default=0.0,
    )


def _probability_metrics(logits: torch.Tensor, gold: torch.Tensor, temperature: float, bins: int) -> dict[str, float]:
    present = gold != IGNORE
    if not bool(present.any()):
        return {}
    values = logits[present].float() / temperature
    gold = gold[present]
    probs = values.softmax(1).numpy()
    truth = gold.numpy()
    return {
        "ece": ece(probs, truth, bins),
        "mce": _mce(probs, truth, bins),
        "brier": brier(probs, truth),
        "nll": float(F.cross_entropy(values, gold)),
        "n": len(truth),
    }


def calibrate_run(run_dir: Path, *, features: FeatureStore | None = None, rows: pd.DataFrame | None = None) -> dict:
    """Write temperatures, review thresholds and before/after dev metrics to a run."""
    run_dir = Path(run_dir)
    config = load(run_dir / "config.yaml", TrainConfig)
    predictor = Predictor.from_run(run_dir)
    feature_store = features or FeatureStore()
    manifest = rows if rows is not None else pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    fold = config.fold.removeprefix("loeo_") if config.fold else None
    dataset = ClassifierDataset(manifest, feature_store, split="dev", fold=fold, use_cib=config.data.use_cib)
    if not dataset:
        raise ValueError("no supervised dev rows available for calibration")
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        collate_fn=ClassifierCollator(predictor.tokenizer, max_text_tokens=config.data.max_text_tokens),
    )
    logits: dict[str, list[torch.Tensor]] = {head: [] for head in MAIN_HEADS}
    targets: dict[str, list[torch.Tensor]] = {head: [] for head in MAIN_HEADS}
    humanitarian_gold = []
    with torch.inference_mode():
        for batch in loader:
            output = predictor.model(**{key: value.to(predictor.device) for key, value in batch["inputs"].items()})
            for head in MAIN_HEADS:
                logits[head].append(output[head].float().cpu())
                targets[head].append(batch["targets"][head])
            humanitarian_gold.append(batch["humanitarian_gold"])
    scores = {head: torch.cat(values) for head, values in logits.items()}
    gold = {head: torch.cat(values) for head, values in targets.items()}
    fitted = {
        head: fit_temperature(
            scores[head],
            gold[head],
            min_dev_rows=config.calibration.min_dev_rows,
            t_min=config.calibration.t_min,
            t_max=config.calibration.t_max,
        )
        for head in MAIN_HEADS
    }
    temperatures = {head: fit.temperature for head, fit in fitted.items()}
    calibration_metrics = {
        head: {
            "fit": asdict(fitted[head]),
            "before": _probability_metrics(scores[head], gold[head], 1.0, config.calibration.ece_bins),
            "after": _probability_metrics(scores[head], gold[head], temperatures[head], config.calibration.ece_bins),
        }
        for head in MAIN_HEADS
    }
    p_inf = (scores["informative"] / temperatures["informative"]).softmax(1).numpy()
    q_hum = (scores["humanitarian"] / temperatures["humanitarian"]).softmax(1).numpy()
    hum_gold = torch.cat(humanitarian_gold).numpy()
    labelled_hum = int((hum_gold != IGNORE).sum())
    if labelled_hum < config.calibration.min_dev_rows:
        log.warning("abstention tuning skipped: %d humanitarian dev labels", labelled_hum)
        thresholds = {"tau_inf": 0.5, "tau_hum": 0.5, "reason": "too_few_rows"}
    else:
        grid = tuple(
            float(value)
            for value in np.arange(
                config.abstention.grid_start,
                config.abstention.grid_stop + config.abstention.grid_step / 2,
                config.abstention.grid_step,
            )
        )
        fit = fit_abstention(p_inf, q_hum, hum_gold, target_coverage=config.abstention.target_coverage, grid=grid)
        thresholds = asdict(fit)
    atomic.text(run_dir / "temperatures.json", json.dumps(temperatures, indent=2) + "\n")
    atomic.text(run_dir / "thresholds.json", json.dumps(thresholds, indent=2) + "\n")
    atomic.text(run_dir / "calibration_metrics.json", json.dumps(calibration_metrics, indent=2) + "\n")
    return {"temperatures": temperatures, "thresholds": thresholds, "metrics": calibration_metrics}
