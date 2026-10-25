"""Publication figures from a frozen official classifier prediction table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sakshi import paths
from sakshi.compute.files import hash_file
from sakshi.data import labels
from sakshi.eval import metrics
from sakshi.models.evaluate_run import _head_frame, _probabilities

HEADS = (
    ("informative", "Informativeness", labels.INFORMATIVE),
    ("humanitarian", "Humanitarian", labels.HUMANITARIAN),
    ("severity", "Severity · gold gated", labels.SEVERITY),
    ("disaster_type", "Disaster type", labels.DISASTER_TYPE),
)
GOLD = {
    "informative": "y_informative",
    "humanitarian": "y_humanitarian",
    "severity": "y_severity",
    "disaster_type": "y_disaster_type",
}


def render_classifier_figures(run_dir: Path, *, output_dir: Path = paths.REPORTS / "figures") -> tuple[Path, Path]:
    """Plot calibrated reliability and humanitarian risk from verified test rows."""
    run_dir = Path(run_dir)
    from sakshi.config import load
    from sakshi.models.train import TrainConfig

    config = load(run_dir / "config.yaml", TrainConfig)
    if config.model.modality != "both":
        raise ValueError("full classifier figures require both modalities")
    seed = config.seed
    marker = json.loads((run_dir / "evaluation_complete.json").read_text(encoding="utf-8"))
    if marker != {name: hash_file(run_dir / name) for name in ("metrics_test.json", "predictions_official.parquet")}:
        raise ValueError("official evaluation artifacts do not match their completion marker")
    request = json.loads((run_dir / "evaluation_request.json").read_text(encoding="utf-8"))
    manifest_path = paths.PROCESSED / "manifest.parquet"
    if hash_file(manifest_path) != request["manifest"]:
        raise ValueError("manifest differs from the frozen evaluation input")
    manifest = pd.read_parquet(manifest_path)
    predictions = pd.read_parquet(run_dir / "predictions_official.parquet")
    saved = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), sharex=True, sharey=True)
    for ax, (head, title, classes) in zip(axes.flat, HEADS, strict=True):
        frame = _head_frame(manifest, predictions, head)
        p = _probabilities(frame, head, classes)
        truth = frame[GOLD[head]].astype(str).to_numpy()
        y = np.array([classes.index(value) for value in truth])
        confidence = p.max(axis=1)
        correct = p.argmax(axis=1) == y
        bins = np.minimum((confidence * 15).astype(int), 14)
        x, observed, sizes = [], [], []
        for number in range(15):
            mask = bins == number
            if mask.any():
                x.append(float(confidence[mask].mean()))
                observed.append(float(correct[mask].mean()))
                sizes.append(int(mask.sum()))
        ax.plot([0, 1], [0, 1], color="#aab7bb", linewidth=1.2, linestyle="--", label="Perfect calibration")
        ax.plot(x, observed, color="#9f4b38", linewidth=1.8, marker="o", markersize=4, label=f"Seed {seed}")
        ax.set_title(f"{title}\n{len(frame):,} posts  ·  ECE {metrics.ece(p, y):.3f}", loc="left", pad=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#e4e9e9", linewidth=0.7)
        ax.text(0.02, 0.97, f"15 bins · {sum(sizes):,} scored", transform=ax.transAxes, va="top", color="#52696d", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Mean predicted confidence")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed accuracy")
    fig.suptitle("Calibration on frozen official test selectors", x=0.08, y=1.01, ha="left", fontsize=16, color="#173b43")
    fig.tight_layout()
    reliability = output_dir / f"classifier_seed{seed}_reliability.png"
    fig.savefig(reliability, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    frame = _head_frame(manifest, predictions, "humanitarian")
    gold = frame.y_humanitarian.astype(str).to_numpy()
    pred = frame.humanitarian_label.astype(str).to_numpy()
    p_inf = _probabilities(frame, "informative", labels.INFORMATIVE)
    p_hum = _probabilities(frame, "humanitarian", labels.HUMANITARIAN)
    conditional = np.divide(
        p_hum[:, [0, 1, 3, 4]].max(axis=1),
        p_inf[:, labels.INFORMATIVE.index("informative")],
        out=np.zeros(len(frame)),
        where=p_inf[:, labels.INFORMATIVE.index("informative")] > 0,
    )
    confidence = np.where(
        frame.informative_label.to_numpy() == "informative",
        np.minimum(p_inf.max(axis=1), conditional),
        p_inf.max(axis=1),
    )
    coverage, risk, aurc = metrics.risk_coverage(confidence, gold == pred)
    if abs(aurc - saved["abstention"]["aurc"]) > 1e-10:
        raise ValueError("risk curve AURC differs from the frozen metric")
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.plot(coverage * 100, risk * 100, color="#9f4b38", linewidth=2.2)
    ax.axhline(100 * (gold != pred).mean(), color="#83979a", linewidth=1.2, linestyle="--", label="All posts")
    ax.set(xlim=(0, 100), ylim=(0, None), xlabel="Coverage (%)", ylabel="Humanitarian error rate (%)")
    ax.set_title("Risk as low-confidence posts enter review", loc="left", pad=14, fontsize=15, color="#173b43")
    ax.text(
        0.98,
        0.07,
        f"Seed {seed} · {len(frame):,} official posts · AURC {aurc:.3f}",
        transform=ax.transAxes,
        ha="right",
        color="#52696d",
    )
    ax.grid(color="#e4e9e9", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.83))
    fig.tight_layout()
    risk_path = output_dir / f"classifier_seed{seed}_risk_coverage.png"
    fig.savefig(risk_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return reliability, risk_path
