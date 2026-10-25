"""Render the frozen official classifier evaluation as a reproducible report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.eval.selectors import official


def _score_cell(score: dict) -> str:
    macro = score["macro_f1"]
    return f"{macro['point']:.3f} [{macro['lo']:.3f}, {macro['hi']:.3f}]"


def render_classifier_report(run_dir: Path, *, output: Path | None = None) -> Path:
    """Write only from a complete, hash-verified official test evaluation."""
    run_dir = Path(run_dir)
    from sakshi.config import load
    from sakshi.models.train import TrainConfig

    config = load(run_dir / "config.yaml", TrainConfig)
    if config.model.modality != "both":
        raise ValueError("full classifier report requires both modalities")
    seed = config.seed
    if output is None:
        output = paths.REPORTS / "results" / ("classifier.md" if seed == 0 else f"classifier_seed{seed}.md")
    marker = json.loads((run_dir / "evaluation_complete.json").read_text(encoding="utf-8"))
    expected = {name: hash_file(run_dir / name) for name in ("metrics_test.json", "predictions_official.parquet")}
    if marker != expected:
        raise ValueError("official evaluation artifacts do not match their completion marker")
    request = json.loads((run_dir / "evaluation_request.json").read_text(encoding="utf-8"))
    manifest_path = paths.PROCESSED / "manifest.parquet"
    if hash_file(manifest_path) != request["manifest"]:
        raise ValueError("manifest differs from the frozen evaluation input")
    manifest = pd.read_parquet(
        manifest_path, columns=["dataset", "split", "split_cib_disaster_types", "y_disaster_type", "source"]
    )
    disaster_sources = manifest.loc[official(manifest, "disaster_type"), "source"].value_counts()
    result = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
    heads = result["heads"]
    selectors = result["selectors"]
    rows = (
        ("Informativeness", "informative", heads["informative"]),
        ("Humanitarian", "humanitarian", heads["humanitarian"]),
        ("Severity, gold gated", "severity", heads["severity"]["gold_gated"]),
        ("Severity, predicted gated", "severity", heads["severity"]["predicted_gated"]),
        ("Severity, DMD", "severity_dmd", heads["severity_dmd"]),
        ("Disaster type, combined", "disaster_type", heads["disaster_type"]["all"]),
        ("Disaster type, AIDR-DT", "aidr_dt", heads["disaster_type"]["aidr_dt"]),
        ("Disaster type, DMD", "dmd", heads["disaster_type"]["dmd"]),
    )
    lines = [
        f"# Full classifier: frozen official test, seed {seed}",
        "",
        f"Run: `{run_dir.name}`. Model: `{result['model_id']}`.",
        f"Bootstrap: {request['bootstrap']:,} item resamples (95% percentile intervals).",
        "The checkpoint, calibration, label maps, manifest, feature cache and bootstrap count were hashed",
        "before test inference. The resulting prediction and metric hashes are recorded in",
        "`evaluation_complete.json`. No test-driven configuration changes were made.",
        "",
        "| Official task | Rows | Macro-F1 [95% CI] | Weighted-F1 | Accuracy | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, selector, score in rows:
        count = selectors[selector] if selector in selectors else int(disaster_sources[selector])
        lines.append(
            f"| {label} | {count:,} | {_score_cell(score)} | {score['weighted_f1']['point']:.3f} | "
            f"{score['accuracy']['point']:.3f} | {score['ece']:.3f} | {score['brier']:.3f} |"
        )
    abstention = result["abstention"]
    lines += [
        "",
        "## Hierarchy and abstention",
        "",
        f"Hierarchy contradiction rate: **{result['contradiction_rate']:.3f}**.",
        f"Humanitarian review coverage: **{abstention['coverage']:.3f}**; selective macro-F1:",
        f"**{abstention['selective_macro_f1']:.3f}**; selective accuracy:",
        f"**{abstention['selective_accuracy']:.3f}**; errors routed to review:",
        f"**{abstention['errors_routed_to_review']:.3f}**; AURC: **{abstention['aurc']:.3f}**.",
        f"Predicted-gated severity coverage: **{heads['severity']['predicted_gated']['coverage']:.3f}**.",
        "",
        "## Per-class results and confusion matrices",
    ]
    for label, _, score in rows:
        lines += [
            "",
            f"### {label}",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for class_name, class_score in score["per_class"].items():
            lines.append(
                f"| {class_name} | {class_score['support']:,} | {class_score['precision']:.3f} | "
                f"{class_score['recall']:.3f} | {class_score['f1']:.3f} |"
            )
        confusion = score["confusion"]
        names = confusion["labels"]
        lines += [
            "",
            "Confusion matrix (gold rows, predicted columns):",
            "",
            "| Gold / predicted | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        for name, counts in zip(names, confusion["matrix"], strict=True):
            lines.append("| " + name + " | " + " | ".join(str(int(value)) for value in counts) + " |")
    lines += [
        "",
        "## Scope",
        "",
        "This is one trained seed on the fixed official selectors. It is an E1/E6 result,",
        "but the planned three-seed summary, B1–B4/B6 baselines, ablations, LOEO and HumAID",
        "runs are still outstanding. Development-set checkpoint selection and calibration",
        "are recorded in this run's `metrics_dev.json`; B5 is reported separately in",
        "`baselines_clip_zeroshot.md`. Do not use this report as a complete E1–E6 table.",
        "",
        "## Figures",
        "",
        f"![Reliability diagrams for the four heads](../figures/classifier_seed{seed}_reliability.png)",
        "",
        f"![Humanitarian risk–coverage curve](../figures/classifier_seed{seed}_risk_coverage.png)",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"sakshi-model evaluate --run runs/classifier/{run_dir.name}",
        f"sakshi-model report --run runs/classifier/{run_dir.name}",
        "```",
        "",
    ]
    atomic.text(Path(output), "\n".join(lines))
    return Path(output)
