"""Reproducible event roles, injection plans and metrics for provenance §8.9."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.compute.files import hash_file
from sakshi.data.features import EMBED_FILE, MODEL_REVISION
from sakshi.images import load_rgb
from sakshi.provenance.detect import IndexImage, QueryImage
from sakshi.provenance.index import ProvenanceIndex
from sakshi.provenance.transforms import NAMES, transform
from sakshi.types import ProvStatus

TUNING_EVENTS = ("hurricane_harvey", "hurricane_irma", "srilanka_floods")
TEST_EVENTS = ("hurricane_maria", "mexico_earthquake", "california_wildfires", "iraq_iran_earthquake")
REVIEW_STATUSES = {ProvStatus.CROSS_EVENT_CONCURRENT, ProvStatus.MATCH_UNDATED, ProvStatus.TEMPLATE_MATCH}
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TauGrid:
    start: float = 0.80
    stop: float = 0.99
    step: float = 0.005

    def values(self) -> list[float]:
        if not -1 <= self.start <= self.stop <= 1 or self.step <= 0:
            raise ValueError("invalid CLIP threshold grid")
        steps = round((self.stop - self.start) / self.step)
        values = [round(self.start + k * self.step, 6) for k in range(steps + 1)]
        if abs(values[-1] - self.stop) > 1e-6:
            raise ValueError("CLIP threshold grid stop must be reached exactly")
        return values


@dataclass(frozen=True)
class BenchmarkConfig:
    phash_max_distance: int = 8
    phash_min_side: int = 64
    tau_clip: float = 0.92
    tau_graphic: float = 0.5
    tuning_events: list[str] = field(default_factory=lambda: list(TUNING_EVENTS))
    test_events: list[str] = field(default_factory=lambda: list(TEST_EVENTS))
    rates: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    transforms: list[str] = field(default_factory=lambda: list(NAMES))
    tau_grid: TauGrid = field(default_factory=TauGrid)
    batch_size: int = 16

    def __post_init__(self) -> None:
        if tuple(self.tuning_events) != TUNING_EVENTS or tuple(self.test_events) != TEST_EVENTS:
            raise ValueError("benchmark tuning and test events must have the fixed §8.9 roles")
        if self.rates != [0.01, 0.05, 0.10, 0.20] or self.seeds != [0, 1, 2]:
            raise ValueError("benchmark rates and seeds must match §8.9")
        if tuple(self.transforms) != NAMES:
            raise ValueError("benchmark must include all nine §8.9 transformations")
        if not 0 <= self.tau_graphic <= 1 or not 0 <= self.phash_max_distance <= 64 or self.phash_min_side < 1:
            raise ValueError("invalid provenance benchmark thresholds")
        if self.batch_size < 1:
            raise ValueError("benchmark batch size must be positive")
        self.tau_grid.values()


@dataclass(frozen=True)
class Injection:
    query_id: str
    source_id: str
    event: str
    time: datetime
    transform_seed: int


@dataclass(frozen=True)
class Outcome:
    status: ProvStatus
    latency_ms: float
    ambiguous: bool = False
    hard_negative: bool = False
    origin_correct: bool = False


@dataclass(frozen=True)
class Prepared:
    query: QueryImage
    candidates: list[IndexImage]
    preparation_ms: float
    source_id: str | None = None


def genuine_ids(index: ProvenanceIndex, event: str) -> list[str]:
    """All CrisisMMD event images in timestamp and stable sample-ID order."""
    if event not in (*TUNING_EVENTS, *TEST_EVENTS):
        raise ValueError(f"invalid benchmark event {event!r}")
    rows = index.manifest
    subset = rows.loc[(rows.dataset == "crisismmd") & (rows.event == event)]
    if subset.posted_at.isna().any() or subset.empty:
        raise ValueError(f"benchmark event {event} has absent timestamps or no images")
    return subset.reset_index(drop=True).sort_values(["posted_at", "sample_id"], kind="stable").sample_id.astype(str).tolist()


def ambiguous_ids(index: ProvenanceIndex) -> set[str]:
    """Components containing images from more than one CrisisMMD event."""
    rows = index.manifest.loc[index.manifest.dataset == "crisismmd"]
    cross = rows.groupby("component").event.nunique() > 1
    return set(rows.loc[rows.component.isin(cross.index[cross]), "sample_id"].astype(str))


def positive_pool(index: ProvenanceIndex, event: str) -> list[str]:
    """Older, non-degenerate, non-graphic sources from other events."""
    if event not in index.onsets:
        raise ValueError(f"event {event!r} has no onset")
    frame = index.frame
    eligible = (
        frame.first_seen.notna()
        & (frame.first_seen < index.onsets[event])
        & (frame.index_event != event)
        & ~frame.degenerate.astype(bool)
        & ~frame.is_graphic.astype(bool)
    )
    return frame.loc[eligible, "sample_id"].astype(str).tolist()


def injection_plan(index: ProvenanceIndex, event: str, *, rate: float, seed: int) -> list[Injection]:
    """Choose fixed source images and times; transformations never enter the index."""
    if not 0 < rate < 1 or seed < 0:
        raise ValueError("injection rate must be between zero and one; seed must be nonnegative")
    ids = genuine_ids(index, event)
    n = round(rate * len(ids) / (1 - rate))
    pool = positive_pool(index, event)
    if n > len(pool):
        raise ValueError(f"event {event} needs {n} positive sources but pool has only {len(pool)}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n, replace=False)
    event_rows = index.manifest.loc[ids]
    first = int(event_rows.posted_at.min().timestamp())
    last = int(event_rows.posted_at.max().timestamp())
    times = rng.integers(first, last + 1, size=n)
    return [
        Injection(
            query_id=f"inj:{event}:{seed}:{k}",
            source_id=str(source),
            event=event,
            time=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
            transform_seed=int(rng.integers(0, 2**32)),
        )
        for k, (source, timestamp) in enumerate(zip(chosen, times, strict=True))
    ]


def score_case(genuine: list[Outcome], injected: list[Outcome]) -> dict[str, float | int | None]:
    """Apply the specified positive, ambiguity, review and origin rules."""
    if not genuine or not injected:
        raise ValueError("a benchmark case needs genuine and injected queries")
    negatives = [outcome for outcome in genuine if not outcome.ambiguous]
    hard = [outcome for outcome in genuine if outcome.hard_negative]
    tp = sum(outcome.status == ProvStatus.RECYCLED for outcome in injected)
    fp = sum(outcome.status == ProvStatus.RECYCLED for outcome in negatives)
    origin_hits = sum(outcome.status == ProvStatus.RECYCLED and outcome.origin_correct for outcome in injected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(injected)
    timings = [outcome.latency_ms for outcome in [*genuine, *injected]]
    if not np.isfinite(timings).all() or min(timings) < 0:
        raise ValueError("latencies must be finite and nonnegative")
    return {
        "n_genuine": len(genuine),
        "n_nonambiguous": len(negatives),
        "n_ambiguous": len(genuine) - len(negatives),
        "n_injected": len(injected),
        "tp": int(tp),
        "fp": int(fp),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "recall_at_origin": origin_hits / len(injected),
        "fpr_genuine": fp / len(negatives) if negatives else None,
        "review_flag_rate": sum(outcome.status in REVIEW_STATUSES for outcome in genuine) / len(genuine),
        "hard_negative_accuracy": (sum(outcome.status != ProvStatus.RECYCLED for outcome in hard) / len(hard) if hard else None),
        "mean_latency_ms": float(np.mean(timings)),
    }


def auprc(points: list[dict[str, float | int | None]]) -> float:
    """Trapezoidal area under the observed threshold-grid PR curve."""
    if not points:
        raise ValueError("PR curve must have at least one point")
    coords = sorted({(float(point["recall"]), float(point["precision"])) for point in points})
    coords.insert(0, (0.0, 1.0))
    return float(sum((right[0] - left[0]) * (left[1] + right[1]) / 2 for left, right in zip(coords, coords[1:], strict=False)))


def prepare_genuine(index: ProvenanceIndex, event: str, *, min_tau_clip: float) -> list[Prepared]:
    """Cache each real query's embedding and candidates once per event."""
    result = []
    for sample_id in genuine_ids(index, event):
        started = perf_counter()
        query = index.prepare_query(sample_id)
        candidates = index.candidates(query, min_tau_clip=min_tau_clip)
        result.append(Prepared(query, candidates, (perf_counter() - started) * 1000))
    return result


def prepare_injected(
    index: ProvenanceIndex,
    plan: list[Injection],
    transformation: str,
    *,
    min_tau_clip: float,
    batch_size: int = 16,
) -> list[Prepared]:
    """Transform source pixels in memory and encode each query exactly once in batches."""
    if batch_size < 1:
        raise ValueError("benchmark batch size must be positive")
    result = []
    for start in range(0, len(plan), batch_size):
        batch = plan[start : start + batch_size]
        transformed_images = []
        transform_ms = []
        try:
            for injection in batch:
                started = perf_counter()
                source = index.manifest.loc[injection.source_id]
                image = load_rgb(paths.image_file(source))
                if image is None:
                    raise FileNotFoundError(f"indexed source image is no longer readable: {injection.source_id}")
                try:
                    transformed_images.append(transform(image, transformation, seed=injection.transform_seed))
                finally:
                    image.close()
                transform_ms.append((perf_counter() - started) * 1000)
            encode_started = perf_counter()
            queries = index.prepare_image_batch(
                transformed_images,
                times=[injection.time for injection in batch],
                events=[injection.event for injection in batch],
            )
            encode_ms = (perf_counter() - encode_started) * 1000 / len(batch)
            for injection, query, image_ms in zip(batch, queries, transform_ms, strict=True):
                lookup_started = perf_counter()
                query = replace(query, post_id=injection.query_id)
                candidates = index.candidates(query, min_tau_clip=min_tau_clip)
                elapsed = image_ms + encode_ms + (perf_counter() - lookup_started) * 1000
                result.append(Prepared(query, candidates, elapsed, injection.source_id))
        finally:
            for image in transformed_images:
                image.close()
    return result


def classify_prepared(
    index: ProvenanceIndex,
    prepared_queries: list[Prepared],
    *,
    tau_clip: float,
    ambiguous: set[str],
    graphic_by_id: dict[str, bool],
) -> list[Outcome]:
    """Classify each cached query once at a threshold; real outcomes can be reused."""
    outcomes = []
    for prepared in prepared_queries:
        started = perf_counter()
        result = index.check_query(prepared.query, candidates=prepared.candidates, tau_clip=tau_clip)
        elapsed = prepared.preparation_ms + (perf_counter() - started) * 1000
        hard_negative = bool(
            prepared.source_id is None and prepared.query.is_graphic and graphic_by_id.get(result.top_match_post_id, False)
        )
        origin_correct = False
        if result.origin_post_id is not None and prepared.source_id is not None:
            origin_correct = result.origin_post_id == prepared.source_id or (
                result.origin_post_id in index.manifest.index
                and index.manifest.loc[result.origin_post_id, "component"] == index.manifest.loc[prepared.source_id, "component"]
            )
        outcomes.append(
            Outcome(
                result.status,
                elapsed,
                ambiguous=prepared.source_id is None and prepared.query.post_id in ambiguous,
                hard_negative=hard_negative,
                origin_correct=origin_correct,
            )
        )
    return outcomes


def evaluate_case(
    index: ProvenanceIndex,
    genuine: list[Prepared],
    injected: list[Prepared],
    *,
    tau_clip: float,
    ambiguous: set[str],
) -> dict[str, float | int | None]:
    """Evaluate cached candidates at a CLIP threshold without re-encoding images."""
    graphic_by_id = dict(zip(index.frame.sample_id.astype(str), index.frame.is_graphic.astype(bool), strict=True))
    real = classify_prepared(index, genuine, tau_clip=tau_clip, ambiguous=ambiguous, graphic_by_id=graphic_by_id)
    positive = classify_prepared(index, injected, tau_clip=tau_clip, ambiguous=ambiguous, graphic_by_id=graphic_by_id)
    return score_case(real, positive)


def _table(frame: pd.DataFrame, key: str) -> list[str]:
    columns = [
        "precision",
        "recall",
        "f1",
        "recall_at_origin",
        "fpr_genuine",
        "review_flag_rate",
        "hard_negative_accuracy",
        "mean_latency_ms",
    ]
    means = frame.groupby(key, dropna=False)[columns].mean()
    lines = [f"### By {key}", "", "| " + key + " | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for label, row in means.iterrows():
        lines.append("| " + str(label) + " | " + " | ".join(f"{value:.3f}" for value in row) + " |")
    return lines


def _write_report(
    config: BenchmarkConfig,
    selected_tau: float,
    tuning_scores: dict[float, list[float]],
    cases: list[dict],
    curves: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    frame = pd.DataFrame(cases)
    payload = {
        "protocol": "closed_world_injection_v1",
        "selected_tau_clip": selected_tau,
        "tau_graphic": config.tau_graphic,
        "tuning_mean_f1": {str(tau): float(np.mean(scores)) for tau, scores in tuning_scores.items()},
        "test_cases": cases,
        "pr_curves": curves,
        "auprc": {name: auprc(points) for name, points in curves.items()},
        "config": asdict(config),
    }
    lines = [
        "# Provenance injection benchmark",
        "",
        "Closed-world test: each injected image's original is present in the index. This cannot measure",
        "recall for recycled images whose originals were never indexed.",
        "",
        f"Selected CLIP threshold on tuning events: **{selected_tau:.3f}**. Graphic threshold: **{config.tau_graphic:.3f}**.",
        f"The {len(cases)} held-out cases cover {len(config.test_events)} events, {len(config.transforms)} transforms,",
        f"{len(config.rates)} injection rates and {len(config.seeds)} seeds. Rates and seeds are macro-averaged.",
        "",
        "### Overall",
        "",
    ]
    for metric in (
        "precision",
        "recall",
        "f1",
        "recall_at_origin",
        "fpr_genuine",
        "review_flag_rate",
        "hard_negative_accuracy",
        "mean_latency_ms",
    ):
        lines.append(f"- {metric}: {frame[metric].mean():.3f}")
    lines.append("")
    lines.append("AUPRC by transformation: " + ", ".join(f"{name} {auprc(points):.3f}" for name, points in curves.items()))
    lines.extend(["", *_table(frame, "transform"), "", *_table(frame, "rate"), "", *_table(frame, "event"), ""])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir.parent / "figures"
    for name, points in curves.items():
        fig, ax = plt.subplots(figsize=(6.7, 4.7), facecolor="#ffffff")
        ordered = sorted(points, key=lambda point: (point["recall"], point["precision"]))
        recall = [point["recall"] for point in ordered]
        precision = [point["precision"] for point in ordered]
        ax.fill_between(recall, precision, color="#d8ecea", alpha=0.7)
        ax.plot(recall, precision, color="#087f78", linewidth=2.3)
        selected = next(point for point in points if point["tau_clip"] == selected_tau)
        ax.scatter([selected["recall"]], [selected["precision"]], s=64, color="#d45d3e", zorder=3)
        ax.annotate(
            "selected threshold",
            (selected["recall"], selected["precision"]),
            xytext=(8, 9),
            textcoords="offset points",
            fontsize=8,
            color="#893b2b",
        )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        ax.set_title(name.replace("_", " ").title(), loc="left", fontsize=16, fontweight="bold", color="#172b3a", pad=15)
        ax.text(
            1,
            1.045,
            f"AUPRC  {auprc(points):.3f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=10,
            color="#087f78",
            fontweight="bold",
        )
        ax.grid(color="#dce4e8", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9aaab4")
        ax.tick_params(colors="#536b79")
        fig.tight_layout(pad=2)
        atomic.write(figure_dir / f"prov_pr_{name}.png", lambda temporary, figure=fig: figure.savefig(temporary, dpi=180))
        plt.close(fig)
    degradation = frame.groupby("transform").recall.mean().sort_values()
    fig, ax = plt.subplots(figsize=(9.2, 5.3), facecolor="#ffffff")
    colors = ["#d45d3e" if value < 0.3 else "#087f78" for value in degradation]
    bars = ax.barh([name.replace("_", " ") for name in degradation.index], degradation.to_numpy(), color=colors, height=0.68)
    ax.invert_yaxis()
    for bar, value in zip(bars, degradation, strict=True):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=9,
            color="#172b3a",
            fontweight="bold",
        )
    ax.set(xlim=(0, 1.11), xlabel="Recall")
    ax.set_title("Recycled-image recall by transformation", loc="left", fontsize=17, fontweight="bold", color="#172b3a", pad=14)
    ax.text(
        0, 1.01, "Held-out events · selected CLIP threshold", transform=ax.transAxes, fontsize=9, color="#536b79", va="bottom"
    )
    ax.grid(axis="x", color="#dce4e8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9aaab4")
    ax.tick_params(axis="y", length=0, pad=8, colors="#172b3a")
    ax.tick_params(axis="x", colors="#536b79")
    fig.tight_layout(pad=2)
    atomic.write(figure_dir / "prov_degradation.png", lambda temporary: fig.savefig(temporary, dpi=180))
    plt.close(fig)
    lines.extend(["### Figures", "", "![Recall by transformation](../figures/prov_degradation.png)", ""])
    lines.extend(
        [
            f"- [{name.replace('_', ' ').title()} PR curve](../figures/prov_pr_{name}.png) (AUPRC {auprc(points):.3f})"
            for name, points in curves.items()
        ]
    )
    lines.append("")
    atomic.text(output_dir / "benchmark.md", "\n".join(lines))
    # JSON is the completion marker: only publish it after every report figure exists.
    atomic.text(output_dir / "benchmark.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run_benchmark(
    config: BenchmarkConfig,
    *,
    index: ProvenanceIndex | None = None,
    output_dir: Path = paths.REPORTS / "provenance",
) -> Path:
    """Tune on three events, freeze τ_clip, then evaluate four held-out events."""
    supplied_index = index is not None
    if index is None:
        index = ProvenanceIndex.load(
            tau_clip=config.tau_clip,
            tau_graphic=config.tau_graphic,
            phash_max_distance=config.phash_max_distance,
            phash_min_side=config.phash_min_side,
        )
    if not np.array_equal(index.frame.is_graphic.to_numpy(dtype=bool), index.frame.p_graphic.to_numpy() >= config.tau_graphic):
        raise ValueError("provenance index graphic flags do not match benchmark tau_graphic; rebuild the index")
    grid = config.tau_grid.values()
    minimum = min(grid)
    cache_dir = None
    if not supplied_index:
        files = {
            "index": paths.PROCESSED / "provenance" / "index.parquet",
            "manifest": paths.PROCESSED / "manifest.parquet",
            "embeddings": paths.PROCESSED / "features" / EMBED_FILE,
            "benchmark_code": Path(__file__),
            "index_code": Path(__file__).with_name("index.py"),
            "decision_code": Path(__file__).with_name("detect.py"),
            "transform_code": Path(__file__).with_name("transforms.py"),
        }
        signature = {
            "config": asdict(config),
            "clip_revision": MODEL_REVISION,
            "files": {name: hash_file(path) for name, path in files.items()},
        }
        key = sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        cache_dir = paths.REPO / "runs" / "provenance" / "benchmark_cache" / key
        cache_dir.mkdir(parents=True, exist_ok=True)
        atomic.text(cache_dir / "signature.json", json.dumps(signature, indent=2) + "\n")

    def scores_for_case(name: str, genuine_by_tau: dict, plan: list[Injection], transformation: str) -> dict:
        cached = cache_dir / f"{name}.json" if cache_dir is not None else None
        if cached is not None and cached.is_file():
            scores = json.loads(cached.read_text(encoding="utf-8"))
            if set(scores) != {str(tau) for tau in grid}:
                raise ValueError(f"incomplete benchmark cache case {cached}")
            return scores
        injected = prepare_injected(index, plan, transformation, min_tau_clip=minimum, batch_size=config.batch_size)
        scores = {
            str(tau): score_case(
                genuine_by_tau[tau],
                classify_prepared(index, injected, tau_clip=tau, ambiguous=ambiguous, graphic_by_id=graphic_by_id),
            )
            for tau in grid
        }
        if cached is not None:
            atomic.text(cached, json.dumps(scores, allow_nan=False) + "\n")
        return scores

    ambiguous = ambiguous_ids(index)
    graphic_by_id = dict(zip(index.frame.sample_id.astype(str), index.frame.is_graphic.astype(bool), strict=True))
    tuning_scores: dict[float, list[float]] = {tau: [] for tau in grid}
    for event in config.tuning_events:
        genuine = prepare_genuine(index, event, min_tau_clip=minimum)
        real_by_tau = {
            tau: classify_prepared(index, genuine, tau_clip=tau, ambiguous=ambiguous, graphic_by_id=graphic_by_id) for tau in grid
        }
        for transformation in config.transforms:
            for seed in config.seeds:
                plan = injection_plan(index, event, rate=0.05, seed=seed)
                scores = scores_for_case(f"tune-{event}-{transformation}-{seed}", real_by_tau, plan, transformation)
                for tau in grid:
                    tuning_scores[tau].append(float(scores[str(tau)]["f1"]))
                log.info("tuning event=%s transform=%s seed=%d", event, transformation, seed)
    # The higher threshold wins a tie to avoid gratuitous false positives.
    selected_tau = max(grid, key=lambda tau: (float(np.mean(tuning_scores[tau])), tau))
    cases: list[dict] = []
    curves: dict[str, list[dict]] = {name: [] for name in config.transforms}
    curve_parts: dict[str, dict[float, list[dict]]] = {name: {tau: [] for tau in grid} for name in config.transforms}
    for event in config.test_events:
        genuine = prepare_genuine(index, event, min_tau_clip=minimum)
        real_by_tau = {
            tau: classify_prepared(index, genuine, tau_clip=tau, ambiguous=ambiguous, graphic_by_id=graphic_by_id) for tau in grid
        }
        for transformation in config.transforms:
            for rate in config.rates:
                for seed in config.seeds:
                    plan = injection_plan(index, event, rate=rate, seed=seed)
                    scores = scores_for_case(
                        f"test-{event}-{transformation}-{rate:.2f}-{seed}", real_by_tau, plan, transformation
                    )
                    for tau in grid:
                        stats = scores[str(tau)]
                        curve_parts[transformation][tau].append(stats)
                        if tau == selected_tau:
                            cases.append({"event": event, "transform": transformation, "rate": rate, "seed": seed, **stats})
                    log.info("test event=%s transform=%s rate=%.2f seed=%d", event, transformation, rate, seed)
    for name, by_tau in curve_parts.items():
        curves[name] = [
            {
                "tau_clip": tau,
                "precision": float(np.mean([part["precision"] for part in parts])),
                "recall": float(np.mean([part["recall"] for part in parts])),
            }
            for tau, parts in by_tau.items()
        ]
    _write_report(config, selected_tau, tuning_scores, cases, curves, output_dir)
    return output_dir / "benchmark.json"
