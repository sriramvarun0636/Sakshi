"""Model-bound aggregate runs with scoped inputs, audit metadata and ranking reports."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml

from sakshi import atomic, paths
from sakshi.aggregate.clusters import MODEL_NAME, MODEL_REVISION, MiniLMEmbedder, TextEmbedder, claim_clusters
from sakshi.aggregate.core import evidence_groups, rollup_samples
from sakshi.aggregate.packets import load_packets, validate_selected
from sakshi.aggregate.ranking import evaluate_window, format_report, gold_relevance
from sakshi.aggregate.replay import ALLOWED_HOURS, aggregate_window, windows
from sakshi.config import environment, seed_everything
from sakshi.data.events import EVENTS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregateConfig:
    model_id: str = ""
    event: str = ""
    mode: str = "loeo"
    seed: int = 0
    window_hours: int = 24
    predictions: str | None = None
    classifier_run: str | None = None
    packets: str | None = None
    manifest: str = "data/processed/manifest.parquet"
    provenance_dir: str = "data/processed/provenance"
    output_root: str = "runs/aggregate"
    report_dir: str = "reports/results"

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.model_id) or self.event not in EVENTS:
            raise ValueError("aggregation requires a path-safe model_id and known event")
        if self.mode not in {"loeo", "heldout", "demo"} or self.window_hours not in ALLOWED_HOURS or self.seed < 0:
            raise ValueError("invalid aggregation mode, window size or seed")
        if self.mode != "demo" and not self.classifier_run:
            raise ValueError("research aggregation requires classifier_run metadata; use demo for a baseline")


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else paths.REPO / path


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    atomic.text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_scope(manifest: pd.DataFrame, config: AggregateConfig) -> pd.DataFrame:
    """Assert the training fold's identity before selecting downstream evaluation rows."""
    config.validate()
    rows = manifest.loc[(manifest.dataset == "crisismmd") & (manifest.event == config.event)].copy()
    if config.mode != "demo":
        run = _path(config.classifier_run)
        run_config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        env = json.loads((run / "env.json").read_text(encoding="utf-8"))
        if env.get("model_id") != config.model_id:
            raise ValueError("classifier run model_id differs from aggregation model_id")
        fold = run_config.get("fold")
        if config.mode == "loeo":
            if not fold or str(fold).removeprefix("loeo_") != config.event:
                raise ValueError("LOEO classifier fold does not match the aggregation event")
            column = f"loeo_{config.event}"
            if column not in rows or not rows[column].eq("test").all():
                raise ValueError("event rows are not all test rows of this LOEO fold")
        else:
            if fold is not None:
                raise ValueError("heldout fallback must use a main classifier run without a fold")
            rows = rows.loc[rows.split == "test"]
    if rows.empty:
        raise ValueError("aggregation scope contains no event samples")
    return rows


def collect_ranking_reports(config: AggregateConfig, root: Path) -> Path:
    """Collect completed runs of the same scope without mixing demo and research results."""
    records = []
    for marker in sorted(root.glob("*/*/complete.json")):
        saved = json.loads((marker.parent / "config.json").read_text(encoding="utf-8"))
        if saved["mode"] == config.mode:
            committed = json.loads(marker.read_text(encoding="utf-8"))
            if _hash(marker.parent / "ranking.json") != committed["artifacts"]["ranking.json"]:
                raise ValueError(f"completed ranking artifact changed at {marker.parent}")
            records.extend(json.loads((marker.parent / "ranking.json").read_text(encoding="utf-8")))
    filename = {"demo": "ranking_demo.md", "heldout": "ranking_heldout.md", "loeo": "ranking.md"}[config.mode]
    output = _path(config.report_dir) / filename
    atomic.text(output, format_report(records, mode=config.mode))
    return output


def run_aggregation(config: AggregateConfig, *, embedder: TextEmbedder | None = None) -> Path:
    """Write aggregates and ranking audits; a final completion marker commits the run."""
    config.validate()
    manifest_path = _path(config.manifest)
    prediction_path = (
        _path(config.predictions) if config.predictions else paths.PROCESSED / "predictions" / f"{config.model_id}.parquet"
    )
    provenance_path = _path(config.provenance_dir) / f"replay_{config.event}.parquet"
    packet_path = _path(config.packets) if config.packets else paths.PROCESSED / "packets" / f"{config.model_id}.jsonl"
    manifest = pd.read_parquet(manifest_path)
    rows = select_scope(manifest, config)
    predictions = pd.read_parquet(prediction_path)
    if predictions.model_id.isna().any() or set(predictions.model_id) != {config.model_id}:
        raise ValueError("prediction table must contain exactly the requested model_id")
    provenance = pd.read_parquet(provenance_path)
    tweets = rollup_samples(rows, predictions, provenance)
    inputs = {"manifest": manifest_path, "predictions": prediction_path, "provenance": provenance_path}
    if config.classifier_run:
        for name in ("config.yaml", "env.json"):
            inputs[f"classifier_{name}"] = _path(config.classifier_run) / name
    evidence_ready = config.mode != "demo" or config.packets is not None
    if evidence_ready:
        packets = load_packets(packet_path, config.model_id)
        validate_selected(set(rows.sample_id), packets, rows, predictions, provenance)
        inputs["packets"] = packet_path
    signature = {name: {"path": str(path.resolve()), "sha256": _hash(path)} for name, path in inputs.items()}
    if embedder is None or isinstance(embedder, MiniLMEmbedder):
        signature["embedder"] = {
            "model": getattr(embedder, "model_name", MODEL_NAME),
            "revision": getattr(embedder, "revision", MODEL_REVISION),
        }
    else:
        signature["embedder"] = {"implementation": f"{type(embedder).__module__}.{type(embedder).__qualname__}"}
    signature["implementation"] = {file.name: _hash(file) for file in sorted(Path(__file__).parent.glob("*.py"))}
    root = _path(config.output_root) / "demo" if config.mode == "demo" else _path(config.output_root)
    directory = root / config.model_id / config.event
    metadata = directory / "config.json"
    complete = directory / "complete.json"
    if directory.exists() and any(directory.iterdir()):
        if not metadata.is_file() or json.loads(metadata.read_text()) != asdict(config):
            raise FileExistsError("aggregation output already contains a different run configuration; choose a new output_root")
        if json.loads((directory / "inputs.json").read_text()) != signature:
            raise FileExistsError("aggregation input artifacts changed; choose a new output_root to preserve the previous run")
        if complete.exists():
            committed = json.loads(complete.read_text())
            if all(
                (directory / name).is_file() and _hash(directory / name) == digest
                for name, digest in committed["artifacts"].items()
            ):
                collect_ranking_reports(config, root)
                return complete
            raise ValueError("completed aggregation artifacts are absent or changed")
    seed_everything(config.seed)
    if embedder is None:
        embedder = MiniLMEmbedder()
    _json(metadata, asdict(config))
    _json(directory / "inputs.json", signature)
    _json(directory / "env.json", environment(model_id=config.model_id, seed=config.seed, run_id=f"aggregate/{config.event}"))
    atomic.text(directory / "config.yaml", yaml.safe_dump(asdict(config), sort_keys=False))
    atomic.parquet(directory / "tweets.parquet", pd.DataFrame([asdict(tweet) for tweet in tweets]))
    gold = gold_relevance(rows)
    ranking = []
    artifacts = ["config.json", "config.yaml", "inputs.json", "env.json", "tweets.parquet", "ranking.json", "ranking.md"]
    for start, end, subset in windows(tweets, hours=config.window_hours):
        clusters, omitted = claim_clusters(subset, embedder)
        groups = evidence_groups(subset)
        aggregate = aggregate_window(
            subset,
            event=config.event,
            model_id=config.model_id,
            start=start,
            end=end,
            embedder=embedder,
            cluster_ids=clusters,
            group_ids=groups,
            omitted_ids=omitted,
        )
        aggregate["input_scope"] = config.mode
        aggregate["evidence_ready"] = evidence_ready
        name = f"{start.strftime('%Y%m%dT%H%M%SZ')}.json"
        _json(directory / name, aggregate)
        artifacts.append(name)
        scores = evaluate_window(
            [tweet for tweet in subset if tweet.tweet_id not in omitted], clusters, groups, gold, seed=config.seed
        )
        ranking.append(
            {
                "model_id": config.model_id,
                "event": config.event,
                "window": aggregate["window"],
                "clustering_omitted": len(omitted),
                **scores,
            }
        )
        log.info(
            "%s %s: %d tweets, %d selected", config.event, aggregate["window"]["start"], len(subset), len(aggregate["selected"])
        )
    _json(directory / "ranking.json", ranking)
    atomic.text(directory / "ranking.md", format_report(ranking, mode=config.mode))
    _json(
        complete,
        {
            "mode": config.mode,
            "windows": len(ranking),
            "tweets": len(tweets),
            "evidence_ready": evidence_ready,
            "artifacts": {name: _hash(directory / name) for name in artifacts},
        },
    )
    collect_ranking_reports(config, root)
    return complete
