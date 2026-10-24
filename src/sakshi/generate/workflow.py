"""Bind WP5 outputs to completed WP4 aggregates and same-model WP3 packets."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml

from sakshi import atomic, paths
from sakshi.aggregate.packets import load_packets
from sakshi.config import environment
from sakshi.data.events import EVENTS
from sakshi.explain.packets import validate_packet
from sakshi.generate.llm import CachedLLM, HFClient, LLMClient, OllamaClient
from sakshi.generate.reports import summary
from sakshi.generate.sitrep import generate_sitrep
from sakshi.generate.verify import DebertaNLI, EntailmentScorer, Verifier


@dataclass(frozen=True)
class LLMSettings:
    backend: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    max_tokens: int = 1024
    retries: int = 2
    tokenizer: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class NLISettings:
    model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tau_entail: float = 0.5
    tau_contra: float = 0.3
    max_premise_tokens: int = 480


@dataclass(frozen=True)
class GenerateConfig:
    aggregate_dir: str = ""
    packets: str = ""
    mode: str = "loeo"
    output_root: str = "runs/sitreps"
    report_path: str = "reports/generation/summary.md"
    cache_root: str = "runs/llm_cache"
    evaluation_plan: str = "runs/generation/evaluation_plan.json"
    llm: LLMSettings = field(default_factory=LLMSettings)
    nli: NLISettings = field(default_factory=NLISettings)

    def validate(self) -> None:
        if not self.aggregate_dir or not self.packets or self.mode not in {"loeo", "heldout", "demo"}:
            raise ValueError("generation requires aggregate_dir, packets and a valid mode")
        if self.llm.backend not in {"ollama", "hf"} or self.llm.retries != 2 or self.llm.temperature < 0:
            raise ValueError("invalid LLM backend, retries or temperature")
        if self.llm.max_tokens < 1 or self.llm.seed < 0:
            raise ValueError("invalid LLM token budget or seed")


def _path(name: str) -> Path:
    value = Path(name)
    return value if value.is_absolute() else paths.REPO / value


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    atomic.text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _selected(
    aggregate: dict, tweets: pd.DataFrame, packets: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, str | None], dict[str, str | None]]:
    by_samples = {}
    for row in tweets.to_dict("records"):
        key = frozenset(row["sample_ids"])
        if key in by_samples:
            raise ValueError("aggregate tweet table contains duplicate sample groups")
        by_samples[key] = row
    selected = {}
    severity = {}
    origin_status = {}
    if [item["pid"] for item in aggregate["selected"]] != [f"P{index}" for index in range(1, len(aggregate["selected"]) + 1)]:
        raise ValueError("aggregate P-IDs must be consecutive in selected order")
    for item in aggregate["selected"]:
        pid = item["pid"]
        if not re.fullmatch(r"P[0-9]{1,3}", pid) or pid in selected:
            raise ValueError("invalid or duplicate aggregate P-ID")
        sample_ids = item["sample_ids"]
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("aggregate selected item needs unique sample IDs")
        row = by_samples.get(frozenset(sample_ids))
        if row is None or row["informative_label"] != "informative" or row["abstain"] or row["excluded"]:
            raise ValueError("selected post does not match an eligible aggregate tweet")
        if row["event"] != aggregate["event"] or row["humanitarian_label"] != item["category"]:
            raise ValueError("selected tweet event or category differs from aggregate")
        representative_id = row["representative_id"]
        if representative_id not in sample_ids or any(sample_id not in packets for sample_id in sample_ids):
            raise ValueError("selected tweet lacks complete same-model packet coverage")
        for sample_id in sample_ids:
            packet = packets[sample_id]
            validate_packet(packet)
            if packet["provenance"]["status"] in {"recycled", "template_match"}:
                raise ValueError("selected tweet contains excluded provenance")
            if packet["event"] != aggregate["event"]:
                raise ValueError("selected packet event differs")
            posted_at = packet["posted_at"]
            start = pd.Timestamp(aggregate["window"]["start"])
            end = pd.Timestamp(aggregate["window"]["end"])
            if (
                posted_at is None
                or not start <= pd.Timestamp(posted_at) < end
                or pd.Timestamp(posted_at) != pd.Timestamp(row["posted_at"])
            ):
                raise ValueError("selected packet is outside aggregate window")
        packet = packets[representative_id]
        if packet["predictions"]["humanitarian"]["label"] != item["category"] or packet["abstain"]:
            raise ValueError("selected packet label or abstention differs from aggregate")
        selected[pid] = packet
        value = row["severity_label"]
        severity[pid] = None if pd.isna(value) else str(value)
        statuses = {packets[sample_id]["provenance"]["status"] for sample_id in sample_ids}
        origin_status[pid] = (
            "cross_event_concurrent"
            if "cross_event_concurrent" in statuses
            else "match_undated"
            if "match_undated" in statuses
            else None
        )
        rolled_status = row.get("provenance_status")
        rolled_status = None if pd.isna(rolled_status) else str(rolled_status)
        if rolled_status != origin_status[pid]:
            raise ValueError("aggregate provenance flag differs from selected packets")
    return selected, severity, origin_status


def build_evaluation_plan(
    aggregate_root: Path, output: Path, *, min_eligible: int = 10, max_windows: int = 60, seed: int = 0
) -> Path:
    """Freeze a balanced random sample of eligible 24-hour LOEO windows across three events."""
    if min_eligible < 1 or max_windows < 1 or seed < 0:
        raise ValueError("evaluation selection limits and seed must be positive")
    events = ("hurricane_harvey", "hurricane_maria", "mexico_earthquake")
    candidates: dict[str, list[dict]] = {event: [] for event in events}
    models: dict[str, set[str]] = {event: set() for event in events}
    for marker in sorted(aggregate_root.glob("*/[a-z]*/complete.json")):
        metadata = json.loads((marker.parent / "config.json").read_text(encoding="utf-8"))
        if metadata["mode"] != "loeo" or metadata["event"] not in candidates:
            continue
        models[metadata["event"]].add(metadata["model_id"])
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if not completed["evidence_ready"]:
            raise ValueError("evaluation aggregate lacks validated packet evidence")
        for name, digest in completed["artifacts"].items():
            if _hash(marker.parent / name) != digest:
                raise ValueError("evaluation aggregate changed after completion")
        for path in sorted(marker.parent.glob("*.json")):
            if not re.fullmatch(r"\d{8}T\d{6}Z\.json", path.name):
                continue
            window = json.loads(path.read_text(encoding="utf-8"))
            eligible = sum(window["counts"]["by_category"].values())
            if window["window"]["hours"] != 24 or eligible < min_eligible or window.get("input_scope") != "loeo":
                continue
            candidates[metadata["event"]].append(
                {
                    "model_id": window["model_id"],
                    "event": window["event"],
                    "window": path.name,
                    "aggregate_sha256": _hash(path),
                    "eligible": eligible,
                }
            )
    if any(not items for items in candidates.values()) or any(len(names) != 1 for names in models.values()):
        raise ValueError("all three evaluation events need eligible, evidence-ready LOEO windows")
    rng = random.Random(seed)
    for items in candidates.values():
        rng.shuffle(items)
    chosen = []
    while len(chosen) < max_windows and any(candidates.values()):
        for event in events:
            if candidates[event] and len(chosen) < max_windows:
                chosen.append(candidates[event].pop())
    plan = {
        "schema_version": "1.0",
        "seed": seed,
        "min_eligible": min_eligible,
        "max_windows": max_windows,
        "sampling": "balanced_round_robin_after_eventwise_shuffle",
        "selected": sorted(chosen, key=lambda item: (item["event"], item["window"])),
    }
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != plan:
            raise FileExistsError("evaluation window plan is already frozen with different inputs")
        return output
    _json(output, plan)
    return output


def run_sitreps(config: GenerateConfig, *, llm: LLMClient | None = None, scorer: EntailmentScorer | None = None) -> Path:
    """Generate every window once, then publish a hash-checked completion marker."""
    config.validate()
    source = _path(config.aggregate_dir)
    complete = json.loads((source / "complete.json").read_text(encoding="utf-8"))
    aggregate_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if aggregate_config["mode"] != config.mode or not complete["evidence_ready"]:
        raise ValueError("aggregation mode differs or selected evidence packets were not validated")
    if config.mode == "loeo" and not aggregate_config.get("classifier_run"):
        raise ValueError("research generation requires a LOEO classifier run")
    for name, digest in complete["artifacts"].items():
        if _hash(source / name) != digest:
            raise ValueError(f"completed aggregation artifact changed: {name}")
    windows = sorted(path for path in source.glob("*.json") if re.fullmatch(r"\d{8}T\d{6}Z\.json", path.name))
    if len(windows) != complete["windows"]:
        raise ValueError("aggregate window count differs from completion marker")
    aggregates = [json.loads(path.read_text(encoding="utf-8")) for path in windows]
    if not aggregates:
        raise ValueError("aggregate run contains no windows")
    model_id, event = aggregates[0]["model_id"], aggregates[0]["event"]
    if event not in EVENTS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_id):
        raise ValueError("aggregate has an unsafe event or model ID")
    if any(
        item["model_id"] != model_id or item["event"] != event or item.get("input_scope") != config.mode for item in aggregates
    ):
        raise ValueError("aggregate windows mix models, events or evaluation scopes")
    if config.mode == "loeo":
        plan = json.loads(_path(config.evaluation_plan).read_text(encoding="utf-8"))
        if plan.get("schema_version") != "1.0" or len(plan["selected"]) > plan["max_windows"]:
            raise ValueError("invalid frozen evaluation plan")
        if len({(item["event"], item["window"]) for item in plan["selected"]}) != len(plan["selected"]):
            raise ValueError("evaluation plan repeats a window")
        allowed = {
            item["window"]: item["aggregate_sha256"]
            for item in plan["selected"]
            if item["model_id"] == model_id and item["event"] == event
        }
        windows = [path for path in windows if path.name in allowed]
        if not windows or any(_hash(path) != allowed[path.name] for path in windows):
            raise ValueError("LOEO windows differ from the frozen evaluation plan")
        aggregates = [json.loads(path.read_text(encoding="utf-8")) for path in windows]
        if any(
            item["window"]["hours"] != 24 or sum(item["counts"]["by_category"].values()) < plan["min_eligible"]
            for item in aggregates
        ):
            raise ValueError("planned LOEO window does not meet the evaluation eligibility rule")
    packet_path = _path(config.packets)
    aggregate_inputs = json.loads((source / "inputs.json").read_text(encoding="utf-8"))
    if aggregate_inputs.get("packets", {}).get("sha256") != _hash(packet_path):
        raise ValueError("generation packets differ from the file validated by aggregation")
    packets = load_packets(packet_path, model_id)
    tweets = pd.read_parquet(source / "tweets.parquet")
    selected = [_selected(item, tweets, packets) for item in aggregates]
    output_dir = _path(config.output_root) / model_id / event
    signatures = {
        "aggregate_complete": _hash(source / "complete.json"),
        "tweets": _hash(source / "tweets.parquet"),
        "packets": _hash(packet_path),
        "windows": {path.name: _hash(path) for path in windows},
        "implementation": {path.name: _hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        | {"generate_cli.py": _hash(Path(__file__).parents[1] / "generate_cli.py")},
    }
    if config.mode == "loeo":
        signatures["evaluation_plan"] = _hash(_path(config.evaluation_plan))
    metadata = output_dir / "request.json"
    request = {"config": asdict(config), "inputs": signatures}
    if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")) != request:
        raise FileExistsError("generation config or inputs changed; choose a new output_root")
    if output_dir.exists() and any(output_dir.iterdir()) and not metadata.exists():
        raise FileExistsError("generation directory contains unowned files")
    done = output_dir / "complete.json"
    if done.exists():
        finished = json.loads(done.read_text(encoding="utf-8"))
        if all(
            (output_dir / name).exists() and _hash(output_dir / name) == digest for name, digest in finished["artifacts"].items()
        ):
            summary(_path(config.output_root), _path(config.report_path))
            return done
        raise ValueError("completed generation artifacts changed")
    _json(metadata, request)
    _json(output_dir / "env.json", environment(model_id=model_id, seed=config.llm.seed, run_id=f"generate/{event}"))
    atomic.text(output_dir / "config.yaml", yaml.safe_dump(asdict(config), sort_keys=False))
    if llm is None:
        if any(item["selected"] for item in aggregates):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.llm.tokenizer)

            def count_tokens(system: str, user: str, schema: dict) -> int:
                if config.llm.backend == "hf":
                    system += "\nReturn only JSON matching this schema: " + json.dumps(schema, sort_keys=True)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        else:
            count_tokens = None

        raw = (
            OllamaClient(config.llm.model, num_ctx=config.llm.num_ctx)
            if config.llm.backend == "ollama"
            else HFClient(config.llm.model)
        )
        llm = CachedLLM(raw, _path(config.cache_root), retries=config.llm.retries)
    else:
        count_tokens = None
    verifier = Verifier(
        scorer or DebertaNLI(config.nli.model),
        tau_entail=config.nli.tau_entail,
        tau_contra=config.nli.tau_contra,
        max_premise_tokens=config.nli.max_premise_tokens,
    )
    artifacts = ["request.json", "env.json", "config.yaml"]
    for path, aggregate, (packets_for_window, severity, origin_status) in zip(windows, aggregates, selected, strict=True):
        target = output_dir / path.name
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved["input"]["pids"] != {pid: packet["post_id"] for pid, packet in packets_for_window.items()}:
                raise ValueError("resumed sitrep selected packets differ")
        else:
            saved = generate_sitrep(
                aggregate,
                packets_for_window,
                llm,
                verifier,
                llm_name=f"{config.llm.backend}:{config.llm.model}",
                severity_by_pid=severity,
                origin_status_by_pid=origin_status,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                seed=config.llm.seed,
                token_counter=count_tokens,
            )
            _json(target, saved)
        artifacts.append(path.name)
    _json(
        done,
        {
            "model_id": model_id,
            "event": event,
            "mode": config.mode,
            "windows": len(windows),
            "artifacts": {name: _hash(output_dir / name) for name in artifacts},
        },
    )
    summary(_path(config.output_root), _path(config.report_path))
    return done
