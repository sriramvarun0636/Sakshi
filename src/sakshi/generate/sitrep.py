"""Normalize, verify, regenerate once and assemble a cited situation report."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256

from sakshi.explain.packets import validate_packet
from sakshi.generate.llm import GenerationFailure, LLMClient
from sakshi.generate.prompts import REWRITE_SCHEMA, SITREP_SCHEMA, SYSTEM, post_block, rewrite_user, sitrep_user
from sakshi.generate.verify import Verifier, accepted, split_sentences


def _normalize(items: list[dict]) -> list[tuple[str, list[str]]]:
    seen = set()
    normalized = []
    for item in items:
        citations = list(dict.fromkeys(item["citations"]))
        for piece in split_sentences(item["text"].strip()):
            if piece not in seen:
                seen.add(piece)
                normalized.append((piece, citations))
    return normalized


def generate_sitrep(
    aggregate: dict,
    selected: dict[str, dict],
    llm: LLMClient,
    verifier: Verifier,
    *,
    llm_name: str,
    severity_by_pid: dict[str, str | None] | None = None,
    origin_status_by_pid: dict[str, str | None] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
    token_counter: Callable[[str, str, dict], int] | None = None,
) -> dict:
    """Generate one window; `selected` maps WP4 P-IDs to representative WP3 packets."""
    begun = time.monotonic()
    posts = aggregate["selected"]
    if len(posts) > 25 or set(selected) != {item["pid"] for item in posts}:
        raise ValueError("selected packets do not match the aggregate's P-IDs")
    for item in posts:
        packet = selected[item["pid"]]
        validate_packet(packet)
        if packet["model_id"] != aggregate["model_id"] or packet["event"] != aggregate["event"]:
            raise ValueError("selected packet model or event differs from aggregate")
        if packet["provenance"]["status"] in {"recycled", "template_match"}:
            raise ValueError("excluded-provenance post reached generation input")
    window = aggregate["window"]
    severity_by_pid = severity_by_pid or {}
    origin_status_by_pid = origin_status_by_pid or {}
    blocks = [
        post_block(
            item["pid"],
            selected[item["pid"]],
            item["category"],
            severity_by_pid.get(item["pid"], selected[item["pid"]]["predictions"]["severity"]["label"]),
            image_origin_unverified=origin_status_by_pid.get(item["pid"]) in {"cross_event_concurrent", "match_undated"}
            if item["pid"] in origin_status_by_pid
            else None,
        )
        for item in posts
    ]
    user = sitrep_user(aggregate["event"], window["start"][:16].replace("T", " "), window["end"][:16].replace("T", " "), blocks)

    def count_tokens(system: str, prompt: str, schema: dict) -> int:
        if token_counter is not None:
            return token_counter(system, prompt, schema)
        # Offline fixtures do not need a tokenizer. Production runs supply the model tokenizer.
        return len(re.findall(r"\w+|[^\w\s]", system + prompt))

    initial_tokens = count_tokens(SYSTEM, user, SITREP_SCHEMA)
    if initial_tokens > (5_800 if token_counter is not None else 4_000):
        raise ValueError("window exceeds the 6,000-token prompt budget")
    prompt_hash = sha256((SYSTEM + "\n" + user).encode()).hexdigest()
    report = {
        "schema_version": "1.0",
        "event": aggregate["event"],
        "window": {"start": window["start"], "end": window["end"]},
        "models": {"classifier": aggregate["model_id"], "llm": llm_name, "nli": verifier.scorer.model_name},
        "input": {
            "pids": {item["pid"]: selected[item["pid"]]["post_id"] for item in posts},
            "sample_ids": {item["pid"]: item["sample_ids"] for item in posts},
            "prompt_sha256": prompt_hash,
        },
        "system_lines": list(aggregate["system_lines"]),
        "sentences": [],
        "verification_rate": {"initial": None, "final": None},
        "generation_failures": 0,
        "generation_errors": [],
        "ignored_rewrites": [],
        "prompt_tokens": {"round0": initial_tokens, "rewrite": None},
        "latency_s": None,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if not posts:
        report["latency_s"] = round(time.monotonic() - begun, 3)
        return report
    try:
        round0 = llm.generate_json(SYSTEM, user, SITREP_SCHEMA, max_tokens=max_tokens, temperature=temperature, seed=seed)
    except GenerationFailure as exc:
        report["generation_failures"] += 1
        report["generation_errors"].append(str(exc))
        report["system_lines"].append("Situation report generation failed; no LLM claims are displayed.")
        report["latency_s"] = round(time.monotonic() - begun, 3)
        return report
    initial = []
    for index, (sentence, citations) in enumerate(_normalize(round0["sentences"]), 1):
        gates = verifier.verify(
            sentence, citations, selected, severity_by_pid=severity_by_pid, origin_status_by_pid=origin_status_by_pid
        )
        initial.append(
            {
                "id": f"S{index}",
                "kind": "llm",
                "round": 0,
                "text": sentence,
                "citations": citations,
                "gates": gates,
                "accepted": accepted(gates),
                "replaces": None,
            }
        )
    rejected = [item for item in initial if not item["accepted"]]
    rewrites: dict[str, dict] = {}
    if rejected:
        rewrite_prompt = user + "\nRound-0 JSON:\n" + json.dumps(round0, ensure_ascii=False) + "\n" + rewrite_user(rejected)
        rewrite_tokens = count_tokens(SYSTEM, rewrite_prompt, REWRITE_SCHEMA)
        report["prompt_tokens"]["rewrite"] = rewrite_tokens
        if rewrite_tokens > (5_800 if token_counter is not None else 4_000):
            report["generation_failures"] += 1
            report["generation_errors"].append("rewrite prompt exceeds the 6,000-token budget")
            report["sentences"] = [item for item in initial if item["accepted"]] + rejected
            report["verification_rate"] = {
                "initial": round(sum(item["accepted"] for item in initial) / len(initial), 6),
                "final": round(sum(item["accepted"] for item in initial) / len(initial), 6),
            }
            report["latency_s"] = round(time.monotonic() - begun, 3)
            return report
        try:
            generated = llm.generate_json(
                SYSTEM, rewrite_prompt, REWRITE_SCHEMA, max_tokens=max_tokens, temperature=temperature, seed=seed
            )
            valid_ids = {item["id"] for item in rejected}
            for proposal in generated["rewrites"]:
                target = proposal["replaces"]
                if target not in valid_ids or target in rewrites:
                    report["ignored_rewrites"].append(target)
                    continue
                if proposal["text"] is None:
                    rewrites[target] = None
                    continue
                pieces = _normalize([{"text": proposal["text"], "citations": proposal["citations"]}])
                if len(pieces) != 1:
                    report["ignored_rewrites"].append(target)
                    continue
                sentence, citations = pieces[0]
                gates = verifier.verify(
                    sentence, citations, selected, severity_by_pid=severity_by_pid, origin_status_by_pid=origin_status_by_pid
                )
                rewrites[target] = {
                    "id": f"S{len(initial) + len(rewrites) + 1}",
                    "kind": "llm",
                    "round": 1,
                    "text": sentence,
                    "citations": citations,
                    "gates": gates,
                    "accepted": accepted(gates),
                    "replaces": target,
                }
        except GenerationFailure as exc:
            report["generation_failures"] += 1
            report["generation_errors"].append(str(exc))
    accepted_rows = []
    for item in initial:
        if item["accepted"]:
            accepted_rows.append(item)
        elif (replacement := rewrites.get(item["id"])) is not None and replacement["accepted"]:
            accepted_rows.append(replacement)
    rejected_rows = [item for item in initial if not item["accepted"]]
    rejected_rows.extend(item for item in rewrites.values() if item is not None and not item["accepted"])
    report["sentences"] = accepted_rows + rejected_rows
    count = len(initial)
    report["verification_rate"] = {
        "initial": round(sum(item["accepted"] for item in initial) / count, 6) if count else None,
        "final": round(len(accepted_rows) / count, 6) if count else None,
    }
    report["latency_s"] = round(time.monotonic() - begun, 3)
    return report
