"""Validate WP3 packets before they can accompany selected aggregation evidence."""

from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker

from sakshi.data import labels
from sakshi.text import clean_for_display

PRIVATE = re.compile(r"@\w|http|\.(?:jpe?g|png)\b", flags=re.IGNORECASE)
FORBIDDEN_KEYS = {"tweet_id", "username", "usernames", "url", "image_path"}
HEADS = {
    "informative": labels.INFORMATIVE,
    "humanitarian": labels.HUMANITARIAN,
    "severity": labels.SEVERITY,
    "disaster_type": labels.DISASTER_TYPE,
}


def _privacy(value: object, key: str = "") -> None:
    if key in FORBIDDEN_KEYS:
        raise ValueError(f"packet contains private field {key}")
    if isinstance(value, dict):
        for name, child in value.items():
            _privacy(child, name)
    elif isinstance(value, list):
        for child in value:
            _privacy(child, key)
    elif isinstance(value, str) and key != "post_id" and PRIVATE.search(value):
        raise ValueError("packet contains a handle, URL or image filename")


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def load_packets(path: Path, model_id: str) -> dict[str, dict]:
    """Load schema-validated, privacy-checked JSONL packets for exactly one model."""
    schema = json.loads(files("sakshi").joinpath("schemas/evidence_packet.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    packets = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                packet = json.loads(line, parse_constant=_invalid_constant)
                validator.validate(packet)
                _privacy(packet)
                if packet["model_id"] != model_id:
                    raise ValueError("packet model_id differs from aggregation predictions")
                if packet["post_id"] in packets:
                    raise ValueError("duplicate packet post_id")
                packets[packet["post_id"]] = packet
            except Exception as exc:
                raise ValueError(f"invalid packet at {path}:{line_number}: {exc}") from exc
    return packets


def validate_selected(
    sample_ids: set[str], packets: dict[str, dict], rows: pd.DataFrame, predictions: pd.DataFrame, provenance: pd.DataFrame
) -> None:
    """Reject stale or misaligned packet labels, confidences, context and provenance."""
    if missing := sample_ids - packets.keys():
        raise ValueError(f"selected evidence is missing {len(missing)} same-model packets")
    manifest = rows.set_index("sample_id")
    predicted = predictions.set_index("sample_id")
    statuses = provenance.set_index("post_id")
    for sample_id in sample_ids:
        packet = packets[sample_id]
        row = manifest.loc[sample_id]
        prediction = predicted.loc[sample_id]
        expected_event = None if pd.isna(row.event) else str(row.event)
        expected_time = None if pd.isna(row.posted_at) else pd.Timestamp(row.posted_at)
        actual_time = None if packet["posted_at"] is None else pd.Timestamp(packet["posted_at"])
        if packet["event"] != expected_event or actual_time != expected_time:
            raise ValueError(f"packet context differs for {sample_id}")
        if packet["abstain"] != bool(prediction.abstain) or packet["provenance"]["status"] != statuses.loc[sample_id, "status"]:
            raise ValueError(f"packet abstention or provenance differs for {sample_id}")
        expected_text = clean_for_display(row.text) if isinstance(row.text, str) else None
        if packet["has_text"] and packet.get("text_clean") != expected_text:
            raise ValueError(f"packet text differs from cleaned source for {sample_id}")
        if "text_clean" in packet and packet["text_clean"] != expected_text:
            raise ValueError(f"packet text differs from cleaned source for {sample_id}")
        for name in ("has_text", "has_image"):
            if name in prediction and packet[name] != bool(prediction[name]):
                raise ValueError(f"packet {name} differs for {sample_id}")
        for modality in ("text", "image"):
            if f"alpha_{modality}" in prediction:
                expected_weight = float(prediction[f"alpha_{modality}"])
                if (
                    not math.isfinite(expected_weight)
                    or not 0 <= expected_weight <= 1
                    or not math.isclose(packet["modality_weights"][modality], expected_weight, abs_tol=1e-6)
                ):
                    raise ValueError(f"packet modality weights differ for {sample_id}")
        for head, classes in HEADS.items():
            output = packet["predictions"][head]
            value = prediction[f"{head}_label"]
            expected_label = None if pd.isna(value) else str(value)
            expected_status = prediction.get(f"{head}_status", "predicted")
            if output["label"] != expected_label or output["status"] != expected_status:
                raise ValueError(f"packet {head} decision differs for {sample_id}")
            if expected_label is not None and expected_label not in classes:
                raise ValueError(f"unknown packet {head} label for {sample_id}")
            expected_p = None if expected_label is None else round(float(prediction[f"p_{head}__{expected_label}"]), 2)
            if (
                output["p"] != expected_p
                or expected_p is not None
                and (not math.isfinite(expected_p) or not 0 <= expected_p <= 1)
            ):
                raise ValueError(f"packet {head} probability differs for {sample_id}")
