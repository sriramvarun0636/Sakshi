"""Condense predictions and explanations into the existing WP4/WP5 packet contract."""

from __future__ import annotations

import json
import math
from importlib.resources import files

from jsonschema import Draft202012Validator, FormatChecker

from sakshi.aggregate.packets import HEADS, _privacy
from sakshi.explain.common import MAIN_HEADS, Example
from sakshi.types import Explanation, ProvenanceResult


def validate_packet(packet: dict) -> None:
    schema = json.loads(files("sakshi").joinpath("schemas/evidence_packet.json").read_text(encoding="utf-8"))
    # JSON Schema regards NaN as a number; strict JSON serialization rejects it first.
    json.dumps(packet, allow_nan=False)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(packet)
    _privacy(packet)
    weights = packet["modality_weights"]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in weights.values()) or not math.isclose(
        weights["text"] + weights["image"], 1, abs_tol=1e-6
    ):
        raise ValueError("invalid modality weights")
    for name in ("text", "image"):
        if not packet[f"has_{name}"] and weights[name] != 0:
            raise ValueError("missing modality received nonzero weight")
    if not packet["has_text"] and not packet["has_image"]:
        raise ValueError("packet has no input modality")
    for head, classes in HEADS.items():
        decision = packet["predictions"][head]
        if decision["label"] is None:
            if decision["p"] is not None or decision["status"] == "predicted":
                raise ValueError("null head must be gated/unavailable with null confidence")
        elif decision["label"] not in classes or decision["p"] is None or not 0 <= decision["p"] <= 1:
            raise ValueError("invalid head label or confidence")
    inf = packet["predictions"]["informative"]
    hum = packet["predictions"]["humanitarian"]
    sev = packet["predictions"]["severity"]
    disaster = packet["predictions"]["disaster_type"]
    if inf["status"] != "predicted" or inf["label"] is None:
        raise ValueError("informativeness must be predicted")
    if inf["label"] == "informative" and (hum["label"] in (None, "not_humanitarian") or hum["status"] != "predicted"):
        raise ValueError("informative post needs a conditional humanitarian prediction")
    if inf["label"] == "not_informative" and hum["status"] != "gated":
        raise ValueError("noninformative humanitarian output must be gated")
    if packet["has_image"]:
        if disaster["status"] != "predicted" or disaster["label"] is None:
            raise ValueError("image disaster type is ungated")
        if inf["label"] == "informative" and (sev["label"] is None or sev["status"] != "predicted"):
            raise ValueError("informative image needs predicted severity")
    elif any(output["status"] != "unavailable" or output["label"] is not None for output in (sev, disaster)):
        raise ValueError("image heads must be unavailable without an image")
    if packet["predictions"]["informative"]["label"] == "not_informative":
        if packet["predictions"]["humanitarian"]["label"] != "not_humanitarian":
            raise ValueError("packet violates the humanitarian hierarchy")
        if packet["predictions"]["severity"]["status"] == "predicted":
            raise ValueError("packet violates severity gating")
    identifiers = [item["id"] for item in packet["evidence"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate evidence IDs")
    for item in packet["evidence"]:
        if not 0 <= item["score"] <= 1:
            raise ValueError("evidence score must be a share in [0, 1]")
        if item["head"] not in MAIN_HEADS or packet["predictions"][item["head"]]["status"] != "predicted":
            raise ValueError("evidence targets a gated/unavailable head")
        if item["type"] == "image_region":
            left, top, right, bottom = item["bbox"]
            if not packet["has_image"] or not (0 <= left < right and 0 <= top < bottom):
                raise ValueError("invalid image region")
        elif not packet["has_text"]:
            raise ValueError("text evidence needs usable text")


def make_packet(
    example: Example,
    *,
    event: str | None,
    posted_at,
    provenance: ProvenanceResult,
    explanations: list[Explanation],
    image_caption: dict | None = None,
) -> dict:
    prediction = example.prediction
    if prediction.post_id != example.post_id or provenance.post_id != example.post_id:
        raise ValueError("prediction/provenance post IDs differ")
    if len({item.head for item in explanations}) != len(explanations):
        raise ValueError("duplicate explanation heads")
    for explanation in explanations:
        decision = getattr(prediction, explanation.head)
        if (
            explanation.post_id != example.post_id
            or explanation.model_id != prediction.model_id
            or explanation.target_label != decision.label
            or decision.status != "predicted"
        ):
            raise ValueError("explanation differs from this post's predicted head")
    lookup = {item.head: item for item in explanations}
    informative = prediction.informative.label == "informative"
    text_head = "humanitarian" if informative else "informative"
    image_head = "severity" if informative and example.pixels is not None else "humanitarian"
    evidence = []
    if text_head in lookup and example.model_text is not None:
        for index, span in enumerate(lookup[text_head].text_spans[:3], 1):
            if span.head != text_head:
                raise ValueError("span head differs from explanation")
            evidence.append({"id": f"T{index}", "type": "text_span", "head": text_head, "value": span.text, "score": span.score})
    if image_head in lookup and example.pixels is not None:
        for index, region in enumerate(lookup[image_head].image_regions[:2], 1):
            if region.head != image_head:
                raise ValueError("region head differs from explanation")
            evidence.append(
                {
                    "id": f"I{index}",
                    "type": "image_region",
                    "head": image_head,
                    "bbox": list(region.bbox_xyxy),
                    "score": region.score,
                    "caption": region.caption,
                }
            )
    context = {
        "status": provenance.status.value,
        "origin_event": provenance.origin_event,
        "origin_first_seen": provenance.origin_first_seen.date().isoformat() if provenance.origin_first_seen else None,
        "similarity": round(provenance.similarity, 2) if provenance.similarity is not None else None,
    }
    packet = {
        "schema_version": "1.0",
        "post_id": example.post_id,
        "event": event,
        "posted_at": posted_at.isoformat() if posted_at is not None else None,
        "has_text": example.model_text is not None,
        "has_image": example.pixels is not None,
        "image_caption": image_caption,
        "evidence": evidence,
        "provenance": context,
        "modality_weights": prediction.modality_weights,
        "abstain": prediction.abstain,
        "abstain_reason": prediction.abstain_reason,
        "model_id": prediction.model_id,
        "predictions": {
            head: {
                "label": (output := getattr(prediction, head)).label,
                "p": round(output.probs[output.label], 2) if output.label is not None else None,
                "status": output.status,
            }
            for head in MAIN_HEADS
        },
    }
    if example.text_clean is not None:
        packet["text_clean"] = example.text_clean
    validate_packet(packet)
    return packet
