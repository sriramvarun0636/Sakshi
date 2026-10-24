"""Condense predictions and explanations into the existing WP4/WP5 packet contract."""

from __future__ import annotations

import json
from importlib.resources import files

from jsonschema import Draft202012Validator, FormatChecker

from sakshi.aggregate.packets import _privacy
from sakshi.explain.common import MAIN_HEADS, Example
from sakshi.types import Explanation, ProvenanceResult


def validate_packet(packet: dict) -> None:
    schema = json.loads(files("sakshi").joinpath("schemas/evidence_packet.json").read_text(encoding="utf-8"))
    # JSON Schema regards NaN as a number; strict JSON serialization rejects it first.
    json.dumps(packet, allow_nan=False)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(packet)
    _privacy(packet)


def make_packet(example: Example, *, event: str | None, posted_at, provenance: ProvenanceResult,
                explanations: list[Explanation], image_caption: dict | None = None) -> dict:
    prediction = example.prediction
    if prediction.post_id != example.post_id or provenance.post_id != example.post_id:
        raise ValueError("prediction/provenance post IDs differ")
    if len({item.head for item in explanations}) != len(explanations):
        raise ValueError("duplicate explanation heads")
    for explanation in explanations:
        decision = getattr(prediction, explanation.head)
        if (explanation.post_id != example.post_id or explanation.model_id != prediction.model_id
                or explanation.target_label != decision.label
                or decision.status != "predicted"):
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
            evidence.append({"id": f"T{index}", "type": "text_span", "head": text_head,
                             "value": span.text, "score": span.score})
    if image_head in lookup and example.pixels is not None:
        for index, region in enumerate(lookup[image_head].image_regions[:2], 1):
            if region.head != image_head:
                raise ValueError("region head differs from explanation")
            evidence.append({"id": f"I{index}", "type": "image_region", "head": image_head,
                             "bbox": list(region.bbox_xyxy), "score": region.score, "caption": region.caption})
    context = {"status": provenance.status.value,
               "origin_event": provenance.origin_event,
               "origin_first_seen": provenance.origin_first_seen.date().isoformat() if provenance.origin_first_seen else None,
               "similarity": round(provenance.similarity, 2) if provenance.similarity is not None else None}
    packet = {"schema_version": "1.0", "post_id": example.post_id, "event": event,
              "posted_at": posted_at.isoformat() if posted_at is not None else None,
              "has_text": example.model_text is not None, "has_image": example.pixels is not None,
              "image_caption": image_caption, "evidence": evidence, "provenance": context,
              "modality_weights": prediction.modality_weights, "abstain": prediction.abstain,
              "abstain_reason": prediction.abstain_reason, "model_id": prediction.model_id,
              "predictions": {head: {"label": (output := getattr(prediction, head)).label,
                                     "p": round(output.probs[output.label], 2) if output.label is not None else None,
                                     "status": output.status} for head in MAIN_HEADS}}
    if example.text_clean is not None:
        packet["text_clean"] = example.text_clean
    validate_packet(packet)
    return packet
