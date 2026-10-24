"""Optional per-post explanation: deterministic prefix plus verified rationale."""

from __future__ import annotations

import json
from hashlib import sha256

from sakshi.explain.packets import validate_packet
from sakshi.generate.llm import GenerationFailure, LLMClient
from sakshi.generate.numbers import unsupported
from sakshi.generate.verify import EntailmentScorer, split_sentences

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rationale"],
    "properties": {"rationale": {"type": "string", "minLength": 1, "maxLength": 500}},
}
SYSTEM = (
    "You explain automatic classifications of disaster-related social-media posts. "
    "Use only the information given. Do not add facts."
)


def explain_post(packet: dict, llm: LLMClient, scorer: EntailmentScorer) -> dict:
    validate_packet(packet)
    packet_sha = sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    predictions = packet["predictions"]
    labels = [
        (name, item["label"], item["p"])
        for name, item in predictions.items()
        if item["status"] == "predicted" and item["label"] is not None
    ]
    decisions = ", ".join(f"{label.replace('_', ' ')} ({probability:.0%})" for _, label, probability in labels)
    image_weight = packet["modality_weights"]["image"]
    prefix = f"Classified as {decisions}; {image_weight:.0%} image-driven."
    caption = packet.get("image_caption")
    description = caption["text"] if caption else "(none)"
    spans = [item["value"] for item in packet["evidence"] if item["type"] == "text_span"]
    user = (
        f"Post: {packet.get('text_clean') or '(no text)'}\nImage description: {description}\n"
        f"Key phrases: {'; '.join(spans) or '(none)'}\nPredicted labels: {decisions}\n"
        "In one or two sentences, explain which parts of the post support the predicted labels."
    )
    result = {
        "post_id": packet["post_id"],
        "model_id": packet["model_id"],
        "packet_sha256": packet_sha,
        "prefix": prefix,
        "rationale": None,
        "rejected_rationale": [],
        "generation_failures": 0,
    }
    try:
        generated = llm.generate_json(SYSTEM, user, SCHEMA, max_tokens=256)
    except GenerationFailure:
        result["generation_failures"] = 1
        return result
    premise = f"Post: {packet.get('text_clean') or '(no text)'} Image: {description}. Key phrases: {'; '.join(spans)}"
    pieces = split_sentences(generated["rationale"].strip())
    if len(pieces) > 2:
        result["rejected_rationale"] = [{"text": part, "reason": "more than two sentences"} for part in pieces]
        return result
    for sentence in pieces:
        p_entail, p_contra = scorer.score(premise, sentence)
        missing = unsupported(sentence, premise)
        if p_entail < 0.5 or p_contra >= 0.3 or missing:
            result["rejected_rationale"].append(
                {"text": sentence, "p_entail": p_entail, "p_contra": p_contra, "unsupported_numbers": missing}
            )
    if not result["rejected_rationale"]:
        result["rationale"] = " ".join(pieces)
    return result
