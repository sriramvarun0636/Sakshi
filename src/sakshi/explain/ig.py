"""BERTweet embedding Integrated Gradients and whitespace-word evidence spans."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from sakshi.aggregate.packets import PRIVATE
from sakshi.explain.common import TEXT_HEADS, Example, expand_inputs, gradients
from sakshi.models.fusion import HEAD_CLASSES
from sakshi.types import TextSpan

log = logging.getLogger(__name__)


def merge_subwords(tokens: list[str], scores: list[float], *, special_tokens: set[str] | None = None) -> list[tuple[str, float]]:
    """Sum trailing-@@ BPE continuations, including a truncated final word."""
    if len(tokens) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("tokens and finite scores must have matching lengths")
    result, word, score = [], "", 0.0
    for token, value in zip(tokens, scores, strict=True):
        if token in (special_tokens or set()):
            if word:
                result.append((word, score))
                word, score = "", 0.0
            continue
        word += token[:-2] if token.endswith("@@") else token
        score += float(value)
        if not token.endswith("@@"):
            result.append((word, score))
            word, score = "", 0.0
    if word:
        result.append((word, score))
    return result


def word_groups(tokenizer, text: str, input_ids: list[int]) -> list[tuple[str, list[int]]]:
    """Align even unknown BPEs to whitespace words, retaining the observed truncated prefix."""
    groups = []
    special = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    positions = [index for index, token in enumerate(input_ids) if token not in special]
    cursor = 0
    for word in text.split():
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        selected = positions[cursor : cursor + len(ids)]
        if not selected:
            break
        if [input_ids[index] for index in selected] != ids[: len(selected)]:
            raise ValueError("BERTweet whitespace/BPE alignment differs from encoded input")
        visible = word
        if len(selected) < len(ids):
            visible = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids[: len(selected)]))
        groups.append((visible, selected))
        cursor += len(selected)
    if cursor != len(positions):
        raise ValueError("unmapped text tokens")
    return groups


def extract_spans(words: list[tuple[str, float]], head: str) -> list[TextSpan]:
    """Select up to three positive contiguous runs, with at most eight words each."""
    if not words or not np.isfinite([score for _, score in words]).all():
        return []
    total = sum(max(0.0, score) for _, score in words)
    maximum = max(score for _, score in words)
    if maximum <= 0:
        return []
    runs, start = [], None
    for index in range(len(words) + 1):
        active = index < len(words) and words[index][1] > 0 and words[index][1] >= 0.1 * maximum
        if active:
            word = words[index][0]
            active = word.strip(".,:;!?()").upper() not in {"@USER", "HTTPURL", "RT", "[USER]", "[LINK]"} and not PRIVATE.search(
                word
            )
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index
            if end - start > 8:
                start = max(range(start, end - 7), key=lambda i: sum(score for _, score in words[i : i + 8]))
                end = start + 8
            runs.append((start, end, sum(score for _, score in words[start:end]) / total))
            start = None
    selected = sorted(runs, key=lambda run: (-run[2], run[0]))[:3]
    return [
        TextSpan(f"T{i}", head, " ".join(word for word, _ in words[start:end]), start, end, score)
        for i, (start, end, score) in enumerate(selected, 1)
    ]


@dataclass
class TextAttribution:
    words: list[tuple[str, float]]
    groups: list[list[int]]
    delta: float
    output_difference: float
    steps: int
    flags: list[str]


@contextmanager
def fixed_positions(text_encoder, input_ids):
    """Hold RoBERTa position IDs fixed when its word baseline contains pad tokens.

    Otherwise pad IDs also change the positional embedding, which is outside the
    attributed word-embedding layer and invalidates its completeness comparison.
    """
    position_embeddings = getattr(getattr(text_encoder, "embeddings", None), "position_embeddings", None)
    padding_idx = getattr(position_embeddings, "padding_idx", None)
    if padding_idx is None:
        yield
        return
    mask = input_ids.ne(padding_idx).long()
    positions = mask.cumsum(1) * mask + padding_idx

    def before_forward(module, args, kwargs):
        kwargs["position_ids"] = positions.expand(len(kwargs["input_ids"]), -1)
        return args, kwargs

    handle = text_encoder.register_forward_pre_hook(before_forward, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def integrated_gradients(model, tokenizer, example: Example, head: str) -> TextAttribution | None:
    """Explain a predicted fused-head logit with the real image features held fixed."""
    from captum.attr import LayerIntegratedGradients

    if head not in TEXT_HEADS or example.model_text is None or getattr(example.prediction, head).status != "predicted":
        return None
    target = HEAD_CLASSES[head].index(getattr(example.prediction, head).label)
    inputs = example.inputs
    ids = inputs["input_ids"]
    baseline = ids.clone()
    protected = torch.zeros_like(ids, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        protected |= ids == token_id
    baseline[~protected] = tokenizer.pad_token_id

    def forward(input_ids):
        data = expand_inputs(inputs, len(input_ids))
        data["input_ids"] = input_ids
        return model(**data)[head][:, target]

    with gradients(model), fixed_positions(model.text_encoder, ids):
        layer = model.text_encoder.get_input_embeddings()
        lig = LayerIntegratedGradients(forward, layer)
        difference = float((forward(ids) - forward(baseline)).detach().cpu().item())
        for steps in (50, 150):
            attribution, delta = lig.attribute(
                ids, baselines=baseline, n_steps=steps, internal_batch_size=16, return_convergence_delta=True
            )
            delta_value = float(delta.detach().cpu().item())
            if abs(delta_value) <= 0.05 * abs(difference):
                break
    scores = attribution.sum(-1)[0].detach().float().cpu().numpy()
    if not np.isfinite(scores).all() or not np.isfinite(delta_value):
        raise ValueError("IG produced nonfinite attributions")
    groups = word_groups(tokenizer, example.model_text, ids[0].tolist())
    words = [(word, float(scores[indices].sum())) for word, indices in groups]
    flags = []
    if example.text_clean is not None and len(example.model_text.split()) == len(example.text_clean.split()):
        display = example.text_clean.split()
        words = [(display[i] if word == example.model_text.split()[i] else word, score) for i, (word, score) in enumerate(words)]
    else:
        flags.append("word_alignment_fallback")
        log.warning("%s: display/model word counts differ; using normalised words", example.post_id)
    if abs(delta_value) > 0.05 * abs(difference):
        flags.append("ig_unconverged")
    if not words or max(score for _, score in words) <= 0:
        flags.append("no_positive_attribution")
    return TextAttribution(words, [indices for _, indices in groups], delta_value, difference, steps, flags)
