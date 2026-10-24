"""Shared types at the boundaries between SAKSHI's processing layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal


class ProvStatus(str, Enum):
    NOVEL = "novel"
    DUPLICATE = "duplicate"
    RECYCLED = "recycled"
    CROSS_EVENT_CONCURRENT = "cross_event_concurrent"
    TEMPLATE_MATCH = "template_match"
    MATCH_UNDATED = "match_undated"
    SEEN_BEFORE = "seen_before"
    NO_IMAGE = "no_image"
    UNSUPPORTED_IMAGE = "unsupported_image"


@dataclass(frozen=True)
class Post:
    post_id: str
    source: str
    text: str | None
    image_path: Path | None
    posted_at: datetime | None
    event: str | None
    tweet_id: str | None = None


@dataclass(frozen=True)
class HeadOutput:
    label: str | None
    probs: dict[str, float]
    status: Literal["predicted", "gated", "unavailable"]


@dataclass(frozen=True)
class Prediction:
    post_id: str
    informative: HeadOutput
    humanitarian: HeadOutput
    severity: HeadOutput
    disaster_type: HeadOutput
    modality_weights: dict[str, float]
    abstain: bool
    abstain_reason: str | None
    model_id: str


@dataclass(frozen=True)
class ProvenanceResult:
    post_id: str
    status: ProvStatus
    tier: Literal["phash", "clip"] | None
    similarity: float | None
    phash_distance: int | None
    origin_post_id: str | None
    origin_event: str | None
    origin_first_seen: datetime | None
    top_match_post_id: str | None
    query_time: datetime | None
    query_is_graphic: bool | None
    index_size: int


@dataclass(frozen=True)
class TextSpan:
    id: str
    head: str
    text: str
    start_word: int
    end_word: int
    score: float


@dataclass(frozen=True)
class ImageRegion:
    id: str
    head: str
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    caption: str | None = None


@dataclass(frozen=True)
class Explanation:
    post_id: str
    head: str
    target_label: str
    word_attributions: list[tuple[str, float]]
    text_spans: list[TextSpan]
    image_heatmap_path: Path | None
    image_regions: list[ImageRegion]
    ig_convergence_delta: float | None
    flags: list[str] = field(default_factory=list)
    model_id: str | None = None


@dataclass(frozen=True)
class GateResult:
    passed: bool
    detail: str
    score: float | None = None


@dataclass
class SitrepSentence:
    id: str
    text: str
    citations: list[str]
    kind: Literal["llm", "system"]
    round: int
    gates: dict[str, GateResult] = field(default_factory=dict)
    accepted: bool = False
    replaces: str | None = None
