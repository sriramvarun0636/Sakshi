"""Deterministic, time-visible two-tier provenance decisions (spec §8.3–8.6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from sakshi.types import ProvenanceResult, ProvStatus


@dataclass(frozen=True)
class QueryImage:
    post_id: str
    tweet_id: str | None
    event: str | None
    time: datetime | None
    has_image: bool
    unreadable: bool
    degenerate: bool
    min_side: int | None
    phash: int | None
    embedding: np.ndarray | None
    is_graphic: bool | None


@dataclass(frozen=True)
class IndexImage:
    post_id: str
    tweet_id: str | None
    index_event: str
    first_seen: datetime | None
    degenerate: bool
    min_side: int
    phash: int | None
    embedding: np.ndarray
    is_graphic: bool


@dataclass(frozen=True)
class Match:
    item: IndexImage
    tier: Literal["phash", "clip"]
    similarity: float
    phash_distance: int | None


def visible(query: QueryImage, item: IndexImage) -> bool:
    """Exclude the query, sibling images of its tweet and future dated items."""
    if query.post_id == item.post_id:
        return False
    if query.tweet_id is not None and item.tweet_id == query.tweet_id:
        return False
    return item.first_seen is None or query.time is None or item.first_seen < query.time


def match(
    query: QueryImage,
    item: IndexImage,
    *,
    phash_max_distance: int = 8,
    phash_min_side: int = 64,
    tau_clip: float = 0.92,
) -> Match | None:
    """Return the strongest applicable tier while retaining the CLIP similarity."""
    if query.embedding is None:
        raise ValueError("a usable query needs a CLIP embedding")
    if not 0 <= phash_max_distance <= 64 or not -1 <= tau_clip <= 1:
        raise ValueError("invalid matching thresholds")
    q_embed = np.asarray(query.embedding, dtype=np.float32)
    i_embed = np.asarray(item.embedding, dtype=np.float32)
    if q_embed.shape != (512,) or i_embed.shape != (512,) or not np.isfinite(q_embed).all() or not np.isfinite(i_embed).all():
        raise ValueError("CLIP embeddings must be finite 512-vectors")
    similarity = float(np.dot(q_embed, i_embed))
    phash_distance = None
    if (
        query.phash is not None
        and item.phash is not None
        and not query.degenerate
        and not item.degenerate
        and query.min_side is not None
        and min(query.min_side, item.min_side) >= phash_min_side
    ):
        phash_distance = (int(query.phash) ^ int(item.phash)).bit_count()
        if phash_distance <= phash_max_distance:
            return Match(item, "phash", similarity, phash_distance)
    if similarity >= tau_clip:
        return Match(item, "clip", similarity, phash_distance)
    return None


def decide(
    query: QueryImage,
    items: list[IndexImage],
    *,
    onsets: dict[str, datetime],
    phash_max_distance: int = 8,
    phash_min_side: int = 64,
    tau_clip: float = 0.92,
    index_size: int | None = None,
) -> ProvenanceResult:
    """Select the earliest dated origin, then classify its temporal relationship."""
    base = {
        "post_id": query.post_id,
        "origin_post_id": None,
        "origin_event": None,
        "origin_first_seen": None,
        "top_match_post_id": None,
        "query_time": query.time,
        "query_is_graphic": query.is_graphic,
        "index_size": len(items) if index_size is None else index_size,
    }
    if not query.has_image:
        return ProvenanceResult(status=ProvStatus.NO_IMAGE, tier=None, similarity=None, phash_distance=None, **base)
    if query.unreadable or query.degenerate or query.embedding is None:
        return ProvenanceResult(status=ProvStatus.UNSUPPORTED_IMAGE, tier=None, similarity=None, phash_distance=None, **base)
    matches = []
    for item in items:
        if visible(query, item):
            candidate = match(
                query, item, phash_max_distance=phash_max_distance, phash_min_side=phash_min_side, tau_clip=tau_clip
            )
            if candidate is not None:
                matches.append(candidate)
    if not matches:
        return ProvenanceResult(status=ProvStatus.NOVEL, tier=None, similarity=None, phash_distance=None, **base)
    top = max(matches, key=lambda candidate: (candidate.similarity, candidate.item.post_id))
    dated = [candidate for candidate in matches if candidate.item.first_seen is not None]
    origin = min(dated, key=lambda candidate: (candidate.item.first_seen, candidate.item.post_id)) if dated else top
    if query.is_graphic and origin.item.is_graphic:
        status = ProvStatus.TEMPLATE_MATCH
    elif query.event is None:
        status = ProvStatus.SEEN_BEFORE
    elif origin.item.first_seen is None:
        status = ProvStatus.MATCH_UNDATED
    elif origin.item.index_event == query.event:
        status = ProvStatus.DUPLICATE
    else:
        if query.event not in onsets:
            raise ValueError(f"unknown onset for query event {query.event!r}")
        status = ProvStatus.RECYCLED if origin.item.first_seen < onsets[query.event] else ProvStatus.CROSS_EVENT_CONCURRENT
    return ProvenanceResult(
        status=status,
        tier=origin.tier,
        similarity=origin.similarity,
        phash_distance=origin.phash_distance,
        origin_post_id=origin.item.post_id,
        origin_event=origin.item.index_event,
        origin_first_seen=origin.item.first_seen,
        top_match_post_id=top.item.post_id,
        post_id=query.post_id,
        query_time=query.time,
        query_is_graphic=query.is_graphic,
        index_size=len(items) if index_size is None else index_size,
    )
