"""Category-local claim clusters over MiniLM text embeddings (§10.5)."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from sakshi.aggregate.core import CATEGORY_URGENCY, Tweet

log = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class TextEmbedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return unit-normalised N×D text vectors."""


class MiniLMEmbedder:
    """Load the fixed MiniLM model only when actual clustering is requested."""

    def __init__(self, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("claim clustering requires the 'aggregate' extra") from exc
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        result = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(result, dtype=np.float32)


def claim_clusters(
    tweets: list[Tweet],
    embedder: TextEmbedder,
    *,
    max_tweets: int = 1000,
    distance_threshold: float = 0.35,
) -> tuple[dict[str, int], set[str]]:
    """Cluster at most the 1,000 highest-confidence eligible tweets; no-text singletons."""
    if max_tweets < 1 or not 0 < distance_threshold < 1:
        raise ValueError("invalid clustering cap or cosine-distance threshold")
    eligible = [tweet for tweet in tweets if tweet.eligible]
    eligible.sort(key=lambda tweet: (-tweet.confidence, tweet.tweet_id))
    excluded = {tweet.tweet_id for tweet in eligible[max_tweets:]}
    if excluded:
        log.warning("clustering capped at %d eligible tweets; %d omitted", max_tweets, len(excluded))
    selected = eligible[:max_tweets]
    result: dict[str, int] = {}
    next_cluster = 0
    for category in CATEGORY_URGENCY:
        part = [tweet for tweet in selected if tweet.humanitarian_label == category]
        with_text = [tweet for tweet in part if tweet.text_clean]
        without_text = [tweet for tweet in part if not tweet.text_clean]
        if with_text:
            vectors = np.asarray(embedder.encode([tweet.text_clean for tweet in with_text]), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(with_text) or not np.isfinite(vectors).all():
                raise ValueError("MiniLM embedder returned invalid text vectors")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
                raise ValueError("MiniLM text vectors must have unit norm")
            if len(with_text) == 1:
                labels = np.array([0])
            else:
                labels = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric="cosine",
                    linkage="average",
                ).fit_predict(vectors)
            mapping = {int(label): next_cluster + offset for offset, label in enumerate(sorted(set(labels)))}
            for tweet, label in zip(with_text, labels, strict=True):
                result[tweet.tweet_id] = mapping[int(label)]
            next_cluster += len(mapping)
        for tweet in without_text:
            result[tweet.tweet_id] = next_cluster
            next_cluster += 1
    if len(result) != len(selected):
        raise ValueError("an eligible tweet has an unsupported humanitarian category")
    return result, excluded
