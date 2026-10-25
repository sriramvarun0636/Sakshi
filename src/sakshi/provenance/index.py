"""Build the time-aware provenance index over cached CLIP embeddings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, time, timezone
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from PIL import Image

from sakshi import atomic, paths
from sakshi.data.events import EVENTS
from sakshi.data.features import EMBED_FILE, MODEL_NAME, MODEL_REVISION, load_index
from sakshi.data.phash import DEGENERATE_STD
from sakshi.images import normalise_rgb
from sakshi.provenance.detect import IndexImage, QueryImage, decide
from sakshi.types import ProvenanceResult

GRAPHIC_PROMPTS = (
    "a weather forecast map",
    "a map",
    "a chart or infographic",
    "a logo",
    "a screenshot of a web page or app",
    "an image of text",
    "a placeholder image saying the image is not available",
    "a meme with text",
    "a poster or flyer",
)
PHOTO_PROMPTS = (
    "a photograph of a real scene",
    "a photo of a street",
    "a photo of people",
    "a photo of buildings",
    "a photo of a landscape",
    "a photo of flood water",
    "a photo of damage",
)


class GraphicScorer(Protocol):
    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return one graphic probability per unit-normalised image embedding."""


class ClipGraphicScorer:
    """CLIP zero-shot graphic probability using the fixed Appendix E.8 prompts."""

    def __init__(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoTokenizer, CLIPModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        if len(tokenizer) < 1000:
            raise RuntimeError("CLIP tokenizer vocabulary is incomplete; download the pinned tokenizer files")
        model = CLIPModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(device).eval()
        encoded = tokenizer(
            [*GRAPHIC_PROMPTS, *PHOTO_PROMPTS], padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        if torch.unique(encoded["input_ids"], dim=0).shape[0] < 2:
            raise RuntimeError("CLIP graphic and photo prompts collapsed to identical token IDs")
        with torch.inference_mode():
            vectors = model.get_text_features(**encoded)
            if not isinstance(vectors, torch.Tensor):
                vectors = vectors.pooler_output
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
        self.prompts = vectors.cpu().numpy()

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        images = np.asarray(embeddings, dtype=np.float32)
        if images.ndim != 2 or images.shape[1] != 512:
            raise ValueError("CLIP image embeddings must be N×512")
        logits = 100 * images @ self.prompts.T
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, : len(GRAPHIC_PROMPTS)].sum(axis=1)


def event_onsets(rows: pd.DataFrame) -> dict[str, datetime]:
    """Collection starts for CrisisMMD and first dated post for each AIDR folder."""
    onsets = {name: datetime.combine(event.collection_start, time.min, tzinfo=timezone.utc) for name, event in EVENTS.items()}
    aidr = rows.loc[(rows.source == "aidr_dt") & rows.folder.notna() & rows.posted_at.notna()]
    for folder, group in aidr.groupby("folder"):
        first_day = group.posted_at.min().date()
        onsets[f"aidr:{folder}"] = datetime.combine(first_day, time.min, tzinfo=timezone.utc)
    return onsets


def build_index(
    rows: pd.DataFrame,
    *,
    feature_dir: Path = paths.PROCESSED / "features",
    output: Path = paths.PROCESSED / "provenance" / "index.parquet",
    scorer: GraphicScorer | None = None,
    tau_graphic: float = 0.5,
) -> Path:
    """Publish one row per usable image, referencing rather than copying CLIP vectors."""
    if not 0 <= tau_graphic <= 1:
        raise ValueError("tau_graphic must lie in [0, 1]")
    required = {
        "sample_id",
        "dataset",
        "source",
        "tweet_id",
        "component",
        "event",
        "folder",
        "posted_at",
        "phash",
        "degenerate",
        "width",
        "height",
    }
    if missing := required - set(rows):
        raise ValueError(f"manifest missing provenance fields: {sorted(missing)}")
    if rows.sample_id.duplicated().any():
        raise ValueError("manifest sample IDs must be unique")
    features = load_index(feature_dir)
    aligned = rows.merge(
        features.rename(columns={"row": "embed_row", "ok": "feature_ok"}), on="sample_id", how="left", validate="one_to_one"
    )
    if aligned.feature_ok.isna().any():
        raise ValueError("feature index does not cover every manifest sample ID")
    usable = aligned.loc[aligned.feature_ok].copy()
    embedding_map = np.load(feature_dir / EMBED_FILE, mmap_mode="r")
    embeddings = np.asarray(embedding_map[usable.embed_row.to_numpy(dtype=int)], dtype=np.float32)
    if embeddings.shape != (len(usable), 512):
        raise ValueError("cached image embeddings have the wrong shape")
    if scorer is None:
        scorer = ClipGraphicScorer()
    probabilities = np.asarray(scorer.score(embeddings), dtype=np.float32)
    if (
        probabilities.shape != (len(usable),)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("graphic scorer must return one finite probability per image")
    usable["index_event"] = usable.event.astype("string")
    aidr = usable.source == "aidr_dt"
    missing_aidr = aidr & usable.index_event.isna()
    usable.loc[missing_aidr, "index_event"] = "aidr:" + usable.loc[missing_aidr, "folder"].astype(str)
    usable["index_event"] = usable.index_event.fillna("dmd")
    usable["first_seen"] = usable.posted_at
    usable["min_side"] = np.minimum(usable.width, usable.height)
    usable["p_graphic"] = probabilities
    usable["is_graphic"] = probabilities >= tau_graphic
    columns = [
        "sample_id",
        "dataset",
        "source",
        "tweet_id",
        "component",
        "index_event",
        "first_seen",
        "phash",
        "degenerate",
        "min_side",
        "p_graphic",
        "is_graphic",
        "embed_row",
    ]
    atomic.parquet(output, usable[columns])
    onset_json = json.dumps({key: value.isoformat() for key, value in event_onsets(rows).items()}, indent=2) + "\n"
    atomic.text(output.with_name("onsets.json"), onset_json)
    return output


def _optional(value: object) -> object | None:
    return None if pd.isna(value) else value


class ProvenanceIndex:
    """Vectorised retrieval followed by the explicit, testable temporal decision tree."""

    def __init__(
        self,
        frame: pd.DataFrame,
        embeddings: np.ndarray,
        manifest: pd.DataFrame,
        onsets: dict[str, datetime],
        *,
        tau_clip: float = 0.92,
        tau_graphic: float = 0.5,
        phash_max_distance: int = 8,
        phash_min_side: int = 64,
        graphic_scorer: GraphicScorer | None = None,
        live_featurizer: object | None = None,
    ) -> None:
        if frame.sample_id.duplicated().any() or manifest.sample_id.duplicated().any():
            raise ValueError("provenance input sample IDs must be unique")
        if embeddings.shape != (len(frame), 512):
            raise ValueError("provenance index embeddings must have shape N×512")
        self.frame = frame.reset_index(drop=True)
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.manifest = manifest.set_index("sample_id", drop=False)
        self.row_lookup = {str(sample_id): i for i, sample_id in enumerate(self.frame.sample_id)}
        self.onsets = onsets
        self.tau_clip = tau_clip
        if not 0 <= tau_graphic <= 1:
            raise ValueError("tau_graphic must lie in [0, 1]")
        self.tau_graphic = tau_graphic
        self.phash_max_distance = phash_max_distance
        self.phash_min_side = phash_min_side
        self.graphic_scorer = graphic_scorer
        self.live_featurizer = live_featurizer
        self.hashes = self.frame.phash.fillna(0).to_numpy(dtype=np.uint64)
        self.hash_valid = self.frame.phash.notna().to_numpy(dtype=bool)
        self.hash_allowed = (~self.frame.degenerate.to_numpy(dtype=bool)) & (
            self.frame.min_side.to_numpy(dtype=int) >= phash_min_side
        )

    @classmethod
    def load(
        cls,
        index_path: Path = paths.PROCESSED / "provenance" / "index.parquet",
        feature_dir: Path = paths.PROCESSED / "features",
        manifest_path: Path = paths.PROCESSED / "manifest.parquet",
        **kwargs,
    ) -> ProvenanceIndex:
        index_path = Path(index_path)
        frame = pd.read_parquet(index_path)
        feature_index = load_index(feature_dir)
        lookup = feature_index.set_index("sample_id")
        if not frame.sample_id.isin(lookup.index).all():
            raise ValueError("provenance index refers to absent feature sample IDs")
        expected = lookup.loc[frame.sample_id, "row"].to_numpy(dtype=int)
        if not np.array_equal(expected, frame.embed_row.to_numpy(dtype=int)):
            raise ValueError("provenance embedding rows differ from the feature cache")
        embeddings = np.load(feature_dir / EMBED_FILE, mmap_mode="r")[expected]
        onsets = {
            name: datetime.fromisoformat(value)
            for name, value in json.loads(index_path.with_name("onsets.json").read_text(encoding="utf-8")).items()
        }
        return cls(frame, embeddings, pd.read_parquet(manifest_path), onsets, **kwargs)

    def _query_from_sample(self, sample_id: str, time: datetime | None, event: str | None) -> QueryImage:
        if sample_id not in self.manifest.index:
            raise KeyError(f"unknown sample ID {sample_id!r}")
        row = self.manifest.loc[sample_id]
        query_time = time if time is not None else _optional(row.posted_at)
        query_event = event if event is not None else _optional(row.event)
        position = self.row_lookup.get(sample_id)
        has_image = _optional(row.image_path) is not None
        if query_event is None and str(row.source) == "aidr_dt" and _optional(row.folder) is not None:
            query_event = f"aidr:{row.folder}"
        if position is None:
            return QueryImage(
                sample_id, _optional(row.tweet_id), query_event, query_time, has_image, has_image, False, None, None, None, None
            )
        indexed = self.frame.iloc[position]
        return QueryImage(
            post_id=sample_id,
            tweet_id=_optional(row.tweet_id),
            event=query_event,
            time=query_time,
            has_image=has_image,
            unreadable=False,
            degenerate=bool(indexed.degenerate),
            min_side=int(indexed.min_side),
            phash=None if pd.isna(indexed.phash) else int(indexed.phash),
            embedding=self.embeddings[position],
            is_graphic=bool(indexed.is_graphic),
        )

    def _query_from_image(self, image: Image.Image, time: datetime | None, event: str | None, tweet_id: str | None) -> QueryImage:
        return self.prepare_image_batch([image], times=[time], events=[event], tweet_ids=[tweet_id])[0]

    def prepare_image_batch(
        self,
        images: Sequence[Image.Image],
        *,
        times: Sequence[datetime | None],
        events: Sequence[str | None],
        tweet_ids: Sequence[str | None] | None = None,
    ) -> list[QueryImage]:
        """Prepare live images together so benchmark transformations share one CLIP pass."""
        import imagehash

        if not images or len(images) != len(times) or len(images) != len(events):
            raise ValueError("image batch, times and events must have the same nonzero length")
        if tweet_ids is None:
            tweet_ids = [None] * len(images)
        if len(tweet_ids) != len(images):
            raise ValueError("image batch and tweet IDs must have the same length")
        rgb_images = [normalise_rgb(image) for image in images]
        try:
            if self.live_featurizer is None:
                from sakshi.data.features import ClipFeaturizer

                self.live_featurizer = ClipFeaturizer()
            if self.graphic_scorer is None:
                self.graphic_scorer = ClipGraphicScorer()
            _, embeddings = self.live_featurizer.encode(rgb_images)
            probabilities = np.asarray(self.graphic_scorer.score(embeddings), dtype=np.float32)
            if embeddings.shape != (len(images), 512) or probabilities.shape != (len(images),):
                raise ValueError("live image features have an unexpected batch shape")
            queries = []
            for rgb, embedding, probability, query_time, event, tweet_id in zip(
                rgb_images, embeddings, probabilities, times, events, tweet_ids, strict=True
            ):
                gray = np.asarray(rgb.convert("L").resize((64, 64)), dtype=np.float32)
                phash = int.from_bytes(np.packbits(imagehash.phash(rgb).hash.ravel()).tobytes(), "big")
                queries.append(
                    QueryImage(
                        post_id="input:" + sha256(rgb.tobytes()).hexdigest()[:16],
                        tweet_id=tweet_id,
                        event=event,
                        time=query_time,
                        has_image=True,
                        unreadable=False,
                        degenerate=float(gray.std()) < DEGENERATE_STD,
                        min_side=min(rgb.size),
                        phash=phash,
                        embedding=embedding,
                        is_graphic=float(probability) >= self.tau_graphic,
                    )
                )
            return queries
        finally:
            for rgb in rgb_images:
                rgb.close()

    def prepare_query(
        self,
        sample_id: str | None = None,
        *,
        image: Image.Image | None = None,
        time: datetime | None = None,
        event: str | None = None,
        tweet_id: str | None = None,
    ) -> QueryImage:
        """Prepare a manifest or live image once for threshold sweeps."""
        if sample_id is not None and image is not None:
            raise ValueError("pass either sample_id or image")
        if sample_id is not None:
            return self._query_from_sample(sample_id, time, event)
        if image is not None:
            return self._query_from_image(image, time, event, tweet_id)
        return QueryImage("input:none", tweet_id, event, time, False, False, False, None, None, None, None)

    def candidates(self, query: QueryImage, *, min_tau_clip: float | None = None) -> list[IndexImage]:
        """Retrieve all visible pHash or CLIP candidates above a minimum sweep threshold."""
        if not query.has_image or query.unreadable or query.degenerate:
            return []
        if query.embedding is None:
            raise ValueError("usable query is missing an embedding")
        threshold = self.tau_clip if min_tau_clip is None else min_tau_clip
        if not -1 <= threshold <= 1:
            raise ValueError("invalid CLIP threshold")
        visible_mask = (self.frame.sample_id != query.post_id).to_numpy(dtype=bool, copy=True)
        if query.tweet_id is not None:
            visible_mask &= (self.frame.tweet_id != query.tweet_id).fillna(True).to_numpy(dtype=bool)
        if query.time is not None:
            visible_mask &= (self.frame.first_seen.isna() | (self.frame.first_seen < query.time)).to_numpy(dtype=bool)
        similarities = self.embeddings @ np.asarray(query.embedding, dtype=np.float32)
        hash_matches = np.zeros(len(self.frame), dtype=bool)
        if query.phash is not None and query.min_side is not None and query.min_side >= self.phash_min_side:
            distances = np.bitwise_count(self.hashes ^ np.uint64(query.phash))
            hash_matches = self.hash_valid & self.hash_allowed & (distances <= self.phash_max_distance)
        positions = np.flatnonzero(visible_mask & (hash_matches | (similarities >= threshold)))
        items = []
        for position in positions:
            row = self.frame.iloc[position]
            items.append(
                IndexImage(
                    post_id=str(row.sample_id),
                    tweet_id=_optional(row.tweet_id),
                    index_event=str(row.index_event),
                    first_seen=_optional(row.first_seen),
                    degenerate=bool(row.degenerate),
                    min_side=int(row.min_side),
                    phash=None if pd.isna(row.phash) else int(row.phash),
                    embedding=self.embeddings[position],
                    is_graphic=bool(row.is_graphic),
                )
            )
        return items

    def check_query(
        self,
        query: QueryImage,
        *,
        candidates: list[IndexImage] | None = None,
        tau_clip: float | None = None,
    ) -> ProvenanceResult:
        """Classify a prepared query; reusable candidates make threshold sweeps cheap."""
        threshold = self.tau_clip if tau_clip is None else tau_clip
        if candidates is None:
            candidates = self.candidates(query, min_tau_clip=threshold)
        return decide(
            query,
            candidates,
            onsets=self.onsets,
            phash_max_distance=self.phash_max_distance,
            phash_min_side=self.phash_min_side,
            tau_clip=threshold,
            index_size=len(self.frame),
        )

    def check(
        self,
        sample_id: str | None = None,
        *,
        image: Image.Image | None = None,
        time: datetime | None = None,
        event: str | None = None,
        tweet_id: str | None = None,
    ) -> ProvenanceResult:
        """Check a manifest image or live PIL image with the same retrieval rules."""
        query = self.prepare_query(sample_id, image=image, time=time, event=event, tweet_id=tweet_id)
        return self.check_query(query)
