"""Frozen CLIP ViT-B/16 features, stored in sample-ID-indexed NumPy arrays."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from PIL import Image

from sakshi import paths
from sakshi.images import load_rgb

log = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-base-patch16"
# Fixed Hugging Face model commit, so a later upstream revision cannot silently change the cache.
MODEL_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
TOKENS_FILE = "clip_vitb16_tokens.f16.npy"
EMBED_FILE = "clip_vitb16_embed.f32.npy"
INDEX_FILE = "features_index.parquet"
TOKEN_SHAPE = (50, 768)
EMBED_DIM = 512
MIN_FREE_BYTES = 5 * 1024**3


class Featurizer(Protocol):
    def encode(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Return post-normalised 50×768 tokens and unit 512-d embeddings."""


def pool_tokens(tokens: np.ndarray) -> np.ndarray:
    """Keep CLS and average adjacent 2×2 CLIP patches into a 7×7 grid."""
    array = np.asarray(tokens)
    if array.ndim != 3 or array.shape[1:] != (197, 768):
        raise ValueError(f"expected B×197×768 CLIP tokens, got {array.shape}")
    patches = array[:, 1:].reshape(-1, 7, 2, 7, 2, 768).mean(axis=(2, 4))
    return np.concatenate((array[:, :1], patches.reshape(-1, 49, 768)), axis=1)


class ClipFeaturizer:
    """The one preprocessing path used by cache generation and live inference."""

    def __init__(self, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION, device: str = "auto") -> None:
        try:
            import torch
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as exc:
            raise ImportError("CLIP features require the 'ml' extra: pip install -e '.[ml]'") from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.processor = CLIPImageProcessor.from_pretrained(model_name, revision=revision)
        self.model = CLIPModel.from_pretrained(model_name, revision=revision).to(device).eval()
        self.model.requires_grad_(False)

    def encode(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            vision = self.model.vision_model(pixel_values=pixels)
            # `last_hidden_state` precedes the vision tower's post-layernorm.
            tower = self.model.vision_model
            post_layernorm = tower.post_layernorm if hasattr(tower, "post_layernorm") else tower.vision_model.post_layernorm
            tokens = post_layernorm(vision.last_hidden_state)
            # This is the same projection used by CLIPModel.get_image_features.
            embeddings = self.model.visual_projection(vision.pooler_output)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return pool_tokens(tokens.float().cpu().numpy()), embeddings.cpu().numpy()


def _paths(out_dir: Path) -> tuple[Path, Path, Path]:
    return out_dir / TOKENS_FILE, out_dir / EMBED_FILE, out_dir / INDEX_FILE


def build_cache(
    rows: pd.DataFrame,
    featurizer: Featurizer,
    out_dir: Path = paths.PROCESSED / "features",
    batch_size: int = 32,
    check_space: bool = True,
) -> Path:
    """Build the cache in manifest order; publish the index last as its commit marker."""
    required = {"sample_id", "dataset", "image_path"}
    if missing := required - set(rows):
        raise ValueError(f"rows missing feature-cache columns: {sorted(missing)}")
    if rows.sample_id.duplicated().any():
        raise ValueError("feature-cache sample_id values must be unique")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    token_path, embed_path, index_path = _paths(out_dir)
    if index_path.exists() and token_path.exists() and embed_path.exists():
        existing = pd.read_parquet(index_path)
        if (
            existing.sample_id.tolist() == rows.sample_id.tolist()
            and np.load(token_path, mmap_mode="r").shape == (len(rows), *TOKEN_SHAPE)
            and np.load(embed_path, mmap_mode="r").shape == (len(rows), EMBED_DIM)
        ):
            log.info("CLIP feature cache already complete: %s rows", len(rows))
            return index_path
        raise ValueError(f"feature cache at {out_dir} differs from input rows; move it aside before rebuilding")
    if check_space and shutil.disk_usage(out_dir).free < MIN_FREE_BYTES:
        raise OSError(f"at least 5 GiB free space is required to build the CLIP feature cache at {out_dir}")

    token_tmp = out_dir / f".{TOKENS_FILE}.tmp.npy"
    embed_tmp = out_dir / f".{EMBED_FILE}.tmp.npy"
    index_tmp = out_dir / f".{INDEX_FILE}.tmp"
    token_map = np.lib.format.open_memmap(token_tmp, mode="w+", dtype=np.float16, shape=(len(rows), *TOKEN_SHAPE))
    embed_map = np.lib.format.open_memmap(embed_tmp, mode="w+", dtype=np.float32, shape=(len(rows), EMBED_DIM))
    records = []
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows.iloc[start : start + batch_size]
            images, positions = [], []
            for offset, (_, row) in enumerate(batch.iterrows()):
                position = start + offset
                image = load_rgb(paths.image_file(row))
                records.append(
                    {
                        "sample_id": row.sample_id,
                        "row": position,
                        "ok": image is not None,
                        "error": None if image is not None else "unreadable image",
                    }
                )
                if image is not None:
                    images.append(image)
                    positions.append(position)
                else:
                    token_map[position] = 0
                    embed_map[position] = 0
            if images:
                try:
                    tokens, embeddings = featurizer.encode(images)
                    if tokens.shape != (len(images), *TOKEN_SHAPE) or embeddings.shape != (len(images), EMBED_DIM):
                        raise ValueError(
                            f"CLIP output shapes disagree with {len(images)} images: {tokens.shape}, {embeddings.shape}"
                        )
                    if not np.isfinite(tokens).all() or not np.isfinite(embeddings).all():
                        raise ValueError("CLIP output contains NaN or infinity")
                    norms = np.linalg.norm(embeddings, axis=1)
                    if not np.allclose(norms, 1.0, atol=1e-3):
                        raise ValueError("CLIP embeddings must have unit norm (±1e-3)")
                    token_map[positions] = tokens.astype(np.float16)
                    embed_map[positions] = embeddings.astype(np.float32)
                finally:
                    for image in images:
                        image.close()
        token_map.flush()
        embed_map.flush()
        del token_map, embed_map
        pd.DataFrame(records, columns=["sample_id", "row", "ok", "error"]).to_parquet(index_tmp, index=False)
        os.replace(token_tmp, token_path)
        os.replace(embed_tmp, embed_path)
        os.replace(index_tmp, index_path)
    finally:
        for path in (token_tmp, embed_tmp, index_tmp):
            path.unlink(missing_ok=True)
    log.info("CLIP feature cache ready: %s rows, %s usable", len(rows), sum(r["ok"] for r in records))
    return index_path


def load_index(out_dir: Path = paths.PROCESSED / "features") -> pd.DataFrame:
    """Load the sample-ID-to-row mapping, checking the two arrays are complete."""
    token_path, embed_path, index_path = _paths(out_dir)
    if not all(p.exists() for p in (token_path, embed_path, index_path)):
        raise FileNotFoundError(f"CLIP feature cache missing at {out_dir}; run 'sakshi-data prepare features'")
    index = pd.read_parquet(index_path)
    if index.sample_id.duplicated().any() or index.row.tolist() != list(range(len(index))):
        raise ValueError("feature index has duplicate sample IDs or non-sequential rows")
    if np.load(token_path, mmap_mode="r").shape != (len(index), *TOKEN_SHAPE):
        raise ValueError("CLIP token array shape differs from feature index")
    if np.load(embed_path, mmap_mode="r").shape != (len(index), EMBED_DIM):
        raise ValueError("CLIP embedding array shape differs from feature index")
    return index
