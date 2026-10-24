"""Dataset, split and cache checks before spending a GPU training session."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sakshi import paths
from sakshi.data.features import EMBED_FILE, TOKEN_SHAPE, TOKENS_FILE, load_index

if TYPE_CHECKING:
    from sakshi.models.train import TrainConfig

EXPECTED_TRAIN_LABELS = {
    "informative": 6006,
    "humanitarian": 1788,
    "severity": 4077,
    "disaster_type": 11105,
    "informative_text": 8729,
    "humanitarian_text": 8729,
    "informative_image": 8729,
    "humanitarian_image": 8729,
}


def inspect_inputs(
    config: TrainConfig, *, processed: Path = paths.PROCESSED, strict: bool = True, scan_arrays: bool = True
) -> dict:
    """Assert cache alignment and split isolation; strict mode checks the fixed release."""
    from sakshi.eval.selectors import HEAD_LABELS, official
    from sakshi.models.data import ClassifierDataset, FeatureStore
    from sakshi.models.fusion import HEAD_CLASSES

    processed = Path(processed)
    rows = pd.read_parquet(processed / "manifest.parquet")
    index = load_index(processed / "features")
    if rows.sample_id.isna().any() or rows.sample_id.duplicated().any() or set(rows.sample_id) != set(index.sample_id):
        raise ValueError("manifest and feature index must cover exactly the same unique sample IDs")
    if index.ok.isna().any() or not index.ok.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("feature availability flags must be booleans")
    if rows.component.isna().any() or not rows.split.isin(("train", "dev", "test")).all():
        raise ValueError("manifest needs components and valid unified split roles")
    column = f"loeo_{config.fold.removeprefix('loeo_')}" if config.fold else "split"
    if column not in rows:
        raise ValueError(f"manifest lacks requested split column {column}")
    visible = rows.loc[rows[column].isin(("train", "dev", "test"))]
    for key in ("component", "tweet_id"):
        if (visible.groupby(key)[column].nunique() > 1).any():
            raise ValueError(f"{key} leaks across train/dev/test roles")
    official_counts = {head: int(official(rows, head).sum()) for head in HEAD_LABELS} if strict else {}
    tokens = np.load(processed / "features" / TOKENS_FILE, mmap_mode="r")
    embeddings = np.load(processed / "features" / EMBED_FILE, mmap_mode="r")
    if tokens.dtype != np.float16 or embeddings.dtype != np.float32:
        raise ValueError("cache dtypes must be float16 tokens and float32 embeddings")
    if scan_arrays:
        for start in range(0, len(index), 256):
            t, e = tokens[start : start + 256], embeddings[start : start + 256]
            if not np.isfinite(t).all() or not np.isfinite(e).all():
                raise ValueError("cached CLIP arrays contain non-finite values")
            ok = index.ok.iloc[start : start + 256].to_numpy(dtype=bool)
            if not np.allclose(np.linalg.norm(e[ok], axis=1), 1, atol=1e-3):
                raise ValueError("usable CLIP embeddings must have unit norm")
            if np.any(t[~ok]) or np.any(e[~ok]):
                raise ValueError("unavailable images must have zero cache rows")
    features = FeatureStore(processed / "features")
    heads = tuple(HEAD_CLASSES if config.model.auxiliary_heads else list(HEAD_CLASSES)[:4])
    fold = config.fold.removeprefix("loeo_") if config.fold else None
    splits = {
        role: ClassifierDataset(rows, features, split=role, fold=fold, enabled_heads=heads, use_cib=config.data.use_cib)
        for role in ("train", "dev")
    }
    if not all(splits.values()):
        raise ValueError("training and dev need supervised samples with usable modalities")
    counts = {head: int(values.sum()) for head, values in splits["train"].class_counts().items()}
    if (
        strict
        and config.fold is None
        and config.data.use_cib
        and config.model.auxiliary_heads
        and index.ok.all()
        and counts != EXPECTED_TRAIN_LABELS
    ):
        raise ValueError(f"training label counts differ from the fixed release: {counts}")
    if tuple(tokens.shape[1:]) != TOKEN_SHAPE:
        raise ValueError("classifier cache token shape differs from the model contract")
    return {
        "manifest_rows": len(rows),
        "usable_images": int(index.ok.sum()),
        "split_column": column,
        "training_rows": len(splits["train"]),
        "dev_rows": len(splits["dev"]),
        "training_label_counts": counts,
        "official_selector_counts": official_counts,
        "scanned_all_arrays": scan_arrays,
        "strict_release_checks": strict,
    }
