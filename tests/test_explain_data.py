"""Opt-in cache equivalence on five private images; default suite stays offline."""

import os

import pandas as pd
import pytest
import torch

from sakshi import paths
from sakshi.data.features import ClipFeaturizer
from sakshi.explain.gradcam import LiveClipFeaturizer
from sakshi.images import load_rgb
from sakshi.models.data import FeatureStore


@pytest.mark.data
@pytest.mark.skipif(
    os.environ.get("SAKSHI_RUN_DATA_TESTS") != "1", reason="set SAKSHI_RUN_DATA_TESTS=1 with private data/model caches"
)
def test_five_real_images_match_existing_frozen_clip_cache():
    torch.set_num_threads(2)
    rows = pd.read_parquet(paths.PROCESSED / "manifest.parquet")
    features = FeatureStore()
    live = LiveClipFeaturizer(ClipFeaturizer(device="cpu"))
    selected = rows.loc[rows.split == "dev"].iloc[:5]
    assert len(selected) == 5
    for _, row in selected.iterrows():
        image = load_rgb(paths.image_file(row))
        assert image is not None
        try:
            error = live.assert_cached(live.pixels(image), torch.from_numpy(features.tokens(str(row.sample_id)))[None])
            assert error <= 1e-2
        finally:
            image.close()
