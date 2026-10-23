import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sakshi import paths
from sakshi.data import features


class FakeFeaturizer:
    def encode(self, images):
        tokens = np.empty((len(images), *features.TOKEN_SHAPE), dtype=np.float32)
        embeddings = np.zeros((len(images), features.EMBED_DIM), dtype=np.float32)
        for i, image in enumerate(images):
            tokens[i] = image.getpixel((0, 0))[0]
            embeddings[i, 0] = 1
        return tokens, embeddings


def test_pool_tokens_preserves_cls_and_averages_patches():
    raw = np.zeros((1, 197, 768), dtype=np.float32)
    raw[:, 0] = 99
    raw[:, 1] = 1
    raw[:, 2] = 2
    raw[:, 15] = 3
    raw[:, 16] = 4
    pooled = features.pool_tokens(raw)
    assert pooled.shape == (1, 50, 768)
    assert pooled[0, 0, 0] == 99
    assert pooled[0, 1, 0] == 2.5
    with pytest.raises(ValueError, match="expected"):
        features.pool_tokens(np.zeros((1, 10, 768)))


def test_feature_cache_indexes_by_sample_id_and_records_unreadable(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    Image.new("RGB", (4, 4), (42, 0, 0)).save(root / "good.png")
    monkeypatch.setattr(paths, "CRISISMMD_ROOT", root)
    rows = pd.DataFrame({"sample_id": ["a", "b"], "dataset": ["crisismmd"] * 2, "image_path": ["good.png", "missing.png"]})
    out = tmp_path / "features"
    index_path = features.build_cache(rows, FakeFeaturizer(), out, batch_size=1, check_space=False)
    index = features.load_index(out)
    assert index_path == out / features.INDEX_FILE
    assert index.sample_id.tolist() == ["a", "b"]
    assert index.ok.tolist() == [True, False]
    assert np.load(out / features.TOKENS_FILE, mmap_mode="r")[0, 0, 0] == 42
    assert np.load(out / features.EMBED_FILE, mmap_mode="r")[1].sum() == 0
    assert features.build_cache(rows, FakeFeaturizer(), out, check_space=False) == index_path
    with pytest.raises(ValueError, match="differs"):
        features.build_cache(rows.iloc[:1], FakeFeaturizer(), out, check_space=False)


def test_clip_featurizer_matches_current_transformers_vision_api():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    text_config = transformers.CLIPTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision_config = transformers.CLIPVisionConfig(
        hidden_size=768,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        image_size=224,
        patch_size=16,
    )
    model_config = transformers.CLIPConfig(text_config=text_config, vision_config=vision_config, projection_dim=512)
    featurizer = object.__new__(features.ClipFeaturizer)
    featurizer.torch = torch
    featurizer.device = "cpu"
    featurizer.processor = transformers.CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224})
    featurizer.model = transformers.CLIPModel(model_config).eval()
    tokens, embeddings = featurizer.encode([Image.new("RGB", (224, 224), (80, 90, 100))])
    assert tokens.shape == (1, *features.TOKEN_SHAPE)
    assert embeddings.shape == (1, features.EMBED_DIM)
    assert np.linalg.norm(embeddings[0]) == pytest.approx(1, abs=1e-4)
