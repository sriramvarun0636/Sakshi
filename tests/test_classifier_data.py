import numpy as np
import pandas as pd
import pytest
import torch

from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE
from sakshi.models.data import ClassifierCollator, ClassifierDataset, FeatureStore, tokenization_stats
from sakshi.models.losses import IGNORE


class FakeTokenizer:
    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        assert padding and truncation and return_tensors == "pt"
        ids = [[1] + [2] * min(len(text.split()), max_length - 1) for text in texts]
        width = max(map(len, ids))
        padded = [row + [0] * (width - len(row)) for row in ids]
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor([[int(token != 0) for token in row] for row in padded]),
        }


def fixture_rows(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    token_map = np.lib.format.open_memmap(root / TOKENS_FILE, mode="w+", dtype=np.float16, shape=(4, 50, 768))
    token_map[:] = 1
    token_map.flush()
    del token_map
    embed_map = np.lib.format.open_memmap(root / EMBED_FILE, mode="w+", dtype=np.float32, shape=(4, 512))
    embed_map[:] = 0
    embed_map.flush()
    del embed_map
    index = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "row": range(4), "ok": [True, True, False, True]})
    index.to_parquet(root / INDEX_FILE)
    rows = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "dataset": ["crisismmd", "cib", "crisismmd", "cib"],
            "split": ["train"] * 4,
            "text": ["Bridge damaged", pd.NA, "Roads blocked", pd.NA],
            "y_informative": ["informative", pd.NA, "not_informative", pd.NA],
            "y_humanitarian": ["infrastructure_and_utility_damage", pd.NA, "not_humanitarian", pd.NA],
            "y_severity": ["mild_damage", pd.NA, pd.NA, pd.NA],
            "y_disaster_type": [pd.NA, "fire", pd.NA, pd.NA],
            "informative_text": ["informative", pd.NA, "not_informative", pd.NA],
            "humanitarian_text": ["infrastructure_and_utility_damage", pd.NA, "not_humanitarian", pd.NA],
            "informative_image": ["informative", pd.NA, pd.NA, pd.NA],
            "humanitarian_image": ["infrastructure_and_utility_damage", pd.NA, pd.NA, pd.NA],
        }
    )
    return rows, FeatureStore(root)


def test_dataset_aligns_cache_masks_labels_and_balances_sources(tmp_path):
    rows, features = fixture_rows(tmp_path)
    dataset = ClassifierDataset(rows, features)
    assert len(dataset) == 3
    assert [item["sample_id"] for item in dataset] == ["a", "b", "c"]
    assert dataset[0]["image_tokens"].shape == (50, 768)
    assert dataset[2]["image_tokens"].sum() == 0
    assert dataset[2]["targets"]["humanitarian"] == IGNORE
    assert dataset[2]["targets"]["severity"] == IGNORE
    assert dataset.class_counts()["informative"].tolist() == [1, 1]
    assert dataset.class_counts()["disaster_type"].sum() == 1
    assert list(dataset.sampler(samples_per_epoch=10).weights) == pytest.approx([0.3, 0.4, 0.3])
    with pytest.raises(ValueError, match="duplicate"):
        ClassifierDataset(pd.concat([rows, rows.iloc[:1]]), features)


def test_collation_never_drops_only_modality_and_masks_affected_heads(tmp_path):
    rows, features = fixture_rows(tmp_path)
    items = list(ClassifierDataset(rows, features))
    collate = ClassifierCollator(FakeTokenizer(), modality_dropout=0.5)
    torch.manual_seed(1)
    batch = collate(items)
    inputs, targets = batch["inputs"], batch["targets"]
    assert inputs["input_ids"].shape[0] == 3
    assert inputs["image_tokens"].shape == (3, 50, 768)
    assert inputs["has_image"][1] and not inputs["has_text"][1]
    assert inputs["has_text"][2] and not inputs["has_image"][2]
    assert int(inputs["has_text"][0]) + int(inputs["has_image"][0]) == 1
    assert targets["informative"][0] != IGNORE and targets["humanitarian"][0] != IGNORE
    if not inputs["has_text"][0]:
        assert targets["informative_text"][0] == IGNORE
    if not inputs["has_image"][0]:
        assert targets["severity"][0] == IGNORE


def test_unimodal_baselines_use_their_own_labels_and_masks(tmp_path):
    rows, features = fixture_rows(tmp_path)
    rows.loc[0, "y_informative"] = "not_informative"
    rows.loc[0, "y_humanitarian"] = "not_humanitarian"
    rows.loc[0, "informative_image"] = "not_informative"
    rows.loc[0, "humanitarian_image"] = "not_humanitarian"
    text_data = ClassifierDataset(rows, features, modality="text", enabled_heads=("informative", "humanitarian"))
    image_data = ClassifierDataset(rows, features, modality="image", enabled_heads=("informative", "humanitarian"))
    assert [item["sample_id"] for item in text_data] == ["a", "c"]
    assert text_data[0]["targets"]["informative"] == 1
    assert text_data[0]["targets"]["humanitarian"] != IGNORE
    assert image_data[0]["targets"]["informative"] == 0
    assert image_data[0]["targets"]["humanitarian"] == IGNORE
    text_batch = ClassifierCollator(FakeTokenizer(), modality="text")(list(text_data))
    image_batch = ClassifierCollator(FakeTokenizer(), modality="image")(list(image_data))
    assert text_batch["inputs"]["has_text"].all() and not text_batch["inputs"]["has_image"].any()
    assert image_batch["inputs"]["has_image"].all() and not image_batch["inputs"]["has_text"].any()
    with pytest.raises(ValueError, match="zero modality dropout"):
        ClassifierCollator(FakeTokenizer(), modality="text", modality_dropout=0.1)


def test_tokenization_counts_include_special_tokens_and_exclude_unusable_text():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert not kwargs["truncation"] and not kwargs["padding"]
            return {"input_ids": [[0] * (len(text.split()) + 2) for text in texts]}

    rows = pd.DataFrame(
        {
            "split": ["train", "train", "train", "dev", "test"],
            "text": ["one two", "one two three", "https://example.org", None, "one"],
        }
    )
    result = tokenization_stats(rows, Tokenizer(), max_tokens=4)
    assert result["splits"]["train"] == {"text_rows": 2, "truncated_rows": 1, "longest_tokens": 5}
    assert result["splits"]["dev"] == {"text_rows": 0, "truncated_rows": 0, "longest_tokens": 0}
    assert result["splits"]["test"]["truncated_rows"] == 0
