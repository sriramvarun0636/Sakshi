"""Manifest-backed classifier samples, cached image tokens and balanced sampling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler

from sakshi import paths
from sakshi.data.features import TOKEN_SHAPE, TOKENS_FILE, load_index
from sakshi.data.labels import HUMANITARIAN
from sakshi.models.fusion import BERTWEET_REVISION, HEAD_CLASSES, HUMANITARIAN_CONDITIONAL
from sakshi.models.losses import IGNORE, mask_targets_for_dropout
from sakshi.text import normalise_for_model

LABEL_COLUMNS = {
    "informative": "y_informative",
    "humanitarian": "y_humanitarian",
    "severity": "y_severity",
    "disaster_type": "y_disaster_type",
    "informative_text": "informative_text",
    "humanitarian_text": "humanitarian_text",
    "informative_image": "informative_image",
    "humanitarian_image": "humanitarian_image",
}


class FeatureStore:
    """Look up cached CLIP tokens by sample ID; keep the large array memory-mapped."""

    def __init__(self, root: Path = paths.PROCESSED / "features") -> None:
        index = load_index(root)
        self.root = Path(root)
        self.lookup = {str(row.sample_id): (int(row.row), bool(row.ok)) for row in index.itertuples()}
        self._tokens: np.memmap | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None  # re-open in a spawned data-loader worker
        return state

    def has_image(self, sample_id: str) -> bool:
        return self.lookup.get(sample_id, (-1, False))[1]

    def tokens(self, sample_id: str) -> np.ndarray:
        row, ok = self.lookup.get(sample_id, (-1, False))
        if not ok:
            return np.zeros(TOKEN_SHAPE, dtype=np.float32)
        if self._tokens is None:
            self._tokens = np.load(self.root / TOKENS_FILE, mmap_mode="r")
        return np.asarray(self._tokens[row], dtype=np.float32)


def encode_targets(row: pd.Series, *, modality: str = "both") -> dict[str, int]:
    """Use -100 for missing labels and for the negative conditional humanitarian class."""
    result = {}
    for head, column in LABEL_COLUMNS.items():
        value = row[column]
        if pd.isna(value) or head == "humanitarian" and value == "not_humanitarian":
            result[head] = IGNORE
        else:
            try:
                result[head] = HEAD_CLASSES[head].index(str(value))
            except ValueError as exc:
                raise ValueError(f"{row.sample_id}: unknown {head} label {value!r}") from exc
    if modality in ("text", "image"):
        result["informative"] = result[f"informative_{modality}"]
        value = row[f"humanitarian_{modality}"]
        result["humanitarian"] = (
            HUMANITARIAN_CONDITIONAL.index(str(value))
            if pd.notna(value) and value != "not_humanitarian"
            else IGNORE
        )
    elif modality != "both":
        raise ValueError("modality must be both, text, or image")
    return result


class ClassifierDataset(Dataset):
    """One manifest row per image; keep rows with an input and enabled supervision."""

    def __init__(
        self,
        rows: pd.DataFrame,
        features: FeatureStore,
        *,
        split: str = "train",
        fold: str | None = None,
        enabled_heads: Sequence[str] = tuple(HEAD_CLASSES),
        use_cib: bool = True,
        modality: str = "both",
    ) -> None:
        if split not in ("train", "dev", "test"):
            raise ValueError("split must be train, dev or test")
        if len(set(enabled_heads)) != len(enabled_heads) or set(enabled_heads) - set(HEAD_CLASSES):
            raise ValueError("enabled_heads contains duplicates or unknown heads")
        if modality not in ("both", "text", "image"):
            raise ValueError("modality must be both, text, or image")
        split_column = f"loeo_{fold}" if fold else "split"
        required = {"sample_id", "dataset", "text", split_column, *LABEL_COLUMNS.values()}
        if missing := required - set(rows):
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        if rows.sample_id.duplicated().any():
            raise ValueError("manifest has duplicate sample IDs")
        selected = rows.loc[rows[split_column] == split]
        if not use_cib:
            selected = selected.loc[selected.dataset != "cib"]
        self.features = features
        self.enabled_heads = tuple(enabled_heads)
        self.samples: list[dict[str, Any]] = []
        for _, row in selected.iterrows():
            sample_id = str(row.sample_id)
            text = normalise_for_model(row.text) if isinstance(row.text, str) else None
            has_image = features.has_image(sample_id)
            if (modality == "text" and text is None) or (modality == "image" and not has_image):
                continue
            if text is None and not has_image:
                continue
            targets = encode_targets(row, modality=modality)
            for head in ("severity", "disaster_type", "informative_image", "humanitarian_image"):
                if not has_image:
                    targets[head] = IGNORE
            for head in ("informative_text", "humanitarian_text"):
                if text is None:
                    targets[head] = IGNORE
            if not any(targets[head] != IGNORE for head in self.enabled_heads):
                continue
            hum_value = row.y_humanitarian if modality == "both" else row[f"humanitarian_{modality}"]
            hum_gold = IGNORE if pd.isna(hum_value) else HUMANITARIAN.index(str(hum_value))
            self.samples.append(
                {"sample_id": sample_id, "dataset": row.dataset, "text": text, "targets": targets, "humanitarian_gold": hum_gold}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        return {
            **item,
            "image_tokens": self.features.tokens(item["sample_id"]),
            "has_image": self.features.has_image(item["sample_id"]),
        }

    def class_counts(self) -> dict[str, Tensor]:
        counts = {head: torch.zeros(len(HEAD_CLASSES[head]), dtype=torch.long) for head in self.enabled_heads}
        for item in self.samples:
            for head in self.enabled_heads:
                target = item["targets"][head]
                if target != IGNORE:
                    counts[head][target] += 1
        return counts

    def sampler(self, *, samples_per_epoch: int = 16000, crisismmd_share: float = 0.6, generator=None) -> WeightedRandomSampler:
        if samples_per_epoch < 1 or not 0 <= crisismmd_share <= 1:
            raise ValueError("invalid sampling settings")
        sources = [item["dataset"] for item in self.samples]
        n_cm = sources.count("crisismmd")
        n_cib = sources.count("cib")
        if not n_cm and not n_cib:
            raise ValueError("dataset is empty")
        if n_cm and n_cib:
            weights = [crisismmd_share / n_cm if source == "crisismmd" else (1 - crisismmd_share) / n_cib for source in sources]
        else:
            weights = [1 / len(sources)] * len(sources)
        return WeightedRandomSampler(weights, samples_per_epoch, replacement=True, generator=generator)


class ClassifierCollator:
    def __init__(
        self, tokenizer: Any, *, max_text_tokens: int = 128, modality_dropout: float = 0.0, modality: str = "both"
    ) -> None:
        if max_text_tokens < 1 or not 0 <= modality_dropout <= 0.5:
            raise ValueError("invalid collation settings")
        if modality not in ("both", "text", "image") or (modality != "both" and modality_dropout):
            raise ValueError("unimodal runs require zero modality dropout")
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.modality_dropout = modality_dropout
        self.modality = modality

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty batch")
        has_text = torch.tensor([item["text"] is not None for item in items], dtype=torch.bool)
        has_image = torch.tensor([item["has_image"] for item in items], dtype=torch.bool)
        if self.modality == "text":
            has_image.zero_()
        elif self.modality == "image":
            has_text.zero_()
        if torch.any(~(has_text | has_image)):
            raise ValueError("a row has neither modality")
        text = [item["text"] or "" for item in items]
        encoded = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_text_tokens, return_tensors="pt")
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "image_tokens": torch.from_numpy(np.stack([item["image_tokens"] for item in items])).float(),
        }
        if self.modality_dropout:
            coin = torch.rand(len(items))
            both = has_text & has_image
            dropped_image = both & (coin < self.modality_dropout)
            dropped_text = both & (coin >= self.modality_dropout) & (coin < 2 * self.modality_dropout)
            has_text = has_text & ~dropped_text
            has_image = has_image & ~dropped_image
        else:
            dropped_image = torch.zeros_like(has_image)
            dropped_text = torch.zeros_like(has_text)
        inputs["has_text"] = has_text
        inputs["has_image"] = has_image
        targets = {head: torch.tensor([item["targets"][head] for item in items], dtype=torch.long) for head in HEAD_CLASSES}
        targets = mask_targets_for_dropout(targets, dropped_text=dropped_text, dropped_image=dropped_image)
        return {
            "sample_ids": [item["sample_id"] for item in items],
            "inputs": inputs,
            "targets": targets,
            "humanitarian_gold": torch.tensor([item["humanitarian_gold"] for item in items], dtype=torch.long),
        }


def load_tokenizer(model_name: str = "vinai/bertweet-base", revision: str = BERTWEET_REVISION) -> Any:
    """Disable BERTweet's own tweet normaliser; texts already use sakshi.text."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, normalization=False, use_fast=False)
    if len(tokenizer) < 1000:
        raise RuntimeError(f"tokenizer vocabulary for {model_name} is incomplete")
    return tokenizer


def tokenization_stats(rows: pd.DataFrame, tokenizer: Any, *, max_tokens: int = 128, fold: str | None = None) -> dict:
    """Count normalized text rows exceeding the token limit, independently per split."""
    column = f"loeo_{fold.removeprefix('loeo_')}" if fold else "split"
    result = {"max_tokens": max_tokens, "split_column": column, "splits": {}}
    for role in ("train", "dev", "test"):
        texts = [normalise_for_model(value) for value in rows.loc[rows[column] == role, "text"] if isinstance(value, str)]
        texts = [value for value in texts if value is not None]
        lengths = []
        for start in range(0, len(texts), 256):
            encoded = tokenizer(
                texts[start : start + 256],
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        result["splits"][role] = {
            "text_rows": len(texts),
            "truncated_rows": sum(length > max_tokens for length in lengths),
            "longest_tokens": max(lengths, default=0),
        }
    return result
