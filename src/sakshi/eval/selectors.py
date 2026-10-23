"""One source of truth for official, extended and development evaluation rows."""

from __future__ import annotations

import pandas as pd

from sakshi.data import labels

# (dataset, official split column, gold column, expected official test count)
_OFFICIAL = {
    "informative": ("crisismmd", "split_informative", "y_informative", 1534),
    "humanitarian": ("crisismmd", "split_humanitarian", "y_humanitarian", 955),
    "severity": ("crisismmd", "split_damage", "y_severity", 529),
    "severity_dmd": ("cib", "split_cib_damage", "y_severity", 1138),
    "disaster_type": ("cib", "split_cib_disaster_types", "y_disaster_type", 3213),
}

HEAD_LABELS = {
    "informative": labels.INFORMATIVE,
    "humanitarian": labels.HUMANITARIAN,
    "severity": labels.SEVERITY,
    "severity_dmd": labels.SEVERITY,
    "disaster_type": labels.DISASTER_TYPE,
}


def _rule(head: str) -> tuple[str, str, str, int]:
    try:
        return _OFFICIAL[head]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation head {head!r}; choose from {list(_OFFICIAL)}") from exc


def official(manifest: pd.DataFrame, head: str) -> pd.Series:
    """Mask for a published-comparable test set; assert its fixed release size."""
    dataset, split_col, gold_col, expected = _rule(head)
    required = {"dataset", "split", split_col, gold_col}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing columns for {head}: {sorted(missing)}")
    mask = (manifest.dataset == dataset) & (manifest[split_col] == "test") & manifest[gold_col].notna()
    count = int(mask.sum())
    if count != expected:
        raise ValueError(f"{head} official test set has {count} rows; expected {expected}")
    if not (manifest.loc[mask, "split"] == "test").all():
        raise ValueError(f"{head} official test rows are not all in the unified test split")
    return mask.rename(head)


def extended(manifest: pd.DataFrame, head: str) -> pd.Series:
    """All held-out rows with a gold label for the requested head."""
    dataset, _, gold_col, _ = _rule(head)
    mask = (manifest.split == "test") & manifest[gold_col].notna()
    if head == "severity_dmd":
        mask &= manifest.dataset == dataset
    return mask.rename(head)


def pool(manifest: pd.DataFrame, head: str, split: str) -> pd.Series:
    """Training or model-selection rows, using only the unified split."""
    if split not in {"train", "dev"}:
        raise ValueError("pool split must be 'train' or 'dev'")
    dataset, _, gold_col, _ = _rule(head)
    mask = (manifest.split == split) & manifest[gold_col].notna()
    if head == "severity_dmd":
        mask &= manifest.dataset == dataset
    return mask.rename(head)
