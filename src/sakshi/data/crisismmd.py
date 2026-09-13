"""Build one row per CrisisMMD image from the fifteen official split files.

The official splits were drawn independently per task (and the agreed-label files are
subsets of the "all" files), so one image can be train for one task and test for
another. This module records every official assignment; `splits.py` resolves them.
"""

from __future__ import annotations

import re
from pathlib import Path

import ftfy
import pandas as pd

from sakshi import paths
from sakshi.data import labels
from sakshi.data.events import EVENTS, snowflake_time

TASKS = ("informative", "humanitarian", "damage")
SPLITS = ("train", "dev", "test")

_FILE_RE = re.compile(r"task_(informative|humanitarian|damage)_text_img_(agreed_lab_)?(train|dev|test)\.tsv$")
_PATH_DATE_RE = re.compile(r"data_image/[^/]+/(\d{1,2})_(\d{1,2})_(\d{4})/")


def read_split_files(dirs: list[Path] | None = None) -> pd.DataFrame:
    """All official split rows, long format: one row per (file, image)."""
    dirs = dirs or [paths.CRISISMMD_SPLITS_ALL, paths.CRISISMMD_SPLITS_AGREED]
    frames = []
    for d in dirs:
        for f in sorted(d.glob("task_*.tsv")):
            m = _FILE_RE.search(f.name)
            if not m:
                continue
            # quoting=3 (QUOTE_NONE): tweet text contains stray quote characters.
            df = pd.read_csv(f, sep="\t", dtype=str, quoting=3, keep_default_na=False, na_values=[""])
            df["task"] = m.group(1)
            df["variant"] = "agreed" if m.group(2) else "all"
            df["split"] = m.group(3)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no CrisisMMD split files under {dirs}")
    return pd.concat(frames, ignore_index=True)


def _collection_date(image_path: str) -> pd.Timestamp | None:
    """Date folder in the image path, e.g. data_image/hurricane_irma/6_9_2017/... (day_month_year)."""
    m = _PATH_DATE_RE.search(image_path)
    if not m:
        return None
    day, month, year = map(int, m.groups())
    return pd.Timestamp(year=year, month=month, day=day)


def _label_table(rows: pd.DataFrame, task: str, variant: str, columns: dict[str, str]) -> pd.DataFrame:
    sub = rows[(rows.task == task) & (rows.variant == variant)]
    if sub.image_id.duplicated().any():
        raise ValueError(f"{task}/{variant}: duplicate image_id within one variant")
    return sub.set_index("image_id")[list(columns)].rename(columns=columns)


def build_manifest(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = read_split_files() if rows is None else rows
    _check_agreed_is_subset(rows)

    base = (
        rows.sort_values(["task", "variant"])  # deterministic pick of text/path when files repeat an image
        .drop_duplicates("image_id")
        .set_index("image_id")[["event_name", "tweet_id", "tweet_text", "image"]]
        .rename(columns={"event_name": "event", "tweet_text": "text_raw", "image": "image_path"})
    )
    unknown_events = set(base.event) - set(EVENTS)
    if unknown_events:
        raise ValueError(f"unknown CrisisMMD events {sorted(unknown_events)}")

    parts = [
        base,
        _label_table(rows, "informative", "all", {"label_text": "informative_text", "label_image": "informative_image"}),
        _label_table(rows, "humanitarian", "all", {"label_text": "humanitarian_text", "label_image": "humanitarian_image"}),
        _label_table(rows, "informative", "agreed", {"label": "y_informative"}),
        _label_table(rows, "humanitarian", "agreed", {"label": "y_humanitarian"}),
        _label_table(rows, "damage", "all", {"label": "y_severity"}),
    ]
    for task in TASKS:
        # Agreed files are subsets of "all" with identical assignments, so "all" is the official split.
        parts.append(_label_table(rows, task, "all", {"split": f"split_{task}"}))
    m = pd.concat(parts, axis=1)
    m.index.name = "image_id"
    m = m.reset_index()

    m["informative_text"] = labels.harmonise(m.informative_text, labels.CRISISMMD_INFORMATIVE, "informative_text")
    m["informative_image"] = labels.harmonise(m.informative_image, labels.CRISISMMD_INFORMATIVE, "informative_image")
    m["humanitarian_text"] = labels.harmonise(m.humanitarian_text, labels.CRISISMMD_HUMANITARIAN, "humanitarian_text")
    m["humanitarian_image"] = labels.harmonise(m.humanitarian_image, labels.CRISISMMD_HUMANITARIAN, "humanitarian_image")
    m["y_informative"] = labels.harmonise(m.y_informative, labels.CRISISMMD_INFORMATIVE, "y_informative")
    m["y_humanitarian"] = labels.harmonise(m.y_humanitarian, labels.CRISISMMD_HUMANITARIAN, "y_humanitarian")
    m["y_severity"] = labels.harmonise(m.y_severity, labels.CRISISMMD_SEVERITY, "y_severity")
    # Disaster type is trained on the Crisis Image Benchmark only, never derived from the event.
    m["y_disaster_type"] = pd.Series(pd.NA, index=m.index, dtype="string")

    m["text"] = m.text_raw.map(lambda t: ftfy.fix_text(t) if isinstance(t, str) else t)
    m["text_repaired"] = m.text != m.text_raw
    m["posted_at"] = pd.to_datetime(m.tweet_id.map(snowflake_time), utc=True)
    m["collection_date"] = m.image_path.map(_collection_date)

    m.insert(0, "dataset", "crisismmd")
    m.insert(1, "sample_id", "crisismmd:" + m.image_id)
    m["source"] = "crisismmd"
    return m.drop(columns=["text_raw"])


def _check_agreed_is_subset(rows: pd.DataFrame) -> None:
    """The agreed files must be a same-split subset of the "all" files with matching labels."""
    for task in ("informative", "humanitarian"):
        agreed = rows[(rows.task == task) & (rows.variant == "agreed")].set_index("image_id")
        full = rows[(rows.task == task) & (rows.variant == "all")].set_index("image_id")
        missing = agreed.index.difference(full.index)
        if len(missing):
            raise ValueError(f"{task}: {len(missing)} agreed images absent from the 'all' files")
        moved = (agreed.split != full.loc[agreed.index, "split"]).sum()
        if moved:
            raise ValueError(f"{task}: {moved} agreed images have a different split in the 'all' files")
        if not (agreed.label_text == agreed.label_image).all():
            raise ValueError(f"{task}: agreed file contains rows whose text and image labels differ")


def check_consistency(m: pd.DataFrame) -> dict[str, int]:
    """Counts the report needs; none of these should be silently assumed to be zero."""
    merged_agree = m.humanitarian_text.notna() & (m.humanitarian_text == m.humanitarian_image)
    return {
        "images": len(m),
        "tweets": m.tweet_id.nunique(),
        "agreed_informative": int(m.y_informative.notna().sum()),
        "agreed_humanitarian": int(m.y_humanitarian.notna().sum()),
        # Text and image agree after the 8->5 merge, yet the pair is absent from the official
        # agreed file. We keep the official file as the authority for comparability.
        "agree_after_merge_but_not_in_agreed_file": int((merged_agree & m.y_humanitarian.isna()).sum()),
        "in_agreed_file_but_disagree_after_merge": int((m.y_humanitarian.notna() & ~merged_agree).sum()),
        "severity_labelled": int(m.y_severity.notna().sum()),
        "severity_but_image_not_humanitarian": int((m.y_severity.notna() & (m.humanitarian_image == "not_humanitarian")).sum()),
        "gate_violation_not_informative_but_humanitarian": int(
            ((m.y_informative == "not_informative") & m.y_humanitarian.notna() & (m.y_humanitarian != "not_humanitarian")).sum()
        ),
        "text_repaired_by_ftfy": int(m.text_repaired.sum()),
        "missing_posted_at": int(m.posted_at.isna().sum()),
        "missing_collection_date": int(m.collection_date.isna().sum()),
        "posted_outside_collection_window": int(_outside_window(m).sum()),
        **split_leaks(m),
    }


def split_leaks(m: pd.DataFrame) -> dict[str, int]:
    """How the official splits would leak if used naively by a multi-task model."""
    cols = [f"split_{t}" for t in TASKS]
    official = m[cols]
    n_splits = official.nunique(axis=1)
    in_test = (official == "test").any(axis=1)
    in_train = (official == "train").any(axis=1)
    within_task_tweets = sum(int((m.groupby("tweet_id")[c].nunique() > 1).sum()) for c in cols)
    return {
        "images_with_conflicting_official_splits": int((n_splits > 1).sum()),
        "images_test_for_one_task_train_for_another": int((in_test & in_train).sum()),
        "tweets_split_across_train_dev_test_within_one_task": within_task_tweets,
    }


def _outside_window(m: pd.DataFrame) -> pd.Series:
    """Tweets whose snowflake time falls outside their event's collection window (±1 day for timezones)."""
    start = m.event.map(lambda e: pd.Timestamp(EVENTS[e].collection_start, tz="UTC") - pd.Timedelta(days=1))
    end = m.event.map(lambda e: pd.Timestamp(EVENTS[e].collection_end, tz="UTC") + pd.Timedelta(days=2))
    return m.posted_at.notna() & ((m.posted_at < start) | (m.posted_at > end))
