"""HumAID (Alam et al., ICWSM 2021): text-only tweets used for the text-robustness evaluation.

HumAID shares five events with CrisisMMD (Harvey, Irma, Maria, the Mexico earthquake and
the Sri Lanka floods). Retweets of one original carry different tweet ids but the same
text, so a HumAID test tweet can be a CrisisMMD training tweet in disguise. `overlap`
finds those tweets so they can be excluded from any HumAID evaluation.
"""

from __future__ import annotations

import re
from pathlib import Path

import ftfy
import pandas as pd

from sakshi import paths

# Normalised texts shorter than this ("#harvey", "wow") match by accident.
MIN_MATCH_LENGTH = 20


def normalise_text(text: str) -> str:
    """Lower-cased tweet text without retweet prefix, URLs, mentions or punctuation."""
    t = ftfy.fix_text(str(text)).lower()
    t = re.sub(r"^rt @\w+:\s*", "", t)
    t = re.sub(r"https?://\S+|@\w+", "", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def read_tweets(root: Path = paths.HUMAID_ROOT) -> pd.DataFrame:
    """All HumAID tweets with their event and official split."""
    frames = []
    for f in sorted((root / "events_set1").glob("*/*.tsv")):
        df = pd.read_csv(f, sep="\t", dtype=str, quoting=3)
        frames.append(df.assign(event=f.parent.name, split=f.stem.rsplit("_", 1)[1]))
    if not frames:
        raise FileNotFoundError(f"no HumAID files under {root}")
    return pd.concat(frames, ignore_index=True)


def overlap(humaid: pd.DataFrame, crisismmd: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """HumAID tweets that repeat a CrisisMMD tweet (same id, or same normalised text).

    Returns the tweets to exclude (tweet_id, event, split) and summary counts.
    """
    h_norm = humaid.tweet_text.map(normalise_text)
    c_norm = crisismmd.drop_duplicates("tweet_id").text.map(normalise_text)
    known_texts = set(c_norm[c_norm.str.len() >= MIN_MATCH_LENGTH])

    same_id = humaid.tweet_id.isin(crisismmd.tweet_id)
    same_text = h_norm.isin(known_texts) & (h_norm.str.len() >= MIN_MATCH_LENGTH)
    exclude = humaid.loc[same_id | same_text, ["tweet_id", "event", "split"]]
    stats = {
        "tweets": len(humaid),
        "events": int(humaid.event.nunique()),
        "labels": humaid.class_label.value_counts().to_dict(),
        "same_tweet_id_as_crisismmd": int(same_id.sum()),
        "same_normalised_text_as_crisismmd": int(same_text.sum()),
        "excluded_from_humaid_evaluation": len(exclude),
    }
    return exclude, stats
