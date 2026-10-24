"""Deterministic tweet rollup, evidence groups and priority (§10.3–10.6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from sakshi.data.humaid import MIN_MATCH_LENGTH, normalise_text
from sakshi.data.labels import HUMANITARIAN, INFORMATIVE, SEVERITY
from sakshi.data.splits import UnionFind
from sakshi.text import clean_for_display, normalise_for_model
from sakshi.types import ProvStatus

CATEGORY_URGENCY = {
    "affected_individuals": 1.0,
    "infrastructure_and_utility_damage": 0.8,
    "rescue_volunteering_or_donation_effort": 0.6,
    "other_relevant_information": 0.3,
}
SEVERITY_URGENCY = {"severe_damage": 1.0, "mild_damage": 0.5, "little_or_no_damage": 0.1, None: 0.3}
SEVERITY_ORDER = {name: i for i, name in enumerate(SEVERITY)}
EXCLUDED = {"recycled", "template_match"}
REVIEW = {"cross_event_concurrent", "match_undated"}


@dataclass(frozen=True)
class Tweet:
    tweet_id: str
    sample_ids: tuple[str, ...]
    representative_id: str
    posted_at: datetime
    event: str
    text_clean: str | None
    informative_label: str
    humanitarian_label: str
    severity_label: str | None
    abstain: bool
    provenance_status: str | None
    excluded: bool
    confidence: float
    informative_probability: float
    component_ids: tuple[int, ...]
    grouping_text: str | None = None

    @property
    def eligible(self) -> bool:
        return self.informative_label == "informative" and not self.abstain and not self.excluded


@dataclass(frozen=True)
class Selected:
    pid: str
    tweet_id: str
    sample_ids: tuple[str, ...]
    priority: float
    category: str
    cluster_id: int
    group_id: int


def _optional_text(value: object) -> str | None:
    return str(value) if pd.notna(value) and str(value).strip() else None


def rollup_samples(manifest: pd.DataFrame, predictions: pd.DataFrame, provenance: pd.DataFrame) -> list[Tweet]:
    """Choose one representative and conservatively combine all images of each tweet."""
    required_manifest = {"sample_id", "tweet_id", "posted_at", "event", "text", "component"}
    required_predictions = {
        "sample_id",
        "informative_label",
        "humanitarian_label",
        "severity_label",
        "abstain",
        "p_informative__informative",
    }
    required_provenance = {"post_id", "status"}
    if required_manifest - set(manifest) or required_predictions - set(predictions) or required_provenance - set(provenance):
        raise ValueError("rollup inputs are missing required manifest, prediction or provenance columns")
    for name, frame, key in (
        ("manifest", manifest, "sample_id"),
        ("predictions", predictions, "sample_id"),
        ("provenance", provenance, "post_id"),
    ):
        if frame[key].duplicated().any():
            raise ValueError(f"{name} has duplicate sample IDs")
    rows = manifest.merge(predictions, on="sample_id", how="left", validate="one_to_one", indicator="pred_merge")
    rows = rows.merge(
        provenance[["post_id", "status"]], left_on="sample_id", right_on="post_id", how="left", validate="one_to_one"
    )
    if not (rows.pred_merge == "both").all() or rows.status.isna().any():
        raise ValueError("every manifest sample needs a prediction and a provenance result")
    if rows.posted_at.isna().any() or rows.event.isna().any():
        raise ValueError("tweet aggregation needs dated, event-labelled rows")
    if not rows.status.isin([status.value for status in ProvStatus]).all():
        raise ValueError("unknown provenance status")
    if not rows.informative_label.isin(INFORMATIVE).all() or not rows.humanitarian_label.isin(HUMANITARIAN).all():
        raise ValueError("aggregation needs canonical informative and humanitarian predictions")
    if not (rows.severity_label.isna() | rows.severity_label.isin(SEVERITY)).all():
        raise ValueError("unknown severity prediction")
    if rows.abstain.isna().any() or not rows.abstain.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("abstention values must be booleans")
    probabilities = pd.to_numeric(rows["p_informative__informative"], errors="raise")
    if not probabilities.between(0, 1).all():
        raise ValueError("informative probabilities must be finite and in [0, 1]")
    if any(value.tzinfo is None for value in rows.posted_at):
        raise ValueError("tweet aggregation needs timezone-aware timestamps")
    rows["tweet_key"] = rows.tweet_id.astype("string").fillna("sample:" + rows.sample_id.astype(str))
    tweets = []
    for tweet_id, group in rows.groupby("tweet_key", sort=False):
        if group.event.nunique() != 1 or group.posted_at.nunique() != 1:
            raise ValueError(f"tweet {tweet_id} has inconsistent event or timestamp")
        ordered = group.sort_values(["p_informative__informative", "sample_id"], ascending=[False, True])
        representative = ordered.iloc[0]
        statuses = set(group.status.astype(str))
        excluded = bool(statuses & EXCLUDED)
        if "recycled" in statuses:
            provenance_status = "recycled"
        elif "template_match" in statuses:
            provenance_status = "template_match"
        elif "cross_event_concurrent" in statuses:
            provenance_status = "cross_event_concurrent"
        elif "match_undated" in statuses:
            provenance_status = "match_undated"
        else:
            provenance_status = None
        eligible_severity = group.loc[
            (group.informative_label == "informative") & ~group.status.isin(EXCLUDED) & group.severity_label.isin(SEVERITY_ORDER)
        ]
        severity = (
            max(eligible_severity.severity_label.astype(str), key=SEVERITY_ORDER.__getitem__) if len(eligible_severity) else None
        )
        category = str(representative.humanitarian_label)
        probability_column = f"p_humanitarian__{category}"
        if probability_column not in representative:
            raise ValueError(f"prediction is missing {probability_column}")
        confidence = float(representative[probability_column])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("representative humanitarian confidence must be in [0, 1]")
        informative_probability = float(representative["p_informative__informative"])
        if not math.isfinite(informative_probability) or not 0 <= informative_probability <= 1:
            raise ValueError("representative informative probability must be in [0, 1]")
        text = _optional_text(representative.text)
        tweets.append(
            Tweet(
                tweet_id=str(tweet_id),
                sample_ids=tuple(sorted(group.sample_id.astype(str))),
                representative_id=str(representative.sample_id),
                posted_at=representative.posted_at.to_pydatetime(),
                event=str(representative.event),
                text_clean=clean_for_display(text) if normalise_for_model(text) is not None else None,
                informative_label=str(representative.informative_label),
                humanitarian_label=category,
                severity_label=severity,
                abstain=bool(representative.abstain),
                provenance_status=provenance_status,
                excluded=excluded,
                confidence=confidence,
                informative_probability=informative_probability,
                component_ids=tuple(sorted(set(group.component.astype(int)))),
                grouping_text=normalise_text(text) if text is not None else None,
            )
        )
    return tweets


def evidence_groups(tweets: list[Tweet]) -> dict[str, int]:
    """Union tweets sharing a pHash component or a sufficiently long normalised text."""
    if len({tweet.tweet_id for tweet in tweets}) != len(tweets):
        raise ValueError("tweet IDs must be unique")
    uf = UnionFind(len(tweets))
    by_component: dict[int, int] = {}
    by_text: dict[str, int] = {}
    for position, tweet in enumerate(tweets):
        for component in tweet.component_ids:
            if component in by_component:
                uf.union(position, by_component[component])
            else:
                by_component[component] = position
        if tweet.grouping_text is not None or tweet.text_clean is not None:
            normalized = tweet.grouping_text if tweet.grouping_text is not None else normalise_text(tweet.text_clean)
            if len(normalized) >= MIN_MATCH_LENGTH:
                if normalized in by_text:
                    uf.union(position, by_text[normalized])
                else:
                    by_text[normalized] = position
    return {tweet.tweet_id: int(uf.find(position)) for position, tweet in enumerate(tweets)}


def priority(tweet: Tweet, corroboration: int) -> float:
    """The fixed confidence, urgency, corroboration and provenance formula."""
    if not tweet.eligible:
        raise ValueError("priority is defined only for eligible tweets")
    if corroboration < 1 or tweet.humanitarian_label not in CATEGORY_URGENCY:
        raise ValueError("invalid corroboration count or humanitarian category")
    u_cat = CATEGORY_URGENCY[tweet.humanitarian_label]
    u_sev = SEVERITY_URGENCY[tweet.severity_label]
    corr = min(1.0, math.log2(1 + corroboration) / 3)
    prov = 0.5 if tweet.provenance_status in REVIEW else 1.0
    return tweet.confidence * (0.5 * u_cat + 0.3 * u_sev + 0.2 * corr) * prov


def rank_posts(
    tweets: list[Tweet],
    cluster_ids: dict[str, int],
    group_ids: dict[str, int],
) -> list[tuple[float, Tweet]]:
    """Rank eligible tweets before diversity caps; share the ranking with evaluation."""
    eligible = [tweet for tweet in tweets if tweet.eligible]
    ids = {tweet.tweet_id for tweet in eligible}
    if len(ids) != len(eligible):
        raise ValueError("tweet IDs must be unique")
    if ids - set(cluster_ids) or ids - set(group_ids):
        raise ValueError("eligible tweets need claim cluster and evidence group IDs")
    cluster_groups: dict[int, set[int]] = {}
    for tweet in eligible:
        cluster_groups.setdefault(cluster_ids[tweet.tweet_id], set()).add(group_ids[tweet.tweet_id])
    scored = [(priority(tweet, len(cluster_groups[cluster_ids[tweet.tweet_id]])), tweet) for tweet in eligible]
    return sorted(scored, key=lambda item: (-item[0], item[1].tweet_id))


def select_posts(
    tweets: list[Tweet],
    cluster_ids: dict[str, int],
    group_ids: dict[str, int],
    *,
    limit: int = 25,
    per_category: int = 8,
    per_group: int = 2,
) -> list[Selected]:
    """Rank by fixed priority, apply diversity caps, then assign packet-local IDs."""
    if min(limit, per_category, per_group) < 1:
        raise ValueError("selection limits must be positive")
    scored = rank_posts(tweets, cluster_ids, group_ids)
    category_counts: dict[str, int] = {}
    group_counts: dict[int, int] = {}
    chosen = []
    for score, tweet in scored:
        category = tweet.humanitarian_label
        group = group_ids[tweet.tweet_id]
        if category_counts.get(category, 0) >= per_category or group_counts.get(group, 0) >= per_group:
            continue
        chosen.append((score, tweet))
        category_counts[category] = category_counts.get(category, 0) + 1
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(chosen) == limit:
            break
    chosen.sort(key=lambda item: (-CATEGORY_URGENCY[item[1].humanitarian_label], -item[0], item[1].tweet_id))
    return [
        Selected(
            pid=f"P{position}",
            tweet_id=tweet.tweet_id,
            sample_ids=tweet.sample_ids,
            priority=score,
            category=tweet.humanitarian_label,
            cluster_id=cluster_ids[tweet.tweet_id],
            group_id=group_ids[tweet.tweet_id],
        )
        for position, (score, tweet) in enumerate(chosen, 1)
    ]
