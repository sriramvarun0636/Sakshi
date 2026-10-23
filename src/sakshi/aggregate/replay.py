"""UTC-window aggregates with evidence counts and packet-local selected IDs (§10.2, E.3)."""

from __future__ import annotations

import json
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from sakshi import atomic, paths
from sakshi.aggregate.clusters import TextEmbedder, claim_clusters
from sakshi.aggregate.core import CATEGORY_URGENCY, Tweet, evidence_groups, select_posts
from sakshi.data.events import EVENTS

ALLOWED_HOURS = {6, 12, 24, 48}


def windows(tweets: list[Tweet], *, hours: int = 24) -> list[tuple[datetime, datetime, list[Tweet]]]:
    """Return half-open UTC windows aligned to midnight of the event's first day."""
    if hours not in ALLOWED_HOURS:
        raise ValueError(f"window hours must be one of {sorted(ALLOWED_HOURS)}")
    if not tweets:
        return []
    if any(tweet.posted_at.tzinfo is None for tweet in tweets):
        raise ValueError("window timestamps must be timezone-aware")
    first = min(tweet.posted_at.astimezone(timezone.utc) for tweet in tweets)
    last = max(tweet.posted_at.astimezone(timezone.utc) for tweet in tweets)
    start = datetime.combine(first.date(), time.min, tzinfo=timezone.utc)
    step = timedelta(hours=hours)
    result = []
    while start <= last:
        end = start + step
        subset = [tweet for tweet in tweets if start <= tweet.posted_at.astimezone(timezone.utc) < end]
        result.append((start, end, subset))
        start = end
    return result


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def aggregate_window(
    tweets: list[Tweet],
    *,
    event: str,
    model_id: str,
    start: datetime,
    end: datetime,
    embedder: TextEmbedder,
) -> dict:
    """Build one schema-shaped aggregate from a half-open event window."""
    if not model_id or not event or start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("aggregate needs a model, event and valid aware window")
    if any(tweet.event != event or not start <= tweet.posted_at < end for tweet in tweets):
        raise ValueError("aggregate contains a tweet outside its event or window")
    eligible = [tweet for tweet in tweets if tweet.eligible]
    groups = evidence_groups(tweets)
    clusters, omitted = claim_clusters(tweets, embedder)
    selectable = [tweet for tweet in eligible if tweet.tweet_id not in omitted]
    selected = select_posts(selectable, clusters, groups)
    by_category = {category: sum(tweet.humanitarian_label == category for tweet in eligible) for category in CATEGORY_URGENCY}
    severity = {
        name: sum(tweet.severity_label == name for tweet in eligible)
        for name in ("severe_damage", "mild_damage", "little_or_no_damage")
    }
    excluded = {name: sum(tweet.provenance_status == name for tweet in tweets) for name in ("recycled", "template_match")}
    flagged = {
        name: sum(tweet.provenance_status == name for tweet in tweets) for name in ("cross_event_concurrent", "match_undated")
    }
    informative = sum(tweet.informative_label == "informative" for tweet in tweets)
    abstained = sum(tweet.abstain for tweet in tweets)
    if eligible:
        lines = [
            f"{informative} informative posts in this window; {by_category['infrastructure_and_utility_damage']} "
            f"report infrastructure damage ({severity['severe_damage']} severe).",
            f"{excluded['recycled']} posts with recycled images and {excluded['template_match']} with template graphics "
            f"were excluded; {abstained} await review.",
        ]
    else:
        lines = ["No verified information in this window."]
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "event": event,
        "window": {"start": _iso(start), "end": _iso(end), "hours": (end - start).total_seconds() / 3600},
        "counts": {
            "tweets": len(tweets),
            "informative": informative,
            "abstained": abstained,
            "by_category": by_category,
            "severity_eligible": severity,
            "excluded_provenance": excluded,
            "flagged_provenance": flagged,
            "clustering_omitted": len(omitted),
        },
        "selected": [
            {
                "pid": item.pid,
                "sample_ids": list(item.sample_ids),
                "priority": item.priority,
                "category": item.category,
                "cluster_id": item.cluster_id,
                "group_id": item.group_id,
            }
            for item in selected
        ],
        "system_lines": lines,
    }


def aggregate_event(
    tweets: list[Tweet],
    *,
    event: str,
    model_id: str,
    embedder: TextEmbedder,
    hours: int = 24,
    output_root: Path = paths.REPO / "runs" / "aggregate",
) -> list[Path]:
    """Write one atomic aggregate JSON per event window."""
    if any(tweet.event != event for tweet in tweets):
        raise ValueError("aggregate_event received tweets from a different event")
    if event not in EVENTS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_id):
        raise ValueError("event must be known and model_id must be path-safe")
    outputs = []
    for start, end, subset in windows(tweets, hours=hours):
        aggregate = aggregate_window(subset, event=event, model_id=model_id, start=start, end=end, embedder=embedder)
        output = output_root / model_id / event / f"{start.strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic.text(output, json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
        outputs.append(output)
    return outputs
