"""Gold-label urgency proxy and common-candidate ranking comparisons (§10.8)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sakshi.aggregate.core import Tweet, rank_posts


def gold_relevance(manifest: pd.DataFrame) -> dict[str, int]:
    """Maximum known urgency over a tweet's agreed sample labels; no event-derived gold."""
    required = {"sample_id", "tweet_id", "y_informative", "y_humanitarian", "y_severity"}
    if required - set(manifest):
        raise ValueError("ranking gold requires agreed informative/humanitarian labels and severity")
    result: dict[str, int] = {}
    for row in manifest.itertuples(index=False):
        tweet_id = str(row.tweet_id) if pd.notna(row.tweet_id) else f"sample:{row.sample_id}"
        if pd.notna(row.y_informative) and row.y_informative == "not_informative":
            grade = 0
        elif pd.notna(row.y_humanitarian) and row.y_humanitarian == "affected_individuals":
            grade = 3
        elif pd.notna(row.y_humanitarian) and row.y_humanitarian == "infrastructure_and_utility_damage":
            grade = 3 if pd.notna(row.y_severity) and row.y_severity == "severe_damage" else 2
        elif pd.notna(row.y_humanitarian) and row.y_humanitarian == "rescue_volunteering_or_donation_effort":
            grade = 2
        elif pd.notna(row.y_humanitarian) and row.y_humanitarian == "other_relevant_information":
            grade = 1
        else:
            continue
        result[tweet_id] = max(result.get(tweet_id, 0), grade)
    return result


def ndcg(grades: list[int], *, k: int) -> float | None:
    """NDCG with gains 2**relevance-1; no positive gold means an undefined window."""
    if k < 1 or any(grade not in (0, 1, 2, 3) for grade in grades):
        raise ValueError("NDCG requires k > 0 and integer relevance grades 0–3")

    def dcg(values: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(position + 2) for position, grade in enumerate(values[:k]))

    ideal = dcg(sorted(grades, reverse=True))
    return dcg(grades) / ideal if ideal else None


def evaluate_window(
    tweets: list[Tweet],
    clusters: dict[str, int],
    groups: dict[str, int],
    gold: dict[str, int],
    *,
    seed: int = 0,
) -> dict:
    """Compare four orderings of the same eligible, clustered, gold-labelled tweets."""
    scored = rank_posts(tweets, clusters, groups)
    candidates = sorted([tweet for _, tweet in scored if tweet.tweet_id in gold], key=lambda tweet: tweet.tweet_id)
    ordering = {
        "priority": [tweet for _, tweet in scored if tweet.tweet_id in gold],
        "recency": sorted(candidates, key=lambda tweet: (-tweet.posted_at.timestamp(), tweet.tweet_id)),
        "confidence": sorted(candidates, key=lambda tweet: (-tweet.confidence, tweet.tweet_id)),
    }
    metrics = {
        name: {f"ndcg@{k}": ndcg([gold[tweet.tweet_id] for tweet in ordered], k=k) for k in (10, 25)}
        for name, ordered in ordering.items()
    }
    draws = {k: [] for k in (10, 25)}
    for random_seed in range(seed, seed + 10):
        permutation = np.random.default_rng(random_seed).permutation(len(candidates))
        for k in draws:
            value = ndcg([gold[candidates[i].tweet_id] for i in permutation], k=k)
            if value is not None:
                draws[k].append(value)
    metrics["random"] = {f"ndcg@{k}": float(np.mean(draws[k])) if draws[k] else None for k in draws}
    return {
        "candidates": len(candidates),
        "unlabelled": len(scored) - len(candidates),
        "positive_gold": sum(gold[tweet.tweet_id] > 0 for tweet in candidates),
        "random_seeds": list(range(seed, seed + 10)),
        "metrics": metrics,
    }


def format_report(records: list[dict], *, mode: str) -> str:
    """Macro-average evaluable windows and expose exclusions rather than dropping them."""
    lines = [
        "# Urgency ranking" + (" — demonstration" if mode == "demo" else ""),
        "",
        f"Prediction scope: `{mode}`. This is a gold-label urgency proxy, not an operational utility study.",
        "All orderings use the same eligible, clustered tweets with known agreed labels. Missing labels are excluded",
        "from every ordering. Diversity caps and final category display ordering are applied after this ranking audit.",
        "Gains are 2^relevance−1. Random ordering is the mean of ten recorded seeds per window.",
        "Windows without positive gold have undefined NDCG and are reported separately.",
        "",
        "| Model / event | Windows | Scored | No positive gold | Unknown labels | Method | NDCG@10 | NDCG@25 |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    keys = sorted({(record["model_id"], record["event"]) for record in records})
    for model_id, event in keys:
        windows = [record for record in records if record["model_id"] == model_id and record["event"] == event]
        valid = [record for record in windows if record["metrics"]["priority"]["ndcg@10"] is not None]
        unknown = sum(record["unlabelled"] for record in windows)
        for method in ("priority", "random", "recency", "confidence"):
            values = []
            for k in (10, 25):
                value = np.mean([record["metrics"][method][f"ndcg@{k}"] for record in valid]) if valid else None
                values.append(f"{value:.3f}" if value is not None else "n/a")
            lines.append(
                f"| {model_id} / {event} | {len(windows)} | {len(valid)} | {len(windows) - len(valid)} | "
                f"{unknown} | {method} | " + " | ".join(values) + " |"
            )
    return "\n".join(lines) + "\n"
