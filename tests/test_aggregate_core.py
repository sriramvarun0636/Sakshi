from dataclasses import replace

import pandas as pd
import pytest

from sakshi.aggregate.core import Tweet, evidence_groups, priority, rollup_samples, select_posts


def example_tweet(tweet_id, **changes):
    base = Tweet(
        tweet_id=tweet_id,
        sample_ids=(tweet_id + ":0",),
        representative_id=tweet_id + ":0",
        posted_at=pd.Timestamp("2017-09-21T12:00:00Z").to_pydatetime(),
        event="hurricane_maria",
        text_clean="A bridge collapsed after the hurricane",
        informative_label="informative",
        humanitarian_label="infrastructure_and_utility_damage",
        severity_label="severe_damage",
        abstain=False,
        provenance_status=None,
        excluded=False,
        confidence=0.8,
        informative_probability=0.9,
        component_ids=(1,),
    )
    return replace(base, **changes)


def test_rollup_excludes_tweet_with_any_recycled_image_and_uses_genuine_severity():
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "tweet_id": ["tweet", "tweet"],
            "posted_at": pd.to_datetime(["2017-09-21T12:00:00Z"] * 2),
            "event": ["hurricane_maria"] * 2,
            "text": ["Bridge damaged", "Bridge damaged"],
            "component": [1, 2],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "informative_label": ["informative", "informative"],
            "humanitarian_label": ["infrastructure_and_utility_damage"] * 2,
            "severity_label": ["severe_damage", "mild_damage"],
            "abstain": [False, False],
            "p_informative__informative": [0.95, 0.8],
            "p_humanitarian__infrastructure_and_utility_damage": [0.7, 0.6],
        }
    )
    provenance = pd.DataFrame({"post_id": ["a", "b"], "status": ["recycled", "novel"]})
    tweet = rollup_samples(manifest, predictions, provenance)[0]
    assert tweet.representative_id == "a" and tweet.excluded
    assert tweet.provenance_status == "recycled"
    assert tweet.severity_label == "mild_damage"
    assert tweet.sample_ids == ("a", "b") and tweet.component_ids == (1, 2)
    assert not tweet.eligible


def test_evidence_union_priority_and_diversity_limits():
    first = example_tweet("a", component_ids=(1,), text_clean="A bridge collapsed after the hurricane")
    same_image = example_tweet("b", component_ids=(1,), text_clean="A completely unrelated long message")
    same_text = example_tweet("c", component_ids=(3,), text_clean="A bridge collapsed after the hurricane")
    different = example_tweet("d", component_ids=(4,), text_clean="Another distinct report from the affected area")
    groups = evidence_groups([first, same_image, same_text, different])
    assert groups["a"] == groups["b"] == groups["c"] != groups["d"]
    flagged = replace(first, provenance_status="cross_event_concurrent")
    assert priority(flagged, 3) == pytest.approx(1 / 3)
    selected = select_posts(
        [first, same_image, same_text, different],
        {"a": 1, "b": 1, "c": 1, "d": 2},
        groups,
        limit=3,
        per_group=2,
    )
    assert len(selected) == 3
    assert sum(item.group_id == groups["a"] for item in selected) == 2
    assert [item.pid for item in selected] == ["P1", "P2", "P3"]
    assert all(item.priority > 0 for item in selected)


def test_selection_rejects_missing_cluster_ids_and_empty_is_safe():
    assert select_posts([], {}, {}) == []
    with pytest.raises(ValueError, match="claim cluster"):
        select_posts([example_tweet("a")], {}, {"a": 0})
