from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from sakshi.aggregate.clusters import claim_clusters
from sakshi.aggregate.core import Tweet


def example_tweet(tweet_id, **changes):
    return replace(
        Tweet(
            tweet_id=tweet_id,
            sample_ids=(tweet_id,),
            representative_id=tweet_id,
            posted_at=datetime(2017, 9, 21, tzinfo=timezone.utc),
            event="hurricane_maria",
            text_clean="bridge damage",
            informative_label="informative",
            humanitarian_label="infrastructure_and_utility_damage",
            severity_label=None,
            abstain=False,
            provenance_status=None,
            excluded=False,
            confidence=0.8,
            informative_probability=0.8,
            component_ids=(1,),
        ),
        **changes,
    )


class FakeEmbedder:
    def encode(self, texts):
        result = np.zeros((len(texts), 2), dtype=np.float32)
        for row, text in enumerate(texts):
            result[row, 0 if "bridge" in text else 1] = 1
        return result


def test_claim_clusters_respect_category_textless_singletons_and_cap():
    tweets = [
        example_tweet("a", text_clean="bridge damage"),
        example_tweet("b", text_clean="bridge damaged"),
        example_tweet("c", text_clean="people need medical care", humanitarian_label="affected_individuals"),
        example_tweet("d", text_clean=None),
        example_tweet("e", text_clean=None),
    ]
    clusters, omitted = claim_clusters(tweets, FakeEmbedder())
    assert not omitted
    assert clusters["a"] == clusters["b"]
    assert clusters["c"] != clusters["a"]
    assert clusters["d"] != clusters["e"]
    capped, omitted = claim_clusters(
        [replace(tweet, confidence=1 - i * 0.1) for i, tweet in enumerate(tweets)], FakeEmbedder(), max_tweets=2
    )
    assert set(capped) == {"a", "b"} and omitted == {"c", "d", "e"}
