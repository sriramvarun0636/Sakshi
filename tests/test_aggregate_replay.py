import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from sakshi.aggregate.core import Tweet
from sakshi.aggregate.replay import aggregate_event, aggregate_window, windows


class Embedder:
    def encode(self, texts):
        result = np.zeros((len(texts), 2), dtype=np.float32)
        result[:, 0] = 1
        return result


def tweet(tweet_id, hour, **changes):
    return replace(
        Tweet(
            tweet_id=tweet_id,
            sample_ids=(tweet_id + ":0",),
            representative_id=tweet_id + ":0",
            posted_at=datetime(2017, 9, 21, hour, tzinfo=timezone.utc),
            event="hurricane_maria",
            text_clean="A bridge collapsed after the storm",
            informative_label="informative",
            humanitarian_label="infrastructure_and_utility_damage",
            severity_label="severe_damage",
            abstain=False,
            provenance_status=None,
            excluded=False,
            confidence=0.8,
            informative_probability=0.9,
            component_ids=(int(hour),),
        ),
        **changes,
    )


def test_windows_are_half_open_and_empty_aggregate_is_explicit(tmp_path):
    first = tweet("first", 0)
    boundary = tweet("boundary", 6, abstain=True)
    sliced = windows([first, boundary], hours=6)
    assert len(sliced) == 2
    assert [item.tweet_id for item in sliced[0][2]] == ["first"]
    assert [item.tweet_id for item in sliced[1][2]] == ["boundary"]
    empty = aggregate_window(
        [], event="hurricane_maria", model_id="fixture", start=sliced[0][0], end=sliced[0][1], embedder=Embedder()
    )
    assert empty["selected"] == [] and empty["system_lines"] == ["No verified information in this window."]
    output = aggregate_event(
        [first, boundary], event="hurricane_maria", model_id="fixture", embedder=Embedder(), hours=6, output_root=tmp_path
    )
    assert len(output) == 2
    records = [json.loads(path.read_text()) for path in output]
    assert records[0]["selected"][0]["pid"] == "P1"
    assert records[0]["counts"]["severity_eligible"]["severe_damage"] == 1
    assert records[1]["selected"] == []
    assert records[1]["system_lines"] == ["No verified information in this window."]
