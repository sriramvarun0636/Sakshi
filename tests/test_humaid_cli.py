import pandas as pd
import pytest

from sakshi import cli
from sakshi.data import humaid


def test_normalise_text_strips_retweet_urls_mentions_and_mojibake():
    raw = "RT @NWS: Weâ€™re tracking #Irma — stay safe @everyone https://t.co/abc"
    assert humaid.normalise_text(raw) == "we re tracking irma stay safe"


def test_overlap_matches_ids_and_retweeted_text_but_not_short_texts():
    crisismmd = pd.DataFrame(
        {
            "tweet_id": ["1", "2", "3"],
            "text": ["Bridge on I-10 washed away near Houston, avoid the area", "#harvey", "unrelated"],
        }
    )
    tweets = pd.DataFrame(
        {
            "tweet_id": ["1", "10", "11", "12"],
            "tweet_text": [
                "anything",  # same id
                "RT @abc: Bridge on I-10 washed away near Houston, avoid the area https://t.co/x",  # retweet
                "#Harvey",  # too short to count as a match
                "a genuinely different tweet about flooding",
            ],
            "event": "hurricane_harvey_2017",
            "split": "test",
            "class_label": "infrastructure_and_utility_damage",
        }
    )
    exclude, stats = humaid.overlap(tweets, crisismmd)
    assert exclude.tweet_id.tolist() == ["1", "10"]
    assert stats["same_tweet_id_as_crisismmd"] == 1
    assert stats["same_normalised_text_as_crisismmd"] == 1


@pytest.mark.parametrize("argv", [["prepare", "not_a_step"], ["download", "not_a_source"]])
def test_cli_rejects_unknown_names(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)


@pytest.mark.parametrize(
    "argv", [["prepare", "--max-distance", "13"], ["prepare", "--workers", "0"], ["download", "--parts", "0"]]
)
def test_cli_rejects_invalid_resource_options(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)
