"""The seven CrisisMMD events, their dates, and timestamp helpers.

Collection windows are from the CrisisMMD dataset page
(https://crisisnlp.qcri.org/crisismmd). They are the only dates the data itself
vouches for, so the provenance rule uses `collection_start` as the
event's onset. It is a conservative proxy: real onsets were a few days earlier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone


@dataclass(frozen=True)
class Event:
    name: str
    disaster_type: str  # used ONLY for label audits, never as a training label (it would leak the event)
    collection_start: date
    collection_end: date


EVENTS: dict[str, Event] = {
    e.name: e
    for e in [
        Event("srilanka_floods", "flood", date(2017, 5, 31), date(2017, 7, 3)),
        Event("hurricane_harvey", "hurricane", date(2017, 8, 25), date(2017, 9, 20)),
        Event("hurricane_irma", "hurricane", date(2017, 9, 6), date(2017, 9, 21)),
        Event("hurricane_maria", "hurricane", date(2017, 9, 20), date(2017, 11, 13)),
        Event("mexico_earthquake", "earthquake", date(2017, 9, 20), date(2017, 10, 6)),
        Event("california_wildfires", "fire", date(2017, 10, 10), date(2017, 10, 27)),
        Event("iraq_iran_earthquake", "earthquake", date(2017, 11, 13), date(2017, 11, 19)),
    ]
}

# Twitter snowflake IDs encode creation time: milliseconds since this epoch, shifted left 22 bits.
_TWITTER_EPOCH_MS = 1288834974657
# Snowflakes began on 4 Nov 2010; anything decoding outside this range is not a snowflake.
# The lower bound is a day later so that small non-tweet integers (which decode to within
# hours of the epoch) are rejected.
_SNOWFLAKE_MIN = datetime(2010, 11, 5, tzinfo=timezone.utc)
_SNOWFLAKE_MAX = datetime(2025, 1, 1, tzinfo=timezone.utc)


def snowflake_time(tweet_id: str | int | None) -> datetime | None:
    """UTC creation time of a tweet, decoded from its ID. None if the ID is not a plausible snowflake."""
    if tweet_id is None:
        return None
    try:
        n = int(tweet_id)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    t = datetime.fromtimestamp(((n >> 22) + _TWITTER_EPOCH_MS) / 1000, tz=timezone.utc)
    return t if _SNOWFLAKE_MIN <= t < _SNOWFLAKE_MAX else None
