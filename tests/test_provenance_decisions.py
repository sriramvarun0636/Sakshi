from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from sakshi.provenance.detect import IndexImage, QueryImage, decide, match, visible
from sakshi.types import ProvStatus

T0 = datetime(2017, 9, 1, tzinfo=timezone.utc)
ONSET = datetime(2017, 9, 20, tzinfo=timezone.utc)
T1 = datetime(2017, 9, 21, tzinfo=timezone.utc)
T2 = datetime(2017, 9, 22, tzinfo=timezone.utc)


def vector(x=1.0):
    result = np.zeros(512, dtype=np.float32)
    result[0] = x
    return result


def query(**changes):
    base = QueryImage("query", "q-tweet", "maria", T2, True, False, False, 224, 0xFFFF, vector(), False)
    return replace(base, **changes)


def item(**changes):
    base = IndexImage("older", "other-tweet", "irma", T0, False, 224, 0xFFFF, vector(), False)
    return replace(base, **changes)


ONSETS = {"maria": ONSET}


def status(q, items):
    return decide(q, items, onsets=ONSETS)


def test_all_provenance_decision_branches_and_visibility():
    assert status(query(has_image=False), []).status == ProvStatus.NO_IMAGE
    assert status(query(unreadable=True), []).status == ProvStatus.UNSUPPORTED_IMAGE
    assert status(query(degenerate=True), []).status == ProvStatus.UNSUPPORTED_IMAGE
    assert status(query(), []).status == ProvStatus.NOVEL
    assert status(query(), [item(tweet_id="q-tweet")]).status == ProvStatus.NOVEL
    assert status(query(), [item(post_id="query")]).status == ProvStatus.NOVEL
    assert status(query(), [item(first_seen=T2)]).status == ProvStatus.NOVEL  # strict earlier-than-query rule
    assert visible(query(time=None), item(first_seen=T2))

    assert status(query(is_graphic=True), [item(is_graphic=True)]).status == ProvStatus.TEMPLATE_MATCH
    assert status(query(event=None, time=None), [item()]).status == ProvStatus.SEEN_BEFORE
    assert status(query(), [item(first_seen=None)]).status == ProvStatus.MATCH_UNDATED
    assert status(query(), [item(index_event="maria", first_seen=T1)]).status == ProvStatus.DUPLICATE
    assert status(query(), [item()]).status == ProvStatus.RECYCLED
    assert status(query(), [item(first_seen=ONSET)]).status == ProvStatus.CROSS_EVENT_CONCURRENT


def test_earliest_dated_origin_wins_over_more_similar_repost():
    earlier = item(post_id="prior-event", first_seen=T0, phash=0xFFFC, embedding=vector(0.95))
    latest = item(post_id="same-event-repost", index_event="maria", first_seen=T1)
    result = status(query(), [latest, earlier])
    assert result.status == ProvStatus.RECYCLED
    assert result.origin_post_id == "prior-event"
    assert result.top_match_post_id == "same-event-repost"
    assert result.tier == "phash" and result.phash_distance == 2
    assert result.similarity == np.float32(0.95)


def test_phash_size_gate_and_clip_fallback():
    small = item(min_side=32, phash=0xFFFF)
    result = match(query(), small)
    assert result is not None and result.tier == "clip"
    distant_hash = item(phash=0)
    assert match(query(), distant_hash).tier == "clip"
    no_match = item(phash=0, embedding=vector(0.5))
    assert match(query(), no_match) is None
