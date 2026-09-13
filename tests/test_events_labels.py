from datetime import datetime, timezone

import pandas as pd
import pytest

from sakshi.data import labels
from sakshi.data.events import EVENTS, snowflake_time

_EPOCH_MS = 1288834974657


def _snowflake(t: datetime) -> int:
    return (int(t.timestamp() * 1000) - _EPOCH_MS) << 22


def test_snowflake_round_trip():
    t = datetime(2017, 9, 7, 12, 30, 5, 250000, tzinfo=timezone.utc)
    assert snowflake_time(_snowflake(t)) == t
    assert snowflake_time(str(_snowflake(t) + 12345)) == t  # low bits are worker/sequence, not time


@pytest.mark.parametrize("bad", [None, "", "abc", "12345", 0, -5, "12345_1"])
def test_snowflake_rejects_non_snowflakes(bad):
    assert snowflake_time(bad) is None


def test_events_have_ordered_windows():
    assert len(EVENTS) == 7
    for e in EVENTS.values():
        assert e.collection_start < e.collection_end
        assert e.disaster_type in labels.DISASTER_TYPE


def test_humanitarian_merge_is_onto_the_five_classes():
    assert set(labels.CRISISMMD_HUMANITARIAN.values()) == set(labels.HUMANITARIAN)
    raw = pd.Series(["vehicle_damage", "missing_or_found_people", None])
    out = labels.harmonise(raw, labels.CRISISMMD_HUMANITARIAN, "h")
    assert out.iloc[0] == "infrastructure_and_utility_damage"
    assert out.iloc[1] == "affected_individuals"
    assert pd.isna(out.iloc[2])


def test_harmonise_rejects_unknown_labels():
    with pytest.raises(ValueError, match="unmapped"):
        labels.harmonise(pd.Series(["severe_damage", "catastrophic"]), labels.CRISISMMD_SEVERITY, "s")


def test_cib_maps_land_in_canonical_spaces():
    assert set(labels.CIB_SEVERITY.values()) <= set(labels.SEVERITY)
    assert set(labels.CIB_DISASTER_TYPE.values()) <= set(labels.DISASTER_TYPE)
