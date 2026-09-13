"""Canonical label spaces and the maps from each dataset's raw labels into them.

Every map is total over the raw values we have seen; `harmonise` raises on anything
unexpected rather than silently producing NaN, so a new raw value fails loudly.
"""

from __future__ import annotations

import pandas as pd

INFORMATIVE = ["not_informative", "informative"]

HUMANITARIAN = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]

SEVERITY = ["little_or_no_damage", "mild_damage", "severe_damage"]

DISASTER_TYPE = [
    "earthquake",
    "fire",
    "flood",
    "hurricane",
    "landslide",
    "not_disaster",
    "other_disaster",
]

# CrisisMMD's 8 raw humanitarian classes -> the 5 used by the agreed-label benchmark
# (Ofli et al., ISCRAM 2020): minority classes merged into their nearest neighbour.
CRISISMMD_HUMANITARIAN = {
    "affected_individuals": "affected_individuals",
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "infrastructure_and_utility_damage": "infrastructure_and_utility_damage",
    "vehicle_damage": "infrastructure_and_utility_damage",
    "not_humanitarian": "not_humanitarian",
    "other_relevant_information": "other_relevant_information",
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
}

CRISISMMD_INFORMATIVE = {v: v for v in INFORMATIVE}

CRISISMMD_SEVERITY = {v: v for v in SEVERITY}

# Crisis Image Benchmark spellings (checked against its task TSVs).
CIB_SEVERITY = {
    "little_or_none": "little_or_no_damage",
    "mild": "mild_damage",
    "severe": "severe_damage",
}

CIB_DISASTER_TYPE = {v: v for v in DISASTER_TYPE}


def harmonise(raw: pd.Series, mapping: dict[str, str], name: str) -> pd.Series:
    """Map raw labels to canonical ones. Missing stays missing; unknown values raise."""
    present = raw.dropna()
    unknown = sorted(set(present) - set(mapping))
    if unknown:
        raise ValueError(f"{name}: unmapped raw labels {unknown}")
    return raw.map(mapping).astype("string")
