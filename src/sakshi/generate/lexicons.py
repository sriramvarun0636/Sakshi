"""Fixed WP5 lexical safeguards from design spec Appendix E.13."""

from __future__ import annotations

import re

SEVERE = frozenset(
    [
        "destroyed",
        "destroy",
        "destroys",
        "destruction",
        "collapsed",
        "collapse",
        "collapses",
        "flattened",
        "leveled",
        "levelled",
        "razed",
        "devastated",
        "devastating",
        "devastation",
        "obliterated",
        "demolished",
        "ruined",
        "uninhabitable",
    ]
    + [
        "wiped out",
        "washed away",
        "swept away",
        "in ruins",
        "severe damage",
        "severely damaged",
        "heavily damaged",
        "badly damaged",
        "extensive damage",
        "major damage",
        "catastrophic damage",
        "torn apart",
        "ripped apart",
    ]
)
NO_DAMAGE = frozenset(
    [
        "no damage",
        "little damage",
        "minimal damage",
        "minor damage",
        "slight damage",
        "undamaged",
        "intact",
        "largely spared",
        "spared",
    ]
)
CASUALTY = frozenset(
    [
        "dead",
        "death",
        "deaths",
        "died",
        "dies",
        "killed",
        "kills",
        "fatality",
        "fatalities",
        "casualty",
        "casualties",
        "bodies",
        "injured",
        "injury",
        "injuries",
        "wounded",
        "hurt",
        "missing",
        "trapped",
        "stranded",
        "displaced",
        "homeless",
    ]
)
NEGATION = frozenset(["no", "not", "without", "never", "none", "zero", "nobody"])
_WORDS = re.compile(r"\b[\w']+\b")


def terms(text: str, vocabulary: frozenset[str]) -> list[tuple[str, bool]]:
    """Find whole-word terms and whether a cue occurs in the preceding three words."""
    lower = text.lower()
    tokens = list(_WORDS.finditer(lower))
    found = []
    for term in sorted(vocabulary, key=len, reverse=True):
        pattern = re.compile(r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)")
        for match in pattern.finditer(lower):
            before = [token.group() for token in tokens if token.end() <= match.start()][-3:]
            negated = any(word in NEGATION or word.endswith("n't") for word in before)
            found.append((term, negated))
    return found
