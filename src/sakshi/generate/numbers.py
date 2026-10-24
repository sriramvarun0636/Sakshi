"""Conservative numeric support checks for generated claims (spec Appendix E.6)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[?\bP\d+(?:\s*,\s*P\d+)*\b\]?")
_NUMERAL = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:billion|million|thousand|bn|[km])?(?:st|nd|rd|th|am|pm)?\b|"
    r"(?<![\w.])\d+(?:\.\d+)?%"
)
_TOKEN = re.compile(r"[a-z]+")
_UNITS = {
    **dict.fromkeys(("zero",), 0),
    **dict(
        zip(
            [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ],
            range(1, 20),
            strict=True,
        )
    ),
    **dict(zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10), strict=True)),
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
}
_VAGUE = {"dozens": 12, "hundreds": 100, "thousands": 1_000, "millions": 1_000_000}


@dataclass(frozen=True)
class Numbers:
    exact: frozenset[float]
    vague: frozenset[str]


def _word_number(words: list[tuple[str, int, int]], source: str, start: int) -> tuple[int, int] | None:
    total = current = 0
    index = start
    used = False
    while index < len(words):
        word, left, _ = words[index]
        if used and not re.fullmatch(r"[\s-]*", source[words[index - 1][2] : left]):
            break
        if (
            word == "and"
            and used
            and total + current >= 100
            and index + 1 < len(words)
            and words[index + 1][0] in _UNITS | _SCALES
        ):
            index += 1
            continue
        if word == "a" and index + 1 < len(words) and words[index + 1][0] in _SCALES:
            current += 1
        elif word in _UNITS:
            if used and total == 0 and current < 20:
                break
            current += _UNITS[word]
        elif word == "hundred":
            current = max(1, current) * 100
        elif word in ("thousand", "million", "billion"):
            total += max(1, current) * _SCALES[word]
            current = 0
        else:
            break
        used = True
        index += 1
    return (total + current, index) if used else None


def extract(text: str) -> Numbers:
    """Extract exact values and vague lower bounds without interpreting the claim."""
    lower = _CITATION.sub(" ", text.lower())
    exact: set[float] = set()
    for match in _NUMERAL.finditer(lower):
        item = match.group().strip().replace(",", "")
        parsed = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|[km])?(?:st|nd|rd|th|am|pm|%)?", item)
        if parsed:
            exact.add(round(float(parsed[1]) * _MULTIPLIERS.get(parsed[2] or "", 1), 6))
    # Word tokens that overlap a numeral suffix (as in 1.2k) must not become a second value.
    masked = list(lower)
    for match in _NUMERAL.finditer(lower):
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    masked_text = "".join(masked)
    words = [(match.group(), match.start(), match.end()) for match in _TOKEN.finditer(masked_text)]
    vague = {word for word, _, _ in words if word in _VAGUE}
    index = 0
    while index < len(words):
        parsed = _word_number(words, masked_text, index)
        if parsed is None:
            index += 1
        else:
            value, end = parsed
            exact.add(float(value))
            index = end
    return Numbers(frozenset(exact), frozenset(vague))


def unsupported(claim: str, evidence: str) -> list[str]:
    """Return unsupported exact numbers and vague quantities."""
    asked, given = extract(claim), extract(evidence)
    missing = [str(int(value)) if value.is_integer() else str(value) for value in sorted(asked.exact - given.exact)]
    missing.extend(
        word
        for word in sorted(asked.vague)
        if word not in given.vague and not any(value >= _VAGUE[word] for value in given.exact)
    )
    return missing
