"""Tweet text preparation for model input and privacy-safe display."""

from __future__ import annotations

import html
import re
import unicodedata

import ftfy

_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w]", re.UNICODE)
_PLACEHOLDERS = {"rt", "user", "httpurl"}


def _shared_clean(text: str | None) -> str | None:
    if text is None:
        return None
    result = ftfy.fix_text(html.unescape(text))
    result = _MENTION.sub("@USER", result)
    result = _URL.sub("HTTPURL", result)
    result = unicodedata.normalize("NFC", _SPACE.sub(" ", result).strip())
    return result or None


def clean_for_display(text: str | None) -> str | None:
    """Repair text and remove handles and links while retaining visible emoji."""
    result = _shared_clean(text)
    # Packet/UI strings must not contain an @-handle or an http substring (INV-07).
    return result.replace("@USER", "[USER]").replace("HTTPURL", "[LINK]") if result else None


def normalise_for_model(text: str | None) -> str | None:
    """Prepare BERTweet input, or return None when only placeholders remain.

    Emoji conversion uses the optional ``ml`` extra. It is imported here so the base
    data-preparation package never needs model dependencies.
    """
    result = _shared_clean(text)
    if result is None:
        return None
    try:
        import emoji
    except ImportError as exc:
        raise ImportError("text normalisation requires the 'ml' extra: pip install -e '.[ml]'") from exc
    result = _SPACE.sub(" ", emoji.demojize(result)).strip()
    meaningful = [_PUNCTUATION.sub("", word).lower() for word in result.split()]
    return result if any(word and word not in _PLACEHOLDERS for word in meaningful) else None
