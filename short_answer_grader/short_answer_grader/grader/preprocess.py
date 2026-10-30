import re
import unicodedata

_SUBS = {"co₂": "co2", "h₂o": "h2o", "o₂": "o2"}


def normalize(text: str) -> str:
    """Lower-case, normalise unicode/whitespace, keep meaningful terms (no stop-word removal)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    for k, v in _SUBS.items():
        text = text.replace(k, v)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str):
    return normalize(text).split()


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]
