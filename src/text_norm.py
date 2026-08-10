"""Shared text normalization for train, eval, and submission.

Must be identical for references and hypotheses so WER/CER are comparable.
"""

from __future__ import annotations

import re
import unicodedata
import math

# Collapse runs of whitespace
_WS_RE = re.compile(r"\s+")
# Keep letters (all scripts), digits, and apostrophes; drop other punctuation
_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)
# Unicode letter/number class cleanup for leftover symbols
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def normalize_text(text: str | None, *, lowercase: bool = True) -> str:
    """Normalize a transcription for training targets and metric computation.

    Steps:
      1. None / non-str -> empty string
      2. Unicode NFKC
      3. Strip control characters
      4. Optional lowercasing (default on; African orthographies vary in case use)
      5. Remove punctuation except apostrophes inside words
      6. Collapse whitespace and strip
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub("", text)
    if lowercase:
        text = text.lower()
    # Treat common serialized missing-value sentinels as missing predictions.
    # This prevents a pandas NaN/null from becoming a valid transcript token.
    if text.strip() in {"nan", "null", "none"}:
        return ""
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    # Strip dangling apostrophes
    text = re.sub(r"\s+'\s+", " ", text)
    text = text.strip("' ")
    return text


def tokenize_words(text: str) -> list[str]:
    """Word tokens after normalization."""
    norm = normalize_text(text)
    if not norm:
        return []
    return norm.split()
