"""Conservative text-only postprocess rules for MMS CTC hypotheses.

No re-decode. No test-gold learning. Safe to apply to any split.
"""

from __future__ import annotations

import re
from typing import Mapping

from src.text_norm import normalize_text

# Train-supported orthography seeds (extend only with train/val evidence).
ORTHO_MAP: dict[str, dict[str, str]] = {
    "lin": {
        "eglize": "eglise",
        "katolikue": "catholique",
        "katholique": "catholique",
        "folele": "fololo",
        "molai": "molayi",
        "coulera": "culere",
    },
    "lug": {},
    "sna": {
        "mhombe": "mombe",
        "mudzumai": "mudzimai",
        "madzumai": "madzimai",
        "zvitsuku": "zvitsvuku",
    },
}

FUNCTION_SHORT = {
    "lin": {"ya", "na", "pe", "te", "ba", "ko", "se", "po", "mpe", "nde", "eza"},
    "lug": {"ne", "mu", "ku", "wa", "ya", "nga", "te", "si", "no", "wo", "ze", "be"},
    "sna": {"ne", "mu", "ku", "wa", "ya", "nga", "se", "pa", "ha", "ndi", "na", "uye"},
}


def is_stutter_garbage(words: list[str]) -> bool:
    if not words:
        return False
    if all(len(w) <= 1 for w in words) and len(words) >= 3:
        return True
    if len(words) >= 4 and len(set(words)) == 1 and len(words[0]) <= 2:
        return True
    short = sum(1 for w in words if len(w) <= 2)
    if len(words) >= 5 and short / len(words) >= 0.85 and len(set(words)) <= 3:
        return True
    return False


def collapse_stutter(text: str) -> str:
    """Rule 1: pure CTC noise → '.' (reduces WER>1 insertion tails)."""
    words = normalize_text(text).split()
    if is_stutter_garbage(words):
        return "."
    return text


def apply_ortho_map(text: str, lang: str, extra: Mapping[str, str] | None = None) -> str:
    """Rule 2: high-precision orthography map."""
    m = dict(ORTHO_MAP.get(lang, {}))
    if extra:
        m.update(extra)
    if not m:
        return text
    words = normalize_text(text).split()
    if not words:
        return text
    return " ".join(m.get(w, w) for w in words)


def split_ba_prefix(text: str, lexicon: Mapping[str, int], min_stem: int = 3) -> str:
    """Rule 3 (lin): ba+stem → ba stem when fused OOV and stem in train lexicon."""
    words = normalize_text(text).split()
    out: list[str] = []
    for w in words:
        if (
            w.startswith("ba")
            and len(w) >= 2 + min_stem
            and lexicon.get(w, 0) == 0
            and lexicon.get(w[2:], 0) >= 3
        ):
            out.extend(["ba", w[2:]])
        else:
            out.append(w)
    return " ".join(out) if out else text


def dedupe_consecutive_function(text: str, lang: str) -> str:
    """Rule 4: drop repeated short / function words only."""
    words = normalize_text(text).split()
    fw = FUNCTION_SHORT.get(lang, set())
    out: list[str] = []
    for w in words:
        if out and out[-1] == w and (len(w) <= 3 or w in fw):
            continue
        out.append(w)
    return " ".join(out) if out else text


def join_split_majority(
    text: str,
    join_map: Mapping[str, str] | None = None,
    split_map: Mapping[str, str] | None = None,
) -> str:
    """Rule 5: apply train-majority join/split maps (bigram → joined or reverse)."""
    s = normalize_text(text)
    if not s:
        return text
    if join_map:
        for a, b in sorted(join_map.items(), key=lambda x: -len(x[0])):
            s = s.replace(a, b)
    words = s.split()
    if split_map:
        out: list[str] = []
        for w in words:
            if w in split_map:
                out.extend(split_map[w].split())
            else:
                out.append(w)
        words = out
    return " ".join(words) if words else text


def postprocess_hypothesis(
    text: str,
    lang: str,
    *,
    lexicon: Mapping[str, int] | None = None,
    join_map: Mapping[str, str] | None = None,
    split_map: Mapping[str, str] | None = None,
    extra_ortho: Mapping[str, str] | None = None,
) -> str:
    """Apply Rules 1–5 in order. Empty/stutter → '.'."""
    if text is None:
        return "."
    raw = str(text).strip()
    if raw in ("", ".", "nan"):
        return "."

    # Rule 1
    norm = normalize_text(raw)
    words = norm.split()
    if is_stutter_garbage(words):
        return "."

    # Rule 2
    fixed = apply_ortho_map(norm, lang, extra_ortho)

    # Rule 3
    if lang == "lin" and lexicon is not None:
        fixed = split_ba_prefix(fixed, lexicon)

    # Rule 4
    fixed = dedupe_consecutive_function(fixed, lang)

    # Rule 5
    fixed = join_split_majority(fixed, join_map=join_map, split_map=split_map)

    fixed = normalize_text(fixed)
    return fixed if fixed else "."
