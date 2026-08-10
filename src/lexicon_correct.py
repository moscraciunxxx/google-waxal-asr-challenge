"""Train-vocabulary word correction for CTC ASR hypotheses.

Lexicon from train/validation only (never test gold).
Fast: exact match or edit-distance ≤1–2 within first-char + length buckets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.text_norm import normalize_text


def _levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if not a:
        return len(b) if len(b) <= max_dist else max_dist + 1
    if not b:
        return len(a) if len(a) <= max_dist else max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


class LexiconCorrector:
    def __init__(self, word_counts: dict[str, int], max_edit: int = 2):
        self.counts = dict(word_counts)
        self.max_edit = max_edit
        # key: (first_char, length) -> words sorted by freq desc
        self.buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        for w in self.counts:
            if not w:
                continue
            self.buckets[(w[0], len(w))].append(w)
        for k in self.buckets:
            self.buckets[k].sort(key=lambda w: -self.counts[w])

    @classmethod
    def from_texts(cls, texts: Iterable[str], min_count: int = 1) -> "LexiconCorrector":
        c: Counter[str] = Counter()
        for t in texts:
            for w in normalize_text(t).split():
                if w:
                    c[w] += 1
        if min_count > 1:
            c = Counter({w: n for w, n in c.items() if n >= min_count})
        return cls(dict(c))

    def _max_dist_for(self, word: str) -> int:
        n = len(word)
        if n <= 3:
            return 0
        if n <= 5:
            return 1
        return min(self.max_edit, 2)

    def correct_word(self, word: str) -> str:
        word = word.strip()
        if not word:
            return word
        if word in self.counts:
            return word
        if word.isdigit():
            return word
        max_d = self._max_dist_for(word)
        if max_d == 0:
            return word
        L = len(word)
        first = word[0]
        best_w, best_d, best_freq = word, max_d + 1, -1
        # first char ± length band; also allow first-char substitution via nearby buckets of same length
        keys = []
        for dlen in range(L - max_d, L + max_d + 1):
            keys.append((first, dlen))
        # also same-length any first char is too big; only try common alphabet neighbors if needed later
        for key in keys:
            bucket = self.buckets.get(key)
            if not bucket:
                continue
            for cand in bucket[:2500]:
                dist = _levenshtein(word, cand, max_dist=max_d)
                if dist > max_d:
                    continue
                freq = self.counts[cand]
                if dist < best_d or (dist == best_d and freq > best_freq):
                    best_w, best_d, best_freq = cand, dist, freq
                    if best_d == 0:
                        return best_w
        if best_w != word and (best_freq >= 2 or (best_d == 1 and best_freq >= 1)):
            return best_w
        return word

    def correct_text(self, text: str) -> str:
        norm = normalize_text(text)
        if not norm or norm == ".":
            return norm or "."
        fixed = [self.correct_word(w) for w in norm.split()]
        return " ".join(fixed).strip() or "."


def load_lang_texts(
    meta_dir: Path, lang: str, splits: tuple[str, ...] = ("train", "validation")
) -> list[str]:
    texts: list[str] = []
    for split in splits:
        p = meta_dir / f"{lang}_{split}.parquet"
        if not p.exists():
            continue
        import pandas as pd

        df = pd.read_parquet(p)
        col = "Target" if "Target" in df.columns else "transcription"
        texts.extend(df[col].astype(str).tolist())
    return texts


def build_correctors(
    meta_dir: Path,
    languages: tuple[str, ...] = ("lin", "lug", "sna"),
    min_count: int = 1,
) -> dict[str, LexiconCorrector]:
    out: dict[str, LexiconCorrector] = {}
    for lang in languages:
        texts = load_lang_texts(meta_dir, lang)
        out[lang] = LexiconCorrector.from_texts(texts, min_count=min_count)
    return out


def correct_predictions_df(
    df,
    correctors: dict[str, LexiconCorrector],
    lang_col: str = "language",
    pred_col: str = "prediction",
):
    import pandas as pd

    def _row(r):
        corr = correctors.get(r[lang_col])
        return corr.correct_text(str(r[pred_col])) if corr is not None else r[pred_col]

    out = df.copy()
    out[pred_col] = out.apply(_row, axis=1)
    return out
