"""WER / CER and official 0.5/0.5 weighted score."""

from __future__ import annotations

from typing import Iterable, Sequence

from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer

from src.config import CER_WEIGHT, WER_WEIGHT
from src.text_norm import normalize_text


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level Word Error Rate after shared normalization."""
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    # jiwer raises on empty refs; guard with a sentinel empty-safe path
    if not refs:
        return 0.0
    # Replace empty refs with a placeholder so jiwer doesn't crash; empty hyp vs empty ref = 0
    safe_refs = [r if r else " " for r in refs]
    safe_hyps = [h if h else "" for h in hyps]
    # For empty-reference rows, force both to empty-equivalent
    for i, r in enumerate(refs):
        if not r:
            safe_refs[i] = "<empty>"
            safe_hyps[i] = "<empty>" if not hyps[i] else hyps[i]
    return float(jiwer_wer(safe_refs, safe_hyps))


def compute_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level Character Error Rate after shared normalization."""
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    if not refs:
        return 0.0
    safe_refs = [r if r else " " for r in refs]
    safe_hyps = [h if h else "" for h in hyps]
    for i, r in enumerate(refs):
        if not r:
            safe_refs[i] = "<empty>"
            safe_hyps[i] = "<empty>" if not hyps[i] else hyps[i]
    return float(jiwer_cer(safe_refs, safe_hyps))


def weighted_score(wer: float, cer: float) -> float:
    """Official challenge score: 0.5 * WER + 0.5 * CER (lower is better)."""
    return float(WER_WEIGHT * wer + CER_WEIGHT * cer)


def score_pairs(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> dict[str, float]:
    """Return wer, cer, and weighted score for a list of pairs."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps"
        )
    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)
    return {
        "wer": wer,
        "cer": cer,
        "score": weighted_score(wer, cer),
        "n": float(len(references)),
    }


def score_by_language(
    references: Sequence[str],
    hypotheses: Sequence[str],
    languages: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Overall + per-language metrics."""
    if not (len(references) == len(hypotheses) == len(languages)):
        raise ValueError("references, hypotheses, languages must have equal length")

    overall = score_pairs(references, hypotheses)
    by_lang: dict[str, dict[str, float]] = {"overall": overall}

    buckets: dict[str, list[tuple[str, str]]] = {}
    for ref, hyp, lang in zip(references, hypotheses, languages):
        buckets.setdefault(lang, []).append((ref, hyp))

    for lang, pairs in sorted(buckets.items()):
        refs = [p[0] for p in pairs]
        hyps = [p[1] for p in pairs]
        by_lang[lang] = score_pairs(refs, hyps)

    return by_lang
