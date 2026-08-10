"""Pure multi-family fusion helpers (no model I/O).

Fusion is full-row selection across families — not residual conf thr surgery on a floor CSV.
"""

from __future__ import annotations

from typing import Any

from src.text_norm import normalize_text


def mean_error(wer: float, cer: float) -> float:
    """Official lower-is-better mean error: 0.5*(WER+CER)."""
    return 0.5 * float(wer) + 0.5 * float(cer)


def beats_baseline(
    own: dict[str, float],
    baseline: dict[str, float],
) -> bool:
    """True if own strictly beats baseline on zindi mean-error, or both WER and CER lower."""
    o_me = mean_error(own["wer"], own["cer"])
    b_me = mean_error(baseline["wer"], baseline["cer"])
    if o_me < b_me:
        return True
    # allow tie on one metric if both strict lower on the other path via mean already handled
    return own["wer"] < baseline["wer"] and own["cer"] < baseline["cer"]


def fuse_row(
    mms_hyp: str,
    whisper_hyp: str,
    *,
    mms_score: float | None = None,
    whisper_score: float | None = None,
    decode_lang: str = "lug",
    lid_lang: str = "",
    lid_p1: float | None = None,
    prefer_mms_langs: frozenset[str] | None = None,
    mms_conf_floor: float = -0.12,
) -> dict[str, Any]:
    """Select fused hyp from MMS specialist vs Whisper family.

    Policy (documented; full multi-family selection, not residual conf thr surgery):
    1. Prefer non-empty hyps only.
    2. **Open-set Luo (lid=luo, p1>=0.55)** → Whisper family (router-aligned open-set path).
    3. Weak MMS CTC conf (score < mms_conf_floor) → Whisper if non-empty.
    4. Strong LID on domain WAXAL langs (lin/sna/lug/ach/nyn/…) + OK MMS conf → MMS.
    5. Else prefer MMS (domain match) when both present.
    """
    prefer_mms_langs = prefer_mms_langs or frozenset(
        {"lin", "sna", "lug", "ach", "nyn", "sog", "mas"}
    )
    mms = normalize_text(mms_hyp) or ""
    wh = normalize_text(whisper_hyp) or ""
    if not mms and not wh:
        return {
            "fused_hyp": ".",
            "fusion_source": "empty_fallback",
            "fusion_reason": "both_empty",
        }
    if mms and not wh:
        return {"fused_hyp": mms, "fusion_source": "mms", "fusion_reason": "whisper_empty"}
    if wh and not mms:
        return {"fused_hyp": wh, "fusion_source": "whisper", "fusion_reason": "mms_empty"}

    lid = (lid_lang or "").lower()
    dlang = (decode_lang or "").lower()
    strong_lid = lid_p1 is not None and float(lid_p1) >= 0.9
    mms_ok_conf = mms_score is None or float(mms_score) >= mms_conf_floor

    # Open-set Luo mass: use Whisper family (non-trivial multi-family selection)
    if lid == "luo" and lid_p1 is not None and float(lid_p1) >= 0.55 and wh:
        return {
            "fused_hyp": wh,
            "fusion_source": "whisper",
            "fusion_reason": "open_set_luo_router_whisper",
        }
    if mms_score is not None and float(mms_score) < mms_conf_floor and wh:
        return {
            "fused_hyp": wh,
            "fusion_source": "whisper",
            "fusion_reason": "mms_conf_weak",
        }
    if dlang in prefer_mms_langs and strong_lid and mms_ok_conf:
        return {
            "fused_hyp": mms,
            "fusion_source": "mms",
            "fusion_reason": "domain_specialist_strong_lid",
        }
    return {
        "fused_hyp": mms,
        "fusion_source": "mms",
        "fusion_reason": "default_domain_mms",
    }


def pack_fusion_submission_row(sample_id: str, fused_hyp: str, **meta: Any) -> dict[str, Any]:
    row = {"ID": sample_id, "Target": normalize_text(fused_hyp) or "."}
    row.update(meta)
    return row
