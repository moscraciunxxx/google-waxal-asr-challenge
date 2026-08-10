"""Shortlist-driven floor-first gates for new Luo signals (pure, no model I/O).

Implements accept/reject logic aligned with SHORTLIST_RANK priorities:
  S01-lite LID mass composition (acoustic-router proxy without GPU)
  S05-lite hyp structure (blank/garbage path proxy on text)
  S08   char-LM Δ + orthography residual scores
  S09   soft dual band only as co-gate with S08 (never thr expand alone)

Banned as primary: residual CTC conf thr, dual thr>0.15 alone, mass multi-adapter,
blind all-luo, wordmerge/sna/corrector-only rehash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.prize_pack import char_sim, is_banned_mass_rewrite, length_guard_ok
from src.text_norm import normalize_text

# Junk lang2 codes that weaken true-Luo mass composition
_ACH_CODES = frozenset({"ach", "lwo", "alz"})
_JUNK_CODES = frozenset({"eng", "fas", "wol", "ckb", "spa", "fra", "deu"})


@dataclass(frozen=True)
class ResidualFeatures:
    """Feature bundle for one residual lid=luo∩decode=ach candidate."""

    id: str
    p1: float
    lang2: str
    p2: float
    ortho_mms: float
    lp_luo_mms: float
    lp_ach_mms: float
    cer_mc: float
    floor_hyp: str
    luo_hyp: str  # MMS-1B / dual-pool mms hyp
    pick_lang: str = ""
    conf_luo: float = 0.0
    conf_ach: float = 0.0
    already_dual: bool = False


def lp_delta(feat: ResidualFeatures) -> float:
    return float(feat.lp_luo_mms) - float(feat.lp_ach_mms)


def s01_lid_mass_ok(
    feat: ResidualFeatures,
    *,
    min_p1: float = 0.99,
    reject_ach_lang2: bool = True,
    reject_junk_lang2: bool = True,
) -> bool:
    """S01-lite / S04: high luo p1 mass; reject Ach-adjacent and junk lang2 when required.

    Junk lang2 (eng/fas/wol/ckb/…) weakens true-Luo mass composition — reject by default.
    """
    if feat.already_dual:
        return False
    if float(feat.p1) < min_p1:
        return False
    lang2 = (feat.lang2 or "").lower()
    if reject_ach_lang2 and lang2 in _ACH_CODES:
        return False
    if reject_junk_lang2 and lang2 in _JUNK_CODES:
        return False
    return True


def s05_structure_ok(
    luo_hyp: str,
    *,
    min_unique_ratio: float = 0.55,
    min_avg_word_len: float = 2.5,
    max_word_freq_frac: float = 0.25,
    max_consecutive_repeats: int = 1,
    min_bigram_unique: float = 0.55,
    floor_hyp: str | None = None,
    max_uniq_drop_vs_floor: float = 0.15,
) -> bool:
    """S05-lite: reject stutter/garbage / collapsed / repetitive Luo hyp paths.

    Stronger than unique-ratio alone so residual mass is not a no-op co-gate.
    Optional floor_hyp: reject if Luo unique-token ratio drops sharply vs floor.
    """
    hyp = normalize_text(luo_hyp)
    words = hyp.split()
    if not words:
        return False
    if all(len(w) <= 1 for w in words) and len(words) >= 3:
        return False
    if len(words) >= 4 and len(set(words)) == 1 and len(words[0]) <= 2:
        return False
    uniq = len(set(words)) / len(words)
    if uniq < min_unique_ratio:
        return False
    # excessive single-char fraction
    short = sum(1 for w in words if len(w) <= 1)
    if len(words) >= 5 and short / len(words) >= 0.7:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < min_avg_word_len:
        return False
    # dominant single token (loop/garbage) — only on longer hyps to avoid
    # killing short natural repetitions of function words.
    from collections import Counter

    if len(words) >= 8:
        maxf = Counter(words).most_common(1)[0][1] / len(words)
        if maxf > max_word_freq_frac:
            return False
    consec = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    if consec > max_consecutive_repeats:
        return False
    if len(words) >= 8:
        bigrams = list(zip(words, words[1:]))
        ubg = len(set(bigrams)) / max(1, len(bigrams))
        if ubg < min_bigram_unique:
            return False
    if floor_hyp is not None:
        fw = normalize_text(floor_hyp).split()
        if fw:
            floor_uniq = len(set(fw)) / len(fw)
            if uniq < floor_uniq - max_uniq_drop_vs_floor:
                return False
    return True

def s08_ortho_charlm_ok(
    feat: ResidualFeatures,
    *,
    min_ortho: float = 2.4,
    min_lp_delta: float = 0.0,
) -> bool:
    """S08: orthography score + char-LM luo−ach logprob delta (not CTC conf thr)."""
    if float(feat.ortho_mms) < min_ortho:
        return False
    if lp_delta(feat) <= min_lp_delta:
        return False
    return True


def s09_soft_dual_with_ortho_ok(
    feat: ResidualFeatures,
    *,
    cer_lo: float = 0.15,
    cer_hi: float = 0.22,
    min_ortho: float = 2.8,
    min_lp_delta: float = 0.0,
) -> bool:
    """S09: soft dual CER band only when S08-strict co-gates (never thr alone)."""
    cer = float(feat.cer_mc)
    if not (cer_lo < cer <= cer_hi):
        return False
    return s08_ortho_charlm_ok(feat, min_ortho=min_ortho, min_lp_delta=min_lp_delta)


def conf_primary_banned(feat: ResidualFeatures, *, margin: float = 0.0) -> bool:
    """True if a gate would reduce to residual conf primary (banned pattern detector)."""
    # Used only as red-team helper: conf_luo>conf_ach alone is banned primary.
    return (float(feat.conf_luo) - float(feat.conf_ach)) > margin


def stack_accept(
    feat: ResidualFeatures,
    *,
    min_ortho: float = 2.4,
    min_lp_delta: float = 0.0,
    min_p1: float = 0.99,
    require_lang2_not_ach: bool = True,
    reject_junk_lang2: bool = True,
    allow_soft_dual: bool = True,
    max_char_sim_skip: float | None = None,
    min_char_sim: float | None = 0.35,
) -> tuple[bool, str]:
    """Primary shortlist stack: S01-lite (junk reject) ∧ S05 ∧ S08 (optional S09).

    Returns (accept, reason_tag). Fail closed → (False, reason).
    """
    if feat.already_dual:
        return False, "already_dual"
    fl = normalize_text(feat.floor_hyp)
    hyp = normalize_text(feat.luo_hyp)
    if not hyp or hyp == "." or hyp == fl:
        return False, "empty_or_same"
    if not length_guard_ok(fl, hyp):
        return False, "length_guard"
    if not s05_structure_ok(hyp, floor_hyp=fl):
        return False, "s05_structure_fail"
    if not s01_lid_mass_ok(
        feat,
        min_p1=min_p1,
        reject_ach_lang2=require_lang2_not_ach,
        reject_junk_lang2=reject_junk_lang2,
    ):
        return False, "s01_lid_mass_fail"

    sim = char_sim(fl, hyp)
    if min_char_sim is not None and sim < min_char_sim:
        # allow very-low-sim only with stricter ortho (true-Luo rewrite)
        if float(feat.ortho_mms) < 3.0 or lp_delta(feat) <= 0.05:
            return False, "low_char_sim_without_strict_ortho"
    if max_char_sim_skip is not None and sim >= max_char_sim_skip:
        return False, "char_sim_too_high_noop"

    path_a = s08_ortho_charlm_ok(feat, min_ortho=min_ortho, min_lp_delta=min_lp_delta)
    path_b = allow_soft_dual and s09_soft_dual_with_ortho_ok(feat)
    if path_a and path_b:
        return True, "S01+S05+S08+S09"
    if path_a:
        return True, "S01+S05+S08"
    if path_b:
        return True, "S01+S05+S09"
    return False, "s08_s09_fail"


def score_candidate(feat: ResidualFeatures) -> float:
    """Ranking score: higher = more likely true-Luo under S08 features."""
    return float(feat.ortho_mms) + 0.5 * lp_delta(feat)


def select_replaces(
    features: Sequence[ResidualFeatures],
    *,
    max_n: int = 25,
    min_ortho: float = 2.4,
    min_lp_delta: float = 0.0,
    min_char_sim: float | None = 0.35,
) -> list[dict[str, Any]]:
    """Score residual features; return bounded replace dicts for prize_pack.apply_replace_set."""
    accepted: list[tuple[float, dict[str, Any]]] = []
    for feat in features:
        ok, reason = stack_accept(
            feat,
            min_ortho=min_ortho,
            min_lp_delta=min_lp_delta,
            min_char_sim=min_char_sim,
        )
        if not ok:
            continue
        hyp = normalize_text(feat.luo_hyp)
        accepted.append(
            (
                score_candidate(feat),
                {
                    "ID": feat.id,
                    "own_hyp": hyp,
                    "floor_hyp": normalize_text(feat.floor_hyp),
                    "reason": reason,
                    "signals": reason,
                    "ortho_mms": float(feat.ortho_mms),
                    "lp_delta": lp_delta(feat),
                    "cer_mc": float(feat.cer_mc),
                    "p1": float(feat.p1),
                    "score": score_candidate(feat),
                    "char_sim": char_sim(normalize_text(feat.floor_hyp), hyp),
                },
            )
        )
    accepted.sort(key=lambda x: -x[0])
    out = [r for _, r in accepted[: max(0, max_n)]]
    return out


def ban_check_verdict(n_replace: int, n_total: int = 1500) -> dict[str, Any]:
    """Document ban-compliance for meta."""
    return {
        "n_replace": n_replace,
        "mass_rewrite": is_banned_mass_rewrite(n_replace, n_total),
        "bans_respected": [
            "no residual conf thr primary",
            "no dual thr>0.15 alone (S09 requires S08 ortho co-gate)",
            "no mass lid=luo multi-adapter",
            "no blind all-luo",
            "no wordmerge/sna/corrector-only primary",
            "floor_default_except_replace_set",
        ],
        "verdict": "FAIL_mass_rewrite" if is_banned_mass_rewrite(n_replace, n_total) else "PASS_ban_check",
        "shortlist_family": [
            "S01-lite+junk_lang2_reject",
            "S05-lite_structure",
            "S08_ortho_charlm",
            "S09-optional_soft_dual_x_ortho",
        ],
    }


def features_from_row(row: Mapping[str, Any]) -> ResidualFeatures:
    """Build ResidualFeatures from a joined dict/Series-like mapping."""
    return ResidualFeatures(
        id=str(row["ID"]),
        p1=float(row.get("p1") or 0),
        lang2=str(row.get("lang2") or ""),
        p2=float(row.get("p2") or 0),
        ortho_mms=float(row.get("ortho_mms") or -999),
        lp_luo_mms=float(row.get("lp_luo_mms") or -999),
        lp_ach_mms=float(row.get("lp_ach_mms") or -999),
        cer_mc=float(row.get("cer_mc") or 1.0),
        floor_hyp=str(row.get("floor_hyp") or row.get("floor") or ""),
        luo_hyp=str(row.get("luo_hyp") or row.get("mms") or row.get("prediction") or ""),
        pick_lang=str(row.get("pick_lang") or ""),
        conf_luo=float(row.get("conf_luo") or 0),
        conf_ach=float(row.get("conf_ach") or 0),
        already_dual=bool(row.get("already_dual") in (True, "True", "true", 1, "1")),
    )
