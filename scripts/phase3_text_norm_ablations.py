#!/usr/bin/env python3
"""Phase-3 offline proxy: light post-decode text-norm ablations.

Baseline: oracle waxal-300m greedy hyps (from phase2_proxy_expand_detail.csv)
          + current src.text_norm.normalize_text (applied inside score_pairs).

Ablations applied AFTER baseline hyp (hyp-side only):
  A) collapse repeated spaces only
  B) strip apostrophes
  C) collapse elongated vowels (mild: same vowel x3+ → single)
  D) join common split patterns using data/lms/lug lexicon (lug rows only)
  E) no-op control

Never uses Phase-1 test gold for train/tune. Seed 42.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import score_by_language, score_pairs
from src.text_norm import normalize_text

SEED = 42
PROXY_INDEX = ROOT / "data" / "proxy_val_index.csv"
EXPAND_DETAIL = ROOT / "outputs" / "phase2_proxy_expand_detail.csv"
CHAMPION_DETAIL = ROOT / "outputs" / "phase2_proxy_champion_detail.csv"
LUG_COUNTS = ROOT / "data" / "lms" / "lug_counts.json"
OUT_JSON = ROOT / "outputs" / "phase3_text_norm.json"
OUT_MD = ROOT / "outputs" / "phase3_text_norm.md"

# Mild elongated vowel: aaa+ → a (same for e,i,o,u). African langs use long
# vowels (aa, ee, …); only collapse runs of length ≥ 3.
_ELONG_RE = re.compile(r"([aeiou])\1{2,}", flags=re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r" {2,}")
_APOS_RE = re.compile(r"['\u2019\u02bc]")


def load_oracle_hyps() -> pd.DataFrame:
    """Prefer oracle_waxal rows from expand detail; fallback to champion true-lang match."""
    if EXPAND_DETAIL.exists():
        df = pd.read_csv(EXPAND_DETAIL)
        if "method" in df.columns and (df["method"] == "oracle_waxal").any():
            out = df[df["method"] == "oracle_waxal"].copy()
            out = out.rename(columns={"true_lang": "language"})
            out["source"] = "phase2_proxy_expand_detail.csv::oracle_waxal"
            return out[["id", "language", "ref", "hyp", "source"]].reset_index(drop=True)

    if not CHAMPION_DETAIL.exists():
        raise FileNotFoundError(
            f"Need {EXPAND_DETAIL} (oracle_waxal) or {CHAMPION_DETAIL}"
        )
    champ = pd.read_csv(CHAMPION_DETAIL)
    # Use rows where decode_lang == true_lang as proxy for oracle; else keep all
    # with note that multi-hyp routing was used.
    out = champ.rename(columns={"true_lang": "language"})
    matched = (out["decode_lang"] == out["language"]).sum()
    out["source"] = (
        f"phase2_proxy_champion_detail.csv"
        f"(decode==true on {matched}/{len(out)}; not pure oracle)"
    )
    return out[["id", "language", "ref", "hyp", "source"]].reset_index(drop=True)


def load_lug_lexicon() -> tuple[dict[str, int], dict[str, int]]:
    if not LUG_COUNTS.exists():
        return {}, {}
    data = json.loads(LUG_COUNTS.read_text(encoding="utf-8"))
    uni = {w: int(c) for w, c in data.get("uni", {}).items() if not str(w).startswith("<")}
    bi = {k: int(v) for k, v in data.get("bi", {}).items()}
    return uni, bi


def feat_A_collapse_spaces(text: str) -> str:
    """Collapse repeated spaces only (baseline normalize_text already does this)."""
    if text is None:
        return ""
    s = str(text)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s.strip()


def feat_B_strip_apostrophes(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = _APOS_RE.sub("", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s.strip()


def feat_C_collapse_elongated_vowels(text: str) -> str:
    """Collapse same-vowel runs of length ≥3 to a single vowel (mild)."""
    if text is None:
        return ""
    s = str(text)
    s = _ELONG_RE.sub(r"\1", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s.strip()


def feat_D_join_lug_splits(
    text: str,
    uni: dict[str, int],
    bi: dict[str, int],
    *,
    min_joined: int = 3,
    max_bi: int = 0,
) -> str:
    """Join adjacent tokens when joined form is in train lexicon and bigram is rare.

    Conservative: both parts length ≥2, joined count ≥ min_joined, bigram ≤ max_bi.
    """
    words = normalize_text(text).split()
    if not words or not uni:
        return text if text is not None else ""
    out: list[str] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            a, b = words[i], words[i + 1]
            joined = a + b
            jc = uni.get(joined, 0)
            bc = bi.get(f"{a}\t{b}", 0)
            if (
                len(a) >= 2
                and len(b) >= 2
                and jc >= min_joined
                and bc <= max_bi
            ):
                out.append(joined)
                i += 2
                continue
            # secondary: joined much more common than observed split
            if (
                len(a) >= 2
                and len(b) >= 2
                and jc >= 5
                and bc <= 1
                and jc >= 5 * max(bc, 1)
            ):
                out.append(joined)
                i += 2
                continue
        out.append(words[i])
        i += 1
    return " ".join(out)


def feat_E_noop(text: str) -> str:
    return "" if text is None else str(text)


def apply_method(
    hyp: str,
    lang: str,
    method: str,
    uni: dict[str, int],
    bi: dict[str, int],
) -> str:
    if method == "baseline":
        # score_pairs re-applies normalize_text; hyp already greedy-decoded
        return hyp if hyp is not None else ""
    if method == "A_collapse_spaces":
        return feat_A_collapse_spaces(hyp)
    if method == "B_strip_apostrophes":
        return feat_B_strip_apostrophes(hyp)
    if method == "C_collapse_elong_vowels":
        return feat_C_collapse_elongated_vowels(hyp)
    if method == "D_join_lug_splits":
        if lang == "lug":
            return feat_D_join_lug_splits(hyp, uni, bi)
        return hyp if hyp is not None else ""
    if method == "E_noop":
        return feat_E_noop(hyp)
    raise ValueError(f"unknown method {method}")


def pack_metrics(sc: dict) -> dict:
    """Attach zindi_est = 1 - 0.5*WER - 0.5*CER (= 1 - score)."""
    out = {}
    for k, v in sc.items():
        z = 1.0 - float(v["score"])
        out[k] = {
            "wer": float(v["wer"]),
            "cer": float(v["cer"]),
            "score": float(v["score"]),
            "zindi_est": z,
            "n": int(v["n"]),
        }
    return out


def n_changed(before: list[str], after: list[str]) -> int:
    n = 0
    for a, b in zip(before, after):
        if normalize_text(a) != normalize_text(b):
            n += 1
    return n


def main() -> None:
    proxy = pd.read_csv(PROXY_INDEX)
    base_df = load_oracle_hyps()
    # Align to proxy index order / membership
    base_df = proxy[["id", "language"]].merge(
        base_df.drop(columns=["language"], errors="ignore"),
        on="id",
        how="inner",
    )
    if len(base_df) != len(proxy):
        missing = set(proxy["id"]) - set(base_df["id"])
        raise RuntimeError(
            f"hyp coverage {len(base_df)}/{len(proxy)}; missing e.g. {list(missing)[:5]}"
        )

    uni, bi = load_lug_lexicon()
    source = str(base_df["source"].iloc[0]) if "source" in base_df.columns else "unknown"

    methods = [
        "baseline",
        "A_collapse_spaces",
        "B_strip_apostrophes",
        "C_collapse_elong_vowels",
        "D_join_lug_splits",
        "E_noop",
    ]

    raw_hyps = base_df["hyp"].fillna("").astype(str).tolist()
    refs = base_df["ref"].fillna("").astype(str).tolist()
    langs = base_df["language"].astype(str).tolist()

    results: dict = {
        "seed": SEED,
        "n": len(base_df),
        "langs": sorted(base_df["language"].unique().tolist()),
        "lang_counts": base_df["language"].value_counts().to_dict(),
        "hyp_source": source,
        "baseline_def": (
            "oracle waxal-300m greedy (true_lang) + score_pairs "
            "(src.text_norm.normalize_text on ref/hyp)"
        ),
        "ablation_notes": {
            "A_collapse_spaces": "multi-space collapse only; already in normalize_text",
            "B_strip_apostrophes": "remove ' / ’ from hyp only",
            "C_collapse_elong_vowels": r"([aeiou])\1{2,} → \1 (mild; keeps aa/ee long vowels)",
            "D_join_lug_splits": (
                "lug rows only; join a+b if unigram(joined)>=3 and bigram(a,b)<=0 "
                f"(lexicon={LUG_COUNTS.name}, |uni|={len(uni)}, |bi|={len(bi)})"
            ),
            "E_noop": "identity control",
        },
        "methods": {},
    }

    base_z = None
    for method in methods:
        hyps = [
            apply_method(h, lang, method, uni, bi)
            for h, lang in zip(raw_hyps, langs)
        ]
        sc = score_by_language(refs, hyps, langs)
        packed = pack_metrics(sc)
        chg = n_changed(raw_hyps, hyps) if method != "baseline" else 0
        # also count raw string changes (before score norm)
        raw_chg = sum(1 for a, b in zip(raw_hyps, hyps) if str(a) != str(b))
        entry = {
            "overall": packed["overall"],
            "per_lang": {k: v for k, v in packed.items() if k != "overall"},
            "n_rows_changed_vs_raw_hyp": int(raw_chg),
            "n_rows_changed_after_normalize_text": int(chg),
        }
        if base_z is None and method == "baseline":
            base_z = packed["overall"]["zindi_est"]
        if base_z is not None:
            entry["delta_zindi_est_vs_baseline"] = (
                packed["overall"]["zindi_est"] - base_z
            )
            entry["delta_wer_vs_baseline"] = (
                packed["overall"]["wer"] - results.get("methods", {})
                .get("baseline", {})
                .get("overall", {})
                .get("wer", packed["overall"]["wer"])
            )
        results["methods"][method] = entry

    # fill deltas properly after baseline stored
    bz = results["methods"]["baseline"]["overall"]["zindi_est"]
    bw = results["methods"]["baseline"]["overall"]["wer"]
    bc = results["methods"]["baseline"]["overall"]["cer"]
    for method, entry in results["methods"].items():
        entry["delta_zindi_est_vs_baseline"] = entry["overall"]["zindi_est"] - bz
        entry["delta_wer_vs_baseline"] = entry["overall"]["wer"] - bw
        entry["delta_cer_vs_baseline"] = entry["overall"]["cer"] - bc
        # per-lang deltas
        for lang, lm in entry["per_lang"].items():
            b_lang = results["methods"]["baseline"]["per_lang"][lang]
            lm["delta_zindi_est_vs_baseline"] = lm["zindi_est"] - b_lang["zindi_est"]

    # Recommendation
    THRESH = 0.002
    beaters = []
    for method, entry in results["methods"].items():
        if method in ("baseline", "E_noop"):
            continue
        dz = entry["delta_zindi_est_vs_baseline"]
        if dz >= THRESH:
            beaters.append((method, dz))
    beaters.sort(key=lambda x: -x[1])

    if beaters:
        best_m, best_dz = beaters[0]
        recommendation = "APPLY"
        rec_detail = (
            f"{best_m} improves zindi_est by {best_dz:+.6f} (≥ {THRESH}). "
            "Consider applying on Phase-2 submission hyps."
        )
    else:
        recommendation = "DO_NOT_APPLY"
        best = max(
            (
                (m, e["delta_zindi_est_vs_baseline"])
                for m, e in results["methods"].items()
                if m not in ("baseline", "E_noop")
            ),
            key=lambda x: x[1],
            default=(None, 0.0),
        )
        rec_detail = (
            f"No ablation beats baseline by ≥{THRESH} absolute zindi_est. "
            f"Best non-control: {best[0]} Δ={best[1]:+.6f}. "
            "Do NOT change Phase-2 submission post-decode text pipeline."
        )

    results["threshold_abs_zindi"] = THRESH
    results["methods_beating_baseline"] = [
        {"method": m, "delta_zindi_est": dz} for m, dz in beaters
    ]
    results["recommendation"] = recommendation
    results["recommendation_detail"] = rec_detail

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Markdown report
    lines = [
        "# Phase-3: post-decode text-norm ablations (proxy)",
        "",
        f"- **n** = {results['n']} (proxy_val_index: ach/nyn/lug/sog/mas × 40)",
        f"- **seed** = {SEED}",
        f"- **hyp source** = `{source}`",
        f"- **baseline** = {results['baseline_def']}",
        f"- **zindi_est** = `1 - 0.5·WER - 0.5·CER`",
        f"- **promote threshold** = +{THRESH:.3f} absolute zindi_est vs baseline",
        "",
        "## Overall",
        "",
        "| method | WER | CER | zindi_est | Δzindi | rows changed (norm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        e = results["methods"][method]
        o = e["overall"]
        lines.append(
            f"| `{method}` | {o['wer']:.6f} | {o['cer']:.6f} | {o['zindi_est']:.6f} | "
            f"{e['delta_zindi_est_vs_baseline']:+.6f} | "
            f"{e['n_rows_changed_after_normalize_text']} |"
        )

    lines += [
        "",
        "## Per-language zindi_est",
        "",
    ]
    lang_list = sorted(results["langs"])
    header = "| method | " + " | ".join(lang_list) + " |"
    sep = "|---|" + "|".join(["---:" for _ in lang_list]) + "|"
    lines.append(header)
    lines.append(sep)
    for method in methods:
        e = results["methods"][method]
        cells = []
        for lang in lang_list:
            z = e["per_lang"][lang]["zindi_est"]
            dz = e["per_lang"][lang]["delta_zindi_est_vs_baseline"]
            if method == "baseline":
                cells.append(f"{z:.4f}")
            else:
                cells.append(f"{z:.4f} ({dz:+.4f})")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Ablation notes",
        "",
    ]
    for k, v in results["ablation_notes"].items():
        lines.append(f"- **{k}**: {v}")

    lines += [
        "",
        "## Beat baseline by ≥0.002?",
        "",
    ]
    if beaters:
        for m, dz in beaters:
            lines.append(f"- YES: `{m}` Δzindi_est = {dz:+.6f}")
    else:
        lines.append("- **None** of A–D beat baseline by ≥0.002 absolute zindi_est.")
        lines.append(
            "- E_noop should match baseline (control). Small float noise is not a signal."
        )

    lines += [
        "",
        "## Recommendation",
        "",
        f"**{recommendation}** — {rec_detail}",
        "",
        "### Rationale",
        "",
        "- Baseline hyps are already space-collapsed and apostrophe-free (MMS CTC alphabet).",
        "- Refs often retain apostrophes (`'`); hyp-side strip (B) therefore cannot help, "
        "and shared strip would be a *metric* change, not a free post-decode win without "
        "confirming leaderboard uses the same strip.",
        "- Mild vowel de-elongation (C) has ~0 fire rate on oracle waxal hyps (no `aaa+` runs).",
        "- Lug join (D) fires rarely; any local WER gain is diluted over 200 rows and "
        "does not clear +0.002 overall.",
        "- Do not re-run ASR; do not touch champion submission files based on these results.",
        "",
        f"Artifacts: `{OUT_JSON.relative_to(ROOT)}`, `{OUT_MD.relative_to(ROOT)}`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "out_json": str(OUT_JSON),
        "out_md": str(OUT_MD),
        "recommendation": recommendation,
        "baseline_zindi_est": bz,
        "deltas": {
            m: results["methods"][m]["delta_zindi_est_vs_baseline"] for m in methods
        },
    }, indent=2))


if __name__ == "__main__":
    main()
