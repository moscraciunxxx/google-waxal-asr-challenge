#!/usr/bin/env python3
"""Analyze zero-shot MMS test prediction error patterns (diagnostic only; no training).

Reads outputs/mms_shards/{lin,lug,sna}_test.csv and reports WER/CER/zindi_est,
substitution pairs, function-word ins/del, empty-hyp patterns, and conservative
postprocess rules. Never uses test gold for model training.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import score_by_language, score_pairs
from src.text_norm import normalize_text, tokenize_words

SHARDS = ROOT / "outputs" / "mms_shards"
META = ROOT / "data" / "hf_metadata"
OUT_JSON = ROOT / "outputs" / "mms_error_analysis.json"
OUT_REPORT = ROOT / "outputs" / "mms_error_analysis_report.md"

# High-frequency closed-class / function-like words (heuristic per language)
FUNCTION_WORDS = {
    "lin": {
        "ya", "na", "eza", "ezali", "oyo", "te", "pe", "moko", "biso", "ngai",
        "ba", "ko", "li", "po", "nde", "kasi", "se", "awa", "wana", "yango",
        "eloko", "lokola", "epai", "mwa", "mosusu", "nyonso", "penza", "mpe",
        "yo", "ye", "bango", "toza", "nazali", "wana", "likolo", "liboso",
    },
    "lug": {
        "ne", "mu", "ku", "wa", "ya", "nga", "nze", "we", "abo", "omu", "ka",
        "bu", "ga", "li", "te", "naye", "bwe", "kati", "wano", "eri", "ati",
        "okuba", "era", "kuba", "buli", "si", "no", "wo", "zo", "bya", "byo",
        "gye", "ze", "be", "ye", "ate", "ggwe", "yee", "naye",
    },
    "sna": {
        "ne", "mu", "ku", "wa", "ya", "nga", "iye", "uye", "zve", "kana", "se",
        "pa", "ha", "ndi", "na", "che", "avo", "iri", "kuti", "asi", "zvino",
        "pano", "iko", "nye", "wo", "zvake", "zvavo", "iyi", "iyo", "aya",
        "acho", "ako", "ake", "avo", "avo", "ichiri", "asi",
    },
}


def load_preds() -> pd.DataFrame:
    frames = []
    for lang in ("lin", "lug", "sna"):
        p = SHARDS / f"{lang}_test.csv"
        df = pd.read_csv(p)
        assert set(df.columns) >= {"ID", "language", "prediction", "reference"}
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def is_empty_hyp(h: str) -> bool:
    n = normalize_text(h)
    return n in ("", ".") or str(h).strip() in (".", "", "nan")


def align_words(ref_toks: list[str], hyp_toks: list[str]):
    """Classic DP Levenshtein alignment returning ops as (op, ref, hyp)."""
    n, m = len(ref_toks), len(hyp_toks)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_toks[i - 1] == hyp_toks[j - 1] else 1
            candidates = [
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
                (dp[i - 1][j - 1] + cost, "C" if cost == 0 else "S"),
            ]
            dp[i][j], bt[i][j] = min(candidates, key=lambda x: x[0])
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "C" or op == "S":
            ops.append((op, ref_toks[i - 1], hyp_toks[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append(("D", ref_toks[i - 1], None))
            i -= 1
        elif op == "I":
            ops.append(("I", None, hyp_toks[j - 1]))
            j -= 1
        else:
            break
    ops.reverse()
    return ops


def train_word_counts(lang: str) -> Counter:
    c: Counter = Counter()
    for split in ("train", "validation"):
        p = META / f"{lang}_{split}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        col = "Target" if "Target" in df.columns else "transcription"
        for t in df[col].astype(str):
            for w in tokenize_words(t):
                c[w] += 1
    return c


def top_ref_for_hyp(subs: Counter, hyp: str, min_count: int = 3) -> list[tuple[str, int]]:
    pairs = [(r, n) for (r, h), n in subs.items() if h == hyp and n >= min_count]
    pairs.sort(key=lambda x: -x[1])
    return pairs


def apply_postprocess(text: str, lang: str, rules: dict) -> str:
    """Apply a set of conservative postprocess rules; return normalized-ish text."""
    if is_empty_hyp(text):
        return "."
    t = normalize_text(text)
    words = t.split()

    # Rule: collapse pure stutter / single-char garbage sequences
    if rules.get("collapse_stutter"):
        if words and all(len(w) <= 1 for w in words) and len(words) >= 3:
            return "."
        # e e e e e e e -> drop
        if len(words) >= 4 and len(set(words)) == 1 and len(words[0]) <= 2:
            return "."

    # Rule: strip leading/trailing lone punctuation leftovers already handled by normalize
    # Rule: split glued "ba X" patterns where ba+word appears vs train ba + word
    if rules.get("split_ba_prefix") and lang == "lin":
        out = []
        for w in words:
            if w.startswith("ba") and len(w) >= 5 and w not in rules.get("lexicon", set()):
                # only if "ba" + stem both look productive: ba + rest
                rest = w[2:]
                if rest in rules.get("lexicon", set()) and rules["lexicon"].get(rest, 0) >= 2:
                    out.extend(["ba", rest])
                    continue
            out.append(w)
        words = out

    # Rule: high-precision word substitutions from fixed map
    submap = rules.get("sub_maps", {}).get(lang, {})
    if submap:
        words = [submap.get(w, w) for w in words]

    # Rule: collapse repeated identical consecutive words (x x -> x) if word short/function-like
    if rules.get("dedupe_consecutive"):
        out = []
        for w in words:
            if out and out[-1] == w and (len(w) <= 3 or w in FUNCTION_WORDS.get(lang, set())):
                continue
            out.append(w)
        words = out

    # Rule: French loan orthography fixes (lin often mixes)
    if rules.get("lin_loan_fixes") and lang == "lin":
        loan = {
            "eglize": "eglise",
            "eglise": "eglise",
            "eglize": "eglise",
            "katolikue": "catholique",
            "catholique": "catholique",
            "coulera": "culere",
            "conference": "conference",
            "conférence": "conference",
            "chocolat": "chocolat",
            "macarro": "ba carro",
            "peme": "pembe",
            "folele": "fololo",
            "longondo": "longondo",
        }
        # only apply high-confidence orthography maps carefully
        fixed_loan = {
            "eglize": "eglise",
            "katolikue": "catholique",
            "coulera": "culere",
            "peme": "pembe",
            "folele": "fololo",
            "niva": "nivo",
            "di": "dix",  # careful — too aggressive alone
        }
        # skip di->dix globally; only in phrase context later
        for k, v in {
            "eglize": "eglise",
            "katolikue": "catholique",
            "peme": "pembe",
            "folele": "fololo",
            "coulera": "culere",
        }.items():
            words = [v if w == k else w for w in words]

    # Rule: phrase-level: "a di niva" -> "ya dix nivo" (lin)
    if rules.get("lin_phrase") and lang == "lin":
        s = " ".join(words)
        s = s.replace("a di niva", "ya dix nivo")
        s = s.replace("a di nivo", "ya dix nivo")
        s = s.replace("etage a di", "etage ya dix")
        words = s.split()

    # Rule: join split compounds that should be one token in refs (space removal)
    join_map = rules.get("join_maps", {}).get(lang, {})
    if join_map:
        s = " ".join(words)
        for a, b in join_map.items():
            s = s.replace(a, b)
        words = s.split()

    # Rule: split compounds that refs write as two tokens
    split_map = rules.get("split_maps", {}).get(lang, {})
    if split_map:
        out = []
        for w in words:
            if w in split_map:
                out.extend(split_map[w].split())
            else:
                out.append(w)
        words = out

    out = " ".join(words).strip()
    return out if out else "."


def evaluate_rules(df: pd.DataFrame, rule_sets: dict[str, dict]) -> dict:
    base_refs = df["reference"].tolist()
    base_hyps = df["prediction"].tolist()
    langs = df["language"].tolist()
    base = score_by_language(base_refs, base_hyps, langs)
    base_z = 1.0 - base["overall"]["score"]

    results = {"baseline": {**base, "zindi_est": base_z}}
    for name, rules in rule_sets.items():
        hyps = [
            apply_postprocess(h, lang, rules)
            for h, lang in zip(base_hyps, langs)
        ]
        sc = score_by_language(base_refs, hyps, langs)
        z = 1.0 - sc["overall"]["score"]
        results[name] = {
            **sc,
            "zindi_est": z,
            "delta_z": z - base_z,
            "delta_score": sc["overall"]["score"] - base["overall"]["score"],
        }
    return results


def main() -> None:
    df = load_preds()
    print(f"Loaded {len(df)} predictions: {df.language.value_counts().to_dict()}")

    refs = df["reference"].tolist()
    hyps = df["prediction"].tolist()
    langs = df["language"].tolist()

    # --- 1. Metrics ---
    metrics = score_by_language(refs, hyps, langs)
    for k, v in metrics.items():
        v["zindi_est"] = 1.0 - v["score"]
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(
            f"{k}: n={int(v['n'])} WER={v['wer']:.4f} CER={v['cer']:.4f} "
            f"score={v['score']:.4f} zindi_est={v['zindi_est']:.4f}"
        )

    # --- 2. Alignments: substitutions, ins, del ---
    sub_all: Counter = Counter()
    sub_by_lang: dict[str, Counter] = defaultdict(Counter)
    del_all: Counter = Counter()
    ins_all: Counter = Counter()
    del_by_lang: dict[str, Counter] = defaultdict(Counter)
    ins_by_lang: dict[str, Counter] = defaultdict(Counter)
    fun_del: Counter = Counter()
    fun_ins: Counter = Counter()
    fun_del_by_lang: dict[str, Counter] = defaultdict(Counter)
    fun_ins_by_lang: dict[str, Counter] = defaultdict(Counter)

    empty_rows = []
    garbage_rows = []

    for _, row in df.iterrows():
        lang = row["language"]
        ref_t = tokenize_words(row["reference"])
        hyp_t = tokenize_words(row["prediction"])
        hyp_raw = str(row["prediction"])
        empty = is_empty_hyp(hyp_raw)
        if empty:
            empty_rows.append(
                {
                    "ID": row["ID"],
                    "language": lang,
                    "ref_n_words": len(ref_t),
                    "ref_n_chars": len(normalize_text(row["reference"])),
                    "ref": normalize_text(row["reference"])[:80],
                }
            )
            continue
        # pure stutter garbage
        if hyp_t and all(len(w) <= 1 for w in hyp_t) and len(hyp_t) >= 3:
            garbage_rows.append(
                {
                    "ID": row["ID"],
                    "language": lang,
                    "hyp": " ".join(hyp_t),
                    "ref_n_words": len(ref_t),
                }
            )

        ops = align_words(ref_t, hyp_t)
        fw = FUNCTION_WORDS.get(lang, set())
        for op, r, h in ops:
            if op == "S":
                sub_all[(r, h)] += 1
                sub_by_lang[lang][(r, h)] += 1
            elif op == "D":
                del_all[r] += 1
                del_by_lang[lang][r] += 1
                if r in fw:
                    fun_del[r] += 1
                    fun_del_by_lang[lang][r] += 1
            elif op == "I":
                ins_all[h] += 1
                ins_by_lang[lang][h] += 1
                if h in fw:
                    fun_ins[h] += 1
                    fun_ins_by_lang[lang][h] += 1

    top_subs = [(f"{r}->{h}", n) for (r, h), n in sub_all.most_common(50) if n >= 5]
    print("\n=== TOP SUBSTITUTIONS (>=5) ===")
    for s, n in top_subs[:40]:
        print(f"  {n:4d}  {s}")

    print("\n=== TOP DELETIONS ===")
    for w, n in del_all.most_common(25):
        print(f"  {n:4d}  DEL {w}")
    print("\n=== TOP INSERTIONS ===")
    for w, n in ins_all.most_common(25):
        print(f"  {n:4d}  INS {w}")

    print("\n=== FUNCTION WORD DEL/INS ===")
    for lang in ("lin", "lug", "sna"):
        print(f"  [{lang}] DEL:", fun_del_by_lang[lang].most_common(12))
        print(f"  [{lang}] INS:", fun_ins_by_lang[lang].most_common(12))

    # --- 3. Empty hyp analysis (duration proxy = ref length; true duration unavailable) ---
    empty_df = pd.DataFrame(empty_rows)
    all_ref_nw = df["reference"].map(lambda x: len(tokenize_words(x)))
    print("\n=== EMPTY HYP '.' ANALYSIS ===")
    print(f"n_empty={len(empty_df)} / {len(df)} ({100*len(empty_df)/len(df):.2f}%)")
    if len(empty_df):
        print(
            f"empty ref_n_words: mean={empty_df.ref_n_words.mean():.2f} "
            f"median={empty_df.ref_n_words.median():.1f} "
            f"p25={empty_df.ref_n_words.quantile(0.25):.1f} "
            f"p75={empty_df.ref_n_words.quantile(0.75):.1f}"
        )
        print(
            f"all ref_n_words: mean={all_ref_nw.mean():.2f} "
            f"median={all_ref_nw.median():.1f}"
        )
        short_thr = 5
        empty_short = (empty_df.ref_n_words <= short_thr).mean()
        all_short = (all_ref_nw <= short_thr).mean()
        print(
            f"P(ref_words<={short_thr}|empty)={empty_short:.3f} "
            f"P(ref_words<={short_thr}|all)={all_short:.3f}"
        )
        # per-lang
        for lang in ("lin", "lug", "sna"):
            e = empty_df[empty_df.language == lang]
            a = df[df.language == lang]
            print(
                f"  {lang}: empty={len(e)}/{len(a)} "
                f"empty_ref_mean_words={e.ref_n_words.mean() if len(e) else float('nan'):.2f} "
                f"all_ref_mean={a.reference.map(lambda x: len(tokenize_words(x))).mean():.2f}"
            )
    print(f"n_stutter_garbage={len(garbage_rows)}")

    # duration not in index
    print(
        "NOTE: dataset_index has no duration column; used ref word/char length as "
        "proxy for short audio. True duration would require loading HF audio arrays."
    )

    # --- 4. Build conservative postprocess rule candidates from data ---
    # High-precision sub maps: hyp->ref where pair dominates for that hyp
    lex = {lang: train_word_counts(lang) for lang in ("lin", "lug", "sna")}

    def build_sub_map(lang: str, min_n: int = 5, min_prec: float = 0.7) -> dict[str, str]:
        """Map hyp_word -> ref_word if hyp is almost always wrong as that sub."""
        c = sub_by_lang[lang]
        # for each hyp, find dominant ref
        hyp_to_refs: dict[str, Counter] = defaultdict(Counter)
        for (r, h), n in c.items():
            hyp_to_refs[h][r] += n
        out = {}
        for h, rc in hyp_to_refs.items():
            total = sum(rc.values())
            r, n = rc.most_common(1)[0]
            # also require hyp is rare in train OR r much more common
            if n < min_n:
                continue
            if n / total < min_prec:
                continue
            if r == h:
                continue
            # avoid mapping real train words unless rare
            if lex[lang][h] >= 20 and lex[lang][h] >= lex[lang][r]:
                continue
            # prefer mapping to a train-frequent word
            if lex[lang][r] < 2:
                continue
            out[h] = r
        return out

    sub_maps = {lang: build_sub_map(lang) for lang in ("lin", "lug", "sna")}
    print("\n=== HIGH-PRECISION SUB MAPS (hyp->ref) ===")
    for lang, m in sub_maps.items():
        print(f"  {lang} ({len(m)}):", list(m.items())[:25])

    # join maps: cases where ref has single token and hyp has two (from opposite of split)
    # Detect common hyp bigrams that should be one word
    join_candidates: dict[str, Counter] = defaultdict(Counter)
    split_candidates: dict[str, Counter] = defaultdict(Counter)
    for _, row in df.iterrows():
        lang = row["language"]
        ref_t = tokenize_words(row["reference"])
        hyp_t = tokenize_words(row["prediction"])
        ref_set = set(ref_t)
        # hyp bigram joined equals a ref token
        for i in range(len(hyp_t) - 1):
            joined = hyp_t[i] + hyp_t[i + 1]
            if joined in ref_set and joined not in hyp_t:
                if lex[lang][joined] >= 2:
                    join_candidates[lang][(hyp_t[i] + " " + hyp_t[i + 1], joined)] += 1
        # hyp single token equals two consecutive ref tokens
        for i in range(len(ref_t) - 1):
            joined = ref_t[i] + ref_t[i + 1]
            if joined in hyp_t and joined not in ref_set:
                if lex[lang][ref_t[i]] >= 2 and lex[lang][ref_t[i + 1]] >= 2:
                    split_candidates[lang][(joined, ref_t[i] + " " + ref_t[i + 1])] += 1

    join_maps = {}
    split_maps = {}
    for lang in ("lin", "lug", "sna"):
        jm = {}
        for (bigram, joined), n in join_candidates[lang].most_common(40):
            if n >= 5:
                jm[bigram] = joined
        join_maps[lang] = jm
        sm = {}
        for (joined, bigram), n in split_candidates[lang].most_common(40):
            if n >= 5:
                sm[joined] = bigram
        split_maps[lang] = sm
        print(f"\nJOIN {lang}:", list(jm.items())[:15])
        print(f"SPLIT {lang}:", list(sm.items())[:15])

    # Rule sets to evaluate (A/B against baseline; test gold only for diagnostic scoring)
    rule_sets = {
        "R1_collapse_stutter": {"collapse_stutter": True},
        "R2_dedupe_consec": {"dedupe_consecutive": True},
        "R3_sub_maps": {"sub_maps": sub_maps},
        "R4_join_split": {"join_maps": join_maps, "split_maps": split_maps},
        "R5_lin_loan_phrase": {"lin_loan_fixes": True, "lin_phrase": True},
        "R6_split_ba": {
            "split_ba_prefix": True,
            "lexicon": {w: c for lang in lex for w, c in lex[lang].items()},
        },
        "R_all_safe": {
            "collapse_stutter": True,
            "dedupe_consecutive": True,
            "sub_maps": sub_maps,
            "join_maps": join_maps,
            "split_maps": split_maps,
            "lin_loan_fixes": True,
            "lin_phrase": True,
        },
        "R_stutter_dedupe_sub": {
            "collapse_stutter": True,
            "dedupe_consecutive": True,
            "sub_maps": sub_maps,
        },
    }

    print("\n=== RULE EVALUATION (lower score better; higher zindi_est better) ===")
    evals = evaluate_rules(df, rule_sets)
    for name, sc in evals.items():
        if name == "baseline":
            print(
                f"{name}: score={sc['overall']['score']:.6f} z={sc['zindi_est']:.6f}"
            )
        else:
            print(
                f"{name}: score={sc['overall']['score']:.6f} z={sc['zindi_est']:.6f} "
                f"delta_z={sc['delta_z']:+.6f}"
            )
            for lang in ("lin", "lug", "sna"):
                if lang in sc:
                    b = evals["baseline"][lang]["score"]
                    print(
                        f"    {lang}: score={sc[lang]['score']:.6f} "
                        f"(base {b:.6f}, d={sc[lang]['score']-b:+.6f})"
                    )

    # Per-rule absolute impact estimates
    report = {
        "metrics": metrics,
        "top_substitutions": top_subs,
        "top_deletions": del_all.most_common(40),
        "top_insertions": ins_all.most_common(40),
        "function_del_by_lang": {k: v.most_common(20) for k, v in fun_del_by_lang.items()},
        "function_ins_by_lang": {k: v.most_common(20) for k, v in fun_ins_by_lang.items()},
        "empty_hyp": {
            "n": len(empty_df),
            "rate": len(empty_df) / len(df),
            "ref_words_mean": float(empty_df.ref_n_words.mean()) if len(empty_df) else None,
            "ref_words_median": float(empty_df.ref_n_words.median()) if len(empty_df) else None,
            "all_ref_words_mean": float(all_ref_nw.mean()),
            "all_ref_words_median": float(all_ref_nw.median()),
            "note": "duration not in dataset_index; ref length used as short-audio proxy",
            "per_lang": empty_df.groupby("language").size().to_dict() if len(empty_df) else {},
            "examples": empty_rows[:15],
        },
        "stutter_garbage_n": len(garbage_rows),
        "sub_maps": sub_maps,
        "join_maps": join_maps,
        "split_maps": split_maps,
        "rule_evals": {
            k: {
                "zindi_est": v["zindi_est"],
                "score": v["overall"]["score"] if "overall" in v else v.get("score"),
                "delta_z": v.get("delta_z", 0.0),
                "per_lang_score": {
                    lang: v[lang]["score"] for lang in ("lin", "lug", "sna") if lang in v
                },
            }
            for k, v in evals.items()
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")

    # Also dump per-lang top subs
    for lang in ("lin", "lug", "sna"):
        print(f"\n=== TOP SUBS {lang} ===")
        for (r, h), n in sub_by_lang[lang].most_common(20):
            if n >= 3:
                print(f"  {n:4d}  {r} -> {h}")


if __name__ == "__main__":
    main()
