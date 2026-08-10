#!/usr/bin/env python3
"""Validation-only non-model decoding search for goal 2026-08-10.

The search reuses existing validation hypothesis tables. It never reads a
Phase2 label/transcript/audio file and never mutates an existing route cache.
Any promoted cache would have to be an exact-ID prediction-only source and
would be written only after the strict gate in ``evaluate_language`` passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jiwer import cer as raw_cer
from jiwer import wer as raw_wer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/goal_2026_08_10/decoding_search"
SEED = 20260810
BOOTSTRAP_DRAWS = 2000

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import score_pairs
from src.text_norm import normalize_text, tokenize_words


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jdump(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=jdump) + "\n", encoding="utf-8")


def norm_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(normalize_text)


def metric(refs: pd.Series | list[str], hyps: pd.Series | list[str]) -> dict[str, float]:
    result = score_pairs([str(x) for x in refs], [str(x) for x in hyps])
    result["zindi"] = 1.0 - float(result["score"])
    return {k: float(v) for k, v in result.items()}


def row_error(ref: str, hyp: str) -> float:
    return float(score_pairs([normalize_text(ref)], [normalize_text(hyp)])['score'])


def speaker_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    speakers = frame.speaker_id.astype(str)
    if speakers.eq("").any():
        raise RuntimeError("speaker-disjoint split requires speaker_id on every row")
    keyed = sorted((hashlib.sha256(f"{SEED}:split:{s}".encode()).hexdigest(), s) for s in speakers.unique())
    n_hold = max(1, min(len(keyed) - 1, round(len(keyed) * 0.5)))
    hold_speakers = {s for _, s in keyed[:n_hold]}
    hold = np.flatnonzero(speakers.isin(hold_speakers).to_numpy())
    tune = np.flatnonzero(~speakers.isin(hold_speakers).to_numpy())
    overlap = sorted(set(speakers.iloc[tune]) & set(speakers.iloc[hold]))
    if not len(tune) or not len(hold) or overlap:
        raise RuntimeError(f"invalid speaker split: {overlap}")
    return tune, hold, {
        "tune_rows": int(len(tune)),
        "holdout_rows": int(len(hold)),
        "tune_speakers": int(speakers.iloc[tune].nunique()),
        "holdout_speakers": int(speakers.iloc[hold].nunique()),
        "speaker_overlap": overlap,
        "rule": "sha256(seed:split:speaker_id), whole speakers assigned before rows",
    }


def oof_folds(frame: pd.DataFrame, n_folds: int = 5) -> list[np.ndarray]:
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for s in sorted(frame.speaker_id.astype(str).unique()):
        digest = int(hashlib.sha256(f"{SEED}:oof:{s}".encode()).hexdigest()[:12], 16)
        buckets[digest % n_folds].append(s)
    folds = [np.flatnonzero(frame.speaker_id.astype(str).isin(b).to_numpy()) for b in buckets]
    if any(len(x) == 0 for x in folds):
        raise RuntimeError("empty speaker OOF fold")
    return folds


def speaker_bootstrap(frame: pd.DataFrame, base: pd.Series, cand: pd.Series, indices: np.ndarray) -> dict[str, Any]:
    sub = frame.iloc[indices].copy()
    b = base.iloc[indices].reset_index(drop=True)
    c = cand.iloc[indices].reset_index(drop=True)
    refs = sub.reference.astype(str).reset_index(drop=True)
    groups = {s: np.flatnonzero(sub.speaker_id.astype(str).to_numpy() == s) for s in sorted(sub.speaker_id.astype(str).unique())}
    speakers = list(groups)
    rng = np.random.default_rng(SEED + len(indices))
    # The challenge score is additive over utterance-level WER/CER edit
    # contributions. Precompute paired row deltas so the bootstrap remains
    # exactly paired and speaker-clustered without re-running jiwer 2,000 x.
    row_delta = np.asarray([row_error(r, bb) - row_error(r, cc) for r, bb, cc in zip(refs, b, c)], dtype=float)
    group_values = {s: row_delta[idx] for s, idx in groups.items()}
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for i in range(BOOTSTRAP_DRAWS):
        selected = rng.choice(speakers, size=len(speakers), replace=True)
        values[i] = float(np.mean(np.concatenate([group_values[s] for s in selected])))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "delta_mean": float(values.mean()),
        "delta_p05": float(np.quantile(values, 0.05)),
        "delta_p50": float(np.quantile(values, 0.50)),
        "delta_p95": float(np.quantile(values, 0.95)),
        "probability_delta_positive": float(np.mean(values > 0.0)),
        "resampling_unit": "speaker",
    }


def load_vocab(lang: str) -> set[str]:
    for path in (ROOT / "data/lms_phase2_domain" / f"{lang}_counts.json", ROOT / "data/lms" / f"{lang}_counts.json"):
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            uni = payload.get("uni", {}) if isinstance(payload, dict) else {}
            return {normalize_text(str(x)) for x in uni if normalize_text(str(x))}
    return set()


def oov(text: str, lexicon: set[str]) -> float:
    toks = tokenize_words(text)
    return float(sum(t not in lexicon for t in toks) / len(toks)) if toks and lexicon else 0.0


def edit_ratio(left: str, right: str) -> float:
    a, b = normalize_text(left), normalize_text(right)
    # Lightweight normalized character edit distance; only a deployment feature.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return float(prev[-1] / max(1, len(a), len(b)))


def guarded(frame: pd.DataFrame, base: str, alt: str, config: dict[str, Any], lexicon: set[str]) -> pd.Series:
    b = frame[base].fillna("").astype(str)
    a = frame[alt].fillna("").astype(str)
    bw = b.map(lambda x: len(tokenize_words(x)))
    aw = a.map(lambda x: len(tokenize_words(x)))
    ratio = (aw + 1e-6) / (bw + 1e-6)
    gate = ratio.between(float(config["length_lo"]), float(config["length_hi"]))
    gate &= a.map(lambda x: oov(x, lexicon)) <= float(config["oov_max"])
    if config.get("edit_max") is not None:
        gate &= pd.Series([edit_ratio(x, y) for x, y in zip(b, a)], index=frame.index) <= float(config["edit_max"])
    if config.get("confidence_col") and config.get("confidence_min") is not None:
        conf = pd.to_numeric(frame[config["confidence_col"]], errors="coerce").fillna(-np.inf)
        gate &= conf >= float(config["confidence_min"])
    if config.get("duration_col") and config.get("rate_lo") is not None:
        dur = pd.to_numeric(frame[config["duration_col"]], errors="coerce").replace(0, np.nan)
        rate = aw / dur
        gate &= rate.between(float(config["rate_lo"]), float(config["rate_hi"]))
    return a.where(gate, b)


def apply(frame: pd.DataFrame, config: dict[str, Any], lexicon: set[str]) -> pd.Series:
    if config["kind"] == "unconditional":
        return frame[config["policy"]].fillna("").astype(str)
    return guarded(frame, config["base"], config["alt"], config, lexicon)


def names_for(frame: pd.DataFrame, current: str, excluded: set[str]) -> list[str]:
    names = []
    for col in frame.columns:
        low = col.lower()
        if col in excluded or col.startswith("_") or col.endswith("_raw"):
            continue
        if any(token in low for token in ("reference", "target", "oracle", "corrected", "original")):
            continue
        if col.startswith("feature_") or col.endswith("_source") or col.endswith("_reasons"):
            continue
        if any(token in low for token in ("duration", "mean_logprob", "mean_margin", "mean_entropy", "blank_ratio", "frames", "probability", "outer_fold", "anomaly_guard", "gender")):
            continue
        vals = frame[col].fillna("").astype(str)
        numeric = pd.to_numeric(vals, errors="coerce")
        if float(numeric.notna().mean()) >= 0.95:
            continue
        if float((vals.str.strip() != "").mean()) >= 0.98:
            names.append(col)
    if current not in names:
        raise RuntimeError(f"incumbent {current} absent")
    return names


def configs(frame: pd.DataFrame, names: list[str], current: str, train: np.ndarray, lexicon: set[str], confidence_col: str | None, duration_col: str | None) -> list[dict[str, Any]]:
    ranked = []
    for name in names:
        ranked.append((metric(frame.reference.iloc[train], frame[name].iloc[train])['score'], name))
    ranked.sort()
    alts = [name for _, name in ranked if name != current][:6]
    out: list[dict[str, Any]] = [{"kind": "unconditional", "policy": name} for name in names]
    # These grids cover beam/lexicon/fusion guards without using references as deployment features.
    for alt in alts:
        for lo, hi in ((0.85, 1.40), (0.90, 1.20)):
            for oov_max in (0.35, 0.50):
                for edit_max in (None,):
                    cfg = {"kind": "guarded_mixture", "base": current, "alt": alt, "length_lo": lo, "length_hi": hi, "oov_max": oov_max, "edit_max": edit_max, "confidence_col": confidence_col, "duration_col": duration_col, "confidence_min": None, "rate_lo": None, "rate_hi": None}
                    out.append(cfg)
                    if confidence_col and confidence_col in frame:
                        vals = pd.to_numeric(frame[confidence_col], errors="coerce").dropna()
                        for q in (0.50,):
                            c = dict(cfg)
                            c["confidence_min"] = float(vals.quantile(q))
                            out.append(c)
                    if duration_col and duration_col in frame:
                        dur = pd.to_numeric(frame[duration_col], errors="coerce").replace(0, np.nan)
                        rates = frame[alt].fillna("").astype(str).map(lambda x: len(tokenize_words(x))) / dur
                        rates = rates.replace([np.inf, -np.inf], np.nan).dropna()
                        if len(rates):
                            for loq, hiq in ((0.10, 0.90),):
                                d = dict(cfg)
                                d["rate_lo"], d["rate_hi"] = float(rates.quantile(loq)), float(rates.quantile(hiq))
                                out.append(d)
    return out


def config_name(cfg: dict[str, Any]) -> str:
    if cfg["kind"] == "unconditional":
        return str(cfg["policy"])
    c = "none" if cfg.get("confidence_min") is None else f"{cfg['confidence_min']:.5g}"
    d = "none" if cfg.get("rate_lo") is None else f"{cfg['rate_lo']:.3g}-{cfg['rate_hi']:.3g}"
    e = "none" if cfg.get("edit_max") is None else f"{cfg['edit_max']:.2f}"
    return f"guard__{cfg['alt']}__len{cfg['length_lo']:.2f}-{cfg['length_hi']:.2f}__oov{cfg['oov_max']:.2f}__edit{e}__conf{c}__rate{d}"


def select(frame: pd.DataFrame, cfgs: list[dict[str, Any]], indices: np.ndarray, lexicon: set[str]) -> tuple[dict[str, Any], pd.Series, dict[str, float]]:
    best = None
    for cfg in cfgs:
        hyp = apply(frame, cfg, lexicon)
        m = metric(frame.reference.iloc[indices], hyp.iloc[indices])
        key = (m["score"], config_name(cfg))
        if best is None or key < best[0]:
            best = (key, cfg, hyp, m)
    assert best is not None
    return best[1], best[2], best[3]


def evaluation(lang: str, frame: pd.DataFrame, current: str, lexicon: set[str], confidence_col: str | None, duration_col: str | None, source_paths: list[Path]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    tune, hold, split = speaker_split(frame)
    names = names_for(frame, current, {"ID", "speaker_id", "reference", "gender", "fold", "language", "split"})
    cfgs = configs(frame, names, current, tune, lexicon, confidence_col, duration_col)
    selected, selected_hyp, tune_m = select(frame, cfgs, tune, lexicon)
    base_hyp = frame[current].fillna("").astype(str)
    hold_m = metric(frame.reference.iloc[hold], selected_hyp.iloc[hold])
    full_m = metric(frame.reference, selected_hyp)
    base_hold = metric(frame.reference.iloc[hold], base_hyp.iloc[hold])
    base_full = metric(frame.reference, base_hyp)
    boot = speaker_bootstrap(frame, base_hyp, selected_hyp, hold)

    oof_rows = []
    for fold, test_idx in enumerate(oof_folds(frame)):
        train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
        fold_cfgs = configs(frame, names, current, train_idx, lexicon, confidence_col, duration_col)
        fold_cfg, fold_hyp, _ = select(frame, fold_cfgs, train_idx, lexicon)
        cm = metric(frame.reference.iloc[test_idx], fold_hyp.iloc[test_idx])
        bm = metric(frame.reference.iloc[test_idx], base_hyp.iloc[test_idx])
        train_s = set(frame.speaker_id.iloc[train_idx].astype(str))
        test_s = set(frame.speaker_id.iloc[test_idx].astype(str))
        oof_rows.append({"language": lang, "fold": fold, "policy": config_name(fold_cfg), "baseline": current, "n_train": len(train_idx), "n_test": len(test_idx), "train_speakers": len(train_s), "test_speakers": len(test_s), "speaker_overlap": len(train_s & test_s), "candidate_score": cm["score"], "baseline_score": bm["score"], "improvement": bm["score"] - cm["score"], "candidate_wer": cm["wer"], "baseline_wer": bm["wer"], "candidate_cer": cm["cer"], "baseline_cer": bm["cer"]})
    oof = pd.DataFrame(oof_rows)
    pass_checks = {
        "novel_policy": config_name(selected) != current,
        "holdout_gain_at_least_0p005": base_hold["score"] - hold_m["score"] >= 0.005,
        "full_gain_at_least_0p01": base_full["score"] - full_m["score"] >= 0.01,
        "holdout_wer_strictly_better": hold_m["wer"] < base_hold["wer"],
        "holdout_cer_non_regression": hold_m["cer"] <= base_hold["cer"],
        "oof_all_folds_positive": bool((oof.improvement > 0.0).all()),
        "oof_mean_positive": float(oof.improvement.mean()) > 0.0,
        "bootstrap_p05_positive": boot["delta_p05"] > 0.0,
        "bootstrap_probability_at_least_0p95": boot["probability_delta_positive"] >= 0.95,
    }
    result = {
        "language": lang,
        "n": len(frame),
        "speaker_count": int(frame.speaker_id.nunique()),
        "candidate_count": len(names),
        "candidate_names": names,
        "current_route": current,
        "selected_policy": config_name(selected),
        "selected_config": selected,
        "split": split,
        "tune": tune_m,
        "holdout": hold_m,
        "full": full_m,
        "baseline_holdout": base_hold,
        "baseline_full": base_full,
        "holdout_gain": float(base_hold["score"] - hold_m["score"]),
        "full_gain": float(base_full["score"] - full_m["score"]),
        "paired_speaker_bootstrap": boot,
        "oof": {"folds": len(oof), "mean_improvement": float(oof.improvement.mean()), "min_improvement": float(oof.improvement.min()), "all_folds_positive": bool((oof.improvement > 0).all())},
        "pass_checks": pass_checks,
        "strict_pass": bool(all(pass_checks.values())),
        "gate": "novel; holdout gain>=0.005; full gain>=0.01; holdout WER strict better; CER non-regression; all five speaker OOF folds positive; paired speaker bootstrap p05>0 and P(delta>0)>=0.95",
        "source_paths": [str(p.relative_to(ROOT)) for p in source_paths],
    }
    score_rows = []
    for cfg in cfgs:
        hyp = apply(frame, cfg, lexicon)
        tm = metric(frame.reference.iloc[tune], hyp.iloc[tune])
        hm = metric(frame.reference.iloc[hold], hyp.iloc[hold])
        fm = metric(frame.reference, hyp)
        score_rows.append({"language": lang, "policy": config_name(cfg), "kind": cfg["kind"], "tune_zindi": tm["zindi"], "holdout_zindi": hm["zindi"], "full_zindi": fm["zindi"], "tune_wer": tm["wer"], "holdout_wer": hm["wer"], "full_wer": fm["wer"], "tune_cer": tm["cer"], "holdout_cer": hm["cer"], "full_cer": fm["cer"]})
    return result, pd.DataFrame(score_rows).sort_values(["tune_wer", "policy"]), oof


def raw_formatting(lang: str, source_paths: list[Path]) -> dict[str, Any] | None:
    if lang not in {"lin", "sna"}:
        return None
    path = ROOT / "outputs/goal_2026_08_08/scoring_forensics" / f"matched_forensics_{lang}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw_refs, raw_hyps = df.raw_reference.tolist(), df.raw_hypothesis.tolist()
    normalized_refs = [normalize_text(x) for x in raw_refs]
    normalized_hyps = [normalize_text(x) for x in raw_hyps]
    challenge = metric(normalized_refs, normalized_hyps)
    # Diagnostic only: official local scoring normalizes, while this records the
    # case/punctuation cost if a raw-sensitive scorer were used.
    def raw_safe(fn, refs, hyps):
        rr = [unicodedata.normalize("NFKC", str(x)) or " " for x in refs]
        hh = [unicodedata.normalize("NFKC", str(x)) for x in hyps]
        return float(fn(rr, hh))
    return {
        "language": lang,
        "rows": len(df),
        "challenge_normalized": challenge,
        "raw_sensitive_diagnostic": {"wer": raw_safe(raw_wer, raw_refs, raw_hyps), "cer": raw_safe(raw_cer, raw_refs, raw_hyps), "note": "diagnostic only; not the promotion metric"},
        "raw_to_normalized_changed_rows": int(sum(a != b for a, b in zip(raw_hyps, normalized_hyps))),
        "source": str(path.relative_to(ROOT)),
    }


def load_data() -> tuple[dict[str, tuple[pd.DataFrame, str, list[Path], str | None, str | None]], dict[str, Any]]:
    data: dict[str, tuple[pd.DataFrame, str, list[Path], str | None, str | None]] = {}
    audits: dict[str, Any] = {}
    # Lingala and Shona use the existing paired specialist validation tables.
    for lang, table, meta, current in [
        ("lin", "outputs/goal_2026_08_08/sulaiman_public_descendants/validation_w2vbert-lingala-sd3.csv", "data/hf_metadata/lin_validation.parquet", "candidate"),
        ("sna", "outputs/goal_2026_08_08/shona_sd2_parallel/validation_w2vbert-shona-sd2.csv", "data/hf_metadata/sna_validation.parquet", "candidate"),
    ]:
        df = pd.read_csv(ROOT / table, dtype=str, keep_default_na=False)
        m = pd.read_parquet(ROOT / meta, columns=["ID", "speaker_id"])
        df = df.merge(m, on="ID", how="inner", validate="one_to_one")
        df["reference"] = norm_series(df["reference"])
        df["candidate"] = norm_series(df["candidate"])
        df["incumbent"] = norm_series(df["incumbent"])
        raw = pd.read_csv(ROOT / "outputs/goal_2026_08_08/scoring_forensics" / f"raw_hypotheses_{lang}.csv", dtype=str, keep_default_na=False)
        df = df.merge(raw.rename(columns={"raw_hypothesis": "candidate_raw"}), on="ID", how="left", validate="one_to_one")
        df["candidate_raw_norm"] = norm_series(df["candidate_raw"])
        data[lang] = (df, current, [ROOT / table, ROOT / meta, ROOT / "outputs/goal_2026_08_08/scoring_forensics" / f"raw_hypotheses_{lang}.csv"], None, None)
        audits[lang] = {"rows": len(df), "speakers": int(df.speaker_id.nunique()), "speaker_overlap_in_source_fold": int(len(set(df.loc[df.fold == 'tune', 'speaker_id']) & set(df.loc[df.fold == 'holdout', 'speaker_id']))) if 'fold' in df else None}

    lug_table = ROOT / "outputs/goal_2026_08_08/luganda_fusion/matched_hypotheses.csv"
    lug = pd.read_csv(lug_table, dtype=str, keep_default_na=False)
    lug["reference"] = norm_series(lug["corrected_reference"])
    meta = pd.read_parquet(ROOT / "data/hf_metadata/lug_validation.parquet", columns=["ID", "speaker_id"])
    lug = lug.merge(meta, on="ID", how="inner", validate="one_to_one")
    raw = pd.read_csv(ROOT / "outputs/goal_2026_08_08/luganda_fusion/hyps_mms_ft_v3.csv", dtype=str, keep_default_na=False)
    raw = raw[["ID", "mean_logprob", "n_frames"]].rename(columns={"mean_logprob": "confidence", "n_frames": "duration_frames"})
    lug = lug.merge(raw, on="ID", how="left", validate="one_to_one")
    excluded = {"ID", "original_reference", "corrected_reference", "reference", "speaker_id", "confidence", "duration_frames"}
    for col in lug.columns:
        if col not in excluded and lug[col].dtype == object:
            lug[col] = norm_series(lug[col])
    data["lug"] = (lug, "mms_ft_v3_domain_beam_splitjoin", [lug_table, ROOT / "data/hf_metadata/lug_validation.parquet", ROOT / "outputs/goal_2026_08_08/luganda_fusion/hyps_mms_ft_v3.csv"], "confidence", "duration_frames")
    audits["lug"] = {"rows": len(lug), "speakers": int(lug.speaker_id.nunique())}

    nyn_table = ROOT / "outputs/goal_2026_08_08/nyn_ensemble/same_id_hypotheses_and_cv.csv"
    nyn = pd.read_csv(nyn_table, dtype=str, keep_default_na=False)
    nyn["reference"] = norm_series(nyn["reference"])
    for col in nyn.columns:
        if col not in {"ID", "speaker_id", "gender", "reference", "duration_sec", "prod_mean_logp", "prod_mean_margin", "prod_mean_entropy", "prod_blank_ratio", "prod_frames", "outer_fold", "anomaly_guard_pass"} and nyn[col].dtype == object:
            if not col.startswith("feature_") and not col.endswith("_reasons"):
                nyn[col] = norm_series(nyn[col])
    data["nyn"] = (nyn, "nested_selector_hypothesis", [nyn_table], "prod_mean_logp", "duration_sec")
    audits["nyn"] = {"rows": len(nyn), "speakers": int(nyn.speaker_id.nunique())}
    return data, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, audits = load_data()
    results: dict[str, Any] = {}
    formatting: dict[str, Any] = {}
    input_paths: set[Path] = set()
    route_labels = {"lin": "w2vbert-lingala-sd3", "sna": "w2vbert-shona-sd2", "lug": "mms_ft_v3_domain_beam_splitjoin", "nyn": "nested_selector_hypothesis"}
    for lang in ("lin", "sna", "lug", "nyn"):
        frame, current, paths, conf, duration = data[lang]
        input_paths.update(paths)
        result, score_table, oof = evaluation(lang, frame, current, load_vocab(lang), conf, duration, paths)
        result["incumbent_route_label"] = route_labels[lang]
        result["incumbent_column"] = current
        results[lang] = result
        score_table.to_csv(out / f"{lang}_policy_scores.csv", index=False)
        oof.to_csv(out / f"{lang}_oof_scores.csv", index=False)
        fmt = raw_formatting(lang, paths)
        if fmt:
            formatting[lang] = fmt
    manifest = {
        "goal": "Phase2 decoding search 2026-08-10",
        "protocol": {"validation_only": True, "phase2_labels_or_transcripts_used": False, "phase2_audio_used": False, "uploads_performed": False, "existing_candidates_modified": False, "selection": "speaker-disjoint tune/holdout plus five-fold speaker-disjoint OOF", "bootstrap": {"draws": BOOTSTRAP_DRAWS, "unit": "speaker", "seed": SEED}, "promotion": results and "strict robust gate described per-language"},
        "inventory": audits,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(input_paths) if p.exists()},
        "results": results,
        "raw_vs_normalized": formatting,
        "route_caches_written": [],
        "note": "No route cache is emitted by this run unless a novel policy passes every strict gate and an exact-ID prediction-only source is available; validation hypothesis tables alone are not treated as Phase2 cache sources.",
    }
    write_json(out / "manifest.json", manifest)
    lines = ["# Decoding-search audit", "", "Validation-only, non-model search over existing hypothesis tables for Lingala, Shona, Luganda, and Nyanja.", "", "No Phase2 labels/transcripts/audio were read; no upload was attempted; no existing candidate was modified.", "", "| Language | Incumbent | Selected policy | Full gain | Holdout gain | OOF min gain | Bootstrap p05 | Strict pass |", "|---|---|---|---:|---:|---:|---:|---|"]
    for lang in ("lin", "sna", "lug", "nyn"):
        r = results[lang]
        lines.append(f"| {lang} | `{r['incumbent_route_label']}` | `{r['selected_policy']}` | {r['full_gain']:+.6f} | {r['holdout_gain']:+.6f} | {r['oof']['min_improvement']:+.6f} | {r['paired_speaker_bootstrap']['delta_p05']:+.6f} | {r['strict_pass']} |")
    lines += ["", "No route cache was written. See `manifest.json`, per-language `*_policy_scores.csv`, and `*_oof_scores.csv`.", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    rejection = [
        "# Decoding-search promotion decision",
        "",
        "**Rejected: no route promotion and no route cache written.**",
        "",
        "The audit used existing validation hypothesis tables only. Phase2 labels/transcripts/audio were not read, uploads were not attempted, and existing caches/candidates were not modified.",
        "",
        "A candidate had to clear every locked gate: novel policy, holdout score gain >= 0.005, full score gain >= 0.01, strict holdout WER improvement, CER non-regression, positive gain on all five speaker-disjoint OOF folds, and paired speaker-bootstrap P(delta > 0) >= 0.95 with 5th percentile > 0.",
        "",
    ]
    for lang in ("lin", "sna", "lug", "nyn"):
        r = results[lang]
        failed = [k for k, v in r["pass_checks"].items() if not v]
        rejection += [f"## {lang} (`{r['incumbent_route_label']}`)", "", f"Selected `{r['selected_policy']}`; strict pass: **{r['strict_pass']}**.", f"Full gain `{r['full_gain']:+.6f}`, holdout gain `{r['holdout_gain']:+.6f}`, OOF minimum `{r['oof']['min_improvement']:+.6f}`, bootstrap p05 `{r['paired_speaker_bootstrap']['delta_p05']:+.6f}`, P(delta>0) `{r['paired_speaker_bootstrap']['probability_delta_positive']:.4f}`.", f"Failed gates: {', '.join(failed) if failed else 'none'}.", ""]
    rejection += ["No exact-ID prediction-only source for a novel non-model policy was eligible for cache emission.", ""]
    (out / "REJECTION.md").write_text("\n".join(rejection), encoding="utf-8")


if __name__ == "__main__":
    main()
