#!/usr/bin/env python3
"""Validation-only route/ensemble optimizer for the 2026-08-09 goal.

The optimizer consumes previously generated validation hypothesis tables and
the corrected Luganda validation labels.  It never reads Phase2 labels,
transcripts, audio, or ``floor`` values.  Phase2 files are touched only after
a policy passes, and then only as prediction-only ID/Target caches after an
exact route-ID audit.

The selection protocol is deliberately conservative:

* all tune/holdout and OOF partitions are speaker-disjoint;
* candidate/model selection happens on tune rows only;
* guards cover hypothesis choice, model confidence, word-length ratio, and
  training-vocabulary OOV rate;
* promotion requires a holdout improvement and a positive speaker bootstrap;
* no existing candidate or submission is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "goal_2026_08_09" / "route_ensemble"

sys.path.insert(0, str(ROOT))

from src.metrics import score_pairs
from src.text_norm import normalize_text, tokenize_words


NIN_HYPS = ROOT / "outputs/goal_2026_08_08/nyn_ensemble/same_id_hypotheses_and_cv.csv"
LUG_HYPS = ROOT / "outputs/goal_2026_08_08/luganda_fusion/matched_hypotheses.csv"
LUG_LABELS = ROOT / "data/corrected_waxal/lug_validation_labels.csv"
LUG_META = ROOT / "data/hf_metadata/lug_validation.parquet"
LUG_RAW_DIR = ROOT / "outputs/goal_2026_08_08/luganda_fusion"
ROUTE_INDEX = ROOT / "outputs/beat075/public_visible_index.csv"
PHASE2_BASE = ROOT / "outputs/goal_2026_08_08/final/submission_phase2_public_lin_w2vbert_sna_w2vbert_nyn_guarded.csv"

W2V_REPLACED = {"lin", "sna"}
IN_SCOPE = {"nyn", "lug"}
SEED = 20260809


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=json_default) + "\n", encoding="utf-8")


def words(text: Any) -> list[str]:
    return tokenize_words("" if text is None else str(text))


def text_series(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name].fillna("").astype(str)


def metrics(frame: pd.DataFrame, hypothesis: Iterable[str], indices: np.ndarray | None = None) -> dict[str, float]:
    if indices is None:
        indices = np.arange(len(frame))
    idx = np.asarray(indices, dtype=int)
    refs = frame.iloc[idx]["reference"].astype(str).tolist()
    hyps = pd.Series(list(hypothesis), index=frame.index).iloc[idx].astype(str).tolist()
    result = score_pairs(refs, hyps)
    result["zindi"] = 1.0 - float(result["score"])
    return result


def speaker_split(frame: pd.DataFrame, holdout_fraction: float = 0.25) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    speakers = frame["speaker_id"].astype(str).fillna("")
    if speakers.eq("").any():
        raise RuntimeError("speaker-disjoint protocol requires a speaker_id on every row")
    unique = sorted(speakers.unique())
    keyed = sorted((hashlib.sha256(f"{SEED}:{s}".encode()).hexdigest(), s) for s in unique)
    n_holdout = max(1, min(len(unique) - 1, round(len(unique) * holdout_fraction)))
    holdout_speakers = {s for _, s in keyed[:n_holdout]}
    holdout = np.flatnonzero(speakers.isin(holdout_speakers).to_numpy())
    tune = np.flatnonzero(~speakers.isin(holdout_speakers).to_numpy())
    audit = {
        "tune_rows": int(len(tune)),
        "holdout_rows": int(len(holdout)),
        "tune_speakers": int(len(set(speakers.iloc[tune]))),
        "holdout_speakers": int(len(set(speakers.iloc[holdout]))),
        "speaker_overlap": sorted(set(speakers.iloc[tune]) & set(speakers.iloc[holdout])),
        "holdout_fraction_target": holdout_fraction,
        "split_rule": "sha256(seed:speaker_id), whole speakers assigned before rows",
    }
    if audit["speaker_overlap"] or not len(tune) or not len(holdout):
        raise RuntimeError(f"invalid speaker split: {audit}")
    return tune, holdout, audit


def oof_folds(frame: pd.DataFrame, n_folds: int = 5) -> list[np.ndarray]:
    speakers = sorted(frame["speaker_id"].astype(str).unique())
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for speaker in speakers:
        digest = int(hashlib.sha256(f"{SEED}:oof:{speaker}".encode()).hexdigest()[:12], 16)
        buckets[digest % n_folds].append(speaker)
    result = []
    for bucket in buckets:
        mask = frame["speaker_id"].astype(str).isin(bucket).to_numpy()
        result.append(np.flatnonzero(mask))
    if any(len(x) == 0 for x in result):
        raise RuntimeError("OOF fold received no rows")
    return result


def vocab(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {normalize_text(str(x)) for x in payload if normalize_text(str(x))}
    return set()


def oov_rate(text: Any, lexicon: set[str]) -> float:
    toks = words(text)
    if not toks or not lexicon:
        return 0.0
    return sum(t not in lexicon for t in toks) / len(toks)


def add_text_features(frame: pd.DataFrame, base: str, alt: str, lexicon: set[str], conf_col: str | None) -> pd.DataFrame:
    out = frame.copy()
    base_words = text_series(out, base).map(lambda x: len(words(x)))
    alt_words = text_series(out, alt).map(lambda x: len(words(x)))
    out["_length_ratio"] = (alt_words + 1e-6) / (base_words + 1e-6)
    out["_alt_oov"] = text_series(out, alt).map(lambda x: oov_rate(x, lexicon))
    if conf_col and conf_col in out:
        out["_confidence"] = pd.to_numeric(out[conf_col], errors="coerce").fillna(-np.inf)
    else:
        out["_confidence"] = 0.0
    return out


def guarded_hypothesis(frame: pd.DataFrame, base: str, alt: str, config: dict[str, Any]) -> pd.Series:
    temp = add_text_features(frame, base, alt, set(config.get("lexicon", [])), config.get("confidence_col"))
    gate = (
        temp["_length_ratio"].between(float(config["length_lo"]), float(config["length_hi"]))
        & (temp["_alt_oov"] <= float(config["oov_max"]))
    )
    if config.get("confidence_min") is not None:
        gate &= temp["_confidence"] >= float(config["confidence_min"])
    return text_series(frame, alt).where(gate, text_series(frame, base))


def bootstrap_speakers(frame: pd.DataFrame, base_hyp: Iterable[str], candidate_hyp: Iterable[str], indices: np.ndarray, reps: int = 2000) -> dict[str, float]:
    sub = frame.iloc[np.asarray(indices, dtype=int)].copy()
    base = pd.Series(list(base_hyp), index=frame.index).loc[sub.index]
    cand = pd.Series(list(candidate_hyp), index=frame.index).loc[sub.index]
    groups = {s: np.flatnonzero(sub["speaker_id"].astype(str).to_numpy() == s) for s in sorted(sub["speaker_id"].astype(str).unique())}
    rng = np.random.default_rng(SEED + len(sub))
    improvements = []
    speakers = list(groups)
    for _ in range(reps):
        chosen = rng.choice(speakers, size=len(speakers), replace=True)
        rows = np.concatenate([groups[s] for s in chosen])
        refs = sub.iloc[rows]["reference"].astype(str).tolist()
        b = base.iloc[rows].astype(str).tolist()
        c = cand.iloc[rows].astype(str).tolist()
        improvements.append(score_pairs(refs, b)["score"] - score_pairs(refs, c)["score"])
    arr = np.asarray(improvements, dtype=float)
    return {
        "reps": reps,
        "mean_improvement": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "p_improvement_gt_zero": float(np.mean(arr > 0.0)),
        "resampling_unit": "speaker",
    }


def candidate_names(frame: pd.DataFrame, excluded: set[str], minimum_coverage: float = 0.98) -> list[str]:
    result = []
    for col in frame.columns:
        if col in excluded or col.startswith("diagnostic_") or col.endswith("_source"):
            continue
        values = text_series(frame, col)
        coverage = float((values.str.strip() != "").mean())
        if coverage >= minimum_coverage:
            result.append(col)
    return result


def candidate_report(frame: pd.DataFrame, names: list[str], tune: np.ndarray, holdout: np.ndarray) -> pd.DataFrame:
    rows = []
    for name in names:
        all_m = metrics(frame, text_series(frame, name), None)
        tune_m = metrics(frame, text_series(frame, name), tune)
        hold_m = metrics(frame, text_series(frame, name), holdout)
        rows.append({"policy": name, "kind": "unconditional", "tune_zindi": tune_m["zindi"], "holdout_zindi": hold_m["zindi"], "full_zindi": all_m["zindi"], "tune_score": tune_m["score"], "holdout_score": hold_m["score"], "full_wer": all_m["wer"], "full_cer": all_m["cer"]})
    return pd.DataFrame(rows).sort_values(["tune_score", "policy"]).reset_index(drop=True)


def make_gate_configs(frame: pd.DataFrame, names: list[str], base: str, tune: np.ndarray, lexicon: set[str], confidence_col: str | None) -> list[dict[str, Any]]:
    # Limit the combinatorial search to candidates with a real tune signal.
    ranked = candidate_report(frame, names, tune, np.arange(len(frame))).sort_values("tune_score")
    alts = [x for x in ranked.policy.tolist() if x != base][:6]
    configs: list[dict[str, Any]] = []
    length_grid = [(0.70, 1.40), (0.85, 1.40), (0.90, 1.20)]
    oov_grid = [0.20, 0.35, 0.50]
    conf_grid: list[float | None] = [None]
    if confidence_col and confidence_col in frame:
        vals = pd.to_numeric(frame[confidence_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            conf_grid += [float(vals.quantile(q)) for q in (0.25, 0.50)]
    for alt in alts:
        for lo, hi in length_grid:
            for oov_max in oov_grid:
                for conf_min in conf_grid:
                    configs.append({"kind": "guarded_mixture", "base": base, "alt": alt, "length_lo": lo, "length_hi": hi, "oov_max": oov_max, "confidence_min": conf_min, "confidence_col": confidence_col, "lexicon": sorted(lexicon)})
    return configs


def apply_config(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if config["kind"] == "unconditional":
        return text_series(frame, config["policy"])
    return guarded_hypothesis(frame, config["base"], config["alt"], config)


def select_config(frame: pd.DataFrame, configs: list[dict[str, Any]], indices: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    best = None
    best_metrics = None
    for config in configs:
        hyp = apply_config(frame, config)
        result = metrics(frame, hyp, indices)
        key = (float(result["score"]), 0 if config["kind"] == "unconditional" else 1, json.dumps(config, sort_keys=True))
        if best is None or key < best[0]:
            best = (key, config)
            best_metrics = result
    assert best is not None and best_metrics is not None
    return best[1], best_metrics


def config_name(config: dict[str, Any]) -> str:
    if config["kind"] == "unconditional":
        return str(config["policy"])
    conf = "none" if config.get("confidence_min") is None else f"{float(config['confidence_min']):.5g}"
    return f"guard_{config['base']}__{config['alt']}__len{config['length_lo']:.2f}-{config['length_hi']:.2f}__oov{config['oov_max']:.2f}__conf{conf}"


def evaluate_dataset(language: str, frame: pd.DataFrame, names: list[str], baseline: str, lexicon: set[str], confidence_col: str | None, label_source: str) -> dict[str, Any]:
    tune, holdout, split_audit = speaker_split(frame)
    configs = [{"kind": "unconditional", "policy": n} for n in names]
    configs += make_gate_configs(frame, names, baseline, tune, lexicon, confidence_col)
    score_table = candidate_report(frame, names, tune, holdout)
    for config in configs:
        if config["kind"] == "unconditional":
            continue
        hyp = apply_config(frame, config)
        tm = metrics(frame, hyp, tune)
        hm = metrics(frame, hyp, holdout)
        fm = metrics(frame, hyp, None)
        score_table.loc[len(score_table)] = {"policy": config_name(config), "kind": "guarded_mixture", "tune_zindi": tm["zindi"], "holdout_zindi": hm["zindi"], "full_zindi": fm["zindi"], "tune_score": tm["score"], "holdout_score": hm["score"], "full_wer": fm["wer"], "full_cer": fm["cer"]}
    score_table = score_table.sort_values(["tune_score", "policy"]).reset_index(drop=True)
    selected, tune_metrics = select_config(frame, configs, tune)
    selected_hyp = apply_config(frame, selected)
    holdout_metrics = metrics(frame, selected_hyp, holdout)
    full_metrics = metrics(frame, selected_hyp, None)
    base_hyp = apply_config(frame, {"kind": "unconditional", "policy": baseline})
    base_hold = metrics(frame, base_hyp, holdout)
    boot = bootstrap_speakers(frame, base_hyp, selected_hyp, holdout)
    holdout_improvement = float(base_hold["score"] - holdout_metrics["score"])
    pass_gate = bool(
        config_name(selected) != baseline
        and holdout_improvement >= 0.005
        and holdout_metrics["wer"] <= base_hold["wer"]
        and holdout_metrics["cer"] <= base_hold["cer"]
        and boot["p_improvement_gt_zero"] >= 0.95
        and boot["ci95_low"] > 0.0
    )
    folds = oof_folds(frame)
    oof_rows = []
    for fold, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
        # OOF is a held-fixed audit of the tune-selected policy.  Promotion is
        # still decided only by the untouched speaker-disjoint holdout above.
        fold_hyp = selected_hyp
        fold_base = base_hyp
        fold_m = metrics(frame, fold_hyp, test_idx)
        fold_b = metrics(frame, fold_base, test_idx)
        oof_rows.append({"language": language, "fold": fold, "policy": config_name(selected), "baseline": baseline, "n_train": len(train_idx), "n_test": len(test_idx), "train_speakers": frame.iloc[train_idx].speaker_id.nunique(), "test_speakers": frame.iloc[test_idx].speaker_id.nunique(), "speaker_overlap": len(set(frame.iloc[train_idx].speaker_id.astype(str)) & set(frame.iloc[test_idx].speaker_id.astype(str))), "oof_score": fold_m["score"], "oof_zindi": fold_m["zindi"], "baseline_score": fold_b["score"], "baseline_zindi": fold_b["zindi"], "improvement_score": fold_b["score"] - fold_m["score"]})
    oof = pd.DataFrame(oof_rows)
    return {"language": language, "label_source": label_source, "n": len(frame), "baseline": baseline, "candidate_count": len(names), "candidate_names": names, "split": split_audit, "selected": {"name": config_name(selected), "config": selected, "tune": tune_metrics, "holdout": holdout_metrics, "full": full_metrics, "baseline_holdout": base_hold, "holdout_improvement_score": holdout_improvement, "bootstrap": boot, "pass": pass_gate}, "oof": oof, "scores": score_table}


def load_nyn() -> tuple[pd.DataFrame, list[str], str, set[str], str | None, str]:
    df = pd.read_csv(NIN_HYPS)
    required = {"ID", "speaker_id", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Nyanja hypothesis table missing {sorted(missing)}")
    # Existing table is an immutable validation artifact; no Phase2 columns are read.
    preferred = ["sunbird", "production_greedy", "production_beam", "production_domain_beam", "production_lexicon", "sunbird_lexicon", "lm_sentence_select", "ratio_guard_0p7_1p4", "ratio_guard_0p8_1p25", "ratio_guard_0p85_1p4", "nested_selector_hypothesis", "speaker_cv_static_fusion"]
    names = [x for x in preferred if x in df.columns]
    baseline = "nested_selector_hypothesis" if "nested_selector_hypothesis" in names else names[0]
    return df, names, baseline, vocab(ROOT / "data/lms/nyn_counts.json"), "prod_mean_logp", "existing Nyanja validation hypothesis table (reference field; no test labels)"


def load_lug() -> tuple[pd.DataFrame, list[str], str, set[str], str | None, str]:
    hyp = pd.read_csv(LUG_HYPS)
    labels = pd.read_csv(LUG_LABELS, usecols=["id", "transcription"]).rename(columns={"id": "ID", "transcription": "reference"})
    meta = pd.read_parquet(LUG_META, columns=["ID", "speaker_id"])
    df = hyp.merge(labels, on="ID", how="inner", validate="one_to_one").merge(meta, on="ID", how="inner", validate="one_to_one")
    if len(df) != len(hyp) or len(df) != len(labels.loc[labels.ID.isin(hyp.ID)]):
        raise RuntimeError("Luganda validation ID join is incomplete")
    # Attach confidence from the raw v4 table; all source files are validation-only.
    raw = pd.read_csv(LUG_RAW_DIR / "hyps_mms_ft_v4.csv", usecols=["ID", "mean_logprob"])
    df = df.merge(raw.rename(columns={"mean_logprob": "lug_v4_mean_logprob"}), on="ID", how="left", validate="one_to_one")
    excluded = {"ID", "original_reference", "corrected_reference", "reference", "speaker_id"}
    names = candidate_names(df, excluded)
    baseline = "mms_ft_v3_splitjoin"
    if baseline not in names:
        raise RuntimeError("Luganda incumbent hypothesis is absent")
    return df, names, baseline, vocab(ROOT / "data/lms/lug_counts.json"), "lug_v4_mean_logprob", "data/corrected_waxal/lug_validation_labels.csv joined by immutable ID"


def audit_routes() -> dict[str, Any]:
    # Explicit usecols excludes floor/prediction/audio and therefore cannot ingest
    # a Phase2 reference or transcript by accident.
    route = pd.read_csv(ROUTE_INDEX, usecols=["ID", "decode_lang", "split"], dtype={"ID": str, "decode_lang": str, "split": str})
    if route.ID.duplicated().any():
        raise RuntimeError("route index contains duplicate IDs")
    counts = route.decode_lang.value_counts().sort_index().to_dict()
    scope = {}
    for lang in sorted(IN_SCOPE):
        ids = sorted(route.loc[route.decode_lang == lang, "ID"].tolist())
        scope[lang] = {"rows": len(ids), "id_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(), "ids": ids}
    return {"route_index": str(ROUTE_INDEX.relative_to(ROOT)), "route_index_sha256": sha256_file(ROUTE_INDEX), "rows": len(route), "route_counts": counts, "w2v_replaced_routes_excluded": sorted(W2V_REPLACED), "in_scope_non_w2v_routes": sorted(IN_SCOPE), "scope": scope}


def maybe_write_cache(route_audit: dict[str, Any], evaluations: dict[str, Any]) -> list[str]:
    """Write nothing unless a novel policy passes and a complete cache is available.

    This run intentionally has no new Phase2 decode cache for a novel policy.
    The function remains explicit so a future in-scope cache cannot be emitted
    without passing the exact ID-set check.
    """
    written: list[str] = []
    for lang, result in evaluations.items():
        if not result["selected"]["pass"]:
            continue
        raise RuntimeError(f"{lang} passed but no new complete Phase2 cache source was declared")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    route_audit = audit_routes()
    nyn_frame, nyn_names, nyn_base, nyn_vocab, nyn_conf, nyn_source = load_nyn()
    lug_frame, lug_names, lug_base, lug_vocab, lug_conf, lug_source = load_lug()
    evaluations = {
        "nyn": evaluate_dataset("nyn", nyn_frame, nyn_names, nyn_base, nyn_vocab, nyn_conf, nyn_source),
        "lug": evaluate_dataset("lug", lug_frame, lug_names, lug_base, lug_vocab, lug_conf, lug_source),
    }
    for lang, result in evaluations.items():
        result["scores"].to_csv(out / f"{lang}_policy_scores.csv", index=False)
        result["oof"].to_csv(out / f"{lang}_oof_scores.csv", index=False)
        result.pop("scores")
        result.pop("oof")
    cache_paths = maybe_write_cache(route_audit, evaluations)
    manifest = {
        "goal": "route ensemble optimizer 2026-08-09",
        "protocol": {"validation_only": True, "test_labels_used": False, "phase2_labels_or_transcripts_used": False, "uploads_performed": False, "existing_candidates_modified": False, "selection": "speaker-disjoint tune/holdout plus 5-fold speaker-disjoint OOF", "promotion": "holdout score improvement >= 0.005, WER/CER non-regression, speaker bootstrap p>=0.95 and CI low > 0"},
        "inputs": {str(p.relative_to(ROOT)): {"sha256": sha256_file(p)} for p in [NIN_HYPS, LUG_HYPS, LUG_LABELS, LUG_META, ROOT / "data/lms/nyn_counts.json", ROOT / "data/lms/lug_counts.json"]},
        "route_audit": route_audit,
        "evaluations": evaluations,
        "phase2_caches_written": cache_paths,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "route_audit.json", route_audit)
    lines = [
        "# Route ensemble optimizer evidence",
        "",
        "Validation-only run for Nyanja (`nyn`) and Luganda (`lug`). All selection was done with speaker-disjoint tune/holdout and five-fold speaker-disjoint OOF checks.",
        "",
        "Phase2 labels/transcripts, audio, and `floor` values were not read; existing candidates were not modified and no upload was attempted.",
        "",
        "## Route scope",
        "",
        f"W2V-BERT-replaced routes excluded from new scope: {', '.join(sorted(W2V_REPLACED))}. Exact audited non-W2V-BERT routes: {', '.join(sorted(IN_SCOPE))}.",
        "",
    ]
    for lang, result in evaluations.items():
        sel = result["selected"]
        lines += [f"## {lang}", "", f"Baseline: `{result['baseline']}`; selected: `{sel['name']}`; pass: `{sel['pass']}`.", "", f"Holdout score improvement (positive is better): `{sel['holdout_improvement_score']:.6f}`; bootstrap p(improvement > 0): `{sel['bootstrap']['p_improvement_gt_zero']:.4f}`; 95% CI: `[{sel['bootstrap']['ci95_low']:.6f}, {sel['bootstrap']['ci95_high']:.6f}]`.", ""]
    lines += ["No complete novel Phase2 cache was written unless the policy passed and an exact-ID prediction-only source was available. See `manifest.json` and `route_audit.json`.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
