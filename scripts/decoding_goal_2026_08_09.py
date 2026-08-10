#!/usr/bin/env python3
"""Locked decoding/adaptation audit for the 2026-08-09 goal.

This script is prediction-only with respect to Phase2.  It reads validation
references and already-produced hypothesis caches, but never reads a Phase1 or
Phase2 test transcript.  It writes only below
``outputs/goal_2026_08_09/decoding_adaptation``.

The audit covers:

* CTC greedy versus standard/domain KenLM beam hypotheses for Luganda;
* the previously safe, tiny Shona adaptation artifacts and their locked A/B;
* deterministic speaker-disjoint tune/holdout scoring and speaker bootstrap;
* raw ``ID,Target`` route-cache validation; and
* exact SHA-256 hashes for every input and any strong-pass route cache.

The old unsafe full-adaptation trainers are deliberately not imported or run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "goal_2026_08_09" / "decoding_adaptation"
SEED = 20260809
BOOTSTRAP_DRAWS = 2000
STRONG_DELTA = 0.01

LUG_HYPOTHESES = ROOT / "outputs" / "goal_2026_08_08" / "luganda_fusion" / "matched_hypotheses.csv"
LUG_META = ROOT / "data" / "hf_metadata" / "lug_validation.parquet"
LUG_CKPT = ROOT / "checkpoints" / "mms-lug-ft-v3"
LUG_ARPA = ROOT / "data" / "lms_phase2_domain" / "lug_merged_2gram.arpa"
LUG_UNIGRAMS = ROOT / "data" / "lms_phase2_domain" / "lug_unigrams.txt"
SNA_REPORT = ROOT / "outputs" / "goal_2026_08_06" / "sna_non_selfpseudo_ab.json"
SNA_DETAIL = ROOT / "outputs" / "goal_2026_08_08" / "shona_sd2_parallel" / "validation_w2vbert-shona-sd2.csv"
SNA_META = ROOT / "data" / "hf_metadata" / "sna_validation.parquet"
ADAPTATION_AUDIT = ROOT / "outputs" / "goal_2026_08_08" / "adaptation_audit.md"
ROUTE_INDEX = ROOT / "outputs" / "beat075" / "public_visible_index.csv"
PHASE2_BASE = ROOT / "submission_phase2_v2_full.csv"
LUG_PHASE2_CANDIDATE = ROOT / "submission_phase2_beat075_lug_domain_beam.csv"
SAFE_ADAPTATION_CKPT = ROOT / "checkpoints" / "mms-sna-pure-beat-waxal-fixed" / "t11_last1_lm_s12" / "best"

FORBIDDEN_MARKERS = ("test.csv", "/test/", "test_gold", "unsafe-test-gold")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def require_file(path: Path, *, label: str, allow_test_pred_only: bool = False) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing {path}")
    lower = str(path).lower()
    if any(marker in lower for marker in FORBIDDEN_MARKERS):
        if not allow_test_pred_only:
            raise RuntimeError(f"{label}: forbidden test-label path {path}")


def norm(value: object) -> str:
    from src.text_norm import normalize_text

    return normalize_text("" if value is None else str(value))


def metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    from src.metrics import score_pairs

    score = score_pairs(refs, hyps)
    return {
        "n": int(score["n"]),
        "wer": float(score["wer"]),
        "cer": float(score["cer"]),
        "error": float(score["score"]),
        "zindi": float(1.0 - score["score"]),
    }


def row_error(ref: str, hyp: str) -> float:
    return metrics([ref], [hyp])["error"]


def speaker_folds(frame: pd.DataFrame, *, seed: int = SEED) -> pd.Series:
    """Stable speaker split; labels are not used to choose the split."""
    speakers = sorted(frame["speaker_id"].astype(str).unique())
    ranked = sorted(speakers, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    cut = max(1, len(ranked) // 2)
    holdout = set(ranked[:cut])
    if len(holdout) == len(ranked):
        holdout.remove(ranked[-1])
    return frame["speaker_id"].astype(str).map(lambda s: "holdout" if s in holdout else "tune")


def paired_speaker_bootstrap(
    frame: pd.DataFrame,
    baseline_col: str,
    candidate_col: str,
    *,
    seed: int = SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, float | int]:
    """Bootstrap candidate-minus-baseline Zindi gain by speaker."""
    work = frame[["speaker_id", "reference", baseline_col, candidate_col]].copy()
    work["delta"] = [
        row_error(r, b) - row_error(r, c)
        for r, b, c in zip(work.reference, work[baseline_col], work[candidate_col])
    ]
    groups = [g["delta"].to_numpy(dtype=float) for _, g in work.groupby("speaker_id", sort=True)]
    if not groups:
        raise RuntimeError("bootstrap received no speakers")
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for i in range(draws):
        picked = rng.integers(0, len(groups), size=len(groups))
        values[i] = float(np.mean(np.concatenate([groups[j] for j in picked])))
    return {
        "draws": int(draws),
        "seed": int(seed),
        "delta_mean": float(values.mean()),
        "delta_p05": float(np.quantile(values, 0.05)),
        "delta_p50": float(np.quantile(values, 0.50)),
        "delta_p95": float(np.quantile(values, 0.95)),
        "probability_delta_positive": float(np.mean(values > 0.0)),
        "resampling_unit": "speaker",
    }


def evaluate_candidate(
    frame: pd.DataFrame,
    baseline_col: str,
    candidate_col: str,
    *,
    tag: str,
    method_kind: str,
) -> dict[str, Any]:
    overall = metrics(frame.reference.tolist(), frame[candidate_col].tolist())
    base = metrics(frame.reference.tolist(), frame[baseline_col].tolist())
    folds: dict[str, Any] = {}
    for fold in ("tune", "holdout"):
        sub = frame.loc[frame.fold == fold]
        if sub.empty:
            raise RuntimeError(f"{tag}: empty {fold} fold")
        c = metrics(sub.reference.tolist(), sub[candidate_col].tolist())
        b = metrics(sub.reference.tolist(), sub[baseline_col].tolist())
        folds[fold] = {"candidate": c, "baseline": b, "delta_zindi": c["zindi"] - b["zindi"]}
    bootstrap = paired_speaker_bootstrap(frame, baseline_col, candidate_col)
    delta = overall["zindi"] - base["zindi"]
    checks = {
        "overall_delta_at_least_0p01": delta >= STRONG_DELTA,
        "candidate_wer_strictly_better": overall["wer"] < base["wer"],
        "tune_delta_positive": folds["tune"]["delta_zindi"] > 0.0,
        "holdout_delta_positive": folds["holdout"]["delta_zindi"] > 0.0,
        "bootstrap_p05_positive": bootstrap["delta_p05"] > 0.0,
    }
    return {
        "tag": tag,
        "method_kind": method_kind,
        "baseline_column": baseline_col,
        "candidate_column": candidate_col,
        "n": int(len(frame)),
        "speaker_count": int(frame.speaker_id.nunique()),
        "speaker_overlap_tune_holdout": sorted(
            set(frame.loc[frame.fold == "tune", "speaker_id"])
            & set(frame.loc[frame.fold == "holdout", "speaker_id"])
        ),
        "candidate": overall,
        "baseline": base,
        "delta_zindi": float(delta),
        "folds": folds,
        "paired_speaker_bootstrap": bootstrap,
        "pass_checks": checks,
        "strong_pass": bool(all(checks.values())),
        "strong_pass_rule": "delta_zindi>=0.01, WER strictly better, positive tune/holdout deltas, speaker-bootstrap p05>0",
    }


def load_lug_locked() -> tuple[pd.DataFrame, list[Path]]:
    require_file(LUG_HYPOTHESES, label="Luganda hypotheses")
    require_file(LUG_META, label="Luganda validation metadata")
    hyps = pd.read_csv(LUG_HYPOTHESES, dtype=str, keep_default_na=False)
    meta = pd.read_parquet(LUG_META)
    meta = meta.rename(columns={"ID": "ID", "Target": "reference"})
    needed = {"ID", "speaker_id", "reference"}
    if not needed <= set(meta.columns):
        raise RuntimeError(f"Luganda metadata missing {needed - set(meta.columns)}")
    hyp_cols = [
        "ID",
        "original_reference",
        "corrected_reference",
        "mms_ft_v3_splitjoin",
        "mms_ft_v3_standard_beam_splitjoin",
        "mms_ft_v3_domain_beam_splitjoin",
    ]
    if not set(hyp_cols) <= set(hyps.columns):
        raise RuntimeError(f"Luganda hypotheses missing {set(hyp_cols) - set(hyps.columns)}")
    joined = hyps[hyp_cols].merge(meta[["ID", "speaker_id", "reference"]], on="ID", how="inner")
    if len(joined) != len(hyps) or joined.ID.duplicated().any():
        raise RuntimeError("Luganda locked join is not exact and unique")
    joined["reference"] = joined["reference"].map(norm)
    for c in hyp_cols[1:]:
        joined[c] = joined[c].map(norm)
    joined["fold"] = speaker_folds(joined)
    return joined, [
        LUG_HYPOTHESES,
        LUG_META,
        LUG_CKPT / "config.json",
        LUG_CKPT / "vocab.json",
        LUG_CKPT / "model.safetensors",
        LUG_ARPA,
        LUG_UNIGRAMS,
    ]


def load_sna_audit() -> tuple[dict[str, Any], list[Path]]:
    require_file(SNA_REPORT, label="Shona adaptation report")
    require_file(SNA_DETAIL, label="Shona locked detail")
    require_file(SNA_META, label="Shona validation metadata")
    report = json.loads(SNA_REPORT.read_text())
    detail = pd.read_csv(SNA_DETAIL, dtype=str, keep_default_na=False)
    meta = pd.read_parquet(SNA_META)[["ID", "speaker_id"]]
    joined = detail.merge(meta, on="ID", how="inner")
    if len(joined) != len(detail) or joined.ID.duplicated().any():
        raise RuntimeError("Shona locked detail join is not exact and unique")
    joined["fold"] = speaker_folds(joined)
    report["locked_detail_audit"] = {
        "rows": int(len(joined)),
        "speakers": int(joined.speaker_id.nunique()),
        "tune_rows": int((joined.fold == "tune").sum()),
        "holdout_rows": int((joined.fold == "holdout").sum()),
        "speaker_overlap": sorted(
            set(joined.loc[joined.fold == "tune", "speaker_id"])
            & set(joined.loc[joined.fold == "holdout", "speaker_id"])
        ),
        "candidate": "sulaimank/w2vbert-shona-sd2 (existing locked decode)",
    }
    return report, [SNA_REPORT, SNA_DETAIL, SNA_META]


def audit_safe_adaptation() -> dict[str, Any]:
    require_file(ADAPTATION_AUDIT, label="adaptation audit")
    meta_path = SAFE_ADAPTATION_CKPT.parent.parent / "train_meta.json"
    if not meta_path.is_file():
        meta_path = ROOT / "checkpoints" / "mms-sna-pure-beat-waxal-fixed" / "train_meta.json"
    require_file(meta_path, label="safe adaptation metadata")
    # Hash only a compact provenance set; do not copy or rewrite the 1.2 GB model.
    files = [
        meta_path,
        SAFE_ADAPTATION_CKPT / "config.json",
        SAFE_ADAPTATION_CKPT / "vocab.json",
        SAFE_ADAPTATION_CKPT / "model.safetensors",
    ]
    for path in files:
        require_file(path, label="safe adaptation artifact")
    return {
        "method": "mms-sna-pure-beat-waxal-fixed/t11_last1_lm_s12",
        "checkpoint": str(SAFE_ADAPTATION_CKPT),
        "train_meta": json.loads(meta_path.read_text()),
        "artifact_sha256": {str(p): sha256(p) for p in files},
        "existing_locked_evaluation": json.loads(SNA_REPORT.read_text()),
        "launched_this_run": False,
        "unsafe_full_adaptation_launched": False,
        "decision": "reject: existing locked delta is +0.000542 Zindi, below +0.01 strong-pass gate",
    }


def read_prediction_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label=label, allow_test_pred_only=True)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header != ["ID", "Target"]:
        raise RuntimeError(f"{label}: expected exact raw header ID,Target; got {header}")
    frame = pd.read_csv(path, usecols=["ID", "Target"], dtype=str, keep_default_na=False)
    if frame.ID.duplicated().any() or frame.Target.map(lambda v: not str(v).strip()).any():
        raise RuntimeError(f"{label}: duplicate ID or empty raw Target")
    return frame


def materialize_lug_route_cache() -> tuple[Path, dict[str, Any], list[Path]]:
    """Materialize only the already-decoded Luganda route, never a full CSV."""
    index = pd.read_csv(ROUTE_INDEX, dtype=str, keep_default_na=False)
    base = read_prediction_csv(PHASE2_BASE, "Phase2 base prediction")
    candidate = read_prediction_csv(LUG_PHASE2_CANDIDATE, "Luganda beam prediction")
    route = index.loc[index.decode_lang.eq("lug"), ["ID", "decode_lang", "split"]].copy()
    expected = route.ID.tolist()
    if len(set(expected)) != len(expected):
        raise RuntimeError("Luganda route index has duplicate IDs")
    b = base.set_index("ID")["Target"]
    c = candidate.set_index("ID")["Target"]
    if not set(expected) <= set(b.index) or not set(expected) <= set(c.index):
        raise RuntimeError("Luganda route is not fully covered by prediction caches")
    route["Target"] = route.ID.map(c)
    route["changed_vs_base"] = route.ID.map(c).ne(route.ID.map(b)).astype(int)
    out_path = OUT / "phase2_cache_lug_domain_beam.csv"
    route[["ID", "Target"]].to_csv(out_path, index=False)
    check = read_prediction_csv(out_path, "materialized Luganda route cache")
    if check.ID.tolist() != expected:
        raise RuntimeError("materialized route cache order mismatch")
    info = {
        "route": "lug",
        "rows": int(len(check)),
        "changed_vs_phase2_base": int(route.changed_vs_base.sum()),
        "split_counts": {str(k): int(v) for k, v in route.split.value_counts().to_dict().items()},
        "raw_protocol": {
            "header": ["ID", "Target"],
            "unique_ids": int(check.ID.nunique()),
            "empty_targets": int((check.Target.str.strip() == "").sum()),
            "exact_route_id_order": True,
            "full_submission_written": False,
            "upload_performed": False,
        },
    }
    return out_path, info, [ROUTE_INDEX, PHASE2_BASE, LUG_PHASE2_CANDIDATE]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-route-cache", action="store_true", help="skip materializing the strong-pass route cache")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    lug, lug_inputs = load_lug_locked()
    lug_methods = []
    for candidate, tag, kind in [
        ("mms_ft_v3_standard_beam_splitjoin", "lug_ctc_standard_kenlm_beam", "ctc_beam_lm"),
        ("mms_ft_v3_domain_beam_splitjoin", "lug_ctc_domain_kenlm_beam", "ctc_beam_lm"),
    ]:
        lug_methods.append(evaluate_candidate(lug, "mms_ft_v3_splitjoin", candidate, tag=tag, method_kind=kind))

    sna_report, sna_inputs = load_sna_audit()
    adaptation = audit_safe_adaptation()
    route_info = None
    route_path = None
    route_inputs: list[Path] = []
    strong_lug = next(x for x in lug_methods if x["tag"] == "lug_ctc_domain_kenlm_beam")
    if strong_lug["strong_pass"] and not args.no_route_cache:
        route_path, route_info, route_inputs = materialize_lug_route_cache()

    inputs = lug_inputs + sna_inputs + [ADAPTATION_AUDIT]
    inputs += [SAFE_ADAPTATION_CKPT / "config.json", SAFE_ADAPTATION_CKPT / "vocab.json", SAFE_ADAPTATION_CKPT / "model.safetensors"]
    inputs += route_inputs
    input_manifest = {str(path): sha256(path) for path in sorted(set(inputs)) if path.is_file()}
    result = {
        "task": "decoding_and_lightweight_adaptation_goal_2026_08_09",
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "protocol": {
            "competition_metric": "Zindi emulation = 1 - 0.5*WER - 0.5*CER; lower WER/CER before inversion",
            "normalization": "src.text_norm.normalize_text",
            "validation_only": True,
            "phase1_test_labels_read": False,
            "phase2_test_labels_read": False,
            "unsafe_full_adaptation_launched": False,
            "uploads_performed": False,
            "write_scope": str(OUT),
        },
        "locked_samples": {
            "luganda": {
                "rows": int(len(lug)),
                "speakers": int(lug.speaker_id.nunique()),
                "sample_id_sha256": sha256_lines(sorted(lug.ID.tolist())),
                "speaker_id_sha256": sha256_lines(sorted(lug.speaker_id.astype(str).unique().tolist())),
                "fold_counts": {str(k): int(v) for k, v in lug.fold.value_counts().to_dict().items()},
                "speaker_overlap": sorted(set(lug.loc[lug.fold == "tune", "speaker_id"]) & set(lug.loc[lug.fold == "holdout", "speaker_id"])),
            },
            "shona": sna_report.get("locked_detail_audit", {}),
        },
        "decoding": {"luganda": lug_methods},
        "lightweight_adaptation": adaptation,
        "strong_pass_actions": {
            "luganda_domain_kenlm": {
                "strong_pass": bool(strong_lug["strong_pass"]),
                "route_cache_materialized": bool(route_path),
                "route_cache": str(route_path) if route_path else None,
                "route_info": route_info,
            }
        },
        "input_sha256": input_manifest,
        "output_sha256": {
            "script": sha256(Path(__file__)),
            **({str(route_path): sha256(route_path)} if route_path else {}),
        },
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    report_sha = sha256(report_path)
    (OUT / "summary.md").write_text(
        "# Decoding/adaptation audit\n\n"
        f"- Luganda domain KenLM beam strong pass: **{strong_lug['strong_pass']}**\n"
        f"- Luganda route cache written: **{bool(route_path)}**\n"
        f"- Safe Shona adaptation decision: **{adaptation['decision']}**\n"
        "- Test labels read: **false**; uploads: **false**; unsafe full adaptation: **not launched**\n"
        f"- Report SHA-256: `{report_sha}`\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
