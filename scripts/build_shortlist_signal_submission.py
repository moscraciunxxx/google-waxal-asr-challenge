#!/usr/bin/env python3
"""Build floor-first Phase-2 submission from shortlist signal stack (no GPU required).

Uses precomputed residual features:
  - outputs/beat_k63/luo_ortho_residual_scores.csv (S08)
  - outputs/phase2_achluo_dual_pool_detail.csv (mms hyp, cer_mc, already_dual)
  - outputs/phase2_lid126_full.csv (S01-lite mass)
  - outputs/phase2_openset_detail.csv (decode residual pool)
  - outputs/phase2_luo_mms1b_detail.csv (fallback Luo hyp)
  - optional multiadapter (not conf-primary)

Stack: S01-lite ∧ S05-lite ∧ S08 (+ optional S09 soft-dual×ortho).
Defaults to floor; bounded top-N replaces. Does NOT overwrite FINAL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS
from src.prize_pack import align_ids, apply_replace_set, diff_stats, is_banned_mass_rewrite
from src.shortlist_gates import (
    ResidualFeatures,
    ban_check_verdict,
    select_replaces,
)
from src.text_norm import normalize_text


def _load_csv_map(path: Path) -> dict[str, dict]:
    return {r["ID"]: r for r in csv.DictReader(path.open())}


def build_features() -> list[ResidualFeatures]:
    floor_path = ROOT / "submission_phase2_FINAL.csv"
    ortho_path = ROOT / "outputs/beat_k63/luo_ortho_residual_scores.csv"
    pool_path = ROOT / "outputs/phase2_achluo_dual_pool_detail.csv"
    lid_path = ROOT / "outputs/phase2_lid126_full.csv"
    openset_path = ROOT / "outputs/phase2_openset_detail.csv"
    mms1b_path = ROOT / "outputs/phase2_luo_mms1b_detail.csv"
    ma_path = ROOT / "outputs/phase2_mms_multadapter_scores.csv"

    floor = {
        r["ID"]: normalize_text(r["Target"])
        for r in csv.DictReader(floor_path.open())
    }
    ortho = _load_csv_map(ortho_path)
    pool = _load_csv_map(pool_path)
    lid = _load_csv_map(lid_path)
    openset = _load_csv_map(openset_path)
    mms1b = _load_csv_map(mms1b_path) if mms1b_path.exists() else {}
    ma = _load_csv_map(ma_path) if ma_path.exists() else {}

    feats: list[ResidualFeatures] = []
    for sid, o in ortho.items():
        L = lid.get(sid) or {}
        O = openset.get(sid) or {}
        if (L.get("lang1") or "").lower() != "luo":
            continue
        if (O.get("decode_lang") or "").lower() != "ach":
            continue
        P = pool.get(sid) or {}
        already = str(P.get("already_dual", o.get("already_dual", ""))).lower() in (
            "1",
            "true",
            "yes",
        )
        # Luo hyp: dual-pool mms preferred, else mms1b prediction
        luo_hyp = normalize_text(P.get("mms") or "") or normalize_text(
            (mms1b.get(sid) or {}).get("prediction") or ""
        )
        M = ma.get(sid) or {}
        feats.append(
            ResidualFeatures(
                id=sid,
                p1=float(o.get("p1") or L.get("p1") or 0),
                lang2=str(L.get("lang2") or ""),
                p2=float(L.get("p2") or 0),
                ortho_mms=float(o.get("ortho_mms") or -999),
                lp_luo_mms=float(o.get("lp_luo_mms") or -999),
                lp_ach_mms=float(o.get("lp_ach_mms") or -999),
                cer_mc=float(o.get("cer_mc") or P.get("cer_mc") or 1.0),
                floor_hyp=floor.get(sid, ""),
                luo_hyp=luo_hyp,
                pick_lang=str(M.get("pick_lang") or ""),
                conf_luo=float(M.get("conf_luo") or 0),
                conf_ach=float(M.get("conf_ach") or 0),
                already_dual=already,
            )
        )
    return feats


def main() -> int:
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    p = argparse.ArgumentParser()
    p.add_argument("--max-n", type=int, default=25)
    p.add_argument("--min-ortho", type=float, default=2.4)
    p.add_argument("--min-lp-delta", type=float, default=0.0)
    p.add_argument("--min-char-sim", type=float, default=0.35)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/new_signals/submission_phase2_shortlist_signal_stack.csv",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-ids", type=int, default=0, help="if >0, only score first N residual features")
    args = p.parse_args()

    floor_path = ROOT / "submission_phase2_FINAL.csv"
    fail_path = ROOT / "outputs/legit_system/phase2_multifamily_fusion.csv"
    sample = ROOT / "data/phase2/SampleSubmission.csv"

    floor = {
        r["ID"]: normalize_text(r["Target"])
        for r in csv.DictReader(floor_path.open())
    }
    ids = [r["ID"] for r in csv.DictReader(sample.open())]
    feats = build_features()
    if args.max_ids and args.max_ids > 0:
        feats = feats[: args.max_ids]

    replaces = select_replaces(
        feats,
        max_n=args.max_n,
        min_ortho=args.min_ortho,
        min_lp_delta=args.min_lp_delta,
        min_char_sim=args.min_char_sim,
    )
    ban = ban_check_verdict(len(replaces))
    targets = apply_replace_set(floor, replaces)
    rows = align_ids(ids, targets)

    st = diff_stats(floor, targets, ids)
    fail = {
        r["ID"]: normalize_text(r["Target"])
        for r in csv.DictReader(fail_path.open())
    }
    n_same_mf = sum(1 for i in ids if targets[i] == fail[i])

    meta = {
        "out": str(args.out),
        "n_replace": st["n_diff"],
        "n_same_floor": st["n_same"],
        "n_same_failed_multifamily": n_same_mf,
        "mass_rewrite": is_banned_mass_rewrite(st["n_diff"]),
        "byte_identical_failed_mf": False,
        "floor_sha256": hashlib.sha256(floor_path.read_bytes()).hexdigest(),
        "failed_mf_sha256": hashlib.sha256(fail_path.read_bytes()).hexdigest(),
        "method": (
            "floor-first shortlist stack: S01-lite (LID mass) + S05-lite (structure) "
            "+ S08 (ortho+charLM Δ) + optional S09 (soft dual only with S08); "
            "Luo hyp from dual-pool mms / mms1b; no residual conf primary; no thr>0.15 alone"
        ),
        "shortlist_used": ["S01-lite", "S05-lite", "S08", "S09-optional"],
        "catalog_ref": "outputs/new_signals/TEN_PLUS_NEW_SIGNALS.md",
        "shortlist_ref": "outputs/new_signals/SHORTLIST_RANK.md",
        "params": {
            "max_n": args.max_n,
            "min_ortho": args.min_ortho,
            "min_lp_delta": args.min_lp_delta,
            "min_char_sim": args.min_char_sim,
        },
        "ban_check": ban,
        "n_features_scored": len(feats),
        "status": "candidate_ready_for_upload_awaiting_public_score",
        "win_condition": "public score > 0.560605696",
        "floor_public": 0.560605696,
        "public_win_claimed": False,
        "limits": (
            "Full GPU S01 trained router / S02 SSL kNN not run in this env (no torch); "
            "S01-lite uses LID mass composition; S05-lite uses hyp structure on text; "
            "S08 uses precomputed residual ortho/charLM scores"
        ),
    }

    if args.dry_run:
        print(json.dumps({**meta, "sample_replaces": replaces[:5]}, indent=2))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(rows)

    meta["sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta["byte_identical_failed_mf"] = (
        hashlib.sha256(args.out.read_bytes()).digest()
        == hashlib.sha256(fail_path.read_bytes()).digest()
    )
    meta["byte_identical_floor"] = (
        hashlib.sha256(args.out.read_bytes()).digest()
        == hashlib.sha256(floor_path.read_bytes()).digest()
    )

    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    rep_path = args.out.with_name(args.out.stem + "_replaces.json")
    rep_path.write_text(json.dumps(replaces, indent=2))

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
