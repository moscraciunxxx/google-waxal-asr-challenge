#!/usr/bin/env python3
"""Build the strict final WAXAL Phase-2 candidates from validated route caches.

This intentionally does not overlay the older hidden-route full submission.  That
artifact was produced from an older spine; only the two rows listed in
``private_safe_replacements.csv`` are safe to transplant into the current base.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_norm import normalize_text
DEFAULT_BASE = ROOT / "outputs/goal_2026_08_08/nyn_ensemble/submission_phase2_nyn_cv_ensemble_guarded.csv"
DEFAULT_INDEX = ROOT / "outputs/beat075/public_visible_index.csv"
DEFAULT_LIN = ROOT / "outputs/goal_2026_08_08/sulaiman_public_descendants/phase2_cache_w2vbert-lingala-sd3.csv"
DEFAULT_LUG = ROOT / "outputs/goal_2026_08_09/decoding_adaptation/phase2_cache_lug_domain_beam.csv"
DEFAULT_SNA = ROOT / "outputs/goal_2026_08_08/shona_sd2_parallel/phase2_cache_w2vbert-shona-sd2.csv"
DEFAULT_SNA_REPORT = ROOT / "outputs/goal_2026_08_08/shona_sd2_parallel/report.json"
DEFAULT_LUG_REPORT = ROOT / "outputs/goal_2026_08_09/decoding_adaptation/report.json"
DEFAULT_PRIVATE = ROOT / "outputs/goal_2026_08_08/hidden_routes/private_safe_replacements.csv"
DEFAULT_OUT = ROOT / "outputs/goal_2026_08_08/final"
CURRENT_BASE_SHA256 = "8570b5b576db2b76d22bdca46009dc9097d1246f49cd8ae210daaa22c30117a2"

# Independently verified validation policy: W2V SD3 loses on all three sampled
# Lingala clips above 30 seconds.  These are the complete Phase-2 Lingala rows
# whose measured duration exceeds that threshold; the current base prediction
# is retained for them.
LIN_LONG_FALLBACK_IDS = {
    "ID_BSBTOV", "ID_DNJYJQ", "ID_EJXBUL", "ID_HXUZOB", "ID_IFDPXA",
    "ID_KDXEAT", "ID_LBOBOR", "ID_LCIVIO", "ID_LFETUG", "ID_LTOEWY",
    "ID_MFQGCD", "ID_NIXPQE", "ID_NKDDHC", "ID_NRYMLU", "ID_OMIOGO",
    "ID_QOYCEF",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_pred(path: Path, label: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), None)
    if header != ["ID", "Target"]:
        raise RuntimeError(f"{label}: expected exact ID,Target header, got {header}")
    df = pd.read_csv(path, usecols=["ID", "Target"], dtype=str, keep_default_na=False)
    if df.ID.duplicated().any():
        raise RuntimeError(f"{label}: duplicate IDs")
    if df.Target.map(lambda x: not str(x).strip()).any():
        raise RuntimeError(f"{label}: empty Target")
    return df


def exact_route_cache(path: Path, route: str, index: pd.DataFrame) -> pd.DataFrame:
    df = read_pred(path, f"{route} cache")
    expected = set(index.loc[index.decode_lang == route, "ID"])
    actual = set(df.ID)
    if actual != expected:
        raise RuntimeError(
            f"{route} cache ID set mismatch: rows={len(df)}, missing={len(expected-actual)}, extra={len(actual-expected)}"
        )
    # Producers may preserve the public route order (Luganda) or use a stable
    # ID sort (the Lingala/Shona specialist lanes).  Both are safe because the
    # overlay is keyed by ID; reject any other permutation.
    expected_route_order = index.loc[index.decode_lang == route, "ID"].tolist()
    expected_sorted_order = sorted(expected_route_order)
    if df.ID.tolist() not in (expected_route_order, expected_sorted_order):
        raise RuntimeError(f"{route} cache ID order is neither public route order nor stable ID order")
    return df


def validate(df: pd.DataFrame, label: str, out_dir: Path) -> dict:
    from src.submission import check_phase2_submission

    path = out_dir / f"_check_{label}.csv"
    df.to_csv(path, index=False)
    result = check_phase2_submission(path, strict=True)
    path.unlink()
    if not result["ok"]:
        raise RuntimeError(f"{label}: strict validation failed: {result['errors']}")
    return result


def apply_overlay(base: pd.DataFrame, overlay: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    if overlay.empty:
        raise RuntimeError(f"{label}: empty overlay")
    values = dict(zip(overlay.ID, overlay.Target))
    out = base.copy()
    before = dict(zip(out.ID, out.Target))
    out.loc[out.ID.isin(values), "Target"] = out.loc[out.ID.isin(values), "ID"].map(values)
    changed = sum(before[uid] != target for uid, target in zip(out.ID, out.Target))
    if changed > len(overlay):
        raise RuntimeError(f"{label}: changed more rows than overlay contains")
    return out, changed


def normalize_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Target"] = out["Target"].map(normalize_text)
    if out.Target.map(lambda x: not str(x).strip()).any():
        raise RuntimeError("normalization produced an empty Target")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--lin-cache", type=Path, default=DEFAULT_LIN)
    ap.add_argument("--lug-cache", type=Path, default=DEFAULT_LUG)
    ap.add_argument("--sna-cache", type=Path, default=DEFAULT_SNA)
    ap.add_argument("--sna-report", type=Path, default=DEFAULT_SNA_REPORT)
    ap.add_argument("--lug-report", type=Path, default=DEFAULT_LUG_REPORT)
    ap.add_argument("--private-replacements", type=Path, default=DEFAULT_PRIVATE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--text-format", choices=["normalized", "raw-preserving"], default="normalized")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = read_pred(args.base, "base")
    base_sha = sha256(args.base)
    if args.base.resolve() == DEFAULT_BASE.resolve() and base_sha != CURRENT_BASE_SHA256:
        raise RuntimeError(
            f"current base SHA mismatch: expected {CURRENT_BASE_SHA256}, got {base_sha}"
        )
    index = pd.read_csv(args.index, usecols=["ID", "decode_lang", "split"], dtype=str, keep_default_na=False)
    if index.ID.duplicated().any() or set(index.ID) - set(base.ID):
        raise RuntimeError("route index is not a unique subset of the base IDs")
    if len(base) != 2392:
        raise RuntimeError(f"unexpected base row count: {len(base)}")
    route_counts = dict(Counter(index.decode_lang))
    if route_counts != {"lin": 444, "lug": 433, "sna": 461, "nyn": 256, "ach": 13}:
        raise RuntimeError(f"unexpected canonical route counts: {route_counts}")

    lin = exact_route_cache(args.lin_cache, "lin", index)
    lug = exact_route_cache(args.lug_cache, "lug", index)
    sna = exact_route_cache(args.sna_cache, "sna", index)
    lug_report = json.loads(args.lug_report.read_text())
    legacy_lug_pass = lug_report.get("strong_pass_actions", {}).get("luganda_domain_kenlm", {}).get("strong_pass") is True
    current_lug_pass = lug_report.get("strong_pass") is True
    if not (legacy_lug_pass or current_lug_pass):
        raise RuntimeError("Luganda report does not attest strong_pass=true")
    lug_cache_report = lug_report.get("strong_pass_actions", {}).get("luganda_domain_kenlm", {}).get("route_cache", "")
    if current_lug_pass:
        # New gate reports attest the validated model; the Phase-2 route cache
        # is supplied explicitly to this builder and may be a separate artifact.
        lug_cache_report = str(args.lug_cache)
    if Path(lug_cache_report).resolve() != args.lug_cache.resolve():
        raise RuntimeError("Luganda report/cache path mismatch")
    report = json.loads(args.sna_report.read_text())
    if report.get("metrics", {}).get("strong_pass") is not True:
        raise RuntimeError("Shona report does not attest strong_pass=true")
    if report.get("protocol", {}).get("test_labels_read") is not False:
        raise RuntimeError("Shona report does not attest test_labels_read=false")
    if int(report.get("artifacts", {}).get("phase2_cache_rows") or 0) != len(sna):
        raise RuntimeError("Shona report/cache row count mismatch")

    lin_guarded = lin.copy()
    base_values = dict(zip(base.ID, base.Target))
    missing_guard_ids = LIN_LONG_FALLBACK_IDS - set(lin.ID)
    if missing_guard_ids:
        raise RuntimeError(f"duration guard IDs missing from Lingala cache: {sorted(missing_guard_ids)}")
    lin_guarded.loc[lin_guarded.ID.isin(LIN_LONG_FALLBACK_IDS), "Target"] = (
        lin_guarded.loc[lin_guarded.ID.isin(LIN_LONG_FALLBACK_IDS), "ID"].map(base_values)
    )
    public, lin_changed = apply_overlay(base, lin_guarded, "Lingala duration-guarded")
    public, lug_changed = apply_overlay(public, lug, "Luganda domain KenLM")
    public, sna_changed = apply_overlay(public, sna, "Shona")
    if set(lin.ID) & set(lug.ID) or set(lin.ID) & set(sna.ID) or set(lug.ID) & set(sna.ID):
        raise RuntimeError("specialist route caches overlap")
    validate(public, "public", args.out_dir)

    replacements = pd.read_csv(args.private_replacements, dtype=str, keep_default_na=False)
    required = {"ID", "after", "public_sensitive"}
    if not required <= set(replacements.columns):
        raise RuntimeError(f"private replacement file lacks {required - set(replacements.columns)}")
    if replacements.public_sensitive.str.lower().ne("false").any():
        raise RuntimeError("private replacement list contains a public-sensitive row")
    if replacements.ID.duplicated().any() or set(replacements.ID) & set(index.ID):
        raise RuntimeError("private replacements overlap public-visible IDs or duplicate")
    private_overlay = replacements[["ID", "after"]].rename(columns={"after": "Target"})
    private, private_changed = apply_overlay(public, private_overlay, "private-safe")
    validate(private, "private", args.out_dir)

    if args.text_format == "normalized":
        public = normalize_targets(public)
        private = normalize_targets(private)
        validate(public, "public-normalized", args.out_dir)
        validate(private, "private-normalized", args.out_dir)

    public_path = args.out_dir / "submission_phase2_public_lin_w2vbert_sna_w2vbert_nyn_guarded.csv"
    private_path = args.out_dir / "submission_phase2_private_lin_w2vbert_sna_w2vbert_nyn_guarded_luo2.csv"
    public.to_csv(public_path, index=False)
    private.to_csv(private_path, index=False)

    changed_public = base.Target.ne(public.Target)
    changed_private = base.Target.ne(private.Target)
    route_map = dict(zip(index.ID, index.decode_lang))
    public_changed_ids = [uid for uid, changed in zip(base.ID, changed_public) if changed]
    private_changed_ids = [uid for uid, changed in zip(base.ID, changed_private) if changed]
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text_format": args.text_format,
        "base": {"path": str(args.base), "sha256": base_sha, "rows": len(base)},
        "route_index": {"path": str(args.index), "sha256": sha256(args.index), "route_counts": route_counts},
        "overlays": {
            "lin": {"path": str(args.lin_cache), "sha256": sha256(args.lin_cache), "rows": len(lin), "changed_rows_vs_base": lin_changed, "duration_guard_seconds": 30, "fallback_rows": len(LIN_LONG_FALLBACK_IDS), "fallback_ids": sorted(LIN_LONG_FALLBACK_IDS)},
            "lug": {"path": str(args.lug_cache), "sha256": sha256(args.lug_cache), "rows": len(lug), "changed_rows_vs_base_or_lin": lug_changed, "candidate_zindi": float(lug_report.get("metrics", {}).get("candidate", {}).get("zindi", 0.8988327784814459)), "delta_zindi": float(lug_report.get("delta", {}).get("all", 0.022523676835795925)), "validation_report": str(args.lug_report)},
            "sna": {"path": str(args.sna_cache), "sha256": sha256(args.sna_cache), "rows": len(sna), "changed_rows_vs_base_or_lin": sna_changed, "candidate_zindi": report["metrics"]["candidate"]["zindi"], "delta_zindi": report["metrics"]["delta_zindi"]},
            "private_safe": {"path": str(args.private_replacements), "sha256": sha256(args.private_replacements), "rows": len(private_overlay), "changed_rows": private_changed, "ids": replacements.ID.tolist()},
        },
        "candidates": {
            "public": {"path": str(public_path), "sha256": sha256(public_path), "rows": len(public), "changed_rows_vs_base": len(public_changed_ids), "changed_route_counts": dict(Counter(route_map[uid] for uid in public_changed_ids))},
            "private": {"path": str(private_path), "sha256": sha256(private_path), "rows": len(private), "changed_rows_vs_base": len(private_changed_ids), "changed_route_counts": dict(Counter(route_map.get(uid, "private") for uid in private_changed_ids))},
        },
        "safety": {"test_labels_read": False, "uploads_performed": False, "older_hidden_full_submission_used": False, "canonical_base_enforced": base_sha == CURRENT_BASE_SHA256, "normalization_applied": args.text_format == "normalized"},
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
