#!/usr/bin/env python3
"""Evaluate sulaimank/w2vbert-lingala-waxal-cased in an isolated scope.

This is a Lingala-only wrapper around the locked seed-42/n=80 protocol in
``eval_sulaiman_public_descendants.py``.  It changes only the output scope and
candidate specification; the incumbent, validation sampling, metrics,
bootstrap seed, and four-column Phase2 route projection remain identical.

The Phase2 cache is decoded only when the inherited strong-pass gate succeeds.
The route projection contains ID/decode_lang/split/audio only, so no target or
transcription field can be read from the Phase2 index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep all dataset-derived files created by this run in the new scope.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "goal_2026_08_08" / "lin_cased_public"
os.environ.setdefault("HF_DATASETS_CACHE", str(OUT / "hf_datasets_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scripts import eval_sulaiman_public_descendants as base

base.OUT = OUT
base.SPECS = {}

SPEC = base.ModelSpec(
    tag="w2vbert-lingala-waxal-cased",
    model_id="sulaimank/w2vbert-lingala-waxal-cased",
    checkpoint=ROOT / "checkpoints" / "sulaimank-w2vbert-lingala-waxal-cased",
    lang="lin",
    kind="w2vbert",
)


def duration_anomalies(examples: list[dict], detail) -> dict:
    durations = {str(ex["_id"]): len(ex["_array"]) / base.TARGET_SR for ex in examples}
    rows = []
    for row in detail.itertuples(index=False):
        hyp = str(row.candidate)
        ref = str(row.reference)
        duration = float(durations[str(row.ID)])
        rows.append(
            {
                "ID": str(row.ID),
                "duration_s": duration,
                "duration_bucket": "short" if duration < 10 else "medium" if duration < 30 else "long",
                "reference_chars": len(ref),
                "candidate_chars": len(hyp),
                "empty_placeholder": hyp == ".",
                "nonempty_reference_empty_candidate": bool(ref != "." and hyp == "."),
                "candidate_excess_chars_over_ref": len(hyp) - len(ref),
            }
        )
    frame = base.pd.DataFrame(rows)
    bucket = {}
    for name, part in frame.groupby("duration_bucket", sort=False):
        ids = set(part.ID)
        scored = detail[detail.ID.isin(ids)]
        bucket[name] = {
            "n": int(len(part)),
            "duration_min_s": float(part.duration_s.min()),
            "duration_max_s": float(part.duration_s.max()),
            "candidate_empty_placeholders": int(part.empty_placeholder.sum()),
            "candidate": base.metric(scored.candidate, scored.reference),
            "incumbent": base.metric(scored.incumbent, scored.reference),
        }
    return {
        "summary": {
            "n": int(len(frame)),
            "duration_min_s": float(frame.duration_s.min()),
            "duration_median_s": float(frame.duration_s.median()),
            "duration_max_s": float(frame.duration_s.max()),
            "candidate_empty_placeholders": int(frame.empty_placeholder.sum()),
            "candidate_nonempty_reference_empty": int(frame.nonempty_reference_empty_candidate.sum()),
            "candidate_abs_char_delta_p95": float(frame.candidate_excess_chars_over_ref.abs().quantile(0.95)),
        },
        "by_duration_bucket": bucket,
        "detail_path": str(OUT / "validation_duration_anomalies.csv"),
        "detail": frame,
    }


def static_scope_audit() -> dict:
    source = Path(__file__).read_text()
    return {
        "candidate_source": str(Path(__file__).resolve()),
        "locked_protocol_source": str((ROOT / "scripts" / "eval_sulaiman_public_descendants.py").resolve()),
        "validation_loader": "load_hf_asr_split('lin', 'validation')",
        "phase2_usecols": ["ID", "decode_lang", "split", "audio"],
        "phase2_label_columns_read": False,
        "test_transcripts_read": False,
        "submission_built": False,
        "source_contains_test_split_loader": "load_hf_asr_split(\"lin\", \"test\")" in source,
        "source_contains_target_phase2_projection": "usecols=[\"ID\", \"decode_lang\", \"split\", \"audio\"]" not in source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-if-pass", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    device = base.pick_device(args.device)
    audit = base.checkpoint_audit(SPEC)
    examples, manifest = base.validation_sample("lin")
    manifest_path = OUT / "validation_manifest_lin.csv"
    if manifest_path.is_file():
        old = base.pd.read_csv(manifest_path, dtype={"ID": str})
        if old.ID.tolist() != manifest.ID.tolist() or old.reference.tolist() != manifest.reference.tolist():
            raise RuntimeError(f"immutable manifest changed: {manifest_path}")
    else:
        manifest.to_csv(manifest_path, index=False)

    route = base.phase2_route("lin")
    overlap = sorted(set(manifest.ID) & set(route.ID))
    if overlap:
        raise RuntimeError(f"validation/Phase2 ID overlap: {overlap[:5]}")

    detail = base.decode_validation(SPEC, examples, manifest, device, args.batch_size)
    result = base.evaluate(detail)
    anomalies = duration_anomalies(examples, detail)
    anomalies["detail"].to_csv(OUT / "validation_duration_anomalies.csv", index=False)
    result["duration_anomaly_checks"] = {k: v for k, v in anomalies.items() if k != "detail"}
    result["route_audit"] = {
        "rows": int(len(route)),
        "unique_ids": int(route.ID.nunique()),
        "expected_rows": 444,
        "exact_route_id_sha256": base.sha_lines(route.ID.tolist()),
        "validation_phase2_overlap": 0,
    }
    report = {
        "protocol": {
            "dataset": "google/WaxalNLP validation",
            "sample_seed": base.SAMPLE_SEED,
            "n": base.SAMPLE_N,
            "validation_ids_sha256": base.sha_lines(manifest.ID.tolist()),
            "normalization": "src.text_norm.normalize_text",
            "bootstrap_draws": base.BOOTSTRAP_DRAWS,
            "bootstrap_seed": 20260808,
            "test_labels_read": False,
            "submission_built": False,
        },
        "scope_audit": static_scope_audit(),
        "model": {"tag": SPEC.tag, "model_id": SPEC.model_id, "audit": audit},
        "metrics": result,
        "validation_detail": str(OUT / f"validation_{SPEC.tag}.csv"),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.cache_if_pass and result["strong_pass"]:
        cache = base.decode_phase2_cache(SPEC, device, set(manifest.ID), args.batch_size)
        report["phase2_cache"] = {
            "path": str(cache),
            "rows": int(len(base.pd.read_csv(cache))),
            "sha256": base.sha256_file(cache) if hasattr(base, "sha256_file") else None,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    elif args.cache_if_pass:
        print("no strong pass; Phase2 audio was not decoded", flush=True)


if __name__ == "__main__":
    main()
