#!/usr/bin/env python3
"""Isolated, resumable CPU evaluation of Sulaiman Shona SD2.

This lane intentionally writes only under outputs/goal_2026_08_08/
shona_sd2_parallel and never builds or uploads a competition submission.
It reuses the immutable seed-42/n=80 protocol and strict promotion gate from
eval_sulaiman_public_descendants.py, but expands the deployment cache to all
461 public-visible Shona-routed Phase-2 rows (445 new + 16 old).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "goal_2026_08_08" / "shona_sd2_parallel"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", str(OUT / "hf_datasets_cache"))
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoProcessor

from scripts import eval_sulaiman_public_descendants as base

TAG = "w2vbert-shona-sd2"
EXPECTED_ROUTE_ROWS = 461
ROUTES = ROOT / "outputs" / "beat075" / "public_visible_index.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase2_route_all_shona(lang: str) -> pd.DataFrame:
    if lang != "sna":
        raise ValueError(f"this isolated evaluator supports sna only, got {lang!r}")
    # Explicit projection excludes Target/transcription/reference columns.
    frame = pd.read_csv(
        ROUTES,
        usecols=["ID", "decode_lang", "split", "audio"],
        dtype={"ID": str, "decode_lang": str, "split": str, "audio": str},
    )
    frame = frame[frame.decode_lang.eq("sna")].copy()
    frame = frame.sort_values("ID", kind="stable").reset_index(drop=True)
    counts = frame.groupby("split").size().to_dict()
    if counts != {"new": 445, "old": 16}:
        raise RuntimeError(f"unexpected Shona route split counts: {counts}")
    if len(frame) != EXPECTED_ROUTE_ROWS or frame.ID.nunique() != EXPECTED_ROUTE_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_ROUTE_ROWS} unique public-visible Shona IDs, "
            f"got rows={len(frame)} unique={frame.ID.nunique()}"
        )
    missing = [str(path) for path in frame.audio.map(Path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase-2 audio, e.g. {missing[:3]}")
    return frame


def tokenizer_audit(spec: base.ModelSpec) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(str(spec.checkpoint), local_files_only=True)
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    by_id: dict[int, list[str]] = {}
    for token, token_id in vocab.items():
        by_id.setdefault(int(token_id), []).append(str(token))
    duplicate_ids = {str(k): sorted(v) for k, v in by_id.items() if len(v) > 1}
    pad_id = int(tokenizer.pad_token_id)
    delimiter_id = int(tokenizer.convert_tokens_to_ids(tokenizer.word_delimiter_token))
    unk_id = int(tokenizer.unk_token_id)
    synthetic_ids = [vocab["a"], delimiter_id, vocab["b"], pad_id, vocab["c"]]
    synthetic_decode = tokenizer.decode(synthetic_ids)
    audit = {
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_entries": len(vocab),
        "unique_vocab_ids": len(by_id),
        "duplicate_ids": duplicate_ids,
        "model_vocab_size": int(json.loads((spec.checkpoint / "config.json").read_text())["vocab_size"]),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": pad_id,
        "ctc_blank_matches_pad": pad_id
        == int(json.loads((spec.checkpoint / "config.json").read_text())["pad_token_id"]),
        "word_delimiter_token": tokenizer.word_delimiter_token,
        "word_delimiter_id": delimiter_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": unk_id,
        "synthetic_ids": synthetic_ids,
        "synthetic_decode": synthetic_decode,
        "delimiter_decodes_as_space": " " in synthetic_decode,
        "feature_extractor_sampling_rate": int(processor.feature_extractor.sampling_rate),
    }
    if duplicate_ids:
        raise RuntimeError(f"ambiguous tokenizer IDs: {duplicate_ids}")
    if not audit["ctc_blank_matches_pad"]:
        raise RuntimeError("CTC blank does not match tokenizer pad ID")
    if not audit["delimiter_decodes_as_space"]:
        raise RuntimeError(f"word delimiter did not decode as a space: {synthetic_decode!r}")
    if audit["feature_extractor_sampling_rate"] != 16000:
        raise RuntimeError(f"unexpected sampling rate: {audit['feature_extractor_sampling_rate']}")
    return audit


def _read_partial(path: Path, manifest: pd.DataFrame, column: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype={"ID": str, column: str})
    if frame.ID.duplicated().any() or not set(frame.ID).issubset(set(manifest.ID)):
        raise RuntimeError(f"stale or duplicate partial cache: {path}")
    if frame[column].isna().any():
        raise RuntimeError(f"empty hypothesis in partial cache: {path}")
    return dict(zip(frame.ID.astype(str), frame[column].astype(str)))


def _write_partial(path: Path, manifest: pd.DataFrame, done: dict[str, str], column: str) -> None:
    rows = [{"ID": uid, column: done[uid]} for uid in manifest.ID if uid in done]
    pd.DataFrame(rows, columns=["ID", column]).to_csv(path, index=False)


def decode_validation_resumable(
    spec: base.ModelSpec,
    examples: list[dict[str, Any]],
    manifest: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    detail_path = OUT / f"validation_{TAG}.csv"
    if detail_path.is_file():
        detail = pd.read_csv(detail_path, dtype={"ID": str})
        if detail.ID.tolist() != manifest.ID.tolist():
            raise RuntimeError(f"stale final validation cache: {detail_path}")
        return detail

    jobs = [
        ("incumbent", "w2vbert", base.load_incumbent, OUT / "validation_incumbent_partial.csv"),
        ("candidate", spec.kind, lambda _lang, dev: (TAG, *base.load_candidate(spec, dev)), OUT / "validation_candidate_partial.csv"),
    ]
    completed: dict[str, dict[str, str]] = {}
    for column, kind, loader, path in jobs:
        done = _read_partial(path, manifest, column)
        completed[column] = done
        todo = [ex for ex in examples if ex["_id"] not in done]
        if not todo:
            continue
        tag, model, processor = loader(spec.lang, device)
        started = time.time()
        try:
            for start in range(0, len(todo), batch_size):
                chunk = todo[start : start + batch_size]
                hyps = base.decode_many(
                    model, processor, kind, [ex["_array"] for ex in chunk], device
                )
                for ex, hyp in zip(chunk, hyps):
                    done[ex["_id"]] = hyp
                _write_partial(path, manifest, done, column)
                print(
                    f"{tag}: {len(done)}/{len(manifest)} "
                    f"({time.time() - started:.1f}s this run)",
                    flush=True,
                )
        finally:
            base.release(model, processor, device=device)

    for column, done in completed.items():
        if set(done) != set(manifest.ID):
            raise RuntimeError(f"incomplete {column} validation cache: {len(done)}/80")
    detail = manifest.copy()
    detail["incumbent"] = detail.ID.map(completed["incumbent"])
    detail["candidate"] = detail.ID.map(completed["candidate"])
    detail.to_csv(detail_path, index=False)
    return detail


def write_report(
    spec: base.ModelSpec,
    checkpoint: dict[str, Any],
    tokens: dict[str, Any],
    manifest: pd.DataFrame,
    result: dict[str, Any],
    cache: Path | None,
) -> Path:
    route = phase2_route_all_shona("sna")
    report = {
        "protocol": {
            "dataset": "google/WaxalNLP validation",
            "sample_seed": base.SAMPLE_SEED,
            "n": base.SAMPLE_N,
            "device": "cpu",
            "maximum_batch_size": 2,
            "normalization": "src.text_norm.normalize_text",
            "incumbent": "badrex/w2v-bert-2.0-shona-asr greedy CTC",
            "candidate": spec.model_id + " greedy CTC",
            "pass_gate": result["pass_rule"],
            "test_labels_read": False,
            "submission_built": False,
            "submission_uploaded": False,
        },
        "checkpoint_audit": checkpoint,
        "tokenizer_audit": tokens,
        "id_and_leakage_audit": {
            "validation_rows": len(manifest),
            "validation_unique_ids": int(manifest.ID.nunique()),
            "validation_ids_sha256": base.sha_lines(manifest.ID.tolist()),
            "phase2_route_rows": len(route),
            "phase2_route_unique_ids": int(route.ID.nunique()),
            "phase2_route_split_counts": route.groupby("split").size().to_dict(),
            "phase2_route_ids_sha256": base.sha_lines(route.ID.tolist()),
            "validation_phase2_id_overlap": sorted(set(manifest.ID) & set(route.ID)),
            "phase2_columns_read": ["ID", "decode_lang", "split", "audio"],
            "test_transcript_columns_read": [],
            "card_provenance_warning": checkpoint.get("card_provenance_warning"),
        },
        "metrics": result,
        "artifacts": {
            "validation_detail": str(OUT / f"validation_{TAG}.csv"),
            "phase2_cache": str(cache) if cache else None,
            "phase2_cache_rows": EXPECTED_ROUTE_ROWS if cache else 0,
            "phase2_cache_sha256": sha256_file(cache) if cache else None,
        },
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache-if-pass", action="store_true")
    args = parser.parse_args()
    if args.device != "cpu":
        raise ValueError("this parallel lane is explicitly CPU-only")
    if not 1 <= args.batch_size <= 2:
        raise ValueError("--batch-size must be 1 or 2")

    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.phase2_route = phase2_route_all_shona
    spec = base.SPECS[TAG]
    device = torch.device("cpu")

    checkpoint = base.checkpoint_audit(spec)
    tokens = tokenizer_audit(spec)
    examples, manifest = base.validation_sample("sna")
    manifest_path = OUT / "validation_manifest_sna.csv"
    if manifest_path.is_file():
        old = pd.read_csv(manifest_path, dtype={"ID": str})
        if old.ID.tolist() != manifest.ID.tolist() or old.reference.tolist() != manifest.reference.tolist():
            raise RuntimeError("immutable seed-42 manifest changed")
    else:
        manifest.to_csv(manifest_path, index=False)

    route = phase2_route_all_shona("sna")
    overlap = sorted(set(manifest.ID) & set(route.ID))
    if overlap:
        raise RuntimeError(f"validation/Phase-2 ID leakage: {overlap[:5]}")

    detail = decode_validation_resumable(spec, examples, manifest, device, args.batch_size)
    result = base.evaluate(detail)
    cache = None
    write_report(spec, checkpoint, tokens, manifest, result, cache)
    print(json.dumps({"model": TAG, "metrics": result}, indent=2), flush=True)

    if args.cache_if_pass and result["strong_pass"]:
        cache = base.decode_phase2_cache(spec, device, set(manifest.ID), args.batch_size)
        if cache.name != f"phase2_cache_{TAG}.csv":
            raise RuntimeError(f"unexpected cache path: {cache}")
        final = pd.read_csv(cache, dtype={"ID": str, "Target": str})
        if final.ID.tolist() != route.ID.tolist() or final.Target.isna().any():
            raise RuntimeError("final Phase-2 cache failed exact route/order/nonempty audit")
        write_report(spec, checkpoint, tokens, manifest, result, cache)
        print(f"strong pass; complete cache={cache} sha256={sha256_file(cache)}", flush=True)
    elif args.cache_if_pass:
        print("no strong pass; Phase-2 audio was not decoded", flush=True)


if __name__ == "__main__":
    main()
