#!/usr/bin/env python3
"""Train/calibrate S02 SSL kNN gate on open data only.

Non-LID head: frozen SSL encoder mean-pool embeddings → class prototypes
(luo from FLEURS luo_ke, ach from WAXAL ach_asr) → margin score
d(x,ach)−d(x,luo). Threshold calibrated on open **validation** holdout to
maximize TPR s.t. FPR ≤ max_fpr (default 5%).

Hygiene:
  - fit prototypes on **train** splits only
  - calibrate thr on **validation** only
  - never load/use FORBIDDEN_TRAIN_SPLITS (test) for fit or thr
  - does NOT overwrite submission_phase2_FINAL.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, SEED, TARGET_SR
from src.s02_ssl_knn_gate import (
    S02Thresholds,
    calibrate_threshold,
    eval_binary_scores,
    knn_margin_scores,
    l2_normalize,
    mean_prototype,
)
from src.torch_env import describe_torch, pick_torch_device


DEFAULT_MODEL = "facebook/wav2vec2-base"
# Prefer multilingual SSL if already cached / requested
DEFAULT_MODEL_CANDIDATES = (
    "facebook/wav2vec2-xls-r-300m",
    "facebook/wav2vec2-base",
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _array_from_hf(ex) -> tuple[np.ndarray, int]:
    """Decode HF audio without torchcodec (soundfile on path or bytes)."""
    import io

    import soundfile as sf

    audio = ex["audio"]
    if isinstance(audio, dict) and audio.get("array") is not None:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or TARGET_SR)
    elif isinstance(audio, dict) and audio.get("bytes") is not None:
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        sr = int(sr)
    elif isinstance(audio, dict) and audio.get("path") and Path(str(audio["path"])).exists():
        arr, sr = sf.read(str(audio["path"]), dtype="float32", always_2d=False)
        sr = int(sr)
    else:
        raise ValueError(
            f"unsupported audio payload keys="
            f"{list(audio) if isinstance(audio, dict) else type(audio)}"
        )
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)) + 1e-9)
    if peak > 0:
        arr = arr / peak
    return arr, sr


def _resample(arr: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if int(sr) == int(target):
        return arr
    import librosa

    return librosa.resample(arr, orig_sr=int(sr), target_sr=int(target)).astype(np.float32)


class SSLEmbedder:
    """Frozen wav2vec2-family encoder; mean-pool last_hidden_state → L2 emb."""

    def __init__(self, model_id: str, device: torch.device):
        from transformers import AutoFeatureExtractor, Wav2Vec2Model

        self.model_id = model_id
        self.device = device
        try:
            self.fe = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
            self.model = Wav2Vec2Model.from_pretrained(model_id, local_files_only=True)
            self.loaded = "local"
        except Exception:
            self.fe = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id)
            self.loaded = "download"
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def embed(self, array: np.ndarray, sr: int) -> np.ndarray:
        arr = _resample(array, sr, TARGET_SR)
        # cap length for memory (30s @ 16k)
        max_samples = int(30 * TARGET_SR)
        if arr.shape[0] > max_samples:
            arr = arr[:max_samples]
        inputs = self.fe(arr, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # mean pool over time
        h = out.last_hidden_state[0]  # [T, H]
        emb = h.mean(dim=0).detach().cpu().numpy().astype(np.float64)
        return l2_normalize(emb)


def _local_parquet_files(dataset: str, config: str, split: str) -> list[str]:
    """Resolve cached parquet shards for FLEURS / WAXAL open splits (no network)."""
    import os

    cache_root = Path(
        os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub")
    )
    out: list[str] = []

    if dataset == "google/fleurs" and config == "luo_ke":
        hub = cache_root / "datasets--google--fleurs" / "snapshots"
        if hub.is_dir():
            for snap in sorted(hub.iterdir(), reverse=True):
                pq = snap / "parquet-data" / "luo_ke" / f"{split}-00000-of-00001.parquet"
                if pq.resolve().is_file():
                    return [str(pq.resolve())]
                # also accept any matching name
                found = sorted((snap / "parquet-data" / "luo_ke").glob(f"{split}-*.parquet"))
                found = [p for p in found if p.resolve().is_file()]
                if found:
                    return [str(p.resolve()) for p in found]

    if dataset == "google/WaxalNLP" and config == "ach_asr":
        # never pick unlabeled/test
        if split in FORBIDDEN_TRAIN_SPLITS or split == "unlabeled":
            return []
        hub = cache_root / "datasets--google--WaxalNLP" / "snapshots"
        needle = f"ach-{split}-"
        if hub.is_dir():
            for snap in sorted(hub.iterdir(), reverse=True):
                asr_dir = snap / "data" / "ASR" / "ach"
                if not asr_dir.is_dir():
                    continue
                found = sorted(
                    p
                    for p in asr_dir.glob("*.parquet")
                    if needle in p.name and p.resolve().is_file()
                )
                if found:
                    return [str(p.resolve()) for p in found]
    return out


def stream_embeddings(
    *,
    dataset: str,
    config: str,
    split: str,
    max_n: int,
    embedder: SSLEmbedder,
    seed: int,
    label: str,
) -> tuple[np.ndarray, dict]:
    """Load open split (local parquet preferred), extract embeddings.

    Refuses forbidden splits (test) for any use.
    """
    from datasets import Audio, load_dataset

    if split in FORBIDDEN_TRAIN_SPLITS:
        raise RuntimeError(
            f"refusing to load split={split!r} for S02 ({label}): "
            f"FORBIDDEN_TRAIN_SPLITS={sorted(FORBIDDEN_TRAIN_SPLITS)}"
        )

    meta: dict = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "max_n": max_n,
        "label": label,
        "audio_decode": "soundfile_bytes_or_path",
    }

    local_files = _local_parquet_files(dataset, config, split)
    rows_iter = None
    if local_files:
        meta["load_mode"] = "local_parquet"
        meta["parquet_files"] = local_files
        print(f"  {label}: local parquet n_files={len(local_files)}", flush=True)
        ds = load_dataset("parquet", data_files={split: local_files}, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        # subsample with seed for train diversity
        n_total = len(ds)
        if n_total > max_n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_total, size=max_n, replace=False)
            idx = sorted(int(i) for i in idx)
            ds = ds.select(idx)
        rows_iter = (ds[i] for i in range(len(ds)))
        meta["n_available"] = n_total
    else:
        meta["load_mode"] = "streaming_fallback"
        print(f"  {label}: streaming fallback {dataset}/{config}/{split}", flush=True)
        stream = load_dataset(dataset, config, split=split, streaming=True)
        stream = stream.cast_column("audio", Audio(decode=False))
        if split == "train":
            stream = stream.shuffle(seed=seed, buffer_size=min(256, max(max_n * 2, 32)))
        rows_iter = stream

    embs: list[np.ndarray] = []
    errors = 0
    for ex in rows_iter:
        if len(embs) >= max_n:
            break
        try:
            arr, sr = _array_from_hf(ex)
            e = embedder.embed(arr, sr)
            embs.append(e)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"  warn decode/embed skip ({label}): {exc}", flush=True)
            continue
        if (len(embs) % 10) == 0:
            print(f"  {label} {split}: {len(embs)}/{max_n}", flush=True)

    if not embs:
        raise RuntimeError(f"no embeddings extracted for {label} {dataset}/{config}/{split}")
    mat = np.stack(embs, axis=0)
    meta["n"] = int(mat.shape[0])
    meta["dim"] = int(mat.shape[1])
    meta["errors_skipped"] = errors
    return mat, meta


def resolve_model_id(requested: str | None) -> str:
    if requested:
        return requested
    # Prefer local cache when present
    hub = Path.home() / ".cache/huggingface/hub"
    for mid in DEFAULT_MODEL_CANDIDATES:
        slug = "models--" + mid.replace("/", "--")
        if (hub / slug).exists():
            return mid
    return DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    p = argparse.ArgumentParser(description="S02 SSL kNN train+calib (open data only)")
    p.add_argument("--model-id", type=str, default=None, help="HF SSL model id")
    p.add_argument("--max-luo-fit", type=int, default=80, help="FLEURS luo train for prototypes")
    p.add_argument("--max-ach-fit", type=int, default=80, help="WAXAL ach train for prototypes")
    p.add_argument("--max-luo-val", type=int, default=60, help="FLEURS luo val holdout")
    p.add_argument("--max-ach-val", type=int, default=80, help="WAXAL ach val holdout")
    p.add_argument("--max-fpr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/new_signals",
    )
    p.add_argument("--tag", type=str, default="", help="optional suffix for artifact names")
    args = p.parse_args(argv)

    t0 = time.time()
    set_seed(args.seed)
    device = pick_torch_device()
    model_id = resolve_model_id(args.model_id)
    torch_info = describe_torch()
    print(
        json.dumps(
            {
                "event": "s02_start",
                "device": str(device),
                "model_id": model_id,
                "max_fpr": args.max_fpr,
                "seed": args.seed,
                "torch": torch_info,
                "forbidden_train_splits": sorted(FORBIDDEN_TRAIN_SPLITS),
                "non_lid": True,
                "fit_splits": ["train"],
                "calib_splits": ["validation"],
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )

    print(f"loading SSL embedder {model_id} ...", flush=True)
    embedder = SSLEmbedder(model_id, device)
    print(f"embedder ready (loaded={embedder.loaded})", flush=True)

    # --- FIT on train ---
    print("FIT: FLEURS luo_ke train + WAXAL ach_asr train", flush=True)
    luo_fit, luo_fit_meta = stream_embeddings(
        dataset="google/fleurs",
        config="luo_ke",
        split="train",
        max_n=args.max_luo_fit,
        embedder=embedder,
        seed=args.seed,
        label="luo_fit",
    )
    ach_fit, ach_fit_meta = stream_embeddings(
        dataset="google/WaxalNLP",
        config="ach_asr",
        split="train",
        max_n=args.max_ach_fit,
        embedder=embedder,
        seed=args.seed,
        label="ach_fit",
    )
    luo_proto = mean_prototype(luo_fit)
    ach_proto = mean_prototype(ach_fit)
    print(
        f"prototypes: luo_fit n={luo_fit.shape[0]} ach_fit n={ach_fit.shape[0]} dim={luo_proto.shape[0]}",
        flush=True,
    )

    # --- CALIB on validation holdout ---
    print("CALIB: FLEURS luo_ke validation + WAXAL ach_asr validation", flush=True)
    luo_val, luo_val_meta = stream_embeddings(
        dataset="google/fleurs",
        config="luo_ke",
        split="validation",
        max_n=args.max_luo_val,
        embedder=embedder,
        seed=args.seed,
        label="luo_val",
    )
    ach_val, ach_val_meta = stream_embeddings(
        dataset="google/WaxalNLP",
        config="ach_asr",
        split="validation",
        max_n=args.max_ach_val,
        embedder=embedder,
        seed=args.seed,
        label="ach_val",
    )

    scores_luo = knn_margin_scores(luo_val, luo_proto, ach_proto)
    scores_ach = knn_margin_scores(ach_val, luo_proto, ach_proto)
    thr, stats = calibrate_threshold(
        scores_luo.tolist(), scores_ach.tolist(), max_fpr=args.max_fpr
    )
    # re-eval for explicit report
    labels = [True] * len(scores_luo) + [False] * len(scores_ach)
    all_scores = scores_luo.tolist() + scores_ach.tolist()
    reval = eval_binary_scores(all_scores, labels, thr)

    fpr = float(stats.get("fpr", reval["fpr"]))
    tpr = float(stats.get("tpr", reval["tpr"]))
    ok = fpr <= args.max_fpr + 1e-12
    elapsed = time.time() - t0

    floor_path = ROOT / "submission_phase2_FINAL.csv"
    floor_sha = hashlib.sha256(floor_path.read_bytes()).hexdigest() if floor_path.exists() else None

    tag = f"_{args.tag}" if args.tag else ""
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"s02_ssl_knn_prototypes{tag}.npz"
    calib_path = out_dir / f"s02_ssl_knn_calib{tag}.json"

    np.savez_compressed(
        npz_path,
        luo_proto=luo_proto.astype(np.float64),
        ach_proto=ach_proto.astype(np.float64),
        scores_luo_val=scores_luo.astype(np.float64),
        scores_ach_val=scores_ach.astype(np.float64),
        luo_fit_mean_emb=luo_fit.mean(axis=0),
        ach_fit_mean_emb=ach_fit.mean(axis=0),
    )

    artifact = {
        "signal": "S02",
        "name": "ssl_knn_lang_router",
        "non_lid": True,
        "model_id": model_id,
        "embedder_loaded": embedder.loaded,
        "device": str(device),
        "seed": args.seed,
        "max_fpr_target": args.max_fpr,
        "threshold": thr.as_dict(),
        "open_val_metrics": {
            "tpr": tpr,
            "fpr": fpr,
            "tp": reval.get("tp"),
            "fp": reval.get("fp"),
            "tn": reval.get("tn"),
            "fn": reval.get("fn"),
            "n_pos": reval.get("n_pos"),
            "n_neg": reval.get("n_neg"),
            "note": stats.get("note"),
            "candidates_ok": stats.get("candidates_ok"),
        },
        "score_summary": {
            "luo_val_mean": float(np.mean(scores_luo)),
            "luo_val_std": float(np.std(scores_luo)),
            "ach_val_mean": float(np.mean(scores_ach)),
            "ach_val_std": float(np.std(scores_ach)),
            "luo_val_min": float(np.min(scores_luo)),
            "luo_val_max": float(np.max(scores_luo)),
            "ach_val_min": float(np.min(scores_ach)),
            "ach_val_max": float(np.max(scores_ach)),
        },
        "fit": {"luo": luo_fit_meta, "ach": ach_fit_meta},
        "val": {"luo": luo_val_meta, "ach": ach_val_meta},
        "hygiene": {
            "forbidden_train_splits": sorted(FORBIDDEN_TRAIN_SPLITS),
            "test_used": False,
            "fit_splits": ["train"],
            "calib_splits": ["validation"],
            "sources": [
                "google/fleurs:luo_ke:train|validation",
                "google/WaxalNLP:ach_asr:train|validation",
            ],
        },
        "prototypes_npz": str(npz_path),
        "floor_sha256_unchanged": floor_sha,
        "floor_path": str(floor_path),
        "elapsed_sec": round(elapsed, 2),
        "fpr_le_max": ok,
        "criterion_open_val_fpr_le_5pct": ok,
        "status": "PASS_fpr" if ok else "FAIL_fpr_or_fail_closed",
    }
    calib_path.write_text(json.dumps(artifact, indent=2))

    print(
        json.dumps(
            {
                "event": "s02_calib_done",
                "tpr": tpr,
                "fpr": fpr,
                "max_fpr": args.max_fpr,
                "fpr_le_max": ok,
                "thr": thr.as_dict(),
                "n_luo_val": int(len(scores_luo)),
                "n_ach_val": int(len(scores_ach)),
                "n_luo_fit": int(luo_fit.shape[0]),
                "n_ach_fit": int(ach_fit.shape[0]),
                "note": stats.get("note"),
                "calib_json": str(calib_path),
                "prototypes_npz": str(npz_path),
                "floor_sha256": floor_sha,
                "elapsed_sec": round(elapsed, 2),
                "status": artifact["status"],
            },
            indent=2,
        ),
        flush=True,
    )

    if not ok:
        print(
            f"S02_FAIL: open-val fpr={fpr:.6f} > max_fpr={args.max_fpr} "
            f"(or fail_closed thr). See {calib_path}",
            flush=True,
        )
        return 2

    print(
        f"S02_PASS: open-val FPR={fpr:.6f} ≤ {args.max_fpr} TPR={tpr:.6f} thr={thr.min_score}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
