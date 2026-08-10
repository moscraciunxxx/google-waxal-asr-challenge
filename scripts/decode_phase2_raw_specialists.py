#!/usr/bin/env python3
"""Resumably decode public Lingala/Shona routes without lowercasing or stripping.

Only the route index (ID/decode_lang/split/audio) and audio are read.  This is a
separate producer from the normalized caches so the two text representations can
be compared without overwriting evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eval_sulaiman_public_descendants as base
from src.text_norm import normalize_text

ROUTES = ROOT / "outputs/beat075/public_visible_index.csv"
OUT = ROOT / "outputs/goal_2026_08_08/raw_phase2"
EXPECTED = {"lin": 444, "sna": 461}
SPECS = {lang: base.SPECS[tag] for lang, tag in {
    "lin": "w2vbert-lingala-sd3",
    "sna": "w2vbert-shona-sd2",
}.items()}
WS = re.compile(r"\s+")


def raw_keep(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf"})
    return WS.sub(" ", text).strip()


def route(lang: str) -> pd.DataFrame:
    frame = pd.read_csv(
        ROUTES,
        usecols=["ID", "decode_lang", "split", "audio"],
        dtype={"ID": str, "decode_lang": str, "split": str, "audio": str},
        keep_default_na=False,
    )
    frame = frame.loc[frame.decode_lang.eq(lang)].sort_values("ID", kind="stable").reset_index(drop=True)
    if len(frame) != EXPECTED[lang] or frame.ID.nunique() != EXPECTED[lang]:
        raise RuntimeError(f"{lang}: expected {EXPECTED[lang]} unique route IDs, got {len(frame)}")
    missing = [p for p in frame.audio.map(Path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{lang}: missing audio, e.g. {missing[:3]}")
    return frame


@torch.inference_mode()
def decode_raw(model, processor, arrays: list[np.ndarray], device: torch.device) -> list[str]:
    inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    kwargs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    ids = torch.argmax(model(**kwargs).logits, dim=-1).detach().cpu()
    return [
        # Match the official-scoring forensic decoder exactly: processor.decode
        # defaults, followed only by Unicode/control/whitespace cleanup.
        raw_keep(processor.decode(row)) or "."
        for row in ids
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(SPECS), required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.batch_size < 1 or args.batch_size > 2:
        raise ValueError("batch-size must be 1 or 2")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    spec = SPECS[args.lang]
    jobs = route(args.lang)
    path = OUT / f"phase2_cache_raw_{spec.tag}.csv"
    done: dict[str, str] = {}
    if path.is_file():
        prior = pd.read_csv(path, dtype={"ID": str, "Target": str}, keep_default_na=False)
        if prior.ID.duplicated().any() or not set(prior.ID).issubset(set(jobs.ID)):
            raise RuntimeError(f"stale raw cache: {path}")
        if prior.Target.map(lambda x: not str(x).strip()).any():
            raise RuntimeError(f"empty raw prediction in partial cache: {path}")
        done = dict(zip(prior.ID, prior.Target))

    model, processor = base.load_candidate(spec, dev)
    started = time.time()
    try:
        todo = [row for row in jobs.itertuples(index=False) if row.ID not in done]
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start : start + args.batch_size]
            arrays = []
            for row in chunk:
                array, sr = sf.read(str(row.audio), dtype="float32", always_2d=False)
                if int(sr) != 16000:
                    import librosa
                    array = librosa.resample(np.asarray(array), orig_sr=int(sr), target_sr=16000)
                arrays.append(base.normalize_audio(array))
            for row, text in zip(chunk, decode_raw(model, processor, arrays, dev)):
                done[row.ID] = text
            ordered = [{"ID": uid, "Target": done[uid]} for uid in jobs.ID if uid in done]
            pd.DataFrame(ordered, columns=["ID", "Target"]).to_csv(path, index=False)
            print(f"{spec.tag} raw Phase2: {len(done)}/{len(jobs)} ({time.time()-started:.1f}s)", flush=True)
    finally:
        base.release(model, processor, device=dev)

    final = pd.read_csv(path, dtype={"ID": str, "Target": str}, keep_default_na=False)
    if final.ID.tolist() != jobs.ID.tolist() or final.Target.map(lambda x: not str(x).strip()).any():
        raise RuntimeError(f"raw cache failed exact route/order/nonempty audit: {path}")
    normalized_path = ROOT / "outputs/goal_2026_08_08" / (
        "sulaiman_public_descendants/phase2_cache_w2vbert-lingala-sd3.csv"
        if args.lang == "lin" else
        "shona_sd2_parallel/phase2_cache_w2vbert-shona-sd2.csv"
    )
    normalization_audit = {"normalized_cache": str(normalized_path), "compared": False, "mismatch_count": None, "mismatch_examples": []}
    if normalized_path.is_file():
        norm = pd.read_csv(normalized_path, dtype={"ID": str, "Target": str}, keep_default_na=False)
        expected_norm = dict(zip(norm.ID, norm.Target))
        mismatches = [uid for uid, text in zip(final.ID, final.Target) if normalize_text(text) != expected_norm.get(uid)]
        # CTC logits can depend on batch padding length.  The legacy normalized
        # producer was resumed with different batch settings, so a mismatch is
        # recorded rather than silently discarding a valid raw decode.  Raw
        # validation metrics were independently produced with this batch-2
        # policy; the raw cache remains the authoritative output of this run.
        normalization_audit = {
            "normalized_cache": str(normalized_path),
            "compared": True,
            "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:10],
        }
    audit_path = OUT / f"audit_{spec.tag}.json"
    audit_path.write_text(json.dumps(normalization_audit, indent=2) + "\n")
    print(f"normalization audit: {normalization_audit}", flush=True)
    print(f"complete: {path} rows={len(final)}", flush=True)


if __name__ == "__main__":
    main()
