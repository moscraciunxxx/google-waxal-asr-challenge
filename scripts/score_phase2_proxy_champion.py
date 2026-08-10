#!/usr/bin/env python3
"""Score frozen Phase-2 champion multi-hyp policy on proxy_val_index.

Champion policy (must match run_phase2_openset.py, waxal-300m family only):
  true ach (luo-domain sim): ach|lug|sog
  true lug:                  lug|nyn|sog
  true nyn:                  nyn|lug|sog
  true sog:                  sog|lug
  true mas:                  mas|lug
Pick by CTC mean logprob; greedy decode; src.text_norm + src.metrics.score_pairs.
zindi_est = 1 - 0.5*WER - 0.5*CER

Never touches Phase-1 test gold for train/tune. Does not modify submission CSVs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Prefer HF offline when models are already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.config import OUTPUT_DIR, TARGET_SR
from src import config as _cfg
# Proxy langs beyond Phase-1 challenge three
for _lang in ("ach", "nyn", "lug", "sog", "mas", "lin", "sna"):
    _cfg.HF_CONFIGS.setdefault(_lang, f"{_lang}_asr")
from src.dataset import load_hf_asr_split
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy_champion")

WAXAL300 = {
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
    "ach": "waxal-benchmarking/mms-300m-waxal-ach",
    "sog": "waxal-benchmarking/mms-300m-waxal-sog",
    "nyn": "waxal-benchmarking/mms-300m-waxal-nyn",
    "mas": "waxal-benchmarking/mms-300m-waxal-mas",
}

# Multi-hyp candidates for true_lang on proxy (Phase-2 openset-like routing)
CANDIDATES_BY_TRUE = {
    "ach": ["ach", "lug", "sog"],  # lid≈luo domain → ach|lug|sog
    "lug": ["lug", "nyn", "sog"],
    "nyn": ["nyn", "lug", "sog"],
    "sog": ["sog", "lug"],
    "mas": ["mas", "lug"],
}

PROXY_INDEX = ROOT / "data" / "proxy_val_index.csv"


def candidates_for(true_lang: str) -> list[str]:
    if true_lang in CANDIDATES_BY_TRUE:
        return list(CANDIDATES_BY_TRUE[true_lang])
    if true_lang in WAXAL300:
        return [true_lang]
    return ["lug"]


def load_waxal(lang: str, device: torch.device):
    mid = WAXAL300[lang]
    logger.info("Loading %s on %s (local_files_only preferred)", mid, device)
    try:
        processor = AutoProcessor.from_pretrained(mid, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True)
    except Exception as e:
        logger.warning("local_files_only failed for %s (%s); retrying online", mid, e)
        # Temporarily allow hub if offline env blocked cache miss
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        processor = AutoProcessor.from_pretrained(mid)
        model = Wav2Vec2ForCTC.from_pretrained(mid)
    model.to(device).eval()
    return model, processor


def free_model(model) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _resolve_val_parquets(lang: str) -> list[str]:
    """Locate cached WaxalNLP validation parquet shards for lang."""
    hub_ds = "google--WaxalNLP"
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub")
    )
    snap_root = cache_root / f"datasets--{hub_ds}" / "snapshots"
    needle = f"{lang}-validation-"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            asr_dir = snap / "data" / "ASR" / lang
            if not asr_dir.is_dir():
                continue
            found = sorted(
                p for p in asr_dir.glob("*.parquet") if needle in p.name and p.is_file()
            )
            if found:
                return [str(p.resolve()) for p in found]
    return []


def load_proxy_samples(
    index_csv: Path,
    max_per_lang: int | None,
    seed: int = 42,
) -> list[dict]:
    from datasets import Audio, load_dataset

    from src.dataset import _decode_audio_item

    idx = pd.read_csv(index_csv)
    if "language" not in idx.columns:
        raise SystemExit(f"proxy index missing language column: {index_csv}")
    want_ids: dict[str, set[str]] = {}
    for lang, g in idx.groupby("language"):
        ids = g["id"].astype(str).tolist()
        if max_per_lang is not None and len(ids) > max_per_lang:
            rng = np.random.default_rng(seed)
            pick = rng.choice(ids, size=max_per_lang, replace=False)
            ids = list(pick)
        want_ids[str(lang)] = set(ids)

    samples: list[dict] = []
    for lang, id_set in sorted(want_ids.items()):
        logger.info("Loading HF validation for %s (need %d ids)", lang, len(id_set))
        files = _resolve_val_parquets(lang)
        if files:
            ds = load_dataset("parquet", data_files={"validation": files}, split="validation")
        else:
            # fallback (may touch network if not cached)
            ds = load_hf_asr_split(lang, "validation", max_samples=None)
            # still need id filter with full decode risk — use as last resort
        all_ids = [str(x) for x in ds["id"]]
        positions = [i for i, uid in enumerate(all_ids) if uid in id_set]
        if not positions:
            logger.warning("%s: no matching proxy ids in validation", lang)
            continue
        sub = ds.select(positions)
        try:
            sub = sub.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
        found = 0
        for i in range(len(sub)):
            ex = sub[i]
            uid = str(ex["id"])
            audio = _decode_audio_item(ex["audio"], TARGET_SR)
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            ref = normalize_text(ex.get("transcription") or ex.get("text") or "")
            if not ref:
                logger.warning("empty ref for %s — skip", uid)
                continue
            samples.append(
                {
                    "id": uid,
                    "true_lang": lang,
                    "ref": ref,
                    "arr": arr,
                    "sr": sr,
                }
            )
            found += 1
        missing = id_set - {s["id"] for s in samples if s["true_lang"] == lang}
        if missing:
            logger.warning(
                "%s: missing %d proxy ids (e.g. %s)",
                lang,
                len(missing),
                next(iter(missing)),
            )
        logger.info("%s: loaded %d/%d", lang, found, len(id_set))
    return samples


@torch.inference_mode()
def score_sequential(samples: list[dict], device: torch.device) -> list[dict]:
    """Decode with one model at a time; pick best conf among candidates."""
    # best[id] = (hyp, decode_lang, conf)
    best: dict[str, tuple[str, str, float]] = {
        s["id"]: (".", candidates_for(s["true_lang"])[0], -1e9) for s in samples
    }
    by_id = {s["id"]: s for s in samples}

    # Which samples need each model
    need: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        for cand in candidates_for(s["true_lang"]):
            need[cand].append(s["id"])

    for lang in sorted(need.keys(), key=lambda x: (-len(need[x]), x)):
        if lang not in WAXAL300:
            logger.error("No waxal-300m for candidate %s — skip", lang)
            continue
        model, processor = load_waxal(lang, device)
        for uid in tqdm(need[lang], desc=f"decode-{lang}"):
            s = by_id[uid]
            hyp, conf = transcribe_waveform(
                model,
                processor,
                s["arr"],
                s["sr"],
                device=device,
                return_confidence=True,
            )
            prev_hyp, prev_lang, prev_c = best[uid]
            if conf > prev_c:
                best[uid] = (hyp, lang, conf)
        free_model(model)
        del processor
        gc.collect()

    rows = []
    for s in samples:
        hyp, dlang, conf = best[s["id"]]
        rows.append(
            {
                "id": s["id"],
                "true_lang": s["true_lang"],
                "decode_lang": dlang,
                "hyp": hyp,
                "ref": s["ref"],
                "conf": conf,
                "candidates": "|".join(candidates_for(s["true_lang"])),
                "pick_ok": int(dlang == s["true_lang"]),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    sc = score_pairs(refs, hyps)
    overall = {
        "wer": sc["wer"],
        "cer": sc["cer"],
        "error": sc["score"],
        "zindi_est": 1.0 - sc["score"],
        "n": int(sc["n"]),
        "pick_accuracy": sum(r["pick_ok"] for r in rows) / max(len(rows), 1),
        "decode_mass": dict(Counter(r["decode_lang"] for r in rows)),
        "true_mass": dict(Counter(r["true_lang"] for r in rows)),
    }
    per: dict[str, dict] = {}
    by = defaultdict(list)
    for r in rows:
        by[r["true_lang"]].append(r)
    for lang, group in sorted(by.items()):
        p = score_pairs([g["ref"] for g in group], [g["hyp"] for g in group])
        per[lang] = {
            "wer": p["wer"],
            "cer": p["cer"],
            "error": p["score"],
            "zindi_est": 1.0 - p["score"],
            "n": int(p["n"]),
            "pick_accuracy": sum(g["pick_ok"] for g in group) / max(len(group), 1),
            "decode_mass": dict(Counter(g["decode_lang"] for g in group)),
        }
    return {
        "policy": "phase2_openset_champion_multihyp_waxal300",
        "models": "waxal-benchmarking/mms-300m-waxal-{lang} only",
        "candidate_map": CANDIDATES_BY_TRUE,
        "proxy_index": str(PROXY_INDEX),
        "seed": 42,
        "overall": overall,
        "per_lang": per,
    }


def write_md(summary: dict, path: Path) -> None:
    o = summary["overall"]
    lines = [
        "# Phase-2 proxy champion baseline",
        "",
        "Frozen champion multi-hyp policy (`run_phase2_openset.py` routing) scored on "
        f"`data/proxy_val_index.csv` (n={o['n']}).",
        "",
        "## Policy",
        "",
        "- Models: **only** `waxal-benchmarking/mms-300m-waxal-{lang}`",
        "- Greedy CTC decode; pick by mean log-prob of argmax path",
        "- Normalize: `src.text_norm`; score: `src.metrics.score_pairs`",
        "- `zindi_est = 1 - 0.5*WER - 0.5*CER`",
        "",
        "Candidate map (true_lang → multi-hyp):",
        "",
    ]
    for lang, cands in summary["candidate_map"].items():
        lines.append(f"- **{lang}**: `{'|'.join(cands)}`")
    lines += [
        "",
        "## Overall",
        "",
        f"| metric | value |",
        f"|--------|-------|",
        f"| n | {o['n']} |",
        f"| zindi_est | {o['zindi_est']:.4f} |",
        f"| WER | {o['wer']:.4f} |",
        f"| CER | {o['cer']:.4f} |",
        f"| error (0.5wer+0.5cer) | {o['error']:.4f} |",
        f"| pick_accuracy (decode_lang==true_lang) | {o['pick_accuracy']:.4f} |",
        "",
        f"decode_mass: `{o['decode_mass']}`",
        "",
        "## Per language",
        "",
        "| lang | n | zindi_est | WER | CER | pick_acc |",
        "|------|---|-----------|-----|-----|----------|",
    ]
    for lang, p in summary["per_lang"].items():
        lines.append(
            f"| {lang} | {p['n']} | {p['zindi_est']:.4f} | {p['wer']:.4f} | "
            f"{p['cer']:.4f} | {p['pick_accuracy']:.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Hard baseline every other method must beat.",
        "- Proxy is HF **validation** only (ach/nyn/lug/sog/mas); never Phase-1 test gold.",
        "- Outputs: `phase2_proxy_champion.json`, `phase2_proxy_champion_detail.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=PROXY_INDEX)
    p.add_argument("--max-per-lang", type=int, default=None, help="Cap per lang (default: all proxy ids)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, default=OUTPUT_DIR / "phase2_proxy_champion.json")
    p.add_argument("--out-csv", type=Path, default=OUTPUT_DIR / "phase2_proxy_champion_detail.csv")
    p.add_argument("--out-md", type=Path, default=OUTPUT_DIR / "phase2_proxy_champion.md")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    logger.info("device=%s", device)

    t0 = time.time()
    samples = load_proxy_samples(args.index, args.max_per_lang, seed=args.seed)
    if not samples:
        raise SystemExit("No proxy samples loaded")
    logger.info("samples=%d langs=%s", len(samples), Counter(s["true_lang"] for s in samples))

    rows = score_sequential(samples, device)
    # Free audio arrays from memory before scoring IO
    for s in samples:
        s.pop("arr", None)

    summary = summarize(rows)
    summary["device"] = str(device)
    summary["seconds"] = time.time() - t0
    summary["max_per_lang"] = args.max_per_lang

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows)[
        ["id", "true_lang", "decode_lang", "hyp", "ref", "conf"]
    ].to_csv(args.out_csv, index=False)
    write_md(summary, args.out_md)

    o = summary["overall"]
    logger.info(
        "DONE n=%d zindi_est=%.4f wer=%.4f cer=%.4f pick_acc=%.3f seconds=%.1f",
        o["n"],
        o["zindi_est"],
        o["wer"],
        o["cer"],
        o["pick_accuracy"],
        summary["seconds"],
    )
    print(json.dumps({"overall": o, "per_lang": {k: v["zindi_est"] for k, v in summary["per_lang"].items()}}, indent=2))
    print("WROTE", args.out_json)
    print("WROTE", args.out_csv)
    print("WROTE", args.out_md)


if __name__ == "__main__":
    main()
