#!/usr/bin/env python3
"""Multi-system decode of PUBLIC-VISIBLE phase-2 rows (lid!=luo old + all new).

Systems per route (ban-compliant same-route specialists only):
  lug: mms-lug-ft-v3, mms-1b adapter.lug, waxal-300m-lug
  nyn: mms-nyn-ft-v1 greedy, mms-1b adapter.nyn
  lin: mms-1b adapter.lin (floor), waxal-300m-lin
  sna: waxal-300m-sna (floor), mms-1b adapter.sna
  ach: waxal-300m-ach (only ~13 public rows)

Outputs hyp tables under outputs/beat075/ for ROVER / PL / submission merge.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, TARGET_SR
from src.mms_infer import set_lang, transcribe_waveform
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import fix_mms_tokenizer, pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("beat075_decode")

OUT = PROJECT_ROOT / "outputs" / "beat075"
OLD_AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
NEW_AUDIO = PROJECT_ROOT / "newaudios"


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = float(np.max(np.abs(arr)) + 1e-9)
    return arr / peak, int(sr)


def load_ctc(model_id_or_path: str, device: torch.device, lang: str | None = None, is_ft: bool = False):
    p = Path(model_id_or_path)
    local = p.exists()
    try:
        proc = AutoProcessor.from_pretrained(str(model_id_or_path), local_files_only=local or True)
        model = Wav2Vec2ForCTC.from_pretrained(str(model_id_or_path), local_files_only=local or True)
    except Exception:
        proc = AutoProcessor.from_pretrained(str(model_id_or_path))
        model = Wav2Vec2ForCTC.from_pretrained(str(model_id_or_path))
    if is_ft and lang:
        try:
            fix_mms_tokenizer(proc, lang)
        except Exception as e:
            logger.warning("fix_mms_tokenizer: %s", e)
    elif lang and "mms-1b" in str(model_id_or_path):
        try:
            set_lang(model, proc, lang)
        except Exception:
            try:
                proc.tokenizer.set_target_lang(lang)
                model.load_adapter(lang)
            except Exception as e:
                logger.warning("adapter %s: %s", lang, e)
    model.to(device).eval()
    return model, proc


@torch.inference_mode()
def decode_ids(model, proc, ids: list[str], paths: dict[str, Path], device, tag: str) -> dict[str, str]:
    out = {}
    t0 = time.time()
    for k, uid in enumerate(ids):
        path = paths[uid]
        arr, sr = load_wav(path)
        hyp = transcribe_waveform(model, proc, arr, sr, device=device, return_confidence=False)
        out[uid] = normalize_text(hyp) or "."
        if (k + 1) % 25 == 0 or k + 1 == len(ids):
            logger.info("%s %d/%d %.1fs", tag, k + 1, len(ids), time.time() - t0)
    return out


def public_visible_table() -> pd.DataFrame:
    """Old lid!=luo + all new clips, with floor route + text."""
    op = pd.read_csv(PROJECT_ROOT / "outputs" / "phase2_openset_detail.csv")
    op["ID"] = op["ID"].astype(str)
    old = op[op.lid_lang != "luo"][["ID", "lid_lang", "decode_lang", "p1", "prediction"]].copy()
    old["split"] = "old"
    old["audio"] = old["ID"].map(lambda i: str(OLD_AUDIO / f"{i}.wav"))

    new = pd.read_csv(PROJECT_ROOT / "outputs" / "next_iter" / "new_routes.csv")
    new["ID"] = new["ID"].astype(str)
    new = new[["ID", "lid_lang", "decode_lang", "p1", "route_text"]].rename(columns={"route_text": "prediction"})
    new["split"] = "new"
    new["audio"] = new["ID"].map(lambda i: str(NEW_AUDIO / f"{i}.wav"))

    floor = pd.read_csv(PROJECT_ROOT / "submission_phase2_v2_full.csv")
    floor["ID"] = floor["ID"].astype(str)
    floor_map = floor.set_index("ID")["Target"].to_dict()

    tab = pd.concat([old, new], ignore_index=True)
    tab["floor"] = tab["ID"].map(floor_map)
    missing = [p for p in tab.audio if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing audio e.g. {missing[:3]} count={len(missing)}")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="+", default=["lug", "nyn", "lin", "sna"])
    ap.add_argument("--limit-per-route", type=int, default=None, help="debug limit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--systems", default="all", help="all|floor_peers|quick")
    args = ap.parse_args()

    device = pick_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    tab = public_visible_table()
    tab.to_csv(OUT / "public_visible_index.csv", index=False)
    logger.info("public_visible n=%d by route %s", len(tab), tab.decode_lang.value_counts().to_dict())

    # system registry: (route, tag, model_path, is_ft, mms1b_lang)
    systems = []
    if "lug" in args.routes:
        systems += [
            ("lug", "ft_v3", str(PROJECT_ROOT / "checkpoints" / "mms-lug-ft-v3"), True, "lug"),
            ("lug", "mms1b_zs", "facebook/mms-1b-all", False, "lug"),
            ("lug", "waxal300", "waxal-benchmarking/mms-300m-waxal-lug", False, None),
        ]
    if "nyn" in args.routes:
        systems += [
            ("nyn", "ft_v1", str(PROJECT_ROOT / "checkpoints" / "mms-nyn-ft-v1"), True, "nyn"),
            ("nyn", "mms1b_zs", "facebook/mms-1b-all", False, "nyn"),
            ("nyn", "waxal300", "waxal-benchmarking/mms-300m-waxal-nyn", False, None),
        ]
    if "lin" in args.routes:
        systems += [
            ("lin", "mms1b_zs", "facebook/mms-1b-all", False, "lin"),
            ("lin", "waxal300", "waxal-benchmarking/mms-300m-waxal-lin", False, None),
        ]
    if "sna" in args.routes:
        systems += [
            ("sna", "waxal300", "waxal-benchmarking/mms-300m-waxal-sna", False, None),
            ("sna", "mms1b_zs", "facebook/mms-1b-all", False, "sna"),
        ]
    if "ach" in args.routes:
        systems += [
            ("ach", "waxal300", "waxal-benchmarking/mms-300m-waxal-ach", False, None),
        ]

    if args.systems == "quick":
        # one peer per route
        keep = {("lug", "mms1b_zs"), ("nyn", "mms1b_zs"), ("lin", "waxal300"), ("sna", "mms1b_zs")}
        systems = [s for s in systems if (s[0], s[1]) in keep]

    summary = {"device": str(device), "n_public": len(tab), "runs": []}

    for route, tag, mid, is_ft, mms_lang in systems:
        sub = tab[tab.decode_lang == route]
        if args.limit_per_route:
            sub = sub.head(args.limit_per_route)
        if sub.empty:
            continue
        out_path = OUT / f"hyps_{route}_{tag}.csv"
        if out_path.exists() and not args.limit_per_route:
            prev = pd.read_csv(out_path)
            if len(prev) == len(tab[tab.decode_lang == route]):
                logger.info("skip existing %s", out_path.name)
                summary["runs"].append({"route": route, "tag": tag, "skipped": True})
                continue

        logger.info("=== decode route=%s tag=%s model=%s n=%d ===", route, tag, mid, len(sub))
        model, proc = load_ctc(mid, device, lang=mms_lang or route, is_ft=is_ft)
        if mms_lang and "mms-1b" in mid:
            try:
                set_lang(model, proc, mms_lang)
            except Exception:
                pass
        paths = {r.ID: Path(r.audio) for r in sub.itertuples()}
        hyps = decode_ids(model, proc, list(sub.ID), paths, device, f"{route}:{tag}")
        df = pd.DataFrame({"ID": list(hyps.keys()), f"hyp_{tag}": list(hyps.values())})
        # if full route decode without limit, write full; else merge
        if args.limit_per_route:
            df.to_csv(OUT / f"hyps_{route}_{tag}_lim{args.limit_per_route}.csv", index=False)
        else:
            df.to_csv(out_path, index=False)
        summary["runs"].append({"route": route, "tag": tag, "n": len(df), "path": str(out_path)})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    (OUT / "decode_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("done %s", OUT)


if __name__ == "__main__":
    main()
