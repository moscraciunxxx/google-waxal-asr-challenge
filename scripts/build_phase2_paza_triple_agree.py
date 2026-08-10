#!/usr/bin/env python3
"""NEW mechanism after soft-margin public reject (LdgY1mLF 0.5584):

PAZA Whisper Luo as a *third* independent Luo hyp for residual lid=luo∩ach-route.
Unlike dual thr expand (BAN) or soft CTC margin (BAN public), this uses a new
open model family (microsoft/paza-whisper-large-v3-turbo) and only replaces when
PAZA agrees with MMS and/or CLEAR under strict CER + p1 gates.

Base: public floor selective_v3_dual15 (KEEP CHPRnXLG until public beat).
Never rewrite decode_lang=lug. No Phase-1 test gold. Open weights only.

Modes:
  --mode mms_paza   : residual not already dual; CER(MMS,PAZA)<=thr & p1>=min_p1
  --mode clear_paza : CER(CLEAR,PAZA)<=thr (CLEAR source must be clear_luo)
  --mode triple_2of3: at least 2 of {MMS,CLEAR,PAZA} pairwise CER<=thr
  --mode all3       : all three pairwise CER<=thr

Upload only after FLEURS+ach false-pos calib and n_changed risk budget.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.submission import check_submission
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paza_triple")

AUDIO = PROJECT_ROOT / "data" / "phase2" / "audio"
FLOOR = PROJECT_ROOT / "submission_phase2_selective_v3_dual15.csv"
SAMPLE = PROJECT_ROOT / "data" / "phase2" / "SampleSubmission.csv"
DET = OUTPUT_DIR / "phase2_selective_v3_detail.csv"
MMS = OUTPUT_DIR / "phase2_luo_mms1b_detail.csv"
CLR = OUTPUT_DIR / "phase2_selective_clear_allluo_detail.csv"
PAZA_MODEL = "microsoft/paza-whisper-large-v3-turbo"


def cer(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(la, lb)


def load_wav(path: Path):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(-1)
    arr = np.asarray(arr, dtype=np.float32)
    return arr / (float(np.max(np.abs(arr)) + 1e-9)), int(sr)


def run_paza_on_ids(ids: list[str], out_csv: Path, limit: int = 0) -> pd.DataFrame:
    """Transcribe phase2 wavs with PAZA language=luo."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Loading %s on %s", PAZA_MODEL, device)
    try:
        proc = AutoProcessor.from_pretrained(PAZA_MODEL, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            PAZA_MODEL, torch_dtype=torch_dtype, local_files_only=True
        )
    except Exception:
        proc = AutoProcessor.from_pretrained(PAZA_MODEL)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(PAZA_MODEL, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        for _, r in prev.iterrows():
            done[str(r["ID"])] = str(r["prediction"])
        logger.info("Resume PAZA scores n=%d", len(done))

    todo = [i for i in ids if i not in done]
    if limit > 0:
        todo = todo[:limit]
    rows = [{"ID": k, "prediction": v, "source": "paza_luo"} for k, v in done.items()]

    # PAZA adds <|luo|> (id 51867); get_decoder_prompt_ids("luo") is unsupported.
    # Mirror Whisper task/notimestamps ids from a working sw prompt: 50360 / 50364.
    luo_tok = None
    for i in range(len(proc.tokenizer)):
        if proc.tokenizer.convert_ids_to_tokens(i) == "<|luo|>":
            luo_tok = i
            break
    if luo_tok is None:
        luo_tok = 51867
    forced = [(1, int(luo_tok)), (2, 50360), (3, 50364)]
    gen_kwargs = {
        "max_new_tokens": 180,
        "num_beams": 1,
        "do_sample": False,
        "forced_decoder_ids": forced,
        "no_repeat_ngram_size": 3,
    }
    logger.info("PAZA forced_decoder_ids=%s", forced)

    for id_ in tqdm(todo, desc="paza-luo"):
        wav = AUDIO / f"{id_}.wav"
        if not wav.exists():
            logger.warning("missing %s", wav)
            continue
        arr, sr = load_wav(wav)
        if sr != 16000:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            sr = 16000
        inputs = proc(arr, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        if torch_dtype == torch.float16:
            input_features = input_features.half()
        with torch.inference_mode():
            ids_out = model.generate(input_features, **gen_kwargs)
        text = proc.batch_decode(ids_out, skip_special_tokens=True)[0]
        text = normalize_text(text) or "."
        if text.startswith("luo "):
            text = text[4:].strip() or "."
        # Hard length cap vs ~20s clips: reject obvious loops later via CER gate
        if len(text.split()) > 80:
            text = " ".join(text.split()[:80])
        rows.append({"ID": id_, "prediction": text, "source": "paza_luo"})
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def build_candidate(
    paza: pd.DataFrame,
    mode: str,
    thr: float,
    min_p1: float,
    name: str,
) -> dict:
    floor = pd.read_csv(FLOOR)
    det = pd.read_csv(DET)
    mms = pd.read_csv(MMS)
    clr = pd.read_csv(CLR)
    v3 = pd.read_csv(PROJECT_ROOT / "submission_phase2_selective_v3.csv")
    frozen_lug = set(det.loc[det.decode_lang == "lug", "ID"].astype(str))

    m = (
        det.merge(mms[["ID", "prediction"]].rename(columns={"prediction": "mms"}), on="ID")
        .merge(
            clr[["ID", "prediction", "source"]].rename(
                columns={"prediction": "clr", "source": "clr_src"}
            ),
            on="ID",
        )
        .merge(paza[["ID", "prediction"]].rename(columns={"prediction": "paza"}), on="ID")
        .merge(floor.rename(columns={"Target": "floor"}), on="ID")
        .merge(v3.rename(columns={"Target": "sel_v3"}), on="ID")
    )
    m["already_dual"] = m["floor"] != m["sel_v3"]
    m["cer_mp"] = [cer(a, b) for a, b in zip(m.mms, m.paza)]
    m["cer_cp"] = [cer(a, b) for a, b in zip(m.clr, m.paza)]
    m["cer_mc"] = [cer(a, b) for a, b in zip(m.mms, m.clr)]

    # residual ach-route only
    pool = m[
        (m.lid_lang == "luo")
        & (m.decode_lang == "ach")
        & (~m.already_dual)
        & (~m.ID.astype(str).isin(frozen_lug))
    ].copy()

    if mode == "mms_paza":
        accept = pool[(pool.cer_mp <= thr) & (pool.p1 >= min_p1)]
        text_col = "mms"
    elif mode == "clear_paza":
        accept = pool[
            (pool.clr_src == "clear_luo")
            & (pool.cer_cp <= thr)
            & (pool.p1 >= min_p1)
        ]
        text_col = "paza"  # or clr; paza is new
    elif mode == "triple_2of3":
        def ok(r):
            pairs = 0
            if r.cer_mc <= thr:
                pairs += 1
            if r.cer_mp <= thr:
                pairs += 1
            if r.clr_src == "clear_luo" and r.cer_cp <= thr:
                pairs += 1
            return pairs >= 2 and r.p1 >= min_p1

        mask = pool.apply(ok, axis=1)
        accept = pool[mask]
        text_col = "mms"
    elif mode == "all3":
        accept = pool[
            (pool.clr_src == "clear_luo")
            & (pool.cer_mc <= thr)
            & (pool.cer_mp <= thr)
            & (pool.cer_cp <= thr)
            & (pool.p1 >= min_p1)
        ]
        text_col = "mms"
    else:
        raise ValueError(mode)

    ids = set(accept.ID.astype(str))
    text_map = accept.set_index("ID")[text_col]
    out = floor.copy()
    n = 0
    for i, row in out.iterrows():
        if str(row.ID) in ids:
            val = normalize_text(str(text_map.loc[row.ID])) or "."
            if val != str(row.Target):
                out.at[i, "Target"] = val
                n += 1

    path = PROJECT_ROOT / f"submission_phase2_paza_{mode}_t{int(thr*100):02d}.csv"
    out.to_csv(path, index=False)
    chk = check_submission(path, SAMPLE)
    rep = {
        "path": str(path),
        "mode": mode,
        "thr": thr,
        "min_p1": min_p1,
        "n_accept": int(len(accept)),
        "n_changed": int(n),
        "check_ok": chk["ok"],
        "frozen_lug_touched": int(len(ids & frozen_lug)),
        "new_mechanism": "PAZA third-model agree (not dual-thr expand, not soft CTC margin)",
        "base": "selective_v3_dual15",
    }
    rep_path = OUTPUT_DIR / f"phase2_paza_{mode}_t{int(thr*100):02d}_check.json"
    rep_path.write_text(json.dumps(rep, indent=2))
    # detail for risk budget
    accept.to_csv(OUTPUT_DIR / f"phase2_paza_{mode}_t{int(thr*100):02d}_accept.csv", index=False)
    logger.info("%s", json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mms_paza", choices=["mms_paza", "clear_paza", "triple_2of3", "all3"])
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--min-p1", type=float, default=0.99)
    ap.add_argument("--limit", type=int, default=0, help="debug limit PAZA infer")
    ap.add_argument("--skip-infer", action="store_true", help="use existing PAZA scores only")
    ap.add_argument("--name", type=str, default="")
    args = ap.parse_args()

    det = pd.read_csv(DET)
    # Infer PAZA on residual ach-route pool (+ already dual for completeness)
    pool_ids = det.loc[
        (det.lid_lang == "luo") & (det.decode_lang == "ach"), "ID"
    ].astype(str).tolist()
    paza_csv = OUTPUT_DIR / "phase2_paza_luo_scores.csv"
    if args.skip_infer and paza_csv.exists():
        paza = pd.read_csv(paza_csv)
    else:
        paza = run_paza_on_ids(pool_ids, paza_csv, limit=args.limit)

    name = args.name or f"{args.mode}_t{int(args.thr*100):02d}"
    rep = build_candidate(paza, args.mode, args.thr, args.min_p1, name)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
