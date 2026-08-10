#!/usr/bin/env python3
"""Build merged submissions for the expanded Phase-2 test set (1500 old + 892 new).

New-clip decoders (all A/B-validated on n=120 WAXAL val, seed 42):
  lin (444): facebook/mms-1b-all adapter.lin zero-shot greedy (0.7700 vs waxal 0.6935)
  sna (445): waxal-300m-waxal-sna greedy (0.8476; beats every FT/zero-shot variant)
  lug (3):   checkpoints/mms-lug-ft-v3 greedy (route_text from pipeline)

Old-clip variants merged in:
  v2            : submission_phase2_nextiter_nyn.csv  (public best wMjTA7Dz)
  v2_paza_ach   : submission_phase2_paza_ach437_c16.csv
  v2_paza_all   : submission_phase2_paza_all746_c16.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROJECT_ROOT, TARGET_SR
from src.text_norm import normalize_text
from scripts.mms_adapter_ft import pick_device
from scripts.run_phase2_openset import load_wav

NEW_AUDIO = PROJECT_ROOT / "newaudios"
TABLE = OUTPUT_DIR / "next_iter" / "new_clips_table.csv"
LIN_HYPS = OUTPUT_DIR / "next_iter" / "new_lin_mms1b_hyps.csv"

OLD_BASES = {
    "v2": PROJECT_ROOT / "submission_phase2_nextiter_nyn.csv",
    "v2_paza_ach": PROJECT_ROOT / "submission_phase2_paza_ach437_c16.csv",
    "v2_paza_all": PROJECT_ROOT / "submission_phase2_paza_all746_c16.csv",
}


def decode_lin(table: pd.DataFrame, device) -> pd.DataFrame:
    if LIN_HYPS.exists():
        prev = pd.read_csv(LIN_HYPS)
        if len(prev) == (table.decode_lang == "lin").sum():
            return prev
    ids = table[table.decode_lang == "lin"].ID.astype(str).tolist()
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    proc.tokenizer.set_target_lang("lin")
    model.load_adapter("lin")
    model.to(device).eval()
    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for k, uid in enumerate(ids):
            arr, sr = load_wav(NEW_AUDIO / f"{uid}.wav")
            if sr != TARGET_SR:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            inputs = proc(arr, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            tid = torch.argmax(model(inputs.input_values.to(device)).logits, dim=-1)[0]
            rows.append({"ID": uid, "lin_mms1b": normalize_text(proc.decode(tid)) or "."})
            if (k + 1) % 50 == 0:
                print(f"lin {k+1}/{len(ids)} {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(LIN_HYPS, index=False)
    return df


def main():
    device = pick_device(None)
    table = pd.read_csv(TABLE)
    table["ID"] = table["ID"].astype(str)
    lin_df = decode_lin(table, device).set_index("ID")["lin_mms1b"]

    new_pred = {}
    for _, r in table.iterrows():
        uid = str(r.ID)
        if r.decode_lang == "lin" and uid in lin_df.index:
            new_pred[uid] = normalize_text(str(lin_df[uid])) or "."
        else:
            primary = normalize_text(str(r.final_base))
            fallback = normalize_text(str(r.openset_text))
            # A punctuation-only CTC decode is not an uploadable transcript.
            # Prefer the independent open-set hypothesis for that rare case.
            new_pred[uid] = primary if primary and primary != "." else (fallback or "e")

    for tag, base_path in OLD_BASES.items():
        base = pd.read_csv(base_path)
        base["ID"] = base["ID"].astype(str)
        merged = pd.concat(
            [base[["ID", "Target"]],
             pd.DataFrame({"ID": list(new_pred.keys()), "Target": list(new_pred.values())})],
            ignore_index=True,
        )
        merged["Target"] = merged["Target"].astype(str).map(lambda s: normalize_text(s) or ".")
        assert merged.ID.is_unique and len(merged) == len(base) + len(new_pred)
        assert merged.Target.str.strip().ne("").all()
        out = PROJECT_ROOT / f"submission_phase2_{tag}_full.csv"
        merged.to_csv(out, index=False)
        print(tag, "->", out.name, "rows:", len(merged))

    meta = {
        "new_rows": len(new_pred),
        "lin_mms1b_rows": int((table.decode_lang == "lin").sum()),
        "sna_waxal_rows": int((table.decode_lang == "sna").sum()),
        "lug_ftv3_rows": int((table.decode_lang == "lug").sum()),
        "evidence": ["outputs/next_iter/lin_ab.json", "outputs/next_iter/sna_ab.json"],
    }
    (OUTPUT_DIR / "next_iter" / "v2_build_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
