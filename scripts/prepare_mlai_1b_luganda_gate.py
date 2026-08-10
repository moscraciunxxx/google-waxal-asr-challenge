#!/usr/bin/env python3
"""Prepare only the locked Luganda validation audio for the OmniASR gate."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import load_hf_asr_split  # noqa: E402
from src.text_norm import normalize_text  # noqa: E402

TABLE = ROOT / "outputs/goal_2026_08_08/luganda_fusion/matched_hypotheses.csv"
OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate"


def save_wav(path: Path, array: np.ndarray, sample_rate: int) -> None:
    x = np.asarray(array, dtype=np.float32)
    peak = max(1.0, float(np.max(np.abs(x)) or 1.0))
    pcm = np.clip(x / peak, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE, dtype=str).fillna("")
    ids = table["ID"].astype(str).tolist()
    if len(ids) != 40 or len(set(ids)) != 40:
        raise RuntimeError(f"locked Luganda table must contain 40 unique IDs, got {len(ids)}")
    dataset = load_hf_asr_split("lug", "validation")
    by_id = {str(row["id"]): row for row in dataset}
    missing = [uid for uid in ids if uid not in by_id]
    if missing:
        raise RuntimeError(f"missing locked validation audio for {missing[:5]}")
    audio_dir = OUT / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    for uid in ids:
        row = by_id[uid]
        audio = row["audio"]
        path = audio_dir / f"{uid}.wav"
        if not path.exists():
            save_wav(path, np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate") or 16000))
        reference = normalize_text(str(table.loc[table.ID == uid, "corrected_reference"].iloc[0]))
        rows.append({
            "ID": uid,
            "speaker_id": str(row.get("speaker_id") or ""),
            "reference": reference,
            "audio": str(path),
        })
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print(f"prepared {len(rows)} locked rows at {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
