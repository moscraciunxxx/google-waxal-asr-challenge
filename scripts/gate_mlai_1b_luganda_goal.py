#!/usr/bin/env python3
"""Full locked Luganda gate for the public WAXAL 1B OmniASR checkpoint."""

from __future__ import annotations

import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_hf_asr_split  # noqa: E402
from src.metrics import score_pairs  # noqa: E402
from src.text_norm import normalize_text  # noqa: E402

OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval/lug_gate"
TABLE = ROOT / "outputs/goal_2026_08_08/luganda_fusion/matched_hypotheses.csv"
INCUMBENT = "mms_ft_v3_domain_beam_splitjoin"


def metric(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    s = score_pairs(refs, hyps)
    return {
        "n": int(s["n"]),
        "wer": float(s["wer"]),
        "cer": float(s["cer"]),
        "zindi": float(1.0 - s["score"]),
    }


def save_wav(path: Path, array: np.ndarray, sample_rate: int) -> None:
    x = np.asarray(array, dtype=np.float32)
    peak = max(1.0, float(np.max(np.abs(x)) or 1.0)
    )
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
    paths: list[str] = []
    speakers: list[str] = []
    refs: list[str] = []
    for uid in ids:
        row = by_id[uid]
        audio = row["audio"]
        path = audio_dir / f"{uid}.wav"
        if not path.exists():
            save_wav(path, np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate") or 16000))
        paths.append(str(path))
        speakers.append(str(row.get("speaker_id") or ""))
        refs.append(normalize_text(str(table.loc[table.ID == uid, "corrected_reference"].iloc[0])))
    manifest = pd.DataFrame({"ID": ids, "speaker_id": speakers, "reference": refs, "audio": paths})
    manifest.to_csv(OUT / "manifest.csv", index=False)

    # The loader uses the already downloaded public checkpoint and tokenizer.
    import importlib.util

    loader_path = OUT.parent / "omni_cpu_screen.py"
    spec = importlib.util.spec_from_file_location("mlai_omni_cpu_screen", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_pipeline = module.load_pipeline

    pipe = load_pipeline()
    raw = pipe.transcribe(paths, lang=["lug_Latn"] * len(paths), batch_size=1)
    candidate = [normalize_text(x) or "." for x in raw]
    baseline = [normalize_text(x) for x in table[INCUMBENT].tolist()]
    manifest["candidate"] = candidate
    manifest["baseline"] = baseline
    manifest.to_csv(OUT / "hypotheses.csv", index=False)
    tune = np.arange(20)
    holdout = np.arange(20, 40)
    report = {
        "protocol": {
            "dataset": "google/WaxalNLP Luganda validation",
            "sample_size": 40,
            "locked_source": str(TABLE),
            "test_labels_used": False,
            "phase2_labels_or_audio_used": False,
            "candidate_model": "mlai-dante/waxal-omniASR-LLM-1B-v2",
            "checkpoint": "step_1000/model",
        },
        "metrics": {
            "baseline": metric(refs, baseline),
            "candidate": metric(refs, candidate),
            "baseline_tune": metric([refs[i] for i in tune], [baseline[i] for i in tune]),
            "candidate_tune": metric([refs[i] for i in tune], [candidate[i] for i in tune]),
            "baseline_holdout": metric([refs[i] for i in holdout], [baseline[i] for i in holdout]),
            "candidate_holdout": metric([refs[i] for i in holdout], [candidate[i] for i in holdout]),
        },
        "id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }
    report["delta"] = {
        "all": report["metrics"]["candidate"]["zindi"] - report["metrics"]["baseline"]["zindi"],
        "tune": report["metrics"]["candidate_tune"]["zindi"] - report["metrics"]["baseline_tune"]["zindi"],
        "holdout": report["metrics"]["candidate_holdout"]["zindi"] - report["metrics"]["baseline_holdout"]["zindi"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
