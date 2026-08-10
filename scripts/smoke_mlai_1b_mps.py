#!/usr/bin/env python3
"""One-sample MPS feasibility smoke for the public WAXAL 1B checkpoint."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/goal_2026_08_10/mlai_1b_eval"


def main() -> None:
    result = {"device": "mps", "torch_mps_available": bool(torch.backends.mps.is_available())}
    if not result["torch_mps_available"]:
        (OUT / "mps_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    from fairseq2.assets import get_asset_store
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama import get_wav2vec2_llama_model_hub

    card = get_asset_store().retrieve_card("omniASR_LLM_1B_v2")
    hub = get_wav2vec2_llama_model_hub()
    config = hub.get_model_config(card)
    device = torch.device("mps")
    checkpoint = OUT / "checkpoint/ws_1.2a8bfda1/checkpoints/step_1000/model"
    started = time.time()
    model = hub.load_custom_model(checkpoint, config, device=device, dtype=torch.float32, mmap=True, progress=True)
    tokenizer = load_tokenizer("omniASR_LLM_1B_v2")
    pipe = ASRInferencePipeline(None, model=model, tokenizer=tokenizer, device=device, dtype=torch.float32)
    audio = OUT / "locked_screen_audio/lug/00_lug_39010.wav"
    hyp = pipe.transcribe([str(audio)], lang=["lug_Latn"], batch_size=1)[0]
    result.update({"status": "completed", "load_and_decode_seconds": time.time() - started, "audio": str(audio), "hypothesis": str(hyp)})
    (OUT / "mps_smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
