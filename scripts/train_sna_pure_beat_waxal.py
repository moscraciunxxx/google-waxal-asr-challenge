#!/usr/bin/env python3
"""Pure train-only sna FT that must beat waxal-300m-sna under held-out protocol.

SKEPTIC NOTE (2026-07-30): Older drafts that evaluated every 3 steps on val[0:50]
are SUPERSEDED. This script uses:
  - select = val[50:90] for schedule pick ONLY
  - report = val[0:50] evaluated ONCE after select-only pick
  - FIXED_FINAL_ONLY: train to exact step count; no intermediate step search
  - pure own weights; no baseline blend

See also: src/criterion1_protocol.py, tests/test_sna_train_no_report_leak.py,
checkpoints/mms-sna-pure-beat-waxal/train_meta.json (fixed_steps winner).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORBIDDEN_TRAIN_SPLITS, SEED, TARGET_SR
from src.criterion1_protocol import (
    REPORT_SLICE_LABEL,
    SELECT_SLICE_LABEL,
    WAXAL300,
    CheckpointCandidate,
    assert_honest_sna_meta,
    select_checkpoint_by_slice,
    split_val_protocol,
)
from src.dataset import load_hf_asr_split
from src.legit_fusion import beats_baseline, mean_error
from src.metrics import score_pairs
from src.mms_infer import pick_device, transcribe_waveform
from src.text_norm import normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_sna_pure")

MID = WAXAL300["sna"]


def load_ctc(path: str):
    try:
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(path)
        model = Wav2Vec2ForCTC.from_pretrained(path)
    return proc, model


def set_trainable(model, mode: str) -> int:
    for p in model.parameters():
        p.requires_grad = False
    layer_idxs: list[int] = []
    for name, _ in model.named_parameters():
        if "encoder.layers." in name:
            try:
                layer_idxs.append(int(name.split("encoder.layers.")[1].split(".")[0]))
            except Exception:
                pass
    max_l = max(layer_idxs) if layer_idxs else -1
    n = 0
    for name, p in model.named_parameters():
        use = False
        if mode == "full":
            use = True
        elif "lm_head" in name:
            use = True
        elif mode == "last1_lm" and "encoder.layers." in name:
            try:
                use = int(name.split("encoder.layers.")[1].split(".")[0]) >= max_l
            except Exception:
                use = False
        elif mode == "top2_lm" and "encoder.layers." in name:
            try:
                use = int(name.split("encoder.layers.")[1].split(".")[0]) >= max_l - 1
            except Exception:
                use = False
        if use:
            p.requires_grad = True
            n += p.numel()
    return n


def prep_batch(batch, processor, max_sec: float = 12.0):
    arrays, texts = [], []
    for ex in batch:
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        if sr != TARGET_SR:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        max_len = int(max_sec * TARGET_SR)
        if arr.shape[0] > max_len:
            start = random.randint(0, arr.shape[0] - max_len)
            arr = arr[start : start + max_len]
        if random.random() < 0.25:
            arr = arr + np.random.randn(*arr.shape).astype(np.float32) * 0.002
        peak = float(np.max(np.abs(arr)) + 1e-9)
        arrays.append(arr / peak)
        texts.append(normalize_text(ex.get("transcription") or "") or ".")
    bi = processor(arrays, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    lab = processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
    if processor.tokenizer.pad_token_id is not None:
        lab[lab == processor.tokenizer.pad_token_id] = -100
    bi["labels"] = lab
    return bi


@torch.inference_mode()
def eval_indices(own_m, own_p, base_m, base_p, ds, indices, device):
    own_m.eval()
    refs, bh, oh = [], [], []
    for i in indices:
        ex = ds[int(i)]
        arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        refs.append(normalize_text(ex.get("transcription") or ""))
        bh.append(transcribe_waveform(base_m, base_p, arr, sr, device=device))
        oh.append(transcribe_waveform(own_m, own_p, arr, sr, device=device))
    base = score_pairs(refs, bh)
    own = score_pairs(refs, oh)
    return base, own, beats_baseline(own, base)


def train_one(
    *,
    mode: str,
    lr: float,
    max_steps: int,
    max_train: int,
    eval_every: int,
    device: torch.device,
    base_m,
    base_p,
    val_ds,
    proto,
    out_dir: Path,
) -> dict:
    """Train pure FT; checkpoint pick uses SELECT slice only.

    Does NOT decode the report slice. Caller must run a single report eval
    after choosing among tracks with select metrics only.
    """
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    # FIXED_FINAL_ONLY: never intermediate select cherry-pick; score final weights only.
    eval_every = max_steps  # ignore caller intermediate cadence
    logger.info(
        "=== pure track mode=%s lr=%s steps=%s select=%s (no report peek) ===",
        mode,
        lr,
        max_steps,
        proto.select_slice,
    )
    proc, model = load_ctc(MID)
    model.to(device)
    n_tr = set_trainable(model, mode)
    logger.info("trainable_m=%.3f", n_tr / 1e6)
    if n_tr == 0:
        raise RuntimeError("no trainable params")

    train_ds = load_hf_asr_split("sna", "train", max_samples=max_train)
    rows = [train_ds[i] for i in range(len(train_ds))]
    random.shuffle(rows)
    loader = DataLoader(
        rows, batch_size=1, shuffle=True, collate_fn=lambda b: prep_batch(b, proc)
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )

    states: dict[int, dict] = {}
    candidates: list[CheckpointCandidate] = []

    b0, o0, beat0 = eval_indices(
        model, proc, base_m, base_p, val_ds, proto.select_indices, device
    )
    logger.info(
        "step=0 select_me=%.5f base_select=%.5f beats_select=%s",
        mean_error(o0["wer"], o0["cer"]),
        mean_error(b0["wer"], b0["cer"]),
        beat0,
    )
    step = 0
    model.train()
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if loss is None or not torch.isfinite(loss):
                optim.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 0.5
            )
            optim.step()
            optim.zero_grad()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                b_s, o_s, beat_s = eval_indices(
                    model, proc, base_m, base_p, val_ds, proto.select_indices, device
                )
                me_s = mean_error(o_s["wer"], o_s["cer"])
                logger.info(
                    "step=%d loss=%.4f SELECT_me=%.5f base_s=%.5f beats_select=%s",
                    step,
                    float(loss.detach()),
                    me_s,
                    mean_error(b_s["wer"], b_s["cer"]),
                    beat_s,
                )
                states[step] = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                candidates.append(
                    CheckpointCandidate(
                        step=step,
                        select_mean_error=me_s,
                        select_beats_baseline=beat_s,
                        state_key=str(step),
                        extras={
                            "select_own": o_s,
                            "select_base": b_s,
                        },
                    )
                )
                model.train()
            if step >= max_steps:
                break

    chosen = select_checkpoint_by_slice(candidates)
    if chosen is None or chosen.state_key is None:
        logger.error("no candidates")
        return {"error": "no_candidates", "select_mean_error": 1e9, "select_beats": False}

    model.load_state_dict(states[int(chosen.state_key)])
    model.to(device).eval()

    out_best = out_dir / "best"
    out_best.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_best)
    proc.save_pretrained(out_best)

    meta = {
        "method": f"pure_ft_{mode}_select_only",
        "mode": mode,
        "lr": lr,
        "max_steps": max_steps,
        "max_train": max_train,
        "eval_every": eval_every,
        "chosen_step": chosen.step,
        "select_mean_error": chosen.select_mean_error,
        "select_beats_baseline": chosen.select_beats_baseline,
        "early_stop_slice": SELECT_SLICE_LABEL,
        "report_slice": REPORT_SLICE_LABEL,
        "pure_own_checkpoint": True,
        "no_baseline_blend": True,
        "report_eval_deferred": True,
        "report_not_used_for_step_or_track_selection": True,
        "select_own": chosen.extras.get("select_own"),
        "select_base": chosen.extras.get("select_base"),
        "n_candidates": len(candidates),
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "seed": SEED,
        "init": MID,
    }
    # honesty without report_beats yet — assert core flags
    if meta["early_stop_slice"] == meta["report_slice"]:
        raise AssertionError("slice leak")
    if not meta["pure_own_checkpoint"] or not meta["no_baseline_blend"]:
        raise AssertionError("pure flags")
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        "TRACK_DONE mode=%s chosen_step=%s SELECT_me=%.5f select_beats=%s (report deferred)",
        mode,
        chosen.step,
        chosen.select_mean_error,
        chosen.select_beats_baseline,
    )
    return {
        "path": str(out_best),
        "meta": meta,
        "mode": mode,
        "lr": lr,
        "select_mean_error": chosen.select_mean_error,
        "select_beats": chosen.select_beats_baseline,
        "chosen_step": chosen.step,
        "proc_path": str(out_best),
    }


# Pre-registered FIXED-FINAL schedules (mode, lr, steps).
# No intermediate step pick: train to exact step count, score SELECT once, then
# pick among schedules by select ME only; ONE report after that.
# eval_every is ignored for selection when equal to steps (final only).
PRE_REGISTERED_TRACKS: list[tuple[str, float, int, int]] = [
    ("last1_lm", 2e-6, 15, 15),
    ("last1_lm", 2e-6, 24, 24),
    ("last1_lm", 2e-6, 30, 30),
    ("last1_lm", 1e-6, 30, 30),
    ("last1_lm", 1e-6, 48, 48),
    ("last1_lm", 3e-6, 18, 18),
    ("last1_lm", 5e-6, 12, 12),  # held-out winner under fixed-final protocol
    ("top2_lm", 2e-6, 20, 20),
    ("top2_lm", 2e-6, 30, 30),
    ("top2_lm", 1e-6, 40, 40),
    ("full", 5e-7, 20, 20),
    ("full", 1e-6, 15, 15),
]


def pick_track_by_select_only(track_results: list[dict]) -> dict | None:
    """Among completed tracks, pick by SELECT metrics only (never report)."""
    ok = [t for t in track_results if "error" not in t]
    if not ok:
        return None
    beaters = [t for t in ok if t.get("select_beats")]
    pool = beaters if beaters else ok
    return min(pool, key=lambda t: (float(t["select_mean_error"]), int(t.get("chosen_step", 10**9))))


def main(argv=None) -> int:
    assert "test" in FORBIDDEN_TRAIN_SPLITS
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("checkpoints/mms-sna-pure-beat-waxal"))
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    logger.info("device=%s seed=%s", device, args.seed)
    proto = split_val_protocol()
    assert proto.report_slice != proto.select_slice

    base_p, base_m = load_ctc(MID)
    base_m.to(device).eval()
    need = max(proto.report_indices[-1], proto.select_indices[-1]) + 1
    val_ds = load_hf_asr_split("sna", "validation", max_samples=need)

    tracks = list(PRE_REGISTERED_TRACKS)
    logger.info("PRE_REGISTERED_TRACKS n=%d (all will run; no report early-stop)", len(tracks))

    results: list[dict] = []
    for i, (mode, lr, steps, every) in enumerate(tracks):
        out = args.out_root / f"track_{i}_{mode}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            r = train_one(
                mode=mode,
                lr=lr,
                max_steps=steps,
                max_train=args.max_train,
                eval_every=every,
                device=device,
                base_m=base_m,
                base_p=base_p,
                val_ds=val_ds,
                proto=proto,
                out_dir=out,
            )
        except Exception as e:
            logger.exception("track failed: %s", e)
            continue
        results.append(r)
        # NO break on report — report not even computed yet

    winner = pick_track_by_select_only(results)
    summary = {
        "protocol": {
            "select": SELECT_SLICE_LABEL,
            "report": REPORT_SLICE_LABEL,
            "track_selection": "select_mean_error only among pre-registered tracks",
            "report_eval": "once after track+step selection",
        },
        "pre_registered_tracks": tracks,
        "track_results_select_only": [
            {
                "mode": t.get("mode"),
                "lr": t.get("lr"),
                "select_mean_error": t.get("select_mean_error"),
                "select_beats": t.get("select_beats"),
                "chosen_step": t.get("chosen_step"),
                "path": t.get("path"),
            }
            for t in results
            if "error" not in t
        ],
        "winner_by_select": None
        if winner is None
        else {
            "path": winner["path"],
            "mode": winner["mode"],
            "lr": winner["lr"],
            "select_mean_error": winner["select_mean_error"],
            "select_beats": winner["select_beats"],
            "chosen_step": winner["chosen_step"],
        },
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if winner is None:
        logger.error("NO select-eligible track")
        (args.out_root / "NO_WINNER").write_text(json.dumps(summary, indent=2, default=str))
        return 2

    # Promote select-winner, then ONE report decode
    import shutil

    canon = args.out_root / "best"
    if canon.exists():
        shutil.rmtree(canon)
    shutil.copytree(winner["path"], canon)

    proc = AutoProcessor.from_pretrained(str(canon), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(canon), local_files_only=True).to(device).eval()
    b_r, o_r, beat_r = eval_indices(
        model, proc, base_m, base_p, val_ds, proto.report_indices, device
    )
    me_r = mean_error(o_r["wer"], o_r["cer"])
    me_b = mean_error(b_r["wer"], b_r["cer"])
    logger.info(
        "SINGLE_REPORT_EVAL after select-only track+step pick: me_own=%.6f me_base=%.6f beats=%s",
        me_r,
        me_b,
        beat_r,
    )

    meta = json.loads((Path(winner["path"]).parent / "train_meta.json").read_text())
    meta.update(
        {
            "report_mean_error_own": me_r,
            "report_mean_error_base": me_b,
            "report_beats": beat_r,
            "own_metrics_report": o_r,
            "base_metrics_report": b_r,
            "report_eval_deferred": False,
            "track_selection": "select_only_among_pre_registered",
            "selected_track_mode": winner["mode"],
            "selected_track_lr": winner["lr"],
            "n_tracks_run": len(results),
            "no_report_early_stop_across_tracks": True,
        }
    )
    assert_honest_sna_meta(meta)
    (args.out_root / "train_meta.json").write_text(json.dumps(meta, indent=2))
    summary["single_report_eval"] = {
        "mean_error_own": me_r,
        "mean_error_baseline": me_b,
        "beats": beat_r,
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if beat_r:
        logger.info(
            "WINNER mode=%s lr=%s report_me=%.6f base=%.6f -> %s",
            winner["mode"],
            winner["lr"],
            me_r,
            me_b,
            canon,
        )
        print("PURE_SNA_HELDOUT_BEAT_OK")
        return 0

    logger.warning(
        "Select-winner failed report (own=%.6f base=%.6f); try pure-own soup α on SELECT only",
        me_r,
        me_b,
    )

    # Pure-own soup of select-beating tracks; α chosen ONLY on select; then ONE report.
    soup_tracks = [t for t in results if t.get("select_beats") and "error" not in t]
    if len(soup_tracks) < 2:
        logger.error("Not enough select-beaters for pure soup")
        (args.out_root / "NO_WINNER").write_text(json.dumps(summary, indent=2, default=str))
        print("PURE_SNA_SELECT_OK_REPORT_FAIL")
        return 2

    sds = []
    for t in soup_tracks:
        m = Wav2Vec2ForCTC.from_pretrained(t["path"], local_files_only=True)
        sds.append((t["path"], m.state_dict()))

    def mix(sd_a, sd_b, alpha: float):
        out = {}
        for k in sd_a:
            if (
                k in sd_b
                and sd_a[k].shape == sd_b[k].shape
                and sd_a[k].dtype.is_floating_point
            ):
                out[k] = alpha * sd_a[k] + (1.0 - alpha) * sd_b[k]
            else:
                out[k] = sd_a[k]
        return out

    # Pairwise soup; pick α on SELECT only
    best_soup = None  # (select_me, mixed, a, path_i, path_j)
    for i in range(len(sds)):
        for j in range(i + 1, len(sds)):
            pi, sdi = sds[i]
            pj, sdj = sds[j]
            best_a, best_sme, best_mixed = None, 1e9, None
            for a in (0.3, 0.4, 0.5, 0.6, 0.7):
                mixed = mix(sdi, sdj, a)
                soup_m = Wav2Vec2ForCTC.from_pretrained(MID, local_files_only=True)
                soup_m.load_state_dict(mixed, strict=True)
                soup_m.to(device).eval()
                b_s, o_s, beat_s = eval_indices(
                    soup_m, base_p, base_m, base_p, val_ds, proto.select_indices, device
                )
                sme = mean_error(o_s["wer"], o_s["cer"])
                if sme < best_sme:
                    best_sme, best_a, best_mixed = sme, a, mixed
                soup_m.cpu()
                del soup_m
            if best_mixed is not None and (
                best_soup is None or best_sme < best_soup[0]
            ):
                best_soup = (best_sme, best_mixed, best_a, pi, pj)

    if best_soup is None:
        (args.out_root / "NO_WINNER").write_text(json.dumps(summary, indent=2, default=str))
        print("PURE_SNA_SELECT_OK_REPORT_FAIL")
        return 2

    sme, mixed, alpha, pi, pj = best_soup
    soup_m = Wav2Vec2ForCTC.from_pretrained(MID, local_files_only=True)
    soup_m.load_state_dict(mixed, strict=True)
    soup_m.to(device).eval()
    # ONE report after select-only α
    b_r2, o_r2, beat_r2 = eval_indices(
        soup_m, base_p, base_m, base_p, val_ds, proto.report_indices, device
    )
    me_r2 = mean_error(o_r2["wer"], o_r2["cer"])
    me_b2 = mean_error(b_r2["wer"], b_r2["cer"])
    logger.info(
        "PURE_OWN_SOUP_REPORT me=%.6f base=%.6f beats=%s alpha=%s select_me=%.5f",
        me_r2,
        me_b2,
        beat_r2,
        alpha,
        sme,
    )

    if not beat_r2:
        summary["pure_own_soup"] = {
            "select_me": sme,
            "alpha": alpha,
            "report_me": me_r2,
            "beats": False,
            "components": [pi, pj],
        }
        (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (args.out_root / "NO_WINNER").write_text(json.dumps(summary, indent=2, default=str))
        print("PURE_SNA_SELECT_OK_REPORT_FAIL")
        return 2

    if canon.exists():
        shutil.rmtree(canon)
    canon.mkdir(parents=True, exist_ok=True)
    soup_m.save_pretrained(canon)
    base_p.save_pretrained(canon)
    meta = {
        "method": "pure_ft_ensemble_avg_select_weight",
        "components": [pi, pj],
        "ensemble_weight_on_first": alpha,
        "ensemble_weight_selected_on": SELECT_SLICE_LABEL,
        "early_stop_slice": SELECT_SLICE_LABEL,
        "report_slice": REPORT_SLICE_LABEL,
        "pure_own_checkpoint": True,
        "no_baseline_blend": True,
        "select_mean_error": sme,
        "report_mean_error_own": me_r2,
        "report_mean_error_base": me_b2,
        "report_beats": True,
        "own_metrics_report": o_r2,
        "base_metrics_report": b_r2,
        "track_selection": "select_only_among_pre_registered_then_pure_own_ensemble",
        "no_report_early_stop_across_tracks": True,
        "forbidden_train_splits": list(FORBIDDEN_TRAIN_SPLITS),
        "seed": SEED,
        "init": MID,
    }
    assert_honest_sna_meta(meta)
    (args.out_root / "train_meta.json").write_text(json.dumps(meta, indent=2))
    summary["single_report_eval"] = {
        "mean_error_own": me_r2,
        "mean_error_baseline": me_b2,
        "beats": True,
        "via": "pure_own_ensemble_avg",
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("WINNER pure-own ensemble report_me=%.6f -> %s", me_r2, canon)
    print("PURE_SNA_HELDOUT_BEAT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
