"""Static check: sna pure FT never selects on report slice (step or track)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.train_sna_pure_beat_waxal import pick_track_by_select_only


def test_train_one_has_zero_report_indices():
    """train_one must not touch report_indices at all."""
    lines = Path("scripts/train_sna_pure_beat_waxal.py").read_text().splitlines()
    in_fn = False
    report_uses = 0
    select_uses = 0
    for line in lines:
        if line.startswith("def train_one"):
            in_fn = True
            continue
        if in_fn and line.startswith("def "):
            break
        if not in_fn or line.strip().startswith("#"):
            continue
        if "report_indices" in line:
            report_uses += 1
        if "select_indices" in line:
            select_uses += 1
    assert select_uses >= 2
    assert report_uses == 0, f"train_one must not use report_indices, got {report_uses}"


def test_main_has_single_report_after_track_pick():
    script = Path("scripts/train_sna_pure_beat_waxal.py").read_text()
    assert "pick_track_by_select_only" in script
    assert "PRE_REGISTERED_TRACKS" in script
    assert "SINGLE_REPORT_EVAL" in script
    assert "if r.get(\"beats_report\")" not in script
    assert "beats_report" not in script
    # Scoreboard decode of report slice must follow select-only track pick
    i_pick = script.find("pick_track_by_select_only(results)")
    i_eval = script.find("SINGLE_REPORT_EVAL")
    assert i_pick > 0 and i_eval > i_pick


def test_pick_track_by_select_only_ignores_report_fields():
    tracks = [
        {
            "mode": "a",
            "select_mean_error": 0.20,
            "select_beats": True,
            "chosen_step": 10,
            "path": "a",
            "report_me": 0.01,  # better report — must be ignored
        },
        {
            "mode": "b",
            "select_mean_error": 0.15,
            "select_beats": True,
            "chosen_step": 20,
            "path": "b",
            "report_me": 0.99,  # worse report
        },
    ]
    w = pick_track_by_select_only(tracks)
    assert w is not None
    assert w["mode"] == "b"  # better select, not better report


def test_meta_on_disk_honest_if_present():
    from src.criterion1_protocol import assert_honest_sna_meta

    p = Path("checkpoints/mms-sna-pure-beat-waxal/train_meta.json")
    if not p.exists():
        return
    meta = json.loads(p.read_text())
    # Skip incomplete track-only metas (report deferred)
    if meta.get("report_eval_deferred") is True and "report_beats" not in meta:
        return
    assert_honest_sna_meta(meta)
    assert meta["early_stop_slice"] != meta["report_slice"]
    if "report_beats" in meta:
        assert meta.get("no_report_early_stop_across_tracks") is True or str(
            meta.get("track_selection", "")
        ).startswith("select_only")