"""Unit tests for shortlist_gates — drive real shipped functions."""

from __future__ import annotations

from src.shortlist_gates import (
    ResidualFeatures,
    ban_check_verdict,
    conf_primary_banned,
    s01_lid_mass_ok,
    s05_structure_ok,
    s08_ortho_charlm_ok,
    s09_soft_dual_with_ortho_ok,
    select_replaces,
    stack_accept,
)


def _feat(**kwargs) -> ResidualFeatures:
    base = dict(
        id="ID_TEST",
        p1=0.999,
        lang2="lug",
        p2=0.001,
        ortho_mms=3.0,
        lp_luo_mms=-1.0,
        lp_ach_mms=-1.2,
        cer_mc=0.20,
        floor_hyp="pii madit dok i teng yadi",
        luo_hyp="pi madiet teng pi man tie yieri",
        pick_lang="ach",
        conf_luo=0.1,
        conf_ach=0.2,
        already_dual=False,
    )
    base.update(kwargs)
    return ResidualFeatures(**base)


def test_s05_rejects_stutter_and_accepts_normal():
    assert not s05_structure_ok("a a a a a")
    assert not s05_structure_ok("x x x x")
    assert s05_structure_ok("pi madiet teng pi man tie yieri matie")


def test_s08_requires_ortho_and_lp_delta():
    ok = _feat(ortho_mms=3.0, lp_luo_mms=-1.0, lp_ach_mms=-1.2)
    assert s08_ortho_charlm_ok(ok)
    assert not s08_ortho_charlm_ok(_feat(ortho_mms=1.0, lp_luo_mms=-1.0, lp_ach_mms=-1.2))
    assert not s08_ortho_charlm_ok(_feat(ortho_mms=3.0, lp_luo_mms=-1.2, lp_ach_mms=-1.0))


def test_s09_not_dual_thr_alone():
    # thr band without ortho → fail
    assert not s09_soft_dual_with_ortho_ok(_feat(cer_mc=0.18, ortho_mms=1.0, lp_luo_mms=-1.0, lp_ach_mms=-1.2))
    # thr band WITH ortho → pass
    assert s09_soft_dual_with_ortho_ok(_feat(cer_mc=0.18, ortho_mms=3.0, lp_luo_mms=-1.0, lp_ach_mms=-1.2))
    # outside band even with ortho → fail path b
    assert not s09_soft_dual_with_ortho_ok(_feat(cer_mc=0.40, ortho_mms=3.0, lp_luo_mms=-1.0, lp_ach_mms=-1.2))


def test_s01_rejects_ach_lang2_and_low_p1():
    assert s01_lid_mass_ok(_feat(p1=0.999, lang2="lug"))
    assert not s01_lid_mass_ok(_feat(p1=0.5, lang2="lug"))
    assert not s01_lid_mass_ok(_feat(p1=0.999, lang2="ach"))


def test_s01_rejects_junk_lang2_by_default():
    """eng/fas/wol/ckb weaken true-Luo mass — reject with reject_junk_lang2=True (default)."""
    for junk in ("eng", "fas", "wol", "ckb", "spa", "fra", "deu"):
        assert not s01_lid_mass_ok(_feat(p1=0.999, lang2=junk))
        ok, reason = stack_accept(_feat(id="ID_JUNK", p1=0.999, lang2=junk))
        assert not ok, junk
        # s01 runs after structure; structure-ok fixtures must fail on mass gate
        assert reason == "s01_lid_mass_fail", (junk, reason)
    # opt-out still allows junk when explicitly disabled
    assert s01_lid_mass_ok(_feat(p1=0.999, lang2="eng"), reject_junk_lang2=False)
    # clean Bantu-adjacent lang2 still ok
    assert s01_lid_mass_ok(_feat(p1=0.999, lang2="lug"))
    assert s01_lid_mass_ok(_feat(p1=0.999, lang2="nyn"))


def test_s05_rejects_consecutive_repeats_and_low_avg_len():
    # consecutive token loops (residual garbage pattern)
    assert not s05_structure_ok("pi pi pi madiet teng yieri matie")
    # very short avg word length
    assert not s05_structure_ok("a ab a ab a ab a ab")
    # normal hyp still passes
    assert s05_structure_ok(
        "pi madiet teng pi man tie yieri matie",
        floor_hyp="pii madit dok i teng yadi",
    )


def test_stack_accept_and_select_bounded():
    good = _feat(id="ID_GOOD", lang2="lug")
    ok, reason = stack_accept(good)
    assert ok and "S08" in reason
    bad = _feat(id="ID_BAD", ortho_mms=0.5, lang2="lug")
    ok2, _ = stack_accept(bad)
    assert not ok2
    # conf alone is not the stack (conf_primary can be true while stack fails)
    conf_only = _feat(id="ID_CONF", ortho_mms=0.1, conf_luo=0.9, conf_ach=0.1, lang2="lug")
    assert conf_primary_banned(conf_only)
    assert not stack_accept(conf_only)[0]
    # junk lang2 rejected even with strong ortho
    junk = _feat(id="ID_JUNK", lang2="wol", ortho_mms=4.0)
    assert not stack_accept(junk)[0]

    feats = [
        good,
        bad,
        junk,
        _feat(id="ID_G2", ortho_mms=2.9, lp_luo_mms=-1.0, lp_ach_mms=-1.15, lang2="lug"),
    ]
    reps = select_replaces(feats, max_n=1)
    assert len(reps) == 1
    assert reps[0]["ID"] in ("ID_GOOD", "ID_G2")
    assert "own_hyp" in reps[0]
    v = ban_check_verdict(len(reps))
    assert v["verdict"] == "PASS_ban_check"
    assert not v["mass_rewrite"]
