"""Floor-first Phase-2 packing helpers (no model I/O).

Used to build ban-compliant prize candidates that default to floor Targets
and only apply a documented high-precision replace set.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def align_ids(sample_ids: Sequence[str], targets: Mapping[str, str]) -> list[dict[str, str]]:
    """Return submission rows in SampleSubmission order. Missing IDs raise."""
    rows: list[dict[str, str]] = []
    for sid in sample_ids:
        if sid not in targets:
            raise KeyError(f"missing Target for ID={sid}")
        t = (targets[sid] or "").strip() or "."
        rows.append({"ID": sid, "Target": t})
    return rows


def apply_replace_set(
    floor: Mapping[str, str],
    replaces: Sequence[Mapping[str, Any]],
    *,
    id_key: str = "ID",
    hyp_key: str = "own_hyp",
) -> dict[str, str]:
    """Copy floor Targets then overlay replace-set hyps (high-precision only)."""
    out = {k: (v or "").strip() or "." for k, v in floor.items()}
    for r in replaces:
        sid = r[id_key]
        if sid not in out:
            raise KeyError(f"replace ID not in floor: {sid}")
        hyp = (r.get(hyp_key) or "").strip()
        if not hyp:
            continue
        out[sid] = hyp
    return out


def length_guard_ok(floor_t: str, new_t: str, *, lo: float = 0.5, hi: float = 2.0) -> bool:
    """Reject empty or extreme length-ratio replacements vs floor."""
    ft = (floor_t or "").split()
    nt = (new_t or "").split()
    if not nt:
        return False
    if not ft:
        return True
    r = len(nt) / max(1, len(ft))
    return lo <= r <= hi


def char_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio in [0, 1] for orthography-safe replace gates."""
    import difflib

    return difflib.SequenceMatcher(None, (a or "").strip(), (b or "").strip()).ratio()


def char_sim_ok(floor_t: str, new_t: str, *, min_sim: float = 0.6) -> bool:
    """Reject near-total rewrites (e.g. wrong-lang orthography flips)."""
    if not (new_t or "").strip():
        return False
    return char_sim(floor_t, new_t) >= min_sim


def word_level_merge(
    floor_t: str,
    specialist_t: str,
    *,
    token_min_sim: float = 0.72,
    max_token_edits: int | None = None,
    allow_insert: bool = False,
    allow_delete: bool = False,
) -> str:
    """Floor-default word merge: keep floor tokens unless specialist is a high-sim variant.

    Aligns words with SequenceMatcher. Equal spans keep floor. Replace spans swap
    only when lengths match and each token pair has char_sim >= token_min_sim.
    Inserts/deletes default to floor (safer for nyn↔lug orthography flips).
    """
    import difflib

    ft = (floor_t or "").strip()
    st = (specialist_t or "").strip()
    if not st or st == ".":
        return ft or "."
    if not ft or ft == ".":
        return st
    fw = ft.split()
    sw = st.split()
    if not fw:
        return st
    if not sw:
        return ft

    sm = difflib.SequenceMatcher(a=fw, b=sw, autojunk=False)
    out: list[str] = []
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(fw[i1:i2])
            continue
        if tag == "replace":
            f_span = fw[i1:i2]
            s_span = sw[j1:j2]
            if len(f_span) == len(s_span) and all(
                char_sim(a, b) >= token_min_sim for a, b in zip(f_span, s_span)
            ):
                if max_token_edits is not None and edits + len(s_span) > max_token_edits:
                    out.extend(f_span)
                else:
                    out.extend(s_span)
                    edits += sum(1 for a, b in zip(f_span, s_span) if a != b)
            else:
                # mixed-length or low-sim: keep floor tokens (no mass rewrite)
                out.extend(f_span)
            continue
        if tag == "delete":
            if allow_delete:
                # drop floor tokens only if allowed
                pass
            else:
                out.extend(fw[i1:i2])
            continue
        if tag == "insert":
            if allow_insert:
                out.extend(sw[j1:j2])
            # else: skip specialist inserts (floor default)
            continue
    merged = " ".join(out).strip()
    return merged or ft


def diff_stats(
    a: Mapping[str, str],
    b: Mapping[str, str],
    ids: Sequence[str] | None = None,
) -> dict[str, int]:
    """Count same/diff Targets over ids (default: keys of a)."""
    keys = list(ids) if ids is not None else list(a.keys())
    same = sum(1 for k in keys if (a.get(k) or "").strip() == (b.get(k) or "").strip())
    return {"n": len(keys), "n_same": same, "n_diff": len(keys) - same}


def is_banned_mass_rewrite(n_diff: int, n_total: int = 1500, *, max_frac: float = 0.25) -> bool:
    """Heuristic: replace fraction above max_frac is mass-rewrite risk (failed multifamily ~0.98)."""
    if n_total <= 0:
        return True
    return (n_diff / n_total) > max_frac
