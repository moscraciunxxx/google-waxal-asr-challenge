"""Criterion-1 honest protocol: disjoint select vs report val slices.

Selection (early-stop / hyperparams) may use ONLY the select slice.
CRITERION1 scoreboard uses ONLY the report slice, evaluated once after selection.
Never pass report metrics into checkpoint selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Fixed protocol (SEED-independent index ranges into HF validation order).
REPORT_START = 0
REPORT_END = 50  # exclusive → val[0:50]
SELECT_START = 50
SELECT_END = 90  # exclusive → val[50:90]

REPORT_SLICE_LABEL = f"val[{REPORT_START}:{REPORT_END}]"
SELECT_SLICE_LABEL = f"val[{SELECT_START}:{SELECT_END}]"

WAXAL300 = {
    "lin": "waxal-benchmarking/mms-300m-waxal-lin",
    "sna": "waxal-benchmarking/mms-300m-waxal-sna",
    "lug": "waxal-benchmarking/mms-300m-waxal-lug",
}


@dataclass(frozen=True)
class ValProtocol:
    """Fixed disjoint validation partitions for criterion-1."""

    report_indices: tuple[int, ...]
    select_indices: tuple[int, ...]
    report_slice: str = REPORT_SLICE_LABEL
    select_slice: str = SELECT_SLICE_LABEL

    def __post_init__(self) -> None:
        rs, ss = set(self.report_indices), set(self.select_indices)
        if not self.report_indices or not self.select_indices:
            raise ValueError("report and select slices must be nonempty")
        if rs & ss:
            raise ValueError(
                f"report and select slices must be disjoint; overlap={rs & ss}"
            )
        if self.report_slice == self.select_slice:
            raise ValueError("report_slice label must differ from select_slice")


def split_val_protocol(
    n_val: int | None = None,
    *,
    report_start: int = REPORT_START,
    report_end: int = REPORT_END,
    select_start: int = SELECT_START,
    select_end: int = SELECT_END,
) -> ValProtocol:
    """Build fixed report/select index tuples.

    If n_val is set, slices are clipped to available length (still must be nonempty
    and disjoint after clip).
    """
    r0, r1 = report_start, report_end
    s0, s1 = select_start, select_end
    if n_val is not None:
        r1 = min(r1, n_val)
        s1 = min(s1, n_val)
        s0 = min(s0, n_val)
    report = tuple(range(r0, r1))
    select = tuple(range(s0, s1))
    return ValProtocol(report_indices=report, select_indices=select)


@dataclass
class CheckpointCandidate:
    """One pure checkpoint observed during train (select metrics only)."""

    step: int
    select_mean_error: float
    select_beats_baseline: bool
    state_key: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def select_checkpoint_by_slice(
    candidates: Sequence[CheckpointCandidate],
    *,
    # Deliberately NOT accepted for selection — any report_* args are ignored.
    report_mean_error: float | None = None,
    report_beats: bool | None = None,
    report_metrics: Mapping[str, Any] | None = None,
) -> CheckpointCandidate | None:
    """Pick best pure checkpoint using SELECT metrics only.

    Preference:
      1. Among candidates that beat baseline on select, lowest select_mean_error.
      2. Else lowest select_mean_error overall (may still fail report later).

    Report fields are accepted only so accidental call sites cannot feed them into
    selection — they are ignored.
    """
    _ = (report_mean_error, report_beats, report_metrics)  # never used
    if not candidates:
        return None
    beaters = [c for c in candidates if c.select_beats_baseline]
    pool = beaters if beaters else list(candidates)
    return min(pool, key=lambda c: (c.select_mean_error, c.step))


def finalize_report(
    own_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    *,
    beats: bool | None = None,
    own_model: str,
    baseline_model: str,
    lang: str,
    n: int,
    protocol: ValProtocol,
    own_kind: str,
    train_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one language's CRITERION1 report entry from REPORT metrics only."""
    from src.legit_fusion import mean_error

    mean_error_baseline = mean_error(
        float(baseline_metrics["wer"]), float(baseline_metrics["cer"])
    )
    mean_error_own = mean_error(float(own_metrics["wer"]), float(own_metrics["cer"]))
    derived_beats = mean_error_own < mean_error_baseline
    if beats is not None and bool(beats) != derived_beats:
        raise ValueError(
            "beats disagrees with report metrics: "
            f"own={mean_error_own:.8f}, baseline={mean_error_baseline:.8f}"
        )

    return {
        "lang": lang,
        "n": n,
        "split": "validation",
        "baseline_model": baseline_model,
        "baseline_kind": "waxal_mms300",
        "own_model": own_model,
        "own_kind": own_kind,
        "baseline": dict(baseline_metrics),
        "own": dict(own_metrics),
        "beats": derived_beats,
        "mean_error_baseline": mean_error_baseline,
        "mean_error_own": mean_error_own,
        "report_slice": protocol.report_slice,
        "selection_slice": protocol.select_slice,
        "pure_own_no_baseline_blend": True,
        "train_meta": dict(train_meta) if train_meta else None,
    }


def assert_honest_sna_meta(meta: Mapping[str, Any]) -> None:
    """Raise if sna train meta violates honesty protocol."""
    required = (
        "early_stop_slice",
        "report_slice",
        "pure_own_checkpoint",
        "no_baseline_blend",
    )
    missing = [k for k in required if k not in meta]
    if missing:
        raise AssertionError(f"sna train_meta missing keys: {missing}")
    if meta.get("early_stop_slice") == meta.get("report_slice"):
        raise AssertionError(
            "early_stop_slice must differ from report_slice "
            f"(got both={meta.get('report_slice')!r})"
        )
    if not meta.get("pure_own_checkpoint"):
        raise AssertionError("pure_own_checkpoint must be true")
    if not meta.get("no_baseline_blend"):
        raise AssertionError("no_baseline_blend must be true")
    if meta.get("baseline_blend") is True:
        raise AssertionError("baseline_blend must not be true")
    method = str(meta.get("method") or "").lower()
    if "baseline" in method and ("soup" in method or "blend" in method):
        raise AssertionError(f"forbidden baseline blend method: {method!r}")
    # Historical field: weight on WAXAL baseline specialist
    if "alpha_base" in meta and meta.get("own_ft_component"):
        raise AssertionError("alpha_base + own_ft_component implies baseline soup")


def all_languages_beat(per_language: Sequence[Mapping[str, Any]]) -> bool:
    """True only if every language entry has beats=True (report metrics)."""
    if len(per_language) < 3:
        return False
    return all(bool(r.get("beats")) for r in per_language)
