"""Tests for physio/segmentation.py and segment-aware GCT drift."""

from __future__ import annotations

import pytest

from kairos.physio.dynamics import compute_gct_drift, gct_drift_from_segments
from kairos.physio.segmentation import (
    SessionSegment,
    enrich_segments_with_records,
    is_interval_session,
    preceding_work_segment,
    segment_at,
    segment_from_laps,
    segment_from_speed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lap(avg_speed: float, duration: float, hr: float = 140.0) -> dict:
    return {
        "avg_speed": avg_speed,
        "total_elapsed_time": duration,
        "total_timer_time": duration,
        "avg_heart_rate": hr,
        "total_distance": avg_speed * duration,
    }


def _interval_laps(
    n_reps: int = 20,
    work_speed: float = 5.0,
    rec_speed: float = 1.5,
    warmup_speed: float = 2.5,
    work_dur: float = 40.0,
    rec_dur: float = 60.0,
) -> list[dict]:
    laps = [_make_lap(warmup_speed, 300.0)]
    for i in range(n_reps):
        laps.append(_make_lap(work_speed, work_dur))
        if i < n_reps - 1:
            laps.append(_make_lap(rec_speed, rec_dur))
    laps.append(_make_lap(warmup_speed, 300.0))
    return laps


# ---------------------------------------------------------------------------
# Feature C: segment_from_laps
# ---------------------------------------------------------------------------

def test_segment_from_laps_counts() -> None:
    """20×200 m workout → 20 work, 19 recovery, 1 warmup, 1 cooldown."""
    laps = _interval_laps(n_reps=20)
    segs = segment_from_laps(laps)

    work = [s for s in segs if s.kind == "work"]
    rec = [s for s in segs if s.kind == "recovery"]
    warmup = [s for s in segs if s.kind == "warmup"]
    cool = [s for s in segs if s.kind == "cooldown"]

    assert len(work) == 20
    assert len(rec) == 19
    assert len(warmup) == 1
    assert len(cool) == 1


def test_segment_from_laps_no_laps_returns_empty() -> None:
    assert segment_from_laps([]) == []


def test_segment_from_laps_avg_speed_set() -> None:
    laps = _interval_laps(n_reps=5, work_speed=5.0, rec_speed=1.5)
    segs = segment_from_laps(laps)
    work = [s for s in segs if s.kind == "work"]
    assert all(s.avg_speed_ms is not None and s.avg_speed_ms > 4.0 for s in work)


# ---------------------------------------------------------------------------
# Feature C: segment_from_speed (fallback)
# ---------------------------------------------------------------------------

def _interval_records(
    n_reps: int = 20,
    work_speed: float = 5.0,
    rec_speed: float = 1.5,
    work_dur: int = 40,
    rec_dur: int = 60,
) -> list[dict]:
    records = []
    # 5 min warmup
    for t in range(300):
        records.append({"t_s": float(t), "speed": 2.5})
    t0 = 300
    for i in range(n_reps):
        for t in range(work_dur):
            records.append({"t_s": float(t0 + t), "speed": work_speed})
        t0 += work_dur
        if i < n_reps - 1:
            for t in range(rec_dur):
                records.append({"t_s": float(t0 + t), "speed": rec_speed})
            t0 += rec_dur
    # 5 min cooldown
    for t in range(300):
        records.append({"t_s": float(t0 + t), "speed": 2.5})
    return records


def test_segment_speed_fallback_work_count() -> None:
    records = _interval_records(n_reps=20)
    segs = segment_from_speed(records)
    work = [s for s in segs if s.kind == "work"]
    assert len(work) == 20


def test_segment_speed_fallback_no_records() -> None:
    assert segment_from_speed([]) == []


# ---------------------------------------------------------------------------
# Feature C: merge tiny segments
# ---------------------------------------------------------------------------

def test_segment_merges_tiny() -> None:
    """Segments shorter than MIN_SEGMENT_S are merged into the previous segment."""
    from kairos.config import MIN_SEGMENT_S

    tiny = MIN_SEGMENT_S / 2.0
    laps = [
        _make_lap(5.0, 60.0),   # work
        _make_lap(1.5, tiny),    # tiny recovery — should be merged into previous
        _make_lap(5.0, 60.0),   # work
    ]
    segs = segment_from_laps(laps)
    # The tiny recovery should be absorbed; we expect fewer segments than laps
    assert len(segs) < len(laps)


# ---------------------------------------------------------------------------
# Feature C: is_interval_session
# ---------------------------------------------------------------------------

def test_is_interval_session_by_type() -> None:
    assert is_interval_session("interval", []) is True


def test_is_interval_session_by_segment_count() -> None:
    work_segs = [
        SessionSegment(idx=i, kind="work", start_s=0, end_s=40, duration_s=40)
        for i in range(5)
    ]
    assert is_interval_session("easy", work_segs) is True


def test_not_interval_too_few_work() -> None:
    work_segs = [
        SessionSegment(idx=0, kind="work", start_s=0, end_s=40, duration_s=40),
        SessionSegment(idx=1, kind="work", start_s=100, end_s=140, duration_s=40),
    ]
    assert is_interval_session("easy", work_segs) is False


# ---------------------------------------------------------------------------
# Feature C: z_bio excludes recovery segments
# ---------------------------------------------------------------------------

def test_zbio_intervals_excludes_recovery() -> None:
    """
    20 work reps with GCT growing 220→258 ms (drift ≈ 17 %).
    19 recovery segments with GCT = 350 ms.
    The segment-based drift must detect the work-rep increase, not be
    distorted by recovery GCTs.
    """
    segs: list[SessionSegment] = []
    t = 0.0
    for i in range(20):
        segs.append(SessionSegment(
            idx=len(segs), kind="work",
            start_s=t, end_s=t + 40.0, duration_s=40.0,
            avg_speed_ms=5.0,
            avg_gct_ms=220.0 + 2.0 * i,   # 220 → 258 ms across 20 reps
        ))
        t += 40.0
        if i < 19:
            segs.append(SessionSegment(
                idx=len(segs), kind="recovery",
                start_s=t, end_s=t + 60.0, duration_s=60.0,
                avg_speed_ms=1.5,
                avg_gct_ms=350.0,            # high GCT, must not enter drift calc
            ))
            t += 60.0

    drift = gct_drift_from_segments(segs)
    assert drift is not None
    # Expected: (258 - 220) / 220 ≈ 0.173
    assert 0.15 < drift < 0.20


def test_zbio_needs_three_work_segments() -> None:
    segs = [
        SessionSegment(idx=0, kind="work", start_s=0, end_s=40, duration_s=40, avg_gct_ms=230.0),
        SessionSegment(idx=1, kind="work", start_s=100, end_s=140, duration_s=40, avg_gct_ms=235.0),
    ]
    assert gct_drift_from_segments(segs) is None


# ---------------------------------------------------------------------------
# Feature C: continuous session uses legacy drift
# ---------------------------------------------------------------------------

def test_continuous_session_uses_legacy_drift() -> None:
    """Legacy compute_gct_drift works correctly for steady-state records."""
    # 100 samples with gentle GCT increase
    gct_series = [230.0 + i * 0.1 for i in range(100)]
    drift = compute_gct_drift(gct_series)
    assert drift is not None
    # first 20 % ≈ median([230..231.9]) ≈ 231.0
    # last  20 % ≈ median([238..239.9]) ≈ 239.0
    # drift ≈ (239 - 231) / 231 ≈ 0.035
    assert 0.02 < drift < 0.06


# ---------------------------------------------------------------------------
# Feature C: segment_at / preceding_work_segment
# ---------------------------------------------------------------------------

def test_segment_at_finds_correct_segment() -> None:
    segs = [
        SessionSegment(idx=0, kind="work", start_s=0.0, end_s=40.0, duration_s=40.0),
        SessionSegment(idx=1, kind="recovery", start_s=40.0, end_s=100.0, duration_s=60.0),
    ]
    assert segment_at(segs, 20.0).kind == "work"  # type: ignore[union-attr]
    assert segment_at(segs, 50.0).kind == "recovery"  # type: ignore[union-attr]
    assert segment_at(segs, 200.0) is None


def test_preceding_work_segment() -> None:
    segs = [
        SessionSegment(idx=0, kind="work", start_s=0.0, end_s=40.0, duration_s=40.0, avg_speed_ms=5.0),
        SessionSegment(idx=1, kind="recovery", start_s=40.0, end_s=100.0, duration_s=60.0),
    ]
    rec = segs[1]
    work = preceding_work_segment(segs, rec)
    assert work is not None
    assert work.kind == "work"
    assert work.avg_speed_ms == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Feature C: enrich_segments_with_records
# ---------------------------------------------------------------------------

def test_enrich_segments_fills_gct() -> None:
    segs = [
        SessionSegment(idx=0, kind="work", start_s=0.0, end_s=10.0, duration_s=10.0),
    ]
    records = [
        {"t_s": float(i), "ground_contact_time": 230.0 + i} for i in range(10)
    ]
    enriched = enrich_segments_with_records(segs, records)
    assert enriched[0].avg_gct_ms is not None
    assert 230.0 < enriched[0].avg_gct_ms < 240.0
