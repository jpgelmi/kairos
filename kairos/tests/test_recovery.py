"""Tests for physio/recovery.py — HRR60 and τ_HRR computation."""

import math
import pytest
from kairos.physio.recovery import compute_hrr_transition, HRRResult


def _make_recovery_records(
    work_end_t: float,
    hr_peak: float,
    hr_floor: float,
    tau: float,
    duration_s: float,
    dt: float = 1.0,
) -> list[dict]:
    """Synthetic HR series: peak at work_end_t, exponential decay during recovery."""
    records = []
    # Last 10 s of work (at hr_peak)
    for i in range(10):
        t = work_end_t - 10.0 + i
        records.append({"t_s": t, "heart_rate": hr_peak})
    # Recovery: exponential decay
    t = work_end_t + dt
    while t <= work_end_t + duration_s:
        elapsed = t - work_end_t
        hr = (hr_peak - hr_floor) * math.exp(-elapsed / tau) + hr_floor
        records.append({"t_s": t, "heart_rate": hr})
        t += dt
    return records


def test_hrr_tau_robust_to_pause_length() -> None:
    """
    τ estimation works for both short (45 s) and long (120 s) recovery pauses.
    The estimated τ should approximate the true τ regardless of pause length.
    """
    work_end_t = 100.0
    hr_peak = 175.0
    hr_floor = 100.0
    true_tau = 40.0  # seconds

    # Short pause: 45 s
    records_short = _make_recovery_records(work_end_t, hr_peak, hr_floor, true_tau, 45)
    result_short = compute_hrr_transition(
        records_short, work_end_t, work_end_t + 45.0,
        hrmax=None,  # no threshold check
    )
    assert result_short is not None
    assert result_short.tau_s is not None
    # τ estimate should be within 50 % of true τ
    assert abs(result_short.tau_s - true_tau) / true_tau < 0.50

    # Long pause: 120 s
    records_long = _make_recovery_records(work_end_t, hr_peak, hr_floor, true_tau, 120)
    result_long = compute_hrr_transition(
        records_long, work_end_t, work_end_t + 120.0,
        hrmax=None,
    )
    assert result_long is not None
    assert result_long.tau_s is not None
    assert abs(result_long.tau_s - true_tau) / true_tau < 0.50


def test_hrr60_computed_when_pause_long_enough() -> None:
    """HRR60 is the drop at t=60s; available when recovery ≥ 60 s."""
    work_end_t = 0.0
    hr_peak = 180.0
    hr_floor = 110.0
    true_tau = 35.0

    records = _make_recovery_records(work_end_t, hr_peak, hr_floor, true_tau, 90)
    result = compute_hrr_transition(records, work_end_t, 90.0, hrmax=None)
    assert result is not None
    assert result.hrr60 is not None
    # At t=60: HR ≈ (180-110)*exp(-60/35)+110 ≈ 115; drop ≈ 180-115 = 65
    expected_drop = hr_peak - ((hr_peak - hr_floor) * math.exp(-60 / true_tau) + hr_floor)
    assert abs(result.hrr60 - expected_drop) < 10.0


def test_hrr60_none_when_pause_too_short() -> None:
    """Pause < 60 s → hrr60 is None (can't measure 60-s recovery)."""
    work_end_t = 0.0
    records = _make_recovery_records(work_end_t, 175.0, 100.0, 40.0, 45)
    result = compute_hrr_transition(records, work_end_t, 45.0, hrmax=None)
    assert result is not None
    assert result.hrr60 is None


def test_hrr_peak_threshold_filters_low_intensity() -> None:
    """Work bouts that don't reach HRR_MIN_PEAK_PCT × HRmax are filtered out."""
    work_end_t = 0.0
    records = _make_recovery_records(work_end_t, 140.0, 90.0, 40.0, 90)
    result = compute_hrr_transition(
        records, work_end_t, 90.0,
        hrmax=200.0,       # peak 140 < 85 % × 200 = 170 → should be filtered
    )
    assert result is None


def test_hrr_insufficient_recovery_records() -> None:
    """Fewer than 5 recovery records → returns None."""
    records = [{"t_s": float(i), "heart_rate": 170.0 - i} for i in range(3)]
    result = compute_hrr_transition(records, 0.0, 90.0, hrmax=None)
    assert result is None
