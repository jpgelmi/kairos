"""Tests for physio/efficiency.py — Efficiency Factor and decoupling."""

import math
import pytest
from kairos.physio.efficiency import grade_factor, compute_gap_series, compute_ef_session


def test_grade_factor_flat() -> None:
    """Grade 0 % → factor = 1 (no correction)."""
    assert grade_factor(0.0) == pytest.approx(1.0)


def test_grade_factor_uphill() -> None:
    """Uphill → factor > 1 (GAP > raw speed)."""
    assert grade_factor(10.0) > 1.0


def test_grade_factor_downhill() -> None:
    """Downhill → factor < 1 (GAP < raw speed)."""
    assert grade_factor(-10.0) < 1.0


def _make_records(
    n: int,
    speed: float = 3.5,
    hr: float = 140.0,
    altitude: float = 0.0,
) -> list[dict]:
    """Flat records at constant speed and HR."""
    return [
        {
            "t_s": float(i),
            "enhanced_speed": speed,
            "heart_rate": hr,
            "enhanced_altitude": altitude,
        }
        for i in range(n)
    ]


def test_ef_steady_only_gap_corrected() -> None:
    """
    EF is computed on steady aerobic records only (HR < ceiling).
    Records with HR above the aerobic ceiling are excluded.
    GAP correction applied: on flat terrain, GAP == speed.
    """
    hrmax = 190.0
    ceiling = hrmax * 0.85  # 161.5 bpm

    # 1400 steady aerobic records + 100 high-HR records (above ceiling)
    steady = _make_records(1400, speed=3.5, hr=140.0)
    hard = _make_records(100, speed=5.0, hr=170.0)  # above ceiling
    records = steady + hard

    result = compute_ef_session(records, hrmax=hrmax)
    assert result is not None
    assert result["n_records_used"] == 1400   # high-HR excluded

    # EF should be based only on steady segment speed and HR
    expected_ef = 3.5 / 140.0
    assert result["ef"] == pytest.approx(expected_ef, rel=1e-3)


def test_ef_gap_uphill() -> None:
    """With altitude gain, GAP > raw speed → EF computed from GAP, not raw speed."""
    hrmax = 190.0
    # Simulate uphill: each second altitude increases by 0.1 m → grade ~2.9 %
    records = [
        {
            "t_s": float(i),
            "enhanced_speed": 3.0,
            "heart_rate": 140.0,
            "enhanced_altitude": float(i) * 0.1,
        }
        for i in range(1500)
    ]
    result = compute_ef_session(records, hrmax=hrmax)
    assert result is not None
    # GAP-based EF should be higher than raw-speed EF (uphill correction)
    raw_ef = 3.0 / 140.0
    assert result["ef"] > raw_ef


def test_ef_insufficient_data() -> None:
    """Fewer than EF_MIN_DURATION_S steady records → returns None."""
    hrmax = 190.0
    records = _make_records(100, hr=140.0)  # only 100 records
    result = compute_ef_session(records, hrmax=hrmax, min_duration_s=1200.0)
    assert result is None


def test_ef_decoupling_cardiac_drift() -> None:
    """
    Cardiac drift: HR increases over time at constant speed.
    Second half has higher HR → lower EF → positive decoupling.
    """
    hrmax = 190.0
    records = [
        {
            "t_s": float(i),
            "enhanced_speed": 3.5,
            "heart_rate": 130.0 + i * 0.02,  # HR drifts from 130 to ~160
            "enhanced_altitude": 0.0,
        }
        for i in range(1500)
    ]
    result = compute_ef_session(records, hrmax=hrmax)
    assert result is not None
    assert result["decoupling_pct"] is not None
    # EF in first half (lower HR) > second half → positive decoupling
    assert result["decoupling_pct"] > 0
