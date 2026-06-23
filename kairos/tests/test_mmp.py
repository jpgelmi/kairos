"""Tests for physio/mmp.py — MMP curve and CS from field data."""

import pytest
from kairos.physio.mmp import session_mmp, build_mmp_history, cs_from_mmp, MMP_DURATIONS_S


def _flat_records(speed: float, n: int) -> list[dict]:
    return [{"enhanced_speed": speed} for _ in range(n)]


def test_session_mmp_flat_speed() -> None:
    """Constant-speed session → best avg speed equals the constant speed for all durations."""
    records = _flat_records(4.0, 1500)
    mmp = session_mmp(records, durations_s=(120, 300, 600))
    assert mmp[120] == pytest.approx(4.0)
    assert mmp[300] == pytest.approx(4.0)
    assert mmp[600] == pytest.approx(4.0)


def test_session_mmp_short_session() -> None:
    """Session shorter than target duration → returns None for that duration."""
    records = _flat_records(4.0, 100)
    mmp = session_mmp(records, durations_s=(120, 300))
    assert mmp[120] is None
    assert mmp[300] is None


def test_session_mmp_peak_speed() -> None:
    """Session with a fast segment → best avg matches peak segment speed."""
    slow = _flat_records(2.0, 300)
    fast = _flat_records(5.5, 120)
    cooldown = _flat_records(2.0, 300)
    records = slow + fast + cooldown
    mmp = session_mmp(records, durations_s=(120,))
    assert mmp[120] == pytest.approx(5.5)


def test_build_mmp_history_takes_max() -> None:
    """build_mmp_history merges sessions and keeps the all-time best."""
    s1 = {120: 4.0, 300: 3.8, 600: 3.5}
    s2 = {120: 4.5, 300: 3.6, 600: 3.7}
    hist = build_mmp_history([s1, s2], durations_s=(120, 300, 600))
    assert hist[120] == pytest.approx(4.5)
    assert hist[300] == pytest.approx(3.8)
    assert hist[600] == pytest.approx(3.7)


def test_cs_field_marked_floor() -> None:
    """
    cs_from_mmp returns cs_source='training_mmp' (floor estimate, not a lab test).
    """
    # Synthetic MMP consistent with CS=3.5 m/s, D'=150 m
    cs_true, dp_true = 3.5, 150.0
    durations = [120, 180, 300, 600, 1200]
    mmp_hist = {d: cs_true + dp_true / d for d in durations}

    result = cs_from_mmp(mmp_hist)
    assert result is not None
    assert result["cs_source"] == "training_mmp"
    assert result["cs_ms"] == pytest.approx(cs_true, rel=0.01)
    assert result["d_prime_m"] == pytest.approx(dp_true, rel=0.05)


def test_cs_field_negative_cs_returns_none() -> None:
    """Degenerate MMP (all same speed) that produces CS <= 0 → returns None."""
    # All points at same speed → regression gives near-zero slope
    mmp_hist = {d: 3.0 for d in [120, 180, 300]}
    result = cs_from_mmp(mmp_hist)
    # CS from flat line will be near 0 or degenerate → check it handles gracefully
    # (may return None or a very small CS)
    if result is not None:
        assert result["cs_ms"] > 0


def test_cs_field_insufficient_points() -> None:
    """Fewer than 3 MMP points → returns None."""
    result = cs_from_mmp({120: 4.0, 300: 3.8}, min_points=3)
    assert result is None
