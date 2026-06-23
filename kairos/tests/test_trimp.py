"""Tests for physio/trimp.py — TRIMP computation."""

import math
import pytest
from kairos.physio.trimp import (
    edwards_trimp,
    banister_trimp,
    te_proxy_trimp,
    compute_session_trimp,
)


HRMAX = 200.0


def test_edwards_trimp_zones() -> None:
    """Each zone assigns the correct weight; zone-minutes are computed correctly."""
    # 60 s at exactly 75 % HRmax → zone 3 (weight=3) → TRIMP = 60*3/60 = 3.0
    hr_z3 = [0.75 * HRMAX] * 60
    assert edwards_trimp(hr_z3, HRMAX) == pytest.approx(3.0)

    # 60 s at 85 % HRmax → zone 4 (weight=4) → TRIMP = 4.0
    hr_z4 = [0.85 * HRMAX] * 60
    assert edwards_trimp(hr_z4, HRMAX) == pytest.approx(4.0)

    # 60 s at 95 % HRmax → zone 5 (weight=5) → TRIMP = 5.0
    hr_z5 = [0.95 * HRMAX] * 60
    assert edwards_trimp(hr_z5, HRMAX) == pytest.approx(5.0)

    # 60 s at 55 % HRmax → zone 1 (weight=1) → TRIMP = 1.0
    hr_z1 = [0.55 * HRMAX] * 60
    assert edwards_trimp(hr_z1, HRMAX) == pytest.approx(1.0)

    # Below 50 % → weight 0 → TRIMP = 0
    hr_below = [0.40 * HRMAX] * 60
    assert edwards_trimp(hr_below, HRMAX) == pytest.approx(0.0)


def test_edwards_trimp_mixed_zones() -> None:
    """Mixed zone session sums correctly."""
    hr_z3 = [0.75 * HRMAX] * 60   # TRIMP += 3
    hr_z4 = [0.85 * HRMAX] * 120  # TRIMP += 8
    total = edwards_trimp(hr_z3 + hr_z4, HRMAX)
    assert total == pytest.approx(11.0)


def test_edwards_trimp_empty() -> None:
    assert edwards_trimp([], HRMAX) == pytest.approx(0.0)


def test_banister_trimp_positive() -> None:
    """Banister TRIMP should be positive for effort above rest HR."""
    hr_effort = [160.0] * 600  # 10 min at 160 bpm
    result = banister_trimp(hr_effort, hrmax=200.0, hrrest=50.0)
    assert result > 0


def test_banister_trimp_degenerate() -> None:
    """hrmax <= hrrest → returns 0."""
    assert banister_trimp([160.0] * 60, hrmax=50.0, hrrest=60.0) == pytest.approx(0.0)


def test_te_proxy_trimp() -> None:
    """TE=2.5, 60 min → TRIMP = 2.5 × 60 = 150."""
    assert te_proxy_trimp(2.5, 60.0) == pytest.approx(150.0)


def test_te_proxy_trimp_zero_inputs() -> None:
    assert te_proxy_trimp(0.0, 60.0) == pytest.approx(0.0)
    assert te_proxy_trimp(3.0, 0.0) == pytest.approx(0.0)


def test_compute_session_trimp_returns_source() -> None:
    """compute_session_trimp labels source correctly."""
    hr = [0.75 * HRMAX] * 3600  # 60 min zone 3
    trimp, src = compute_session_trimp(hr, HRMAX, method="edwards")
    assert src == "edwards_trimp"
    assert trimp > 0

    trimp_b, src_b = compute_session_trimp(hr, HRMAX, hrrest=50.0, method="banister_trimp")
    assert src_b == "banister_trimp"
    assert trimp_b > 0


def test_compute_session_trimp_empty_hr() -> None:
    trimp, src = compute_session_trimp([], HRMAX)
    assert src == "te_proxy"
    assert trimp == pytest.approx(0.0)
