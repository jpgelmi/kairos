"""
TRIMP — Training Impulse from per-second HR records.

Banister TRIMP (default): continuous formulation using heart-rate reserve (HRr).
Edwards TRIMP (legacy): seconds spent in each of 5 %HRmax zones × zone weight (1-5),
divided by 60 to give zone-minutes.

HRmax detection: percentile 99 of spike-filtered max_hr stored in sessions, or config.HR_MAX override.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kairos.config import DB_PATH
from kairos.config import HR_MAX as _CFG_HRMAX
from kairos.config import HR_REST as _CFG_HRREST

# (lower_bound_pct, upper_bound_pct, weight)
_EDWARDS_ZONES: list[tuple[float, float, int]] = [
    (0.50, 0.60, 1),
    (0.60, 0.70, 2),
    (0.70, 0.80, 3),
    (0.80, 0.90, 4),
    (0.90, 1.01, 5),  # 1.01 captures exactly 100 %
]

_DEFAULT_HRMAX: float = 190.0
_DEFAULT_HRREST: float = 45.0


def edwards_trimp(hr_records: list[float], hrmax: float) -> float:
    """
    Edwards TRIMP from a per-second HR series.
    Returns zone-minutes (seconds in zone × weight / 60).
    Records below 50 % HRmax contribute 0.
    """
    total = 0.0
    for hr in hr_records:
        pct = hr / hrmax
        for lo, hi, weight in _EDWARDS_ZONES:
            if lo <= pct < hi:
                total += weight
                break
    return total / 60.0


def banister_trimp(hr_records: list[float], hrmax: float, hrrest: float) -> float:
    """
    Banister TRIMP: Σ_sec  HRr × 0.64 × exp(1.92 × HRr)  / 60
    where HRr = (HR − HRrest) / (HRmax − HRrest).
    """
    if hrmax <= hrrest:
        return 0.0
    total = 0.0
    for hr in hr_records:
        hrr = (hr - hrrest) / (hrmax - hrrest)
        hrr = max(0.0, min(1.0, hrr))
        total += hrr * 0.64 * np.exp(1.92 * hrr)
    return total / 60.0


def te_proxy_trimp(te: float, duration_min: float) -> float:
    """
    Estimate TRIMP when no HR records are available.
    total_training_effect (0-5) ≈ Edwards zone weight → TRIMP = duration × TE.
    """
    if te <= 0 or duration_min <= 0:
        return 0.0
    zone_weight = max(1.0, min(5.0, te))
    return duration_min * zone_weight


_SPIKE_RATIO: float = 1.35  # max_hr/avg_hr above this → sensor artifact, not physiology
_HRREST_MAX_PLAUSIBLE: float = 80.0  # session avg_hr can never reliably capture true resting HR


def detect_hrmax(db_path: Path = DB_PATH) -> float:
    """
    Percentile 99 of spike-filtered max_hr.

    Spike filter: discard sessions where max_hr > avg_hr * 1.35.
    A genuine max-effort session might hit 1.10–1.20×avg_hr; ratios above 1.35
    are sensor artifacts (e.g., cadence dropout, strap spike).
    Uses percentile 99 (not 99.9) on the filtered set so a handful of artifacts
    cannot inflate the result even if the ratio filter misses them.
    """
    if _CFG_HRMAX is not None:
        return _CFG_HRMAX
    from kairos.db import db_cursor
    try:
        with db_cursor(db_path) as cur:
            rows = cur.execute(
                "SELECT max_hr, avg_hr FROM sessions "
                "WHERE max_hr IS NOT NULL AND avg_hr IS NOT NULL"
            ).fetchall()
        if rows:
            vals = [
                float(r["max_hr"]) for r in rows
                if float(r["max_hr"]) <= float(r["avg_hr"]) * _SPIKE_RATIO
            ]
            if vals:
                return float(np.percentile(vals, 99))
    except Exception:
        pass
    # Try sync_state for cached value from last rebuild-trimp
    try:
        from kairos.ingest.garmin_sync import get_sync_state
        stored = get_sync_state("hrmax_detected", db_path)
        if stored:
            return float(stored)
    except Exception:
        pass
    return _DEFAULT_HRMAX


def detect_hrrest(db_path: Path = DB_PATH) -> float:
    """
    Estimate resting HR from session data.

    Session avg_hr reflects exercise state, not true resting HR — morning HRV
    measurement sessions go to hrv_daily without avg_hr, so the true resting
    value (≈40–65 bpm for trained runners) never appears in sessions.avg_hr.
    If the auto-detected value exceeds _HRREST_MAX_PLAUSIBLE (80 bpm) it means
    the data cannot support the estimate; fall back to the physiological default.
    Set HR_REST in config.py to override with your known morning resting HR.
    """
    if _CFG_HRREST is not None:
        return _CFG_HRREST
    from kairos.db import db_cursor
    try:
        with db_cursor(db_path) as cur:
            rows = cur.execute(
                "SELECT avg_hr FROM sessions WHERE avg_hr IS NOT NULL"
            ).fetchall()
        if rows:
            vals = [float(r["avg_hr"]) for r in rows]
            detected = float(np.percentile(vals, 1))
            if detected <= _HRREST_MAX_PLAUSIBLE:
                return detected
    except Exception:
        pass
    return _DEFAULT_HRREST


def compute_session_trimp(
    hr_records: list[float],
    hrmax: float,
    hrrest: float | None = None,
    method: str | None = None,
) -> tuple[float, str]:
    """
    Compute TRIMP for one session. Returns (trimp_value, load_source_label).
    method defaults to config.LOAD_METRIC when not specified.
    """
    if not hr_records:
        return 0.0, "te_proxy"
    if method is None:
        from kairos.config import LOAD_METRIC
        method = LOAD_METRIC
    if method == "banister_trimp":
        hr = hrrest if hrrest is not None else _DEFAULT_HRREST
        return banister_trimp(hr_records, hrmax, hr), "banister_trimp"
    return edwards_trimp(hr_records, hrmax), "edwards_trimp"
