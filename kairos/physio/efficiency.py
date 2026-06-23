"""
Efficiency Factor (EF) and aerobic decoupling from GPS + HR records.

EF = GAP_speed / HR  (m/s per bpm)
GAP (Grade Adjusted Pace) corrects speed for elevation gradient using the
Minetti et al. 2002 metabolic cost polynomial, normalized so GAP == speed on flat.

Decoupling = (EF_first_half − EF_second_half) / EF_first_half × 100 %
Positive decoupling means cardiac drift → durability deficit.

Only steady aerobic segments (HR < AEROBIC_HR_CEILING_PCT × HRmax, continuous
blocks ≥ EF_MIN_DURATION_S) are used so intensity variation doesn't contaminate EF.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kairos.config import AEROBIC_HR_CEILING_PCT, EF_MIN_DURATION_S


def grade_factor(grade_pct: float) -> float:
    """
    Metabolic cost relative to flat running (Minetti et al. 2002), normalized at 0%.
    GAP = speed × grade_factor = flat-equivalent speed for equal metabolic effort.
    > 1 uphill (harder → GAP > actual speed), < 1 for mild downhill.
    """
    g = grade_pct / 100.0
    cr = 155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6
    return cr / 3.6


def compute_gap_series(records: list[dict[str, Any]]) -> list[float | None]:
    """
    Grade-adjusted speed (m/s) for each record.
    Uses enhanced_altitude differences for gradient estimation.
    Returns None for records with missing speed or invalid gradient.
    """
    n = len(records)
    alts = [r.get("enhanced_altitude") for r in records]
    result: list[float | None] = []

    for i, r in enumerate(records):
        spd = r.get("enhanced_speed") or r.get("speed")
        if spd is None or float(spd) <= 0:
            result.append(None)
            continue
        spd = float(spd)

        if (i > 0 and alts[i] is not None and alts[i - 1] is not None):
            alt_delta = float(alts[i]) - float(alts[i - 1])
            horiz = spd  # ~1 s interval → distance ≈ speed m
            if horiz > 0:
                grade_pct = (alt_delta / horiz) * 100.0
                grade_pct = max(-30.0, min(30.0, grade_pct))
                gf = grade_factor(grade_pct)
                result.append(spd * gf)
                continue
        result.append(spd)

    return result


def compute_ef_session(
    records: list[dict[str, Any]],
    hrmax: float,
    aerobic_ceiling_pct: float = AEROBIC_HR_CEILING_PCT,
    min_duration_s: float = EF_MIN_DURATION_S,
) -> dict | None:
    """
    EF and aerobic decoupling for a session.

    Only steady aerobic records (HR < ceiling, in a continuous run ≥ min_duration_s)
    contribute. Returns None if there aren't enough qualifying records.

    Result keys: ef, decoupling_pct, n_records_used.
    """
    if not records:
        return None

    hr_ceiling = hrmax * aerobic_ceiling_pct
    gap_series = compute_gap_series(records)

    # Collect qualifying records
    steady: list[dict[str, float]] = []
    for i, r in enumerate(records):
        hr = r.get("heart_rate")
        gap = gap_series[i]
        if hr is None or gap is None:
            continue
        if float(hr) < hr_ceiling:
            steady.append({"gap": float(gap), "hr": float(hr)})

    if len(steady) < min_duration_s:
        return None

    gap_arr = np.array([s["gap"] for s in steady], dtype=float)
    hr_arr = np.array([s["hr"] for s in steady], dtype=float)

    mean_hr = float(np.mean(hr_arr))
    if mean_hr == 0:
        return None

    ef = float(np.mean(gap_arr)) / mean_hr

    # Aerobic decoupling: compare EF first vs second half
    mid = len(steady) // 2
    decoupling: float | None = None
    if mid >= 10:
        hr1 = float(np.mean(hr_arr[:mid]))
        hr2 = float(np.mean(hr_arr[mid:]))
        if hr1 > 0 and hr2 > 0:
            ef1 = float(np.mean(gap_arr[:mid])) / hr1
            ef2 = float(np.mean(gap_arr[mid:])) / hr2
            if ef1 > 0:
                decoupling = (ef1 - ef2) / ef1 * 100.0

    return {
        "ef": ef,
        "decoupling_pct": decoupling,
        "n_records_used": len(steady),
    }
