"""
HRR (Heart Rate Recovery) and τ_HRR from work→recovery transitions.

HRR60: HR drop in the first 60 seconds after a work bout ends.
τ_HRR: exponential decay time constant fitted from the recovery HR curve.
       Robust to variable pause lengths — useful when intervals end at different
       durations (e.g. 200 m, 400 m, 800 m reps each followed by different rest).

Only transitions where the work bout peaked at ≥ HRR_MIN_PEAK_PCT × HRmax
contribute valid measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kairos.config import HRR_MIN_PEAK_PCT


@dataclass
class HRRResult:
    hrr60: float | None     # HR drop at 60 s post-work (bpm); None if pause < 60 s
    tau_s: float | None     # exponential decay τ (seconds)
    hr_peak: float          # HR at end of work bout
    hr_floor: float | None  # asymptotic floor during recovery


def _fit_tau(
    t_arr: np.ndarray,
    hr_arr: np.ndarray,
    hr_floor: float,
) -> float | None:
    """
    Fit HR(t) = amplitude × exp(−t/τ) + floor.

    Primary: 3-parameter nonlinear fit (scipy.optimize.curve_fit) — robust to
    incomplete recovery (pause shorter than ~2 × τ) because the floor is
    estimated simultaneously.
    Fallback: log-linear with the supplied hr_floor.
    """
    from scipy.optimize import curve_fit

    n = len(t_arr)
    if n < 5:
        return None

    def _exp3(t: np.ndarray, amp: float, tau: float, floor: float) -> np.ndarray:
        return amp * np.exp(-t / tau) + floor

    amp0 = max(1.0, float(hr_arr[0] - hr_arr[-1]))
    tau0 = max(5.0, float(t_arr[-1] / 3.0))
    floor0 = max(0.0, float(hr_arr[-1]))

    try:
        popt, _ = curve_fit(
            _exp3, t_arr, hr_arr,
            p0=[amp0, tau0, floor0],
            bounds=([0, 1, 0], [500, 600, float(hr_arr.max())]),
            maxfev=2000,
        )
        tau_est = float(popt[1])
        return tau_est if tau_est > 0 else None
    except Exception:
        pass

    # Fallback: log-linear with provided floor
    y = hr_arr - hr_floor
    valid = y > 0
    if valid.sum() < 3:
        return None
    try:
        coeffs = np.polyfit(t_arr[valid], np.log(y[valid]), 1)
        slope = coeffs[0]
        return float(-1.0 / slope) if slope < 0 else None
    except Exception:
        return None


def compute_hrr_transition(
    records: list[dict[str, Any]],
    work_end_t: float,
    recovery_end_t: float,
    hrmax: float | None = None,
    min_peak_pct: float = HRR_MIN_PEAK_PCT,
) -> HRRResult | None:
    """
    Compute HRR for one work→recovery transition.

    work_end_t: t_s of the last record in the work segment.
    recovery_end_t: t_s of the last record in the recovery segment.
    Returns None if the work bout didn't reach min_peak_pct × HRmax.
    """
    # Peak HR: max HR in the last 10 s of the work bout
    work_hrs = [
        float(r["heart_rate"]) for r in records
        if r.get("heart_rate") is not None
        and r.get("t_s") is not None
        and work_end_t - 10.0 <= float(r["t_s"]) <= work_end_t
    ]
    if not work_hrs:
        return None
    hr_peak = max(work_hrs)

    if hrmax is not None and hr_peak < hrmax * min_peak_pct:
        return None

    # Recovery HR series (time zero = work_end_t)
    rec_pairs = [
        (float(r["t_s"]) - work_end_t, float(r["heart_rate"]))
        for r in records
        if r.get("heart_rate") is not None
        and r.get("t_s") is not None
        and work_end_t < float(r["t_s"]) <= recovery_end_t
    ]
    if len(rec_pairs) < 5:
        return None

    t_arr = np.array([p[0] for p in rec_pairs])
    hr_arr = np.array([p[1] for p in rec_pairs])
    hr_floor = float(hr_arr.min())

    # HRR60: HR drop at t = 60 s
    hrr60: float | None = None
    if t_arr[-1] >= 60.0:
        idx = int(np.searchsorted(t_arr, 60.0))
        hr_at_60 = float(hr_arr[min(idx, len(hr_arr) - 1)])
        hrr60 = hr_peak - hr_at_60

    tau = _fit_tau(t_arr, hr_arr, hr_floor)

    return HRRResult(hrr60=hrr60, tau_s=tau, hr_peak=hr_peak, hr_floor=hr_floor)


def session_hrr(
    records: list[dict[str, Any]],
    segments: list,
    hrmax: float | None = None,
) -> list[HRRResult]:
    """
    Extract HRR at every work→recovery transition in a session.
    segments must be sorted by idx (as returned by segmentation.py).
    """
    results: list[HRRResult] = []
    for i, seg in enumerate(segments):
        if seg.kind != "work":
            continue
        next_rec = None
        for j in range(i + 1, len(segments)):
            if segments[j].kind in ("recovery", "rest"):
                next_rec = segments[j]
                break
        if next_rec is None:
            continue
        r = compute_hrr_transition(
            records,
            work_end_t=seg.end_s,
            recovery_end_t=next_rec.end_s,
            hrmax=hrmax,
        )
        if r is not None:
            results.append(r)
    return results
