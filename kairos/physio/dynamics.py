"""
Phase 2 — Running dynamics from Garmin sensor data.

Critical constraint (spec §3.4): use ONLY intra-subject trends, never absolute values.
GCT drift = (GCT_final − GCT_initial) / GCT_initial  — positive = fatigue signal.
A high GCT drift indicates biomechanical fatigue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kairos.physio.segmentation import SessionSegment


def compute_gct_drift(gct_series: list[float]) -> float | None:
    """
    Intra-session GCT drift = (GCT_final − GCT_initial) / GCT_initial.
    Uses median of first and last 20 % of the series to reduce noise.
    Returns None if fewer than 4 points.
    """
    if len(gct_series) < 4:
        return None
    arr = np.asarray(gct_series, dtype=float)
    n = len(arr)
    seg = max(1, n // 5)
    gct_init = float(np.median(arr[:seg]))
    gct_final = float(np.median(arr[-seg:]))
    if gct_init == 0:
        return None
    return (gct_final - gct_init) / gct_init



def gct_drift_from_segments(segments: list[SessionSegment]) -> float | None:
    """
    Compute GCT drift from work segments only (interval sessions).

    Priority grouping:
    1. Same-distance reps (±5 %): most reliable for variable-structure workouts.
    2. Same-pace reps (±COMPARABLE_PACE_TOL): fallback when distance is unavailable.
    3. All work segments: last resort.

    Returns (GCT_last_rep − GCT_first_rep) / GCT_first_rep, or None.
    """
    from kairos.config import COMPARABLE_PACE_TOL

    work = [s for s in segments if s.kind == "work" and s.avg_gct_ms is not None]
    if len(work) < 3:
        return None

    # 1. Group by same distance (±5 %)
    dists = [s.distance_m for s in work if s.distance_m is not None]
    if dists:
        dist_med = float(np.median(dists))
        if dist_med > 0:
            same_dist = [
                s for s in work
                if s.distance_m is not None
                and abs(s.distance_m - dist_med) / dist_med <= 0.05
            ]
            if len(same_dist) >= 3:
                work = same_dist

    # 2. Fall back: group by comparable pace
    if len(work) < 3:
        speeds = [s.avg_speed_ms for s in work if s.avg_speed_ms is not None]
        if speeds:
            med = float(np.median(speeds))
            if med > 0:
                comparable = [
                    s for s in work
                    if s.avg_speed_ms is not None
                    and abs(s.avg_speed_ms - med) / med <= COMPARABLE_PACE_TOL
                ]
                if len(comparable) >= 3:
                    work = comparable

    if len(work) < 3:
        return None

    gcts = [s.avg_gct_ms for s in work]  # type: ignore[union-attr]
    return (gcts[-1] - gcts[0]) / gcts[0]
