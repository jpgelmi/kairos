"""
MMP (Mean Maximal Pace) curve and Critical Speed from field data.

For each target duration d, find the maximum average speed over any continuous
d-second window across all sessions. The resulting MMP history is used to fit a
hyperbolic CS / D' model:

    distance(d) = CS × d + D'   (linear regression on best d × v(d) vs d)

This is a FLOOR estimate — actual laboratory CS tests override it.
"""

from __future__ import annotations

import numpy as np

# Target durations for the MMP curve (seconds)
MMP_DURATIONS_S: tuple[int, ...] = (120, 180, 300, 600, 1200)


def session_mmp(
    records: list[dict],
    durations_s: tuple[int, ...] = MMP_DURATIONS_S,
) -> dict[int, float | None]:
    """
    Best (max) average speed for each target duration in one session.
    Records are treated as 1-second samples (use enhanced_speed or speed).
    Returns {duration_s: best_avg_speed_ms | None}.
    """
    speeds: list[float] = []
    for r in records:
        spd = r.get("enhanced_speed") or r.get("speed")
        if spd is not None:
            speeds.append(float(spd))

    if not speeds:
        return {d: None for d in durations_s}

    spd_arr = np.array(speeds, dtype=float)
    result: dict[int, float | None] = {}

    # Cumulative sum for O(1) rolling mean
    cs = np.cumsum(spd_arr)
    cs = np.insert(cs, 0, 0.0)

    for d in durations_s:
        if len(spd_arr) < d:
            result[d] = None
            continue
        rolling = (cs[d:] - cs[:-d]) / d
        result[d] = float(rolling.max())

    return result


def build_mmp_history(
    session_mmps: list[dict[int, float | None]],
    durations_s: tuple[int, ...] = MMP_DURATIONS_S,
) -> dict[int, float]:
    """
    Merge per-session MMP dicts: keep the all-time best at each duration.
    Returns {duration_s: best_speed_ms}.
    """
    best: dict[int, float] = {}
    for mmp in session_mmps:
        for d in durations_s:
            v = mmp.get(d)
            if v is not None:
                if d not in best or v > best[d]:
                    best[d] = v
    return best


def cs_from_mmp(
    mmp_history: dict[int, float],
    min_points: int = 3,
) -> dict | None:
    """
    Fit CS and D' from MMP curve via linear regression on distance = CS × d + D'.

    Returns dict with cs_ms, d_prime_m, r_squared, cs_source='training_mmp', or None.
    This is a FLOOR estimate — laboratory tests are authoritative.
    """
    pairs = [(d, v) for d, v in mmp_history.items() if v is not None and d > 0]
    if len(pairs) < min_points:
        return None

    t_arr = np.array([d for d, _ in pairs], dtype=float)
    dist_arr = np.array([v * d for d, v in pairs], dtype=float)

    # Linear: dist = CS × t + D'
    A = np.column_stack([t_arr, np.ones_like(t_arr)])
    coeffs, *_ = np.linalg.lstsq(A, dist_arr, rcond=None)
    cs, d_prime = float(coeffs[0]), float(coeffs[1])

    if cs <= 0:
        return None

    dist_pred = cs * t_arr + d_prime
    ss_res = float(np.sum((dist_arr - dist_pred) ** 2))
    ss_tot = float(np.sum((dist_arr - dist_arr.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "cs_ms": cs,
        "d_prime_m": max(0.0, d_prime),
        "r_squared": r_sq,
        "cs_source": "training_mmp",
    }
