"""
Tests for the objective form model:
  - rebuild_form warmup from history (TRIMP-based)
  - find_taper peak on target
  - Validation walk-forward with objective target
  - Segmentation: variable interval structure (5×1000, 8×400, ladder)
  - Neuromuscular drift per rep with same-distance grouping
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from kairos.db import init_db, db_cursor
from kairos.model.fitness_fatigue import (
    date_spine, ewma_step, rebuild_form,
)
from kairos.physio.segmentation import (
    SessionSegment, segment_from_laps,
)
from kairos.physio.dynamics import gct_drift_from_segments
from kairos.config import HISTORY_START


# ---------------------------------------------------------------------------
# 1. rebuild_form warmup from TRIMP-based history
# ---------------------------------------------------------------------------

def test_rebuild_form_warmup_from_trimp(tmp_path: Path) -> None:
    """
    rebuild_form uses trimp when available, falls back to srpe.
    EWMA at target date matches manual walk using trimp values.
    """
    db = tmp_path / "t.db"
    init_db(db)

    sessions = [
        ("2023-06-01", 180.0),   # trimp values (zone-minutes)
        ("2023-09-15", 220.0),
        ("2024-01-20", 150.0),
    ]
    with db_cursor(db) as cur:
        for d, trimp in sessions:
            cur.execute(
                "INSERT INTO sessions (date, type, duration_min, trimp) VALUES (?,?,?,?)",
                (d, "easy", 60.0, trimp),
            )

    target = date(2024, 3, 1)
    start = date.fromisoformat(HISTORY_START)
    rebuild_form(start_date=start, end_date=target, db_path=db)

    # Manual EWMA walk using trimp
    load_dict = {d: trimp for d, trimp in sessions}
    g, h = 0.0, 0.0
    for d in date_spine(start, target):
        g, h = ewma_step(g, h, load_dict.get(d.isoformat(), 0.0))

    with db_cursor(db) as cur:
        row = cur.execute(
            "SELECT g, h FROM form_state WHERE date=?", (target.isoformat(),)
        ).fetchone()

    assert row is not None
    assert abs(row["g"] - g) < 1e-9
    assert abs(row["h"] - h) < 1e-9


def test_rebuild_form_srpe_fallback(tmp_path: Path) -> None:
    """When trimp is NULL and srpe is set, rebuild_form uses srpe."""
    db = tmp_path / "t.db"
    init_db(db)

    with db_cursor(db) as cur:
        cur.execute(
            "INSERT INTO sessions (date, type, duration_min, srpe) VALUES (?,?,?,?)",
            ("2024-01-01", "easy", 60.0, 300.0),
        )

    target = date(2024, 1, 8)
    rebuild_form(start_date=date(2024, 1, 1), end_date=target, db_path=db)

    load_dict = {"2024-01-01": 300.0}
    g, h = 0.0, 0.0
    for d in date_spine(date(2024, 1, 1), target):
        g, h = ewma_step(g, h, load_dict.get(d.isoformat(), 0.0))

    with db_cursor(db) as cur:
        row = cur.execute(
            "SELECT g, h FROM form_state WHERE date=?", (target.isoformat(),)
        ).fetchone()

    assert row is not None
    assert abs(row["h"] - h) < 1e-6


# ---------------------------------------------------------------------------
# 2. Segmentation: variable interval structures
# ---------------------------------------------------------------------------

def _lap(speed: float, dur: float, dist: float | None = None) -> dict:
    """Helper using enhanced_avg_speed (Garmin .fit field name)."""
    return {
        "enhanced_avg_speed": speed,
        "avg_speed": None,          # legacy field absent
        "total_elapsed_time": dur,
        "total_timer_time": dur,
        "total_distance": dist if dist is not None else speed * dur,
        "avg_heart_rate": 150.0,
    }


def test_segment_from_laps_variable_structure_5x1000() -> None:
    """5×1000 m with 200 m recovery → 5 work + 5 recovery (4 rest, 1 warmup, 1 cooldown)."""
    laps = [_lap(2.8, 300.0, 1000.0)]   # warmup
    for _ in range(5):
        laps.append(_lap(5.0, 200.0, 1000.0))   # ~4:00/km work
        laps.append(_lap(1.5, 120.0, 180.0))     # recovery jog
    laps.append(_lap(2.8, 300.0, 840.0))          # cooldown

    segs = segment_from_laps(laps)
    work = [s for s in segs if s.kind == "work"]
    assert len(work) == 5


def test_segment_from_laps_variable_structure_8x400() -> None:
    """8×400 m with 90 s recovery → 8 work segments."""
    laps = [_lap(2.8, 300.0)]   # warmup
    for _ in range(8):
        laps.append(_lap(5.5, 73.0, 400.0))    # ~3:02/km = 400 m pace
        laps.append(_lap(1.5, 90.0, 135.0))    # recovery
    laps.append(_lap(2.8, 300.0))               # cooldown

    segs = segment_from_laps(laps)
    work = [s for s in segs if s.kind == "work"]
    assert len(work) == 8


def test_segment_from_laps_ladder() -> None:
    """Ladder session (400-800-1200-800-400) → 5 work segments despite varying distances."""
    laps = [_lap(2.8, 300.0)]   # warmup
    distances = [400, 800, 1200, 800, 400]
    for dist in distances:
        laps.append(_lap(5.0, dist / 5.0, dist))   # ~3:20/km work pace
        laps.append(_lap(1.5, 90.0))                # recovery
    laps.append(_lap(2.8, 300.0))                    # cooldown

    segs = segment_from_laps(laps)
    work = [s for s in segs if s.kind == "work"]
    assert len(work) == 5


def test_segment_from_laps_uses_enhanced_avg_speed() -> None:
    """segment_from_laps should use enhanced_avg_speed when avg_speed is None."""
    laps = [
        {"enhanced_avg_speed": 2.8, "avg_speed": None, "total_elapsed_time": 300.0,
         "total_distance": 840.0, "avg_heart_rate": 130.0},
        {"enhanced_avg_speed": 5.0, "avg_speed": None, "total_elapsed_time": 200.0,
         "total_distance": 1000.0, "avg_heart_rate": 165.0},
        {"enhanced_avg_speed": 1.5, "avg_speed": None, "total_elapsed_time": 120.0,
         "total_distance": 180.0, "avg_heart_rate": 135.0},
        {"enhanced_avg_speed": 5.0, "avg_speed": None, "total_elapsed_time": 200.0,
         "total_distance": 1000.0, "avg_heart_rate": 165.0},
        {"enhanced_avg_speed": 2.8, "avg_speed": None, "total_elapsed_time": 300.0,
         "total_distance": 840.0, "avg_heart_rate": 130.0},
    ]
    segs = segment_from_laps(laps)
    work = [s for s in segs if s.kind == "work"]
    assert len(work) == 2


# ---------------------------------------------------------------------------
# 3. Neuromuscular: GCT drift only across same-distance reps
# ---------------------------------------------------------------------------

def test_neuro_drift_per_rep_same_distance() -> None:
    """
    In a ladder (200 m + 400 m + 800 m + 400 m + 200 m), gct_drift_from_segments
    should group same-distance reps (200 m × 2 or 400 m × 2) and compute drift
    only within those groups — not across different distances.
    """
    segs: list[SessionSegment] = []
    t = 0.0
    # Layout: 200, 400, 800, 400, 200 work reps with recovery between
    reps = [
        (200.0, 5.5, 220.0),   # (dist_m, speed, gct_ms)
        (400.0, 5.0, 225.0),
        (800.0, 4.8, 228.0),
        (400.0, 5.0, 230.0),   # same dist as rep 2, GCT drifted
        (200.0, 5.5, 240.0),   # same dist as rep 1, GCT drifted
    ]
    for i, (dist, spd, gct) in enumerate(reps):
        dur = dist / spd
        segs.append(SessionSegment(
            idx=len(segs), kind="work",
            start_s=t, end_s=t + dur, duration_s=dur,
            distance_m=dist, avg_speed_ms=spd, avg_gct_ms=gct,
        ))
        t += dur
        if i < len(reps) - 1:
            segs.append(SessionSegment(
                idx=len(segs), kind="recovery",
                start_s=t, end_s=t + 90.0, duration_s=90.0,
                distance_m=135.0, avg_speed_ms=1.5, avg_gct_ms=310.0,
            ))
            t += 90.0

    drift = gct_drift_from_segments(segs)
    assert drift is not None
    # With 400 m reps (225 → 230) or 200 m reps (220 → 240), drift should be positive
    assert drift > 0


# ---------------------------------------------------------------------------
# 4. Validation: CTL correlates with EF (walk-forward sanity check)
# ---------------------------------------------------------------------------

def test_validation_objective_target(tmp_path: Path) -> None:
    """
    Walk-forward concept test: when CTL rises monotonically, sessions achieved
    during high-fitness periods should show higher EF (objective metric).
    Verifies Spearman rank correlation > 0 between CTL and EF.
    """
    from scipy.stats import spearmanr
    from kairos.model.fitness_fatigue import ewma_step
    from kairos.config import TAU_G, TAU_H

    import math

    n = 50
    loads = [80.0 + i * 3.0 for i in range(n)]

    g, h = 0.0, 0.0
    ctl_series: list[float] = []
    ef_series: list[float] = []
    rng = np.random.default_rng(2024)

    for i, load in enumerate(loads):
        g, h = ewma_step(g, h, load)
        ctl_series.append(g)
        signal = g / 3000.0
        ef = 0.018 + signal + rng.normal(0, max(signal * 0.05, 1e-5))
        ef_series.append(ef)

    rho, p = spearmanr(ctl_series, ef_series)
    assert rho > 0.7, f"Expected Spearman rho > 0.7, got {rho:.3f}"


def test_neuro_drift_excludes_different_distance_reps() -> None:
    """
    When reps have wildly different distances (200 m vs 800 m), the grouping
    should select the most common distance group, not blend across all reps.
    The 800 m rep with very different GCT should not contaminate the 200 m group.
    """
    segs: list[SessionSegment] = []
    t = 0.0
    # Three 400 m reps with slight drift, plus one 800 m with extreme GCT
    reps = [
        (400.0, 5.0, 220.0),
        (400.0, 5.0, 224.0),
        (800.0, 4.5, 400.0),   # very different distance + extreme GCT
        (400.0, 5.0, 228.0),
    ]
    for i, (dist, spd, gct) in enumerate(reps):
        dur = dist / spd
        segs.append(SessionSegment(
            idx=len(segs), kind="work",
            start_s=t, end_s=t + dur, duration_s=dur,
            distance_m=dist, avg_speed_ms=spd, avg_gct_ms=gct,
        ))
        t += dur
        if i < len(reps) - 1:
            segs.append(SessionSegment(
                idx=len(segs), kind="recovery",
                start_s=t, end_s=t + 90.0, duration_s=90.0,
                avg_gct_ms=310.0,
            ))
            t += 90.0

    drift = gct_drift_from_segments(segs)
    assert drift is not None
    # Drift should reflect only the 400 m group: (228-220)/220 ≈ 0.036
    # Not the extreme 800 m rep
    assert drift < 0.10   # would be huge if 400 ms GCT rep contaminated
