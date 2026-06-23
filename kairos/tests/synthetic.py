"""
Generate deterministic synthetic data for testing.
Produces ~8 weeks of data so baseline windows are satisfied.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

from kairos.db import db_cursor, init_db


def _date_range(start: date, days: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(days)]


def seed_synthetic(db_path: Path, *, seed: int = 42, days: int = 56) -> None:
    """Populate db_path with 8 weeks of synthetic data."""
    init_db(db_path)
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    dates = _date_range(start, days)

    with db_cursor(db_path) as cur:
        # -----------------------------------------------------------------
        # Sessions (5 per week, 2 rest days)
        # -----------------------------------------------------------------
        session_ids: dict[date, int] = {}
        for i, d in enumerate(dates):
            if d.weekday() in (5, 6):   # Sat/Sun = rest
                continue
            rpe = rng.uniform(4.0, 8.0)
            dur = rng.uniform(40.0, 90.0)
            srpe = rpe * dur
            cur.execute(
                "INSERT INTO sessions (date, type, duration_min, rpe, srpe) "
                "VALUES (?, ?, ?, ?, ?)",
                (d.isoformat(), "easy", dur, rpe, srpe),
            )
            session_ids[d] = cur.lastrowid

        # -----------------------------------------------------------------
        # HRV daily (all days, slight upward trend over 8 weeks)
        # -----------------------------------------------------------------
        ln_base = 3.9   # ~lnRMSSD ≈ 49 ms baseline
        for i, d in enumerate(dates):
            trend = 0.002 * i        # gentle fitness improvement
            noise = rng.gauss(0, 0.08)
            ln_val = ln_base + trend + noise
            rmssd = math.exp(ln_val)
            artifact = rng.uniform(0.005, 0.025)
            cur.execute(
                "INSERT OR REPLACE INTO hrv_daily "
                "(date, ln_rmssd, rmssd, artifact_pct, condition_ok, source) "
                "VALUES (?, ?, ?, ?, 1, 'synthetic')",
                (d.isoformat(), ln_val, rmssd, artifact),
            )

        # -----------------------------------------------------------------
        # Wellness (all days)
        # -----------------------------------------------------------------
        for d in dates:
            items = [rng.randint(3, 5) for _ in range(5)]
            cur.execute(
                "INSERT OR REPLACE INTO wellness_daily "
                "(date, sleep, soreness, stress, mood, motivation, wellness_raw) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (d.isoformat(), *items, sum(items)),
            )

        # -----------------------------------------------------------------
        # Running dynamics (for session days only)
        # -----------------------------------------------------------------
        for d, sid in session_ids.items():
            gct = rng.uniform(220, 260)
            drift = rng.uniform(0.0, 0.06)
            cur.execute(
                "INSERT OR REPLACE INTO running_dynamics "
                "(session_id, gct_mean_ms, gct_drift_pct) "
                "VALUES (?,?,?)",
                (sid, gct, drift),
            )
