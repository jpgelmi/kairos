"""
Tests for rebuild_form and date_spine (historical backfill feature).

Verifies:
  - date_spine produces a gap-free list
  - EWMA walk from HISTORY_START matches manual calculation
  - Rest days (no session) contribute zero load (EWMA decays, not skips)
  - 21-day zero-load gap → h decays by exp(-21/7)
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pytest

from kairos.db import init_db, db_cursor
from kairos.model.fitness_fatigue import (
    date_spine,
    ewma_step,
    rebuild_form,
)
from kairos.config import HISTORY_START


# ---------------------------------------------------------------------------

def test_date_spine_continuous() -> None:
    """date_spine returns every day from start to end with no gaps."""
    spine = date_spine(date(2024, 1, 1), date(2024, 1, 31))
    assert len(spine) == 31
    for i in range(1, len(spine)):
        assert (spine[i] - spine[i - 1]).days == 1


def test_date_spine_single_day() -> None:
    spine = date_spine(date(2024, 6, 9), date(2024, 6, 9))
    assert spine == [date(2024, 6, 9)]


def test_gap_decay() -> None:
    """After 21 days of zero load, h should equal h0 * exp(-21/7)."""
    h0 = 200.0
    g, h = 0.0, h0
    for _ in range(21):
        g, h = ewma_step(g, h, 0.0)
    expected = h0 * math.exp(-21.0 / 7.0)
    assert abs(h - expected) < 1e-6


def test_load_zero_fill_rest_days(tmp_path: Path) -> None:
    """Days with no session contribute zero load; EWMA decays naturally."""
    db = tmp_path / "t.db"
    init_db(db)

    with db_cursor(db) as cur:
        cur.execute(
            "INSERT INTO sessions (date, type, duration_min, srpe) VALUES (?,?,?,?)",
            ("2024-01-01", "easy", 60.0, 300.0),
        )

    rebuild_form(start_date=date(2024, 1, 1), end_date=date(2024, 1, 8), db_path=db)

    # Manually walk: day 1 = 300 load, days 2–8 = 0
    g, h = 0.0, 0.0
    g, h = ewma_step(g, h, 300.0)
    for _ in range(7):
        g, h = ewma_step(g, h, 0.0)

    with db_cursor(db) as cur:
        row = cur.execute(
            "SELECT g, h FROM form_state WHERE date='2024-01-08'"
        ).fetchone()

    assert row is not None
    assert abs(row["h"] - h) < 1e-6
    assert abs(row["g"] - g) < 1e-6


def test_ewma_warmup_from_history(tmp_path: Path) -> None:
    """g/h at target date from rebuild_form matches manual EWMA walk from start."""
    db = tmp_path / "t.db"
    init_db(db)

    sessions = [
        ("2023-06-01", 300.0),
        ("2023-09-15", 250.0),
        ("2024-01-20", 200.0),
    ]
    with db_cursor(db) as cur:
        for d, srpe in sessions:
            cur.execute(
                "INSERT INTO sessions (date, type, duration_min, srpe) VALUES (?,?,?,?)",
                (d, "easy", 60.0, srpe),
            )

    target = date(2024, 3, 1)
    start = date.fromisoformat(HISTORY_START)
    rebuild_form(start_date=start, end_date=target, db_path=db)

    # Manual EWMA walk
    load_dict = {d: srpe for d, srpe in sessions}
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


