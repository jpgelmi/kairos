"""Tests for DB schema and synthetic data seeding."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from kairos.db import db_cursor, drop_all, init_db
from kairos.tests.synthetic import seed_synthetic


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "sessions",
    "hrv_daily",
    "wellness_daily",
    "running_dynamics",
    "session_segments",
    "form_state",
    "race_results",
    "sync_state",
    "lactate_thresholds",  # anclas longitudinales episódicas (final.tex §4 y §5.2)
}


def test_all_tables_created(tmp_db: Path) -> None:
    conn = sqlite3.connect(tmp_db)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        if not row[0].startswith("sqlite_")
    }
    conn.close()
    assert EXPECTED_TABLES == tables


def test_hrv_primary_key_constraint(tmp_db: Path) -> None:
    """Duplicate date on hrv_daily should raise."""
    with db_cursor(tmp_db) as cur:
        cur.execute(
            "INSERT INTO hrv_daily (date, ln_rmssd, rmssd, artifact_pct) "
            "VALUES ('2024-01-01', 3.9, 49.0, 0.01)"
        )
    with pytest.raises(sqlite3.IntegrityError):
        with db_cursor(tmp_db) as cur:
            cur.execute(
                "INSERT INTO hrv_daily (date, ln_rmssd, rmssd, artifact_pct) "
                "VALUES ('2024-01-01', 3.8, 44.0, 0.01)"
            )


def test_drop_all(tmp_db: Path) -> None:
    drop_all(tmp_db)
    conn = sqlite3.connect(tmp_db)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        if not row[0].startswith("sqlite_")
    }
    conn.close()
    assert tables == set()


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def test_synthetic_seed_populates(tmp_path: Path) -> None:
    path = tmp_path / "synth.db"
    seed_synthetic(path, days=56)

    conn = sqlite3.connect(path)

    n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    n_hrv = conn.execute("SELECT COUNT(*) FROM hrv_daily").fetchone()[0]
    n_well = conn.execute("SELECT COUNT(*) FROM wellness_daily").fetchone()[0]
    conn.close()

    assert n_sessions > 30, "Expected ~40 sessions in 8 weeks"
    assert n_hrv == 56
    assert n_well == 56
