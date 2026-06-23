"""
Tests for ingest/quality.py import_report().
"""

from __future__ import annotations

from pathlib import Path

from kairos.db import init_db, db_cursor
from kairos.ingest.quality import import_report


def test_import_report_flags_session_outliers(tmp_path: Path) -> None:
    """import_report flags sessions with implausible RPE or duration."""
    db = tmp_path / "t.db"
    init_db(db)

    with db_cursor(db) as cur:
        # Bad RPE (outside 0-10)
        cur.execute(
            "INSERT INTO sessions (date, type, duration_min, rpe, srpe) VALUES (?,?,?,?,?)",
            ("2024-01-01", "easy", 60.0, 12.0, 720.0),
        )
        # Normal sessions
        for i in range(5):
            d = f"2024-01-{10 + i:02d}"
            cur.execute(
                "INSERT INTO sessions (date, type, duration_min, rpe, srpe) VALUES (?,?,?,?,?)",
                (d, "easy", 60.0, 5.0, 300.0),
            )

    report = import_report(db_path=db)

    assert len(report["session_outliers"]) >= 1
    assert any(o["date"] == "2024-01-01" for o in report["session_outliers"])
    assert "hrv_coverage" in report
    assert report["total_issues"] >= 1


def test_import_report_flags_session_gaps(tmp_path: Path) -> None:
    """import_report flags training gaps > 14 days."""
    db = tmp_path / "t.db"
    init_db(db)

    with db_cursor(db) as cur:
        for d in ["2024-01-01", "2024-01-02", "2024-02-01"]:  # gap > 14 days
            cur.execute(
                "INSERT INTO sessions (date, type, duration_min) VALUES (?,?,?)",
                (d, "easy", 60.0),
            )

    report = import_report(db_path=db)

    assert len(report["session_gaps"]) >= 1
    assert report["session_gaps"][0]["days"] > 14


def test_import_report_clean_db(tmp_path: Path) -> None:
    """An empty database produces zero issues."""
    db = tmp_path / "t.db"
    init_db(db)
    report = import_report(db_path=db)
    assert report["total_issues"] == 0
    assert report["hrv_coverage"]["total_hrv_days"] == 0
