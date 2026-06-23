"""
Data quality checks for Kairós.

Detects: bad session fields, session gaps, and HRV coverage statistics.
Called by the import-report CLI command.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from kairos.config import DB_PATH
from kairos.db import db_cursor

_SESSION_GAP_DAYS: int = 14


def _detect_date_gaps(dates: list[str], threshold_days: int) -> list[dict]:
    """Return gaps > threshold_days in a sorted list of ISO date strings."""
    gaps = []
    for i in range(1, len(dates)):
        d1 = date.fromisoformat(dates[i - 1])
        d2 = date.fromisoformat(dates[i])
        gap = (d2 - d1).days
        if gap > threshold_days:
            gaps.append({"start": dates[i - 1], "end": dates[i], "days": gap})
    return gaps


def import_report(db_path: Path = DB_PATH) -> dict:
    """
    Run quality checks on the database.

    Returns a dict with keys:
      session_outliers  — list of sessions with bad RPE/sRPE/duration
      session_gaps      — list of gaps in session dates
      hrv_coverage      — summary counts (HRV rows from Garmin sync)
      total_issues      — sum of all flagged items
    """
    with db_cursor(db_path) as cur:
        hrv_rows = cur.execute(
            "SELECT date, source FROM hrv_daily ORDER BY date"
        ).fetchall()
        sess_rows = cur.execute(
            "SELECT date, type, srpe, rpe, duration_min "
            "FROM sessions ORDER BY date"
        ).fetchall()

    # ---- Session outliers ----------------------------------------------------
    session_outliers: list[dict] = []
    for r in sess_rows:
        issues: list[str] = []
        rpe = r["rpe"]
        srpe = r["srpe"]
        dur = r["duration_min"]
        if rpe is not None and not (0.0 <= rpe <= 10.0):
            issues.append(f"RPE {rpe:.1f} outside 0–10")
        if srpe is not None and srpe < 0:
            issues.append(f"sRPE {srpe:.1f} < 0")
        if dur is not None and dur > 600:
            issues.append(f"duration {dur:.0f} min > 10 h")
        if issues:
            session_outliers.append({"date": r["date"], "issues": issues})

    # ---- Session gaps --------------------------------------------------------
    session_dates = sorted(set(r["date"] for r in sess_rows))
    session_gaps = _detect_date_gaps(session_dates, _SESSION_GAP_DAYS)

    # ---- HRV coverage (informational — Garmin sync still writes hrv_daily) ---
    fit_days = sum(1 for r in hrv_rows if r["source"] == "fit")
    watch_days = sum(1 for r in hrv_rows if r["source"] == "garmin_sleep_hrv")

    total_issues = len(session_outliers) + len(session_gaps)

    return {
        "session_outliers": session_outliers,
        "session_gaps": session_gaps,
        "hrv_coverage": {
            "fit_days": fit_days,
            "watch_days": watch_days,
            "total_hrv_days": len(hrv_rows),
            "total_session_days": len(session_dates),
            "total_sessions": len(sess_rows),
        },
        "total_issues": total_issues,
    }
