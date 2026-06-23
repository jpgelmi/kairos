"""Tests for garmin_sync — all API calls are mocked."""

from __future__ import annotations

import io
import math
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kairos.db import init_db
from kairos.ingest.garmin_sync import (
    _duration_to_likert,
    _extract_fit_from_zip,
    _score_to_likert,
    sync_activities,
    sync_hrv,
    sync_wellness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zip(fit_content: bytes = b"FIT") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("activity.fit", fit_content)
    return buf.getvalue()


def _mock_client(*, activities=None, hrv=None, sleep=None, stress=None):
    client = MagicMock()
    client.get_activities_by_date.return_value = activities or []
    client.get_hrv_data.return_value = hrv
    client.get_sleep_data.return_value = sleep
    client.get_stress_data.return_value = stress
    client.download_activity.return_value = _make_zip()
    return client


# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------

def test_score_to_likert_normal() -> None:
    assert _score_to_likert(0) == 1
    assert _score_to_likert(100) == 5
    assert _score_to_likert(50) == 3


def test_score_to_likert_inverted() -> None:
    # Low stress score (0) → best (5); high stress (100) → worst (1)
    assert _score_to_likert(0, invert=True) == 5
    assert _score_to_likert(100, invert=True) == 1


def test_score_to_likert_no_data() -> None:
    assert _score_to_likert(-1) == 3    # Garmin "no data" sentinel → neutral


def test_score_to_likert_realistic_sleep() -> None:
    # sleep score 82 → Likert 4: 1 + round(82/25) = 1 + 3 = 4
    assert _score_to_likert(82) == 4


def test_duration_to_likert_ranges() -> None:
    assert _duration_to_likert(None) == 3          # no data → neutral
    assert _duration_to_likert(-1) == 3             # sentinel → neutral
    assert _duration_to_likert(4 * 3600) == 1       # 4 h → very poor
    assert _duration_to_likert(6 * 3600) == 2       # 6 h → poor
    assert _duration_to_likert(7 * 3600) == 3       # 7 h → ok
    assert _duration_to_likert(8 * 3600) == 4       # 8 h → good
    assert _duration_to_likert(9 * 3600) == 5       # 9 h → excellent


def test_sync_wellness_duration_fallback(tmp_path: Path) -> None:
    """When sleepScores.overall is absent, fall back to sleepingSeconds."""
    db = tmp_path / "k.db"
    init_db(db)

    # No 'overall' key → triggers duration fallback (7 h = 25200 s → Likert 3)
    sleep_resp = {"dailySleepDTO": {"sleepingSeconds": 25200, "sleepScores": {}}}
    stress_resp = {"averageStressLevel": 20}
    client = _mock_client(sleep=sleep_resp, stress=stress_resp)

    count = sync_wellness(client, date(2024, 3, 1), date(2024, 3, 1), db)
    assert count == 1

    import sqlite3
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT sleep FROM wellness_daily WHERE date='2024-03-01'").fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 3   # 7 h → Likert 3


def test_extract_fit_from_zip() -> None:
    data = _make_zip(b"FITDATA")
    result = _extract_fit_from_zip(data)
    assert result == b"FITDATA"


def test_extract_fit_from_zip_no_fit_raises() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no fit here")
    with pytest.raises(ValueError, match="No .fit"):
        _extract_fit_from_zip(buf.getvalue())


# ---------------------------------------------------------------------------
# sync_activities
# ---------------------------------------------------------------------------

def test_sync_activities_inserts_new(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    fit_dir = tmp_path / "fit"
    init_db(db)

    activities = [
        {
            "activityId": "111",
            "startTimeLocal": "2024-03-01 08:00:00",
            "duration": 3600,
            "activityType": {"typeKey": "running"},
        }
    ]
    client = _mock_client(activities=activities)

    count = sync_activities(client, date(2024, 3, 1), date(2024, 3, 1), db, fit_dir)

    assert count == 1
    assert any(f.suffix == ".fit" for f in fit_dir.iterdir())


def test_sync_activities_skips_duplicates(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    fit_dir = tmp_path / "fit"
    init_db(db)

    activities = [
        {
            "activityId": "222",
            "startTimeLocal": "2024-03-01 08:00:00",
            "duration": 1800,
            "activityType": {"typeKey": "running"},
        }
    ]
    client = _mock_client(activities=activities)

    sync_activities(client, date(2024, 3, 1), date(2024, 3, 1), db, fit_dir)
    count2 = sync_activities(client, date(2024, 3, 1), date(2024, 3, 1), db, fit_dir)

    assert count2 == 0


def test_sync_activities_no_activities(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    fit_dir = tmp_path / "fit"
    init_db(db)

    client = _mock_client(activities=[])
    count = sync_activities(client, date(2024, 3, 1), date(2024, 3, 1), db, fit_dir)
    assert count == 0


# ---------------------------------------------------------------------------
# sync_hrv
# ---------------------------------------------------------------------------

def test_sync_hrv_inserts_garmin_sleep(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    init_db(db)

    hrv_response = {"hrvSummary": {"lastNight": 45, "weeklyAvg": 48}}
    client = _mock_client(hrv=hrv_response)

    count = sync_hrv(client, date(2024, 3, 1), date(2024, 3, 1), db)

    assert count == 1
    import sqlite3
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT ln_rmssd, source FROM hrv_daily WHERE date='2024-03-01'").fetchone()
    conn.close()
    assert row is not None
    assert abs(row[0] - math.log(45)) < 1e-9
    assert row[1] == "garmin_sleep_hrv"


def test_sync_hrv_skips_if_primary_source_exists(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    init_db(db)

    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO hrv_daily (date, ln_rmssd, rmssd, artifact_pct, condition_ok, source) "
        "VALUES ('2024-03-01', 3.9, 49.0, 0.01, 1, 'fit')"
    )
    conn.commit()
    conn.close()

    hrv_response = {"hrvSummary": {"lastNight": 55}}
    client = _mock_client(hrv=hrv_response)

    count = sync_hrv(client, date(2024, 3, 1), date(2024, 3, 1), db)
    assert count == 0   # primary 'fit' source protected


def test_sync_hrv_no_data(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    init_db(db)
    client = _mock_client(hrv=None)
    count = sync_hrv(client, date(2024, 3, 1), date(2024, 3, 1), db)
    assert count == 0


# ---------------------------------------------------------------------------
# sync_wellness
# ---------------------------------------------------------------------------

def test_sync_wellness_sleep_and_stress(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    init_db(db)

    sleep_resp = {"dailySleepDTO": {"sleepScores": {"overall": {"value": 75}}}}
    stress_resp = {"averageStressLevel": 30}   # low stress → good
    client = _mock_client(sleep=sleep_resp, stress=stress_resp)

    count = sync_wellness(client, date(2024, 3, 1), date(2024, 3, 1), db)
    assert count == 1

    import sqlite3
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT sleep, stress FROM wellness_daily WHERE date='2024-03-01'"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == _score_to_likert(75)
    assert row[1] == _score_to_likert(30, invert=True)


def test_sync_wellness_preserves_manual_entries(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    init_db(db)

    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO wellness_daily (date, sleep, soreness, stress, mood, motivation, wellness_raw) "
        "VALUES ('2024-03-01', 4, 3, 4, 4, 5, 20)"
    )
    conn.commit()
    conn.close()

    sleep_resp = {"dailySleepDTO": {"sleepScores": {"overall": {"value": 60}}}}
    stress_resp = {"averageStressLevel": 40}
    client = _mock_client(sleep=sleep_resp, stress=stress_resp)

    count = sync_wellness(client, date(2024, 3, 1), date(2024, 3, 1), db)
    assert count == 0   # fully populated row → skipped
