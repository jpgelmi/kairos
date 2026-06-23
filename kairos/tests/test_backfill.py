"""
Tests for backfill resumability and sync_state persistence.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from kairos.db import init_db
from kairos.ingest.garmin_sync import get_sync_state, run_backfill, set_sync_state


def test_sync_state_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    init_db(db)
    set_sync_state("test_key", "hello", db)
    assert get_sync_state("test_key", db) == "hello"
    assert get_sync_state("missing_key", db) is None


def test_backfill_resumable_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backfill saves progress; resuming skips already-processed chunks."""
    db = tmp_path / "t.db"
    init_db(db)

    call_ranges: list[tuple[str, str]] = []

    def mock_sync_all(since, until, **kwargs):
        call_ranges.append((since.isoformat(), until.isoformat()))
        return {"sessions": 0, "hrv_daily": 0, "wellness_daily": 0}

    monkeypatch.setattr("kairos.ingest.garmin_sync.sync_all", mock_sync_all)
    monkeypatch.setattr("kairos.ingest.garmin_sync.time.sleep", lambda _: None)

    since = date(2024, 1, 1)
    until = date(2024, 3, 1)

    # First run: processes all chunks
    run_backfill(since=since, until=until, resume=False, db_path=db)
    n_first = len(call_ranges)
    assert n_first > 0

    last_saved = get_sync_state("backfill_last_chunk_end", db)
    assert last_saved == until.isoformat()

    # Resume run: last checkpoint = until → nothing to do
    call_ranges.clear()
    result = run_backfill(since=since, until=until, resume=True, db_path=db)
    assert len(call_ranges) == 0
    assert result.get("status") == "already_complete"

    # Non-resume run: full re-process (same chunks)
    call_ranges.clear()
    run_backfill(since=since, until=until, resume=False, db_path=db)
    assert len(call_ranges) == n_first
