"""
Pull activities, overnight HRV, sleep score, and stress from Garmin Connect.

Auth flow (garminconnect >= 0.3.5):
  - First call: email+password → tokens saved at GARMIN_TOKENSTORE path.
  - Subsequent calls: tokens loaded from disk, auto-refreshed when expiring.
  - No garth dependency.

Credentials (never hardcoded):
  GARMIN_EMAIL      — Garmin Connect account email
  GARMIN_PASSWORD   — Garmin Connect account password
  GARMIN_TOKENSTORE — token directory (default: ~/.kairos/garmin_tokens)

NOTE on HRV source:
  Garmin overnight HRV (watch) ≠ chest-band morning resting HRV (spec primary).
  Overnight values are stored with source='garmin_sleep_hrv' and carry a higher
  κ penalty.  The chest-band .fit morning measurement remains the primary source
  and will overwrite this row when processed by fit_parser in Phase 1.
"""

from __future__ import annotations

import io
import logging
import math
import os
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from kairos.config import (
    DB_PATH,
    FIT_DIR,
    HISTORY_START,
    SYNC_CHUNK_DAYS,
    SYNC_SLEEP_S,
)
from kairos.db import db_cursor, init_db

log = logging.getLogger(__name__)

_DEFAULT_TOKENSTORE = Path.home() / ".kairos" / "garmin_tokens"
_SLEEP_BETWEEN_CALLS = 0.5   # seconds — polite pacing, not a rate-limit workaround

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_client(
    email: str | None = None,
    password: str | None = None,
    tokenstore: str | Path | None = None,
) -> Any:
    """Return an authenticated Garmin client, reusing saved tokens when possible."""
    try:
        import garminconnect
    except ImportError as exc:
        raise ImportError(
            "garminconnect is required for Garmin sync. "
            "Install it with: pip install garminconnect>=0.3.5"
        ) from exc

    email = email or os.environ.get("GARMIN_EMAIL", "")
    password = password or os.environ.get("GARMIN_PASSWORD", "")
    tokenstore_path = Path(
        tokenstore
        or os.environ.get("GARMIN_TOKENSTORE", _DEFAULT_TOKENSTORE)
    ).expanduser()
    tokenstore_path.mkdir(parents=True, exist_ok=True)

    client = garminconnect.Garmin(email, password)
    mfa_status, _ = client.login(tokenstore=str(tokenstore_path))
    if mfa_status:
        mfa_code = input("Garmin MFA code: ").strip()
        client.resume_login(mfa_code, tokenstore=str(tokenstore_path))

    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_likert(score: float, invert: bool = False) -> int:
    """Map a 0–100 Garmin score to a 1–5 Likert value.

    invert=True is used for stress (higher score = worse = lower Likert).
    """
    if score < 0:
        return 3   # Garmin returns -1 / -2 when no data — use neutral
    v = (100 - score) if invert else score
    return max(1, min(5, 1 + round(v / 25)))


def _duration_to_likert(seconds: float | None) -> int:
    """Map sleep duration in seconds to a 1–5 Likert quality proxy.

    Used only as a fallback when the Garmin sleep score is unavailable.
    Thresholds: <5.5 h → 1, <6.5 h → 2, <7.5 h → 3, <8.5 h → 4, else 5.
    """
    if seconds is None or seconds < 0:
        return 3
    hours = seconds / 3600.0
    if hours < 5.5:
        return 1
    if hours < 6.5:
        return 2
    if hours < 7.5:
        return 3
    if hours < 8.5:
        return 4
    return 5


def _extract_fit_from_zip(zip_bytes: bytes) -> bytes:
    """Extract the single .fit file from a Garmin ORIGINAL download zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        fit_names = [n for n in zf.namelist() if n.lower().endswith(".fit")]
        if not fit_names:
            raise ValueError("No .fit file found in downloaded zip")
        return zf.read(fit_names[0])


# ---------------------------------------------------------------------------
# Individual sync functions
# ---------------------------------------------------------------------------

def sync_activities(
    client: Any,
    since: date,
    until: date | None = None,
    db_path: Path = DB_PATH,
    fit_dir: Path = FIT_DIR,
) -> int:
    """Download running .fit files and upsert session rows. Returns count of new rows."""
    import garminconnect

    until = until or date.today()
    fit_dir.mkdir(parents=True, exist_ok=True)

    activities = client.get_activities_by_date(
        since.isoformat(), until.isoformat(), activitytype="running"
    )
    log.info("Garmin returned %d running activities (%s → %s)", len(activities), since, until)

    inserted = 0
    for act in activities:
        activity_id = str(act.get("activityId", ""))
        act_date = (act.get("startTimeLocal") or "")[:10]
        if not act_date or not activity_id:
            continue

        with db_cursor(db_path) as cur:
            exists = cur.execute(
                "SELECT id FROM sessions WHERE fit_path LIKE ?",
                (f"%{activity_id}%",),
            ).fetchone()
        if exists:
            log.debug("Activity %s already in DB, skipping", activity_id)
            continue

        # Download .fit file
        try:
            zip_bytes = client.download_activity(
                activity_id, dl_fmt=garminconnect.Garmin.ActivityDownloadFormat.ORIGINAL
            )
            fit_bytes = _extract_fit_from_zip(zip_bytes)
        except Exception as exc:
            log.warning("Could not download .fit for activity %s: %s", activity_id, exc)
            fit_bytes = None

        fit_path: str | None = None
        if fit_bytes:
            fit_file = fit_dir / f"{act_date}_{activity_id}.fit"
            fit_file.write_bytes(fit_bytes)
            fit_path = str(fit_file)
            log.debug("Saved .fit → %s", fit_file)

        duration_min = (act.get("duration") or 0) / 60.0
        type_key = (act.get("activityType") or {}).get("typeKey", "run")

        with db_cursor(db_path) as cur:
            cur.execute(
                "INSERT INTO sessions (date, type, duration_min, fit_path) VALUES (?,?,?,?)",
                (act_date, type_key, duration_min or None, fit_path),
            )
        inserted += 1
        time.sleep(_SLEEP_BETWEEN_CALLS)

    return inserted


def sync_hrv(
    client: Any,
    since: date,
    until: date | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """Pull overnight HRV (watch) and insert into hrv_daily as a fallback source.

    Skips dates that already have a higher-confidence 'fit' or 'manual' source.
    """
    until = until or date.today()
    inserted = 0
    current = since

    while current <= until:
        date_str = current.isoformat()
        with db_cursor(db_path) as cur:
            row = cur.execute(
                "SELECT source FROM hrv_daily WHERE date = ?", (date_str,)
            ).fetchone()

        if row and row["source"] in ("fit", "manual"):
            log.debug("HRV %s already has primary source, skipping", date_str)
            current += timedelta(days=1)
            continue

        try:
            data = client.get_hrv_data(date_str)
        except Exception as exc:
            log.warning("HRV fetch failed for %s: %s", date_str, exc)
            current += timedelta(days=1)
            time.sleep(_SLEEP_BETWEEN_CALLS)
            continue

        if not data:
            current += timedelta(days=1)
            continue

        summary = data.get("hrvSummary") or {}
        last_night_ms = summary.get("lastNight")  # average RMSSD during sleep (ms)

        if last_night_ms and last_night_ms > 0:
            ln_val = math.log(last_night_ms)
            with db_cursor(db_path) as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO hrv_daily "
                    "(date, ln_rmssd, rmssd, artifact_pct, condition_ok, source) "
                    "VALUES (?, ?, ?, NULL, 1, 'garmin_sleep_hrv')",
                    (date_str, ln_val, last_night_ms),
                )
            inserted += 1

        current += timedelta(days=1)
        time.sleep(_SLEEP_BETWEEN_CALLS)

    return inserted


def sync_wellness(
    client: Any,
    since: date,
    until: date | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """Pull sleep score and stress from Garmin → pre-fill wellness_daily.

    Only fills sleep and stress; mood and motivation remain manual entry.
    Does not overwrite rows where all five fields are already populated.
    """
    until = until or date.today()
    inserted = 0
    current = since

    while current <= until:
        date_str = current.isoformat()

        with db_cursor(db_path) as cur:
            row = cur.execute(
                "SELECT sleep, soreness, stress, mood, motivation FROM wellness_daily WHERE date = ?",
                (date_str,),
            ).fetchone()

        # Skip if already fully populated
        if row and all(row[k] is not None for k in ("sleep", "soreness", "stress", "mood", "motivation")):
            current += timedelta(days=1)
            continue

        # --- Sleep ---
        sleep_likert: int | None = None
        try:
            sleep_data = client.get_sleep_data(date_str)
            if sleep_data:
                dto = sleep_data.get("dailySleepDTO") or {}
                scores = dto.get("sleepScores") or {}
                overall = (scores.get("overall") or {}).get("value")
                if overall is not None:
                    sleep_likert = _score_to_likert(float(overall))
                else:
                    secs = dto.get("sleepTimeSeconds") or dto.get("sleepingSeconds")
                    if secs is not None:
                        sleep_likert = _duration_to_likert(float(secs))
        except Exception as exc:
            log.warning("Sleep fetch failed for %s: %s", date_str, exc)

        time.sleep(_SLEEP_BETWEEN_CALLS)

        # --- Stress ---
        stress_likert: int | None = None
        try:
            stress_data = client.get_stress_data(date_str)
            if stress_data:
                avg_stress = stress_data.get("avgStressLevel") or stress_data.get("averageStressLevel")
                if avg_stress is not None:
                    stress_likert = _score_to_likert(avg_stress, invert=True)
        except Exception as exc:
            log.warning("Stress fetch failed for %s: %s", date_str, exc)

        time.sleep(_SLEEP_BETWEEN_CALLS)

        if sleep_likert is None and stress_likert is None:
            current += timedelta(days=1)
            continue

        # Merge with existing row (preserve what the user already entered)
        existing = dict(row) if row else {}
        new_sleep = existing.get("sleep") or sleep_likert
        new_stress = existing.get("stress") or stress_likert
        new_soreness = existing.get("soreness")
        new_mood = existing.get("mood")
        new_motivation = existing.get("motivation")
        filled = [x for x in (new_sleep, new_soreness, new_stress, new_mood, new_motivation) if x]
        wellness_raw = sum(filled) if filled else None

        with db_cursor(db_path) as cur:
            cur.execute(
                "INSERT OR REPLACE INTO wellness_daily "
                "(date, sleep, soreness, stress, mood, motivation, wellness_raw) "
                "VALUES (?,?,?,?,?,?,?)",
                (date_str, new_sleep, new_soreness, new_stress, new_mood, new_motivation, wellness_raw),
            )
        inserted += 1
        current += timedelta(days=1)

    return inserted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sync_all(
    since: date | None = None,
    until: date | None = None,
    *,
    email: str | None = None,
    password: str | None = None,
    tokenstore: str | Path | None = None,
    db_path: Path = DB_PATH,
    fit_dir: Path = FIT_DIR,
    skip_activities: bool = False,
    skip_hrv: bool = False,
    skip_wellness: bool = False,
) -> dict[str, int]:
    """Full sync: activities, HRV, and wellness pre-fill.

    Args:
        since: Start date (default: 30 days ago).
        until: End date (default: today).
    Returns:
        dict with counts of new rows per table.
    """
    init_db(db_path)
    since = since or (date.today() - timedelta(days=30))
    until = until or date.today()

    log.info("Garmin sync %s → %s", since, until)
    client = _get_client(email=email, password=password, tokenstore=tokenstore)

    results: dict[str, int] = {}

    if not skip_activities:
        results["sessions"] = sync_activities(client, since, until, db_path, fit_dir)

    if not skip_hrv:
        results["hrv_daily"] = sync_hrv(client, since, until, db_path)

    if not skip_wellness:
        results["wellness_daily"] = sync_wellness(client, since, until, db_path)

    log.info("Sync complete: %s", results)
    return results


# ---------------------------------------------------------------------------
# Sync state (backfill checkpoint)
# ---------------------------------------------------------------------------

def get_sync_state(key: str, db_path: Path = DB_PATH) -> str | None:
    """Read a value from sync_state.  Returns None if key not found."""
    with db_cursor(db_path) as cur:
        row = cur.execute("SELECT value FROM sync_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_sync_state(key: str, value: str, db_path: Path = DB_PATH) -> None:
    """Upsert a value into sync_state."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with db_cursor(db_path) as cur:
        cur.execute(
            "INSERT OR REPLACE INTO sync_state (key, value, updated_at) VALUES (?,?,?)",
            (key, value, now),
        )


# ---------------------------------------------------------------------------
# Local archive ingestion
# ---------------------------------------------------------------------------

def ingest_archive_dir(
    archive_dir: Path,
    db_path: Path = DB_PATH,
) -> dict[str, int]:
    """Ingest all .fit files in archive_dir into the database."""
    from kairos.ingest.loaders import ingest_fit_file

    fit_files = sorted(archive_dir.glob("*.fit"))
    log.info("Found %d .fit files in %s", len(fit_files), archive_dir)

    n_sessions, n_hrv, n_errors = 0, 0, 0
    for fit_path in fit_files:
        try:
            result = ingest_fit_file(fit_path, db_path=db_path)
            msg = result.lower()
            if "hrv" in msg or "resting" in msg:
                n_hrv += 1
            else:
                n_sessions += 1
        except Exception as exc:
            log.warning("Failed to ingest %s: %s", fit_path.name, exc)
            n_errors += 1

    log.info("Archive ingest done: %d sessions, %d HRV, %d errors", n_sessions, n_hrv, n_errors)
    return {"sessions": n_sessions, "hrv": n_hrv, "errors": n_errors}


# ---------------------------------------------------------------------------
# Historical backfill (chunked, resumable, rate-limit aware)
# ---------------------------------------------------------------------------

def run_backfill(
    since: date | None = None,
    until: date | None = None,
    *,
    resume: bool = True,
    archive_dir: Path | None = None,
    email: str | None = None,
    password: str | None = None,
    tokenstore: str | Path | None = None,
    db_path: Path = DB_PATH,
    fit_dir: Path = FIT_DIR,
    skip_activities: bool = False,
    skip_hrv: bool = False,
    skip_wellness: bool = False,
) -> dict[str, Any]:
    """
    Backfill Garmin data in SYNC_CHUNK_DAYS chunks with resumability.

    - Progress is saved to sync_state (key='backfill_last_chunk_end') after each chunk.
    - 429 / rate-limit errors trigger exponential backoff (max 1 h).
    - If archive_dir is provided, local .fit files are ingested first (no API needed).
    - resume=True skips chunks already processed in a previous run.
    """
    init_db(db_path)

    since_date = since or date.fromisoformat(HISTORY_START)
    until_date = until or date.today()

    if resume:
        last = get_sync_state("backfill_last_chunk_end", db_path)
        if last:
            resumed_start = date.fromisoformat(last) + timedelta(days=1)
            if resumed_start > until_date:
                log.info("Backfill already complete (last chunk: %s)", last)
                return {"status": "already_complete", "last_chunk": last}
            log.info("Resuming from %s (checkpoint: %s)", resumed_start, last)
            since_date = resumed_start

    # Ingest local archive first (no API rate limits)
    archive_stats: dict = {}
    if archive_dir is not None and Path(archive_dir).exists():
        log.info("Ingesting archive dir: %s", archive_dir)
        archive_stats = ingest_archive_dir(Path(archive_dir), db_path)

    total: dict[str, int] = {}
    current = since_date

    while current <= until_date:
        chunk_end = min(current + timedelta(days=SYNC_CHUNK_DAYS - 1), until_date)

        delay = 60  # seconds; doubles on each 429 up to max_backoff
        for attempt in range(5):
            try:
                chunk_result = sync_all(
                    since=current, until=chunk_end,
                    email=email, password=password, tokenstore=tokenstore,
                    db_path=db_path, fit_dir=fit_dir,
                    skip_activities=skip_activities,
                    skip_hrv=skip_hrv, skip_wellness=skip_wellness,
                )
                for k, v in chunk_result.items():
                    total[k] = total.get(k, 0) + v
                break
            except Exception as exc:
                err_str = str(exc)
                if any(s in err_str for s in ("429", "Too Many", "TooManyRequests", "rate limit")):
                    log.warning(
                        "Rate limited on %s→%s. Waiting %ds (attempt %d/5)",
                        current, chunk_end, delay, attempt + 1,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 3600)
                else:
                    log.error("Chunk %s→%s failed: %s", current, chunk_end, exc)
                    raise

        set_sync_state("backfill_last_chunk_end", chunk_end.isoformat(), db_path)
        log.info("Chunk %s→%s done. Totals: %s", current, chunk_end, total)
        current = chunk_end + timedelta(days=1)
        if current <= until_date:
            time.sleep(SYNC_SLEEP_S)

    if archive_stats:
        total["archive"] = archive_stats
    return total
