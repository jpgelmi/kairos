"""
SQLite database schema and connection management for Kairós.
All DDL lives here; never scatter CREATE TABLE statements across modules.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from kairos.config import DB_PATH

# TRIMP / objective load columns
_TRIMP_MIGRATIONS: list[str] = [
    "ALTER TABLE sessions ADD COLUMN trimp REAL",
    "ALTER TABLE sessions ADD COLUMN load_source TEXT",
    "ALTER TABLE sessions ADD COLUMN avg_hr REAL",
    "ALTER TABLE sessions ADD COLUMN max_hr REAL",
    "ALTER TABLE running_dynamics ADD COLUMN hrr60 REAL",
    "ALTER TABLE running_dynamics ADD COLUMN tau_hrr REAL",
]

_DOMAIN_MIGRATIONS: list[str] = [
    "ALTER TABLE sessions ADD COLUMN trimp_aerobic REAL",
    "ALTER TABLE sessions ADD COLUMN trimp_hii REAL",
    # form_state domain columns: now in base DDL; ALTER keeps existing DBs in sync
    "ALTER TABLE form_state ADD COLUMN g_aerobic REAL",
    "ALTER TABLE form_state ADD COLUMN h_aerobic REAL",
    "ALTER TABLE form_state ADD COLUMN g_hii REAL",
    "ALTER TABLE form_state ADD COLUMN h_hii REAL",
    "ALTER TABLE form_state ADD COLUMN Pi_abs REAL",
    # lactate_thresholds created via DDL; no ALTER needed here
]


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_cursor(db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_DDL = """
-- -------------------------------------------------------------------------
-- Training sessions
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,          -- ISO-8601 YYYY-MM-DD
    type        TEXT    NOT NULL,          -- e.g. easy, tempo, interval, race, rest
    duration_min REAL,
    rpe         REAL,                      -- Borg CR-10 (0-10)
    srpe        REAL,                      -- sRPE = rpe * duration_min
    fit_path    TEXT,                      -- path to .fit file, nullable
    notes       TEXT
);

-- -------------------------------------------------------------------------
-- HRV (one row per day, morning resting measurement)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hrv_daily (
    date         TEXT PRIMARY KEY,          -- ISO-8601 YYYY-MM-DD
    ln_rmssd     REAL,                      -- natural log of RMSSD (ms)
    rmssd        REAL,                      -- raw RMSSD (ms)
    artifact_pct REAL,                      -- fraction of artefact beats [0,1]
    condition_ok INTEGER NOT NULL DEFAULT 1, -- 1=accepted, 0=rejected
    source       TEXT                        -- 'fit', 'garmin_sleep_hrv', 'manual'
);

-- -------------------------------------------------------------------------
-- Daily wellness questionnaire (Likert 1-5 per item)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS wellness_daily (
    date        TEXT PRIMARY KEY,
    sleep       INTEGER,   -- 1 (poor) … 5 (excellent)
    soreness    INTEGER,   -- 1 (severe) … 5 (none)
    stress      INTEGER,   -- 1 (very high) … 5 (none)
    mood        INTEGER,   -- 1 (very bad) … 5 (excellent)
    motivation  INTEGER,   -- 1 (none) … 5 (very high)
    wellness_raw REAL      -- sum of above (5-25)
);

-- -------------------------------------------------------------------------
-- Running dynamics (per session summary)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS running_dynamics (
    session_id      INTEGER PRIMARY KEY REFERENCES sessions(id),
    gct_mean_ms     REAL,   -- mean GCT (ms); entrada para drift_GCT
    gct_drift_pct   REAL,   -- drift_GCT = (GCT_final-GCT_init)/GCT_init  (ec. 15)
    hrr60           REAL,   -- HRR₆₀ = HR_pico - HR_60s                   (ec. 16)
    tau_hrr         REAL    -- τ_HRR: constante de tiempo del descenso exponencial (ec. 16)
);

-- -------------------------------------------------------------------------
-- Session segments  (Feature C — work/recovery/warmup/cooldown split)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    idx         INTEGER NOT NULL,
    kind        TEXT    NOT NULL,   -- 'warmup'|'work'|'recovery'|'rest'|'cooldown'
    start_s     REAL,
    end_s       REAL,
    duration_s  REAL,
    distance_m  REAL,
    avg_speed_ms REAL,
    avg_hr       REAL,
    avg_gct_ms   REAL,
    gct_cv       REAL,
    created_at   TEXT
);

-- -------------------------------------------------------------------------
-- Sync state (key-value store for backfill resumability)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sync_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);

-- -------------------------------------------------------------------------
-- Form state (final.tex §5: g, h, TSB, Π_rel, Π_abs, acumuladores de dominio)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS form_state (
    date           TEXT PRIMARY KEY,
    g              REAL,    -- fitness EWMA  g(t)        (ec. 5)
    h              REAL,    -- fatigue EWMA  h(t)        (ec. 6)
    freshness      REAL,    -- TSB = g(t-1) - h(t-1)    (ec. 7)
    Pi             REAL,    -- Π_rel(t) ∈ (−1, 1)       (ec. 13)
    Pi_abs         REAL,    -- odómetro Π_abs(t)         (ec. 14)
    g_aerobic      REAL,    -- CTL aeróbico (τ=42)
    h_aerobic      REAL,    -- ATL aeróbico (τ=7)
    g_hii          REAL,    -- CTL alta intensidad (τ=21)
    h_hii          REAL     -- ATL alta intensidad (τ=7)
);

-- -------------------------------------------------------------------------
-- Race results (actual performances for prediction calibration)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS race_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    distance_m  REAL NOT NULL,
    time_s      REAL NOT NULL,
    event       TEXT,
    notes       TEXT
);

-- -------------------------------------------------------------------------
-- Lactate thresholds (anclas longitudinales episódicas — final.tex §4 y §5.2)
-- v_lt2 y hr_lt2 provienen de tests periódicos de escalón de lactato.
-- Para cualquier sesión se usa el test más reciente con date <= session_date.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lactate_thresholds (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    date    TEXT NOT NULL UNIQUE,   -- ISO-8601 YYYY-MM-DD del test
    v_lt2   REAL,                   -- velocidad en LT2 (m/s)
    hr_lt2  REAL,                   -- FC en LT2 (bpm)
    notes   TEXT
);

"""


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive column migrations; silently skip if already present."""
    for sql in _TRIMP_MIGRATIONS + _DOMAIN_MIGRATIONS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables if they do not yet exist, then apply pending migrations."""
    with get_connection(db_path) as conn:
        conn.executescript(_DDL)
        _run_migrations(conn)
        conn.commit()


def drop_all(db_path: Path = DB_PATH) -> None:
    """Drop every table — only used in tests with a temp DB."""
    tables = [
        "race_results", "lactate_thresholds", "form_state", "running_dynamics",
        "session_segments", "wellness_daily",
        "hrv_daily", "sessions", "sync_state",
    ]
    with get_connection(db_path) as conn:
        for t in tables:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.commit()
