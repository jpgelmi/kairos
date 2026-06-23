"""
Phase 1 — DB insertion functions for both parsed .fit data and manual CLI entry.
All functions accept explicit db_path for testability.
ingest_fit_file() defers physio imports to runtime (Phase 2 dependency).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from kairos.config import DB_PATH, FIT_DIR
from kairos.db import db_cursor, init_db


# ---------------------------------------------------------------------------
# Low-level insert helpers
# ---------------------------------------------------------------------------

def insert_session(
    date_: str,
    type_: str,
    duration_min: float | None = None,
    rpe: float | None = None,
    fit_path: str | None = None,
    notes: str | None = None,
    db_path: Path = DB_PATH,
) -> int:
    srpe = (rpe * duration_min) if (rpe is not None and duration_min is not None) else None
    with db_cursor(db_path) as cur:
        cur.execute(
            "INSERT INTO sessions (date, type, duration_min, rpe, srpe, fit_path, notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (date_, type_, duration_min, rpe, srpe, fit_path, notes),
        )
        return cur.lastrowid  # type: ignore[return-value]


def upsert_wellness(
    date_: str,
    sleep: int | None = None,
    soreness: int | None = None,
    stress: int | None = None,
    mood: int | None = None,
    motivation: int | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Insert or merge wellness row.  Existing non-None fields are preserved."""
    with db_cursor(db_path) as cur:
        existing = cur.execute(
            "SELECT sleep, soreness, stress, mood, motivation FROM wellness_daily WHERE date=?",
            (date_,),
        ).fetchone()

    if existing:
        sleep = sleep if sleep is not None else existing["sleep"]
        soreness = soreness if soreness is not None else existing["soreness"]
        stress = stress if stress is not None else existing["stress"]
        mood = mood if mood is not None else existing["mood"]
        motivation = motivation if motivation is not None else existing["motivation"]

    filled = [x for x in (sleep, soreness, stress, mood, motivation) if x is not None]
    wellness_raw = sum(filled) if filled else None

    with db_cursor(db_path) as cur:
        cur.execute(
            "INSERT OR REPLACE INTO wellness_daily "
            "(date, sleep, soreness, stress, mood, motivation, wellness_raw) "
            "VALUES (?,?,?,?,?,?,?)",
            (date_, sleep, soreness, stress, mood, motivation, wellness_raw),
        )


def upsert_hrv(
    date_: str,
    ln_rmssd: float,
    rmssd: float,
    artifact_pct: float,
    condition_ok: bool,
    source: str = "fit",
    db_path: Path = DB_PATH,
) -> None:
    """Insert HRV row. 'fit' source always wins over 'garmin_sleep_hrv'."""
    with db_cursor(db_path) as cur:
        existing = cur.execute(
            "SELECT source FROM hrv_daily WHERE date=?", (date_,)
        ).fetchone()
        if existing and existing["source"] == "fit" and source != "fit":
            return
        cur.execute(
            "INSERT OR REPLACE INTO hrv_daily "
            "(date, ln_rmssd, rmssd, artifact_pct, condition_ok, source) "
            "VALUES (?,?,?,?,?,?)",
            (date_, ln_rmssd, rmssd, artifact_pct, int(condition_ok), source),
        )


def insert_segments(
    session_id: int,
    segments: list,
    db_path: Path = DB_PATH,
) -> None:
    """Persist a list of SessionSegment objects for a session (replaces any existing)."""
    from datetime import datetime as _dt, timezone as _tz
    created_at = _dt.now(_tz.utc).isoformat()
    with db_cursor(db_path) as cur:
        cur.execute("DELETE FROM session_segments WHERE session_id=?", (session_id,))
        for seg in segments:
            cur.execute(
                "INSERT INTO session_segments "
                "(session_id, idx, kind, start_s, end_s, duration_s, "
                " distance_m, avg_speed_ms, avg_hr, avg_gct_ms, gct_cv, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id, seg.idx, seg.kind,
                    seg.start_s, seg.end_s, seg.duration_s,
                    seg.distance_m, seg.avg_speed_ms, seg.avg_hr,
                    seg.avg_gct_ms, seg.gct_cv, created_at,
                ),
            )


def get_segments(session_id: int, db_path: Path = DB_PATH) -> list:
    """Return SessionSegment-like objects for a session, ordered by idx."""
    from kairos.physio.segmentation import SessionSegment
    with db_cursor(db_path) as cur:
        rows = cur.execute(
            "SELECT idx, kind, start_s, end_s, duration_s, "
            "distance_m, avg_speed_ms, avg_hr, avg_gct_ms, gct_cv "
            "FROM session_segments WHERE session_id=? ORDER BY idx",
            (session_id,),
        ).fetchall()
    return [
        SessionSegment(
            idx=r["idx"], kind=r["kind"],
            start_s=r["start_s"] or 0.0, end_s=r["end_s"] or 0.0,
            duration_s=r["duration_s"] or 0.0,
            distance_m=r["distance_m"], avg_speed_ms=r["avg_speed_ms"],
            avg_hr=r["avg_hr"], avg_gct_ms=r["avg_gct_ms"], gct_cv=r["gct_cv"],
        )
        for r in rows
    ]


def update_session_load(
    session_id: int,
    trimp: float | None = None,
    load_source: str | None = None,
    avg_hr: float | None = None,
    max_hr: float | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Update TRIMP / HR summary columns on an existing session row."""
    parts: list[str] = []
    values: list = []
    for col, val in (("trimp", trimp), ("load_source", load_source),
                     ("avg_hr", avg_hr), ("max_hr", max_hr)):
        if val is not None:
            parts.append(f"{col}=?")
            values.append(val)
    if not parts:
        return
    values.append(session_id)
    with db_cursor(db_path) as cur:
        cur.execute(f"UPDATE sessions SET {', '.join(parts)} WHERE id=?", values)


def insert_running_dynamics(
    session_id: int,
    gct_mean_ms: float | None,
    gct_drift_pct: float | None,
    hrr60: float | None = None,
    tau_hrr: float | None = None,
    db_path: Path = DB_PATH,
) -> None:
    with db_cursor(db_path) as cur:
        cur.execute(
            "INSERT OR REPLACE INTO running_dynamics "
            "(session_id, gct_mean_ms, gct_drift_pct, hrr60, tau_hrr) "
            "VALUES (?,?,?,?,?)",
            (session_id, gct_mean_ms, gct_drift_pct, hrr60, tau_hrr),
        )



# ---------------------------------------------------------------------------
# High-level .fit ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_fit_file(
    path: Path | str,
    db_path: Path = DB_PATH,
    fit_dir: Path = FIT_DIR,
) -> dict[str, Any]:
    """
    Parse a .fit file and write session data to the DB.
    Returns a summary dict.
    """
    from kairos.ingest.fit_parser import parse_fit
    from kairos.physio.dynamics import compute_gct_drift, gct_drift_from_segments
    from kairos.physio.segmentation import (
        enrich_segments_with_records,
        is_interval_session,
        segment_from_laps,
        segment_from_speed,
    )

    init_db(db_path)
    path = Path(path)
    data = parse_fit(path)
    result: dict[str, Any] = {"path": str(path), "date": str(data.start_date)}

    fit_dir.mkdir(parents=True, exist_ok=True)
    dest = fit_dir / path.name
    if path.resolve() != dest.resolve() and not dest.exists():
        shutil.copy2(path, dest)

    date_str = data.start_date.isoformat() if data.start_date else None
    if not date_str:
        result["error"] = "no date in .fit"
        return result

    summary = data.session_summary
    duration_min = float(summary.get("total_timer_time") or 0) / 60.0
    sport = str(summary.get("sport") or "running").lower()

    session_id = insert_session(
        date_str, sport, duration_min or None,
        fit_path=str(dest), db_path=db_path,
    )
    result["session_id"] = session_id

    # --- Compute t_s (seconds from session start) for each record ---
    from datetime import datetime as _dt
    start_ts = summary.get("start_time")
    if not isinstance(start_ts, _dt) and data.records:
        start_ts = data.records[0].get("timestamp")
    for i, rec in enumerate(data.records):
        ts = rec.get("timestamp")
        if isinstance(ts, _dt) and isinstance(start_ts, _dt):
            rec["t_s"] = (ts - start_ts).total_seconds()
        elif "t_s" not in rec:
            rec["t_s"] = float(i)

    # --- Segment the session ---
    if data.laps:
        segments = segment_from_laps(data.laps)
    else:
        segments = segment_from_speed(data.records)
    segments = enrich_segments_with_records(segments, data.records)
    if segments:
        insert_segments(session_id, segments, db_path)
        result["segments"] = len(segments)

    gct_vals = [float(r["ground_contact_time"]) for r in data.records
                if r.get("ground_contact_time") is not None]

    # --- Compute TRIMP from per-second HR records ---
    hr_records = [float(r["heart_rate"]) for r in data.records
                  if r.get("heart_rate") is not None]
    if hr_records:
        from kairos.physio.trimp import compute_session_trimp, detect_hrmax, detect_hrrest
        import numpy as _np
        hrmax = detect_hrmax(db_path)
        hrrest = detect_hrrest(db_path)
        trimp_val, load_src = compute_session_trimp(hr_records, hrmax, hrrest=hrrest)
        update_session_load(
            session_id,
            trimp=trimp_val,
            load_source=load_src,
            avg_hr=float(_np.mean(hr_records)),
            max_hr=float(max(hr_records)),
            db_path=db_path,
        )
        result["trimp"] = trimp_val
    else:
        # TE proxy fallback when no HR records
        te = summary.get("total_training_effect")
        if te is not None and duration_min:
            from kairos.physio.trimp import te_proxy_trimp
            trimp_val = te_proxy_trimp(float(te), duration_min)
            update_session_load(session_id, trimp=trimp_val, load_source="te_proxy",
                                db_path=db_path)
            result["trimp"] = trimp_val

    # --- Domain TRIMP split (final.tex ec. 8–9) ---
    # Calcula L_aer y L_hii si hay umbral de lactato registrado para esta fecha.
    if hr_records:
        from kairos.model.fitness_fatigue import (
            get_threshold_at, compute_domain_trimp_from_records
        )
        from kairos.physio.trimp import detect_hrmax as _dhrmax2, detect_hrrest as _dhrrest2
        v_lt2, hr_lt2 = get_threshold_at(date_str, db_path)
        if v_lt2 is not None or hr_lt2 is not None:
            _hrmax2 = _dhrmax2(db_path)
            _hrrest2 = _dhrrest2(db_path)
            l_aer, l_hii = compute_domain_trimp_from_records(
                data.records, _hrmax2, _hrrest2, v_lt2, hr_lt2
            )
            if l_aer is not None:
                with db_cursor(db_path) as _cur:
                    _cur.execute(
                        "UPDATE sessions SET trimp_aerobic=?, trimp_hii=? WHERE id=?",
                        (l_aer, l_hii, session_id),
                    )
                result["trimp_aerobic"] = l_aer
                result["trimp_hii"] = l_hii

    # --- GCT drift and running dynamics ---
    if gct_vals:
        if segments and is_interval_session(sport, segments):
            drift = gct_drift_from_segments(segments)
        else:
            drift = compute_gct_drift(gct_vals)

        hrr60_val: float | None = None
        tau_val: float | None = None
        if segments and hr_records:
            from kairos.physio.recovery import session_hrr
            from kairos.physio.trimp import detect_hrmax as _dhrmax
            hrr_results = session_hrr(
                data.records, segments,
                hrmax=_dhrmax(db_path) if hr_records else None,
            )
            if hrr_results:
                import statistics as _stats
                tau_vals = [r.tau_s for r in hrr_results if r.tau_s is not None]
                hrr60_vals = [r.hrr60 for r in hrr_results if r.hrr60 is not None]
                tau_val = float(_stats.median(tau_vals)) if tau_vals else None
                hrr60_val = float(_stats.median(hrr60_vals)) if hrr60_vals else None

        insert_running_dynamics(
            session_id,
            gct_mean_ms=sum(gct_vals) / len(gct_vals),
            gct_drift_pct=drift,
            hrr60=hrr60_val,
            tau_hrr=tau_val,
            db_path=db_path,
        )
        result["dynamics"] = {"gct_n": len(gct_vals)}

    return result
