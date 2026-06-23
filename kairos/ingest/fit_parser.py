"""
Phase 1 — Parse .fit files and extract R-R intervals, running dynamics, and session metadata.

R-R intervals live in `hrv` messages (values in seconds → ×1000 for ms).
Running dynamics come from `record` messages; Garmin-reported units after fitparse scaling:
  ground_contact_time → ms, vertical_oscillation → mm, stance_time_balance → %
A file is classified as "resting HRV" when no GPS data is present and duration < 5 min.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import date, datetime
from pathlib import Path
from typing import Any

_RESTING_MAX_S = 300    # files under 5 min + no GPS → resting HRV measurement


@dataclass
class FitData:
    source_path: Path
    start_date: date | None
    rr_intervals_ms: list[float]                    # R-R in ms
    records: list[dict[str, Any]]                   # per-second record messages
    session_summary: dict[str, Any]                 # session-level fields
    is_resting_hrv: bool
    laps: list[dict[str, Any]] = _field(default_factory=list)  # lap messages


def parse_fit(path: Path | str) -> FitData:
    from fitparse import FitFile

    path = Path(path)
    ff = FitFile(str(path))

    # --- R-R intervals ---
    rr_ms: list[float] = []
    for msg in ff.get_messages("hrv"):
        for f in msg.fields:
            if f.name == "time" and f.value:
                for v in f.value:
                    if v is not None and v > 0:
                        rr_ms.append(float(v) * 1000.0)

    # --- Per-second records ---
    records: list[dict[str, Any]] = []
    for msg in ff.get_messages("record"):
        d: dict[str, Any] = {}
        for f in msg.fields:
            d[f.name] = f.value
        records.append(d)

    # --- Lap messages ---
    laps: list[dict[str, Any]] = []
    for msg in ff.get_messages("lap"):
        d2: dict[str, Any] = {}
        for f in msg.fields:
            d2[f.name] = f.value
        laps.append(d2)

    # --- Session summary ---
    session_summary: dict[str, Any] = {}
    for msg in ff.get_messages("session"):
        for f in msg.fields:
            session_summary[f.name] = f.value
        break

    # --- Determine start date ---
    start_date: date | None = None
    ts = session_summary.get("start_time")
    if isinstance(ts, datetime):
        start_date = ts.date()
    elif records:
        ts2 = records[0].get("timestamp")
        if isinstance(ts2, datetime):
            start_date = ts2.date()

    # --- Classify as resting HRV if no GPS and short ---
    duration_s = session_summary.get("total_timer_time") or 0
    has_gps = any(r.get("position_lat") is not None for r in records[:20])
    is_resting = not has_gps and float(duration_s) < _RESTING_MAX_S

    return FitData(
        source_path=path,
        start_date=start_date,
        rr_intervals_ms=rr_ms,
        records=records,
        session_summary=session_summary,
        is_resting_hrv=is_resting,
        laps=laps,
    )
