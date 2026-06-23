"""
Phase 3+ — Session segmentation into work/recovery/warmup/cooldown segments.

Priority:
  1. Lap messages from .fit (structured workouts — Garmin writes one lap per rep/rest).
  2. Speed-threshold fallback for unstructured files.

After segmentation, `enrich_segments_with_records()` fills per-segment GCT/HR stats
from the per-second record stream so that `gct_drift_from_segments()` in dynamics.py
can compute drift across work reps only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SessionSegment:
    idx: int
    kind: str           # 'warmup' | 'work' | 'recovery' | 'rest' | 'cooldown'
    start_s: float
    end_s: float
    duration_s: float
    distance_m: float | None = None
    avg_speed_ms: float | None = None
    avg_hr: float | None = None
    avg_gct_ms: float | None = None
    gct_cv: float | None = None


# ---------------------------------------------------------------------------
# Primary: segment from .fit lap messages
# ---------------------------------------------------------------------------

def segment_from_laps(laps: list[dict[str, Any]]) -> list[SessionSegment]:
    """
    Build segments from .fit lap messages (one lap per rep/rest in structured workouts).
    Classification uses the largest speed gap to separate work from recovery.
    """
    if not laps:
        return []

    # Garmin .fit files use enhanced_avg_speed (higher precision); avg_speed is legacy fallback
    speeds = [
        float(lap.get("enhanced_avg_speed") or lap.get("avg_speed") or 0.0)
        for lap in laps
    ]
    active_speeds = sorted({s for s in speeds if s > 0.5})

    threshold = _gap_threshold(active_speeds) if len(active_speeds) >= 2 else 0.0

    segs: list[SessionSegment] = []
    t = 0.0
    for i, lap in enumerate(laps):
        dur = float(lap.get("total_elapsed_time") or lap.get("total_timer_time") or 0.0)
        spd = float(lap.get("enhanced_avg_speed") or lap.get("avg_speed") or 0.0)
        dist = lap.get("total_distance")
        hr = lap.get("avg_heart_rate")

        if dur < 1.0:
            t += dur
            continue

        if spd <= 0.5:
            kind = "rest"
        elif threshold > 0 and spd >= threshold:
            kind = "work"
        else:
            kind = "recovery"

        segs.append(SessionSegment(
            idx=len(segs),
            kind=kind,
            start_s=t,
            end_s=t + dur,
            duration_s=dur,
            distance_m=float(dist) if dist is not None else None,
            avg_speed_ms=spd if spd > 0.5 else None,
            avg_hr=float(hr) if hr is not None else None,
        ))
        t += dur

    segs = _assign_warmup_cooldown(segs)
    return _merge_tiny(segs)


def _gap_threshold(sorted_speeds: list[float]) -> float:
    """Midpoint of the largest gap in the speed distribution."""
    if len(sorted_speeds) < 2:
        return 0.0
    max_gap = 0.0
    threshold = sorted_speeds[-1]  # default: everything is work
    for a, b in zip(sorted_speeds, sorted_speeds[1:]):
        if b - a > max_gap:
            max_gap = b - a
            threshold = (a + b) / 2.0
    return threshold


# ---------------------------------------------------------------------------
# Fallback: segment from per-second speed series
# ---------------------------------------------------------------------------

def segment_from_speed(records: list[dict[str, Any]]) -> list[SessionSegment]:
    """
    Fallback segmentation when no lap messages are available.
    Uses WORK_SPEED_PCTILE (default 60th) of non-rest speeds as threshold.
    Records must have 't_s' (seconds from session start) and 'speed' or 'enhanced_speed'.
    """
    from kairos.config import WORK_SPEED_PCTILE

    ts_list: list[float] = []
    spd_list: list[float] = []
    for r in records:
        t = r.get("t_s")
        spd = r.get("speed") or r.get("enhanced_speed")
        if t is not None and spd is not None:
            ts_list.append(float(t))
            spd_list.append(float(spd))

    if len(spd_list) < 10:
        return []

    spd_arr = np.array(spd_list, dtype=float)
    ts_arr = np.array(ts_list, dtype=float)

    # Rolling median smoothing (window = min 5 samples)
    w = min(5, len(spd_arr))
    smoothed = np.array(
        [float(np.median(spd_arr[max(0, i - w // 2): i + w // 2 + 1]))
         for i in range(len(spd_arr))]
    )

    active = smoothed[smoothed > 0.5]
    if len(active) == 0:
        return []
    threshold = float(np.percentile(active, WORK_SPEED_PCTILE))

    labels: list[str] = []
    for s in smoothed:
        if s <= 0.5:
            labels.append("rest")
        elif s < threshold:
            labels.append("recovery")
        else:
            labels.append("work")

    segs: list[SessionSegment] = []
    i = 0
    n = len(labels)
    while i < n:
        kind = labels[i]
        j = i + 1
        while j < n and labels[j] == kind:
            j += 1

        start_s = float(ts_arr[i])
        end_s = float(ts_arr[j - 1]) + 1.0
        dur = end_s - start_s
        chunk_spd = spd_arr[i:j]

        seg = SessionSegment(
            idx=len(segs),
            kind=kind,
            start_s=start_s,
            end_s=end_s,
            duration_s=dur,
            distance_m=float(np.sum(chunk_spd)),
            avg_speed_ms=float(np.mean(chunk_spd)) if len(chunk_spd) else None,
        )
        segs.append(seg)
        i = j

    segs = _assign_warmup_cooldown(segs)
    return _merge_tiny(segs)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assign_warmup_cooldown(segs: list[SessionSegment]) -> list[SessionSegment]:
    work_indices = [i for i, s in enumerate(segs) if s.kind == "work"]
    if not work_indices:
        return segs
    first_work = work_indices[0]
    last_work = work_indices[-1]
    for i in range(first_work):
        if segs[i].kind in ("recovery", "rest"):
            segs[i].kind = "warmup"
    for i in range(last_work + 1, len(segs)):
        if segs[i].kind in ("recovery", "rest"):
            segs[i].kind = "cooldown"
    return segs


def _merge_tiny(segs: list[SessionSegment]) -> list[SessionSegment]:
    from kairos.config import MIN_SEGMENT_S

    if not segs:
        return segs

    merged: list[SessionSegment] = []
    for seg in segs:
        if merged and seg.duration_s < MIN_SEGMENT_S:
            prev = merged[-1]
            wa, wb = prev.duration_s, seg.duration_s
            total = wa + wb

            def _wavg(a: float | None, b: float | None) -> float | None:
                if a is None and b is None:
                    return None
                return ((a or 0.0) * wa + (b or 0.0) * wb) / total

            merged[-1] = SessionSegment(
                idx=prev.idx,
                kind=prev.kind,
                start_s=prev.start_s,
                end_s=seg.end_s,
                duration_s=total,
                distance_m=(prev.distance_m or 0.0) + (seg.distance_m or 0.0)
                            if prev.distance_m is not None or seg.distance_m is not None else None,
                avg_speed_ms=_wavg(prev.avg_speed_ms, seg.avg_speed_ms),
                avg_hr=_wavg(prev.avg_hr, seg.avg_hr),
                avg_gct_ms=_wavg(prev.avg_gct_ms, seg.avg_gct_ms),
                gct_cv=prev.gct_cv,
            )
        else:
            merged.append(seg)

    for i, s in enumerate(merged):
        s.idx = i
    return merged


# ---------------------------------------------------------------------------
# Enrich segments with per-second GCT / HR from record stream
# ---------------------------------------------------------------------------

def enrich_segments_with_records(
    segments: list[SessionSegment],
    records: list[dict[str, Any]],
) -> list[SessionSegment]:
    """
    Fill avg_gct_ms, gct_cv, and avg_hr for each segment using per-second records.
    Records must have 't_s'.
    """
    for seg in segments:
        gcts: list[float] = []
        hrs: list[float] = []
        for r in records:
            t = r.get("t_s")
            if t is None:
                continue
            if seg.start_s <= float(t) < seg.end_s:
                gct = r.get("ground_contact_time")
                hr = r.get("heart_rate")
                if gct is not None and float(gct) > 0:
                    gcts.append(float(gct))
                if hr is not None:
                    hrs.append(float(hr))
        if gcts:
            mean_gct = float(np.mean(gcts))
            seg.avg_gct_ms = mean_gct
            if len(gcts) > 1:
                seg.gct_cv = float(np.std(gcts, ddof=1) / mean_gct)
        if hrs and seg.avg_hr is None:
            seg.avg_hr = float(np.mean(hrs))
    return segments


# ---------------------------------------------------------------------------
# Session-level queries
# ---------------------------------------------------------------------------

def is_interval_session(session_type: str, segments: list[SessionSegment]) -> bool:
    from kairos.config import INTERVAL_MIN_WORK_SEGMENTS
    work_count = sum(1 for s in segments if s.kind == "work")
    return session_type == "interval" or work_count >= INTERVAL_MIN_WORK_SEGMENTS


def segment_at(segments: list[SessionSegment], t_s: float) -> SessionSegment | None:
    for seg in segments:
        if seg.start_s <= t_s < seg.end_s:
            return seg
    return None


def preceding_work_segment(
    segments: list[SessionSegment],
    seg: SessionSegment,
) -> SessionSegment | None:
    for s in reversed(segments):
        if s.end_s <= seg.start_s and s.kind == "work":
            return s
    return None
