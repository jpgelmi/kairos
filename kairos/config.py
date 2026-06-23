"""
Fixed constants for Kairós.  NEVER adjust these from data — see §4 of the
design brief.  Constants from the fitness-fatigue literature (Banister 1991,
Hellard et al. 2006) and meta-analyses (Bosquet et al. 2007).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KAIROS_HOME = Path.home() / ".kairos"
DATA_DIR = KAIROS_HOME / "data"
FIT_DIR = DATA_DIR / "fit"
MANUAL_DIR = DATA_DIR / "manual"
DB_PATH = KAIROS_HOME / "kairos.db"

# ---------------------------------------------------------------------------
# PMC / TSB model (EWMA-based, constants fixed from literature)
# This is the Performance Management Chart (CTL/ATL/TSB), NOT the full
# Banister p̂ = p0 + k_g·g − k_h·h.  The gain factors k_g and k_h are not
# implemented because they cannot be reliably estimated from ~150 data points.
# ---------------------------------------------------------------------------
TAU_G: float = 42.0   # fitness time-constant / CTL window (days)
TAU_H: float = 7.0    # fatigue time-constant / ATL window (days)

# ---------------------------------------------------------------------------
# Session segmentation  (Feature C)
# ---------------------------------------------------------------------------
MIN_SEGMENT_S: float = 10.0               # merge segments shorter than this
INTERVAL_MIN_WORK_SEGMENTS: int = 3       # ≥ N work segments → treat as intervals
WORK_SPEED_PCTILE: int = 60              # speed-fallback percentile threshold
COMPARABLE_PACE_TOL: float = 0.05        # ±5% speed tolerance for "same-pace" reps

# ---------------------------------------------------------------------------
# Historical backfill  (ingesta histórica desde 2023-01-01)
# ---------------------------------------------------------------------------
HISTORY_START: str = "2023-01-01"             # earliest date to pull from Garmin
SYNC_CHUNK_DAYS: int = 30                     # days per backfill chunk
SYNC_SLEEP_S: float = 20.0                    # seconds to sleep between backfill chunks

# ---------------------------------------------------------------------------
# TRIMP / load metric  (objective model — Phase refactor)
# ---------------------------------------------------------------------------
LOAD_METRIC: str = "banister_trimp"   # 'edwards_trimp' | 'banister_trimp' | 'te_proxy'
# Banister: continuous HRR exponential — no binning artifact, physiologically grounded.
# Edwards: binned zone-minutes — legacy, kept for reference.
# TE proxy: fallback when no HR stream (unchanged regardless of LOAD_METRIC).
HR_MAX: float | None = None           # override auto-detection (spike-filtered percentile 99 of max_hr)
HR_REST = 62.0          # prior inicial (final.tex §4, Tabla 1); la media de sesión nunca baja de ~76 bpm
                        # → el verdadero reposo no es estimable desde sessions.avg_hr

# ---------------------------------------------------------------------------
# Efficiency Factor / aerobic computation
# ---------------------------------------------------------------------------
AEROBIC_HR_CEILING_PCT: float = 0.85  # FC < 85 % HRmax → steady aerobic state
EF_MIN_DURATION_S: float = 1200.0    # minimum continuous steady-aerobic seconds for EF

# ---------------------------------------------------------------------------
# HRR (Heart Rate Recovery)
# ---------------------------------------------------------------------------
HRR_MIN_PEAK_PCT: float = 0.85        # work bout must reach >= 85 % HRmax for HRR to count

# ---------------------------------------------------------------------------
# Domain-specific load EWMA (aerobic vs. high-intensity)
# τ values fixed from exercise-physiology literature:
#   Aerobic CTL-like: 42 d (chronic adaptation, same as global CTL)
#   Aerobic ATL-like:  7 d (acute aerobic load, same as global ATL)
#   HII chronic: 21 d (high-intensity adaptations are faster to acquire and lose)
#   HII acute:    7 d (same acute window; HII sessions are typically infrequent)
# These are design choices, not estimates. See D-15 in design_decisions.
# ---------------------------------------------------------------------------
TAU_G_AEROBIC: float = 42.0
TAU_H_AEROBIC: float = 7.0
TAU_G_HII: float = 21.0
TAU_H_HII: float = 7.0

