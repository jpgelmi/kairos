"""
PMC / TSB model + domain-specific load accumulators.
Implementa las ecuaciones de final.tex §5.

Carga global (TRIMP de Banister, ec. 2–3):
  g(t) = CTL  τ_g=42   (ec. 5)
  h(t) = ATL  τ_h=7    (ec. 6)
  TSB(t) = g(t-1) - h(t-1)   (ec. 7)

Acumuladores de dominio por EWMA (ec. 8–9):
  Aeróbico (φ(s) < θ): τ_g=42, τ_h=7
  Alta intensidad (φ(s) ≥ θ): τ_g=21, τ_h=7

Los dominios se calculan segundo-a-segundo usando v_LT2 o HR@LT2 de la tabla
lactate_thresholds (anclas longitudinales episódicas).  Las sesiones sin umbral
disponible no contribuyen a los acumuladores de dominio (no se imputa señal).

Índice de forma relativo Π_rel (ec. 10–13):
  r(j)      = (L(j) - g(j-1)) / τ_g
  r̄_7(t)   = media 7 días de r
  z_mom(t)  = (r̄_7 - μ_90[r̄_7]) / max(σ_90[r̄_7], σ_min)
  Π_rel(t)  = tanh(z_mom(t) / 1.5)

Odómetro Π_abs (ec. 14):
  Día con sesión : Π_abs(t-1) + α·r̄_7(t)   [bidireccional — baja si rampa < 0]
  Descanso d>d₀  : Π_abs(t-1)·[1-(1-e^{-(d-d₀)/τ_dt})·ρ]
  Descanso d≤d₀  : Π_abs(t-1)   [zona muerta: pérdida solo de volumen plasmático]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from kairos.config import (
    DB_PATH,
    HISTORY_START,
    TAU_G,
    TAU_G_AEROBIC,
    TAU_G_HII,
    TAU_H,
    TAU_H_AEROBIC,
    TAU_H_HII,
)
from kairos.db import db_cursor

_DG = math.exp(-1.0 / TAU_G)
_DH = math.exp(-1.0 / TAU_H)
_DG_AER = math.exp(-1.0 / TAU_G_AEROBIC)
_DH_AER = math.exp(-1.0 / TAU_H_AEROBIC)
_DG_HII = math.exp(-1.0 / TAU_G_HII)
_DH_HII = math.exp(-1.0 / TAU_H_HII)

# Parámetros de Π_abs (final.tex §5.5, Tabla 1)
_PI_ABS_SCALE: float = 5.0        # α: escala rampa → nivel
_CHRONIC_RATE: float = 0.0025     # ρ: 0.25 %/día (Coyle 1984, Mujika 2000)
_DEAD_ZONE: int = 7               # d₀: días sin pérdida durable
_TAU_DETRAIN: float = 21.0        # τ_dt: timescale de aceleración del desentrenamiento
_FLOOR: float = 0.0               # Π_abs no puede ser negativo

# Parámetros de Π_rel (final.tex §5.4)
_RAMP_SMOOTH: int = 7             # días de suavizado del ramp rate
_SIGMA_FLOOR: float = 0.10        # σ_min: piso de varianza (final.tex ec. 12)


# ---------------------------------------------------------------------------
# Core EWMA step  (final.tex ec. 5–6)
# ---------------------------------------------------------------------------

def ewma_step(
    g_prev: float,
    h_prev: float,
    load: float,
) -> tuple[float, float]:
    """Un paso EWMA diario. Devuelve (g_new, h_new).

    Implementa:
      g(t) = g(t-1)·e^{-1/τ_g} + L(t)·(1-e^{-1/τ_g})   [ec. 5]
      h(t) = h(t-1)·e^{-1/τ_h} + L(t)·(1-e^{-1/τ_h})   [ec. 6]
    """
    g = g_prev * _DG + load * (1 - _DG)
    h = h_prev * _DH + load * (1 - _DH)
    return g, h


def freshness(g_prev: float, h_prev: float) -> float:
    """TSB(t) = g(t-1) - h(t-1)  (ec. 7).  Usa valores ANTERIORES a la carga de hoy."""
    return g_prev - h_prev


# ---------------------------------------------------------------------------
# Simulate forward
# ---------------------------------------------------------------------------

def simulate(loads: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Simula g(t) y h(t) para una secuencia de cargas diarias.
    Devuelve (G, H) arrays de longitud len(loads).
    """
    G = np.empty(len(loads))
    H = np.empty(len(loads))
    g = h = 0.0
    for i, w in enumerate(loads):
        g, h = ewma_step(g, h, w)
        G[i] = g
        H[i] = h
    return G, H


def ewma_step_domain(
    g_aer: float, h_aer: float,
    g_hii: float, h_hii: float,
    load_aer: float, load_hii: float,
) -> tuple[float, float, float, float]:
    """Paso EWMA diario para los acumuladores de dominio (aeróbico/HII)."""
    g_aer = g_aer * _DG_AER + load_aer * (1 - _DG_AER)
    h_aer = h_aer * _DH_AER + load_aer * (1 - _DH_AER)
    g_hii = g_hii * _DG_HII + load_hii * (1 - _DG_HII)
    h_hii = h_hii * _DH_HII + load_hii * (1 - _DH_HII)
    return g_aer, h_aer, g_hii, h_hii


# ---------------------------------------------------------------------------
# Threshold lookup (anclas longitudinales episódicas — final.tex §4 y §5.2)
# ---------------------------------------------------------------------------

def get_threshold_at(
    session_date: str,
    db_path: Path = DB_PATH,
) -> tuple[float | None, float | None]:
    """Devuelve (v_lt2, hr_lt2) del test más reciente con date <= session_date.
    Retorna (None, None) si no hay ningún test registrado.
    """
    with db_cursor(db_path) as cur:
        row = cur.execute(
            "SELECT v_lt2, hr_lt2 FROM lactate_thresholds "
            "WHERE date <= ? ORDER BY date DESC LIMIT 1",
            (session_date,),
        ).fetchone()
    if row is None:
        return None, None
    v = float(row["v_lt2"]) if row["v_lt2"] is not None else None
    hr = float(row["hr_lt2"]) if row["hr_lt2"] is not None else None
    return v, hr


# ---------------------------------------------------------------------------
# Domain TRIMP split  (final.tex ec. 8–9)
# ---------------------------------------------------------------------------

def compute_domain_trimp_from_records(
    records: list[dict],
    hrmax: float,
    hrrest: float,
    v_lt2: float | None,
    hr_lt2: float | None,
) -> tuple[float | None, float | None]:
    """Calcula (L_aer, L_hii) segundo a segundo según final.tex ec. 8–9.

    Criterio de dominio (φ, θ):
      - Si hay velocidad disponible y v_lt2: φ=v(s), θ=v_LT2  (terreno llano)
      - Fallback: si hay HR y hr_lt2: φ=HR(s), θ=HR@LT2  (terreno con desnivel)
    Segundos con speed=0 (parado) se excluyen del cálculo.
    Retorna (None, None) si no hay umbral disponible o no hay registros clasificables.
    """
    if v_lt2 is None and hr_lt2 is None:
        return None, None
    if not records:
        return None, None
    if hrmax <= hrrest:
        return None, None

    b0, b1 = 0.64, 1.92
    l_aer = 0.0
    l_hii = 0.0
    n_classified = 0

    for r in records:
        hr = r.get("heart_rate")
        if hr is None:
            continue
        hr = float(hr)
        x = max(0.0, min(1.0, (hr - hrrest) / (hrmax - hrrest)))
        contrib = x * b0 * math.exp(b1 * x)

        speed = r.get("speed") or r.get("enhanced_speed")
        if speed is not None and v_lt2 is not None and float(speed) > 0.1:
            # Terreno llano: clasificar por velocidad
            if float(speed) < v_lt2:
                l_aer += contrib
            else:
                l_hii += contrib
            n_classified += 1
        elif hr_lt2 is not None:
            # Terreno con desnivel o sin GPS: clasificar por FC
            if hr < hr_lt2:
                l_aer += contrib
            else:
                l_hii += contrib
            n_classified += 1

    if n_classified == 0:
        return None, None

    return l_aer / 60.0, l_hii / 60.0



def _zsafe(arr: np.ndarray, sigma_floor: float = 0.0) -> np.ndarray:
    """Z-score seguro con piso de varianza σ_min (final.tex ec. 12)."""
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sd = max(sd, sigma_floor)
    return (arr - arr.mean()) / sd if sd > 1e-9 else np.zeros_like(arr)


# ---------------------------------------------------------------------------
# Date spine utility
# ---------------------------------------------------------------------------

def date_spine(start: date, end: date) -> list[date]:
    """Devuelve todas las fechas desde start hasta end (inclusive)."""
    spine: list[date] = []
    current = start
    while current <= end:
        spine.append(current)
        current += timedelta(days=1)
    return spine


# ---------------------------------------------------------------------------
# DB-backed state computation
# ---------------------------------------------------------------------------

@dataclass
class FormState:
    date: date
    g: float
    h: float
    freshness: float


def compute_form_state(
    target_date: date,
    db_path: Path = DB_PATH,
) -> FormState:
    """Calcula g(t), h(t) y TSB hasta target_date desde cargas derivadas de .fit."""
    date_str = target_date.isoformat()

    with db_cursor(db_path) as cur:
        sess_rows = cur.execute(
            "SELECT date, trimp, srpe FROM sessions "
            "WHERE date <= ? AND (trimp IS NOT NULL OR srpe IS NOT NULL) ORDER BY date",
            (date_str,),
        ).fetchall()

    load_dict: dict[str, float] = {}
    for r in sess_rows:
        d = r["date"]
        load = float(r["trimp"]) if r["trimp"] is not None else float(r["srpe"])
        load_dict[d] = load_dict.get(d, 0.0) + load

    if not load_dict:
        return FormState(date=target_date, g=0.0, h=0.0, freshness=0.0)

    start = date.fromisoformat(min(load_dict))
    g, h = 0.0, 0.0
    g_prev, h_prev = 0.0, 0.0
    current = start
    while current <= target_date:
        ds = current.isoformat()
        g_prev, h_prev = g, h
        load = load_dict.get(ds, 0.0)
        g, h = ewma_step(g, h, load)
        current += timedelta(days=1)

    return FormState(
        date=target_date,
        g=g,
        h=h,
        freshness=freshness(g_prev, h_prev),
    )


# ---------------------------------------------------------------------------
# Rebuild form state (historical backfill)
# ---------------------------------------------------------------------------

def rebuild_form(
    start_date: date | None = None,
    end_date: date | None = None,
    window_days: int = 90,
    db_path: Path = DB_PATH,
) -> int:
    """Reconstruye form_state con toda la columna vertebral de fechas.

    Calcula g/h/TSB/Π_rel/Π_abs para cada día calendario desde start_date hasta
    end_date usando cargas derivadas de .fit. Persiste en form_state (INSERT OR REPLACE).

    Π_rel implementa las ecuaciones 10–13 de final.tex.
    Π_abs implementa la ecuación 14 de final.tex (bidireccional: rampa negativa
    reduce Π_abs en días de entrenamiento; curva bifásica en descanso real).

    Devuelve el número de filas escritas.
    """
    start = start_date or date.fromisoformat(HISTORY_START)
    end = end_date or date.today()

    with db_cursor(db_path) as cur:
        sess_rows = cur.execute(
            "SELECT id, date, trimp, srpe, trimp_aerobic, trimp_hii FROM sessions "
            "WHERE date <= ? AND (trimp IS NOT NULL OR srpe IS NOT NULL) ORDER BY date",
            (end.isoformat(),),
        ).fetchall()

    load_dict: dict[str, float] = {}
    load_aer_dict: dict[str, float] = {}
    load_hii_dict: dict[str, float] = {}
    for r in sess_rows:
        d = r["date"]
        load = float(r["trimp"]) if r["trimp"] is not None else float(r["srpe"])
        load_dict[d] = load_dict.get(d, 0.0) + load
        if r["trimp_aerobic"] is not None:
            load_aer_dict[d] = load_aer_dict.get(d, 0.0) + float(r["trimp_aerobic"])
        if r["trimp_hii"] is not None:
            load_hii_dict[d] = load_hii_dict.get(d, 0.0) + float(r["trimp_hii"])

    spine = date_spine(start, end)
    g, h = 0.0, 0.0
    g_aer, h_aer = 0.0, 0.0
    g_hii, h_hii = 0.0, 0.0

    rows_to_write: list[tuple] = []
    for d in spine:
        ds = d.isoformat()
        g_prev, h_prev = g, h
        load = load_dict.get(ds, 0.0)
        g, h = ewma_step(g, h, load)
        fresh = freshness(g_prev, h_prev)

        # Acumuladores de dominio: en descanso decaen con load=0;
        # en sesión sin split disponible no se actualiza (no se imputa).
        l_aer = load_aer_dict.get(ds)
        l_hii = load_hii_dict.get(ds)
        has_session = ds in load_dict
        if l_aer is not None or l_hii is not None:
            g_aer, h_aer, g_hii, h_hii = ewma_step_domain(
                g_aer, h_aer, g_hii, h_hii,
                l_aer or 0.0, l_hii or 0.0,
            )
        else:
            # Descanso o sesión sin umbral clasificable: la carga de dominio es 0.
            # Los acumuladores decaen con L_aer=L_hii=0 — no se imputa señal
            # pero el tiempo sigue pasando (ec. 5-6 con L=0).
            g_aer, h_aer, g_hii, h_hii = ewma_step_domain(
                g_aer, h_aer, g_hii, h_hii, 0.0, 0.0,
            )

        rows_to_write.append((ds, g, h, fresh, g_aer, h_aer, g_hii, h_hii))

    # -------------------------------------------------------------------
    # Segunda pasada: calcular Π_rel y Π_abs sobre toda la columna vertebral
    # -------------------------------------------------------------------
    _PI_WINDOW = window_days   # ventana de 90 días para μ₉₀, σ₉₀ (ec. 12)
    G_all = np.array([r[1] for r in rows_to_write])
    L_all = np.array([load_dict.get(r[0], 0.0) for r in rows_to_write])

    pi_abs = 0.0
    consecutive_rest = 0

    final_rows: list[tuple] = []
    for i, row in enumerate(rows_to_write):
        ds = row[0]
        has_session = ds in load_dict

        # --- Π_rel: ec. 10–13 de final.tex ---
        s = max(0, i - _PI_WINDOW + 1)
        G_w = G_all[s:i + 1]
        L_w = L_all[s:i + 1]

        g_before_window = float(G_all[s - 1]) if s > 0 else 0.0
        pi, ramp_today = _calc_pi_rel(G_w, L_w, g_before_window)

        # --- Π_abs: ec. 14 de final.tex ---
        if has_session:
            # Ganancia bidireccional: r̄_7 positivo sube, negativo baja.
            # (Nota: esto difiere de una implementación intermedia que usaba max(0,...);
            # se alinea con el documento que dice explícitamente que el nivel
            # "baja si la rampa es negativa" en días de entrenamiento.)
            consecutive_rest = 0
            pi_abs += _PI_ABS_SCALE * ramp_today
            pi_abs = max(_FLOOR, pi_abs)
        else:
            # Descanso: curva bifásica de desentrenamiento (Coyle 1984, Mujika 2000)
            consecutive_rest += 1
            if consecutive_rest > _DEAD_ZONE:
                decay_factor = 1.0 - math.exp(
                    -(consecutive_rest - _DEAD_ZONE) / _TAU_DETRAIN
                )
                pi_abs -= decay_factor * _CHRONIC_RATE * max(0.0, pi_abs - _FLOOR)

        final_rows.append(row + (pi, pi_abs))

    with db_cursor(db_path) as cur:
        for row in final_rows:
            cur.execute(
                "INSERT OR REPLACE INTO form_state "
                "(date, g, h, freshness, g_aerobic, h_aerobic, g_hii, h_hii, Pi, Pi_abs) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                row,
            )

    return len(final_rows)


def _calc_pi_rel(gw: np.ndarray, lw: np.ndarray, g_before: float = 0.0) -> tuple[float, float]:
    """Calcula Π_rel y r̄_7 para la ventana de datos (gw, lw).

    Implementa las ecuaciones 10–13 de final.tex:
      r(j)     = (L(j) - g(j-1)) / τ_g           [ec. 10]
      r̄_7(j)  = media de r en los 7 días a j      [ec. 11]
      z_mom(t) = (r̄_7 - μ_90) / max(σ_90, σ_min) [ec. 12]
      Π_rel(t) = tanh(z_mom / 1.5)                [ec. 13]

    g_before: valor de g(t-1) para el primer día de la ventana (G_all[s-1] o 0.0
    si la ventana empieza en el origen).  Necesario para que r(j=0) use el g real
    del día anterior al inicio de la ventana en lugar de asumir g=0.

    Devuelve (pi_rel, ramp_7d_last).
    """
    n = len(gw)
    if n < 2:
        return 0.0, 0.0

    ramp = np.array([
        (lw[j] - (gw[j - 1] if j > 0 else g_before)) / TAU_G
        for j in range(n)
    ])
    ramp_7d = np.array([
        np.mean(ramp[max(0, j - _RAMP_SMOOTH + 1):j + 1])
        for j in range(n)
    ])
    z_momentum = float(_zsafe(ramp_7d, sigma_floor=_SIGMA_FLOOR)[-1])
    return float(np.tanh(z_momentum / 1.5)), float(ramp_7d[-1])
