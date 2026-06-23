"""
Phase 5 — Streamlit dashboard for Kairós.
Run with:  streamlit run kairos/dashboard.py
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from kairos.config import DATA_DIR, DB_PATH
from kairos.db import db_cursor, init_db

MARKERS_FILE = DATA_DIR / "race_markers.json"


def _load_markers() -> list[dict]:
    if not MARKERS_FILE.exists():
        return []
    try:
        return json.loads(MARKERS_FILE.read_text())
    except Exception:
        return []


def _save_markers(markers: list[dict]) -> None:
    MARKERS_FILE.write_text(json.dumps(markers, ensure_ascii=False, indent=2))

st.set_page_config(page_title="Kairós", page_icon="🏃", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Kairós")
page = st.sidebar.radio(
    "Navigation",
    ["Today", "Form Trajectory", "Trayectoria Histórica"],
)

st.sidebar.divider()
st.sidebar.subheader("Configuración de Forma")
pi_window = st.sidebar.slider("Ventana Π (días)", min_value=30, max_value=180, value=90, step=15)

st.sidebar.subheader("Rango de visualización")
today_date = date.today()
default_start = today_date - timedelta(days=90)
date_range = st.sidebar.date_input(
    "Selecciona fechas",
    value=(default_start, today_date),
    max_value=today_date
)

# Ensure date_range is a tuple with two dates
if isinstance(date_range, tuple) and len(date_range) == 2:
    view_start, view_end = date_range
else:
    view_start, view_end = default_start, today_date

if st.sidebar.button("Recalcular Forma 🔄"):
    with st.spinner(f"Recalculando con ventana de {pi_window} días..."):
        from kairos.model.fitness_fatigue import rebuild_form
        rebuild_form(window_days=pi_window)
    st.sidebar.success("Forma reconstruida correctamente.")

st.sidebar.divider()
st.sidebar.subheader("Marcadores de Carreras")

_markers = _load_markers()

if _markers:
    for _i, _m in enumerate(_markers):
        _col_lbl, _col_del = st.sidebar.columns([4, 1])
        with _col_lbl:
            st.caption(f"{_m['date']}  {_m['label']}")
        with _col_del:
            if st.button("✕", key=f"del_marker_{_i}", help="Eliminar"):
                _markers.pop(_i)
                _save_markers(_markers)
                st.rerun()
else:
    st.sidebar.caption("Sin marcadores aún.")

with st.sidebar.expander("+ Agregar marcador"):
    _new_date = st.date_input("Fecha", value=date.today(), key="marker_date_input")
    _new_label = st.text_input("Nombre", placeholder="ej. 10K La Dehesa", key="marker_label_input")
    if st.button("Agregar", key="add_marker_btn"):
        if _new_label.strip():
            _markers.append({"date": _new_date.isoformat(), "label": _new_label.strip()})
            _save_markers(_markers)
            st.rerun()
        else:
            st.warning("Ingresa un nombre para el marcador.")

db_path = DB_PATH
init_db(db_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_year_markers(fig: "go.Figure", dates_str: list[str], n_rows: int) -> None:
    """Add a vertical dashed line + year label at Jan 1 of each year in the range."""
    if not dates_str:
        return
    start, end = dates_str[0], dates_str[-1]
    years = range(int(start[:4]), int(end[:4]) + 1)
    for year in years:
        jan1 = f"{year}-01-01"
        if jan1 < start or jan1 > end:
            continue
        fig.add_vline(
            x=jan1,
            line=dict(color="rgba(255,255,255,0.20)", width=1, dash="dot"),
            annotation_text=str(year),
            annotation_position="top left",
            annotation=dict(font=dict(size=11, color="rgba(255,255,255,0.55)"), showarrow=False),
        )


def _add_race_markers(fig: "go.Figure", markers: list[dict], dates_str: list[str]) -> None:
    """Add vertical lines for user-defined race markers within the visible date range."""
    if not markers or not dates_str:
        return
    start, end = dates_str[0], dates_str[-1]
    for m in markers:
        d = m["date"]
        if d < start or d > end:
            continue
        fig.add_vline(
            x=d,
            line=dict(color="rgba(255,210,0,0.75)", width=1.5, dash="dashdot"),
            annotation_text=m["label"],
            annotation_position="top right",
            annotation=dict(
                font=dict(size=10, color="rgba(255,210,0,0.90)"),
                showarrow=False,
                textangle=-90,
            ),
        )


def _load_form_history(start_date: date, end_date: date):
    with db_cursor(db_path) as cur:
        return cur.execute(
            "SELECT date, g, h, freshness, Pi, Pi_abs, g_aerobic, g_hii "
            "FROM form_state WHERE date >= ? AND date <= ? ORDER BY date",
            (start_date.isoformat(), end_date.isoformat()),
        ).fetchall()


# ---------------------------------------------------------------------------
# Page: Today
# ---------------------------------------------------------------------------

if page == "Today":
    st.title("Today's Form")

    today = date.today()
    col1, col2, col3, col4 = st.columns(4)

    from kairos.model.fitness_fatigue import compute_form_state
    try:
        fs = compute_form_state(today, db_path=db_path)

        with col1:
            st.metric("g(t) CTL — Carga crónica", f"{fs.g:.1f}")
        with col2:
            st.metric("h(t) ATL — Carga aguda", f"{fs.h:.1f}")
        with col3:
            sign = "+" if fs.freshness >= 0 else ""
            st.metric("TSB — Frescura", f"{sign}{fs.freshness:.1f}")

        with db_cursor(db_path) as cur:
            today_row = cur.execute(
                "SELECT Pi, Pi_abs, g_aerobic, g_hii FROM form_state WHERE date = ?",
                (today.isoformat(),),
            ).fetchone()

        if today_row:
            pi_abs = float(today_row["Pi_abs"] or 0.0)
            pi_rel = float(today_row["Pi"] or 0.0)
            g_aer  = today_row["g_aerobic"]
            g_hii_ = today_row["g_hii"]
        else:
            pi_abs, pi_rel, g_aer, g_hii_ = 0.0, 0.0, None, None

        with col4:
            st.metric("Π(t) — Tendencia", f"{pi_rel:.3f}", help="Positivo = mejorando condición física (CTL subiendo). Negativo = perdiendo forma o sobreentrenando. Basado en momentum de CTL a 28 días.")
            st.metric("Π_abs — Nivel de corredor", f"{pi_abs:.0f}", help="Nivel acumulado de forma física. Sube cada día que entrenás (más rápido si Π > 0), sube lento en descanso. Primer año ≈ 100–200, tras años de entrenamiento sostenido ≈ 700–900.")

        if g_aer is not None or g_hii_ is not None:
            col5, col6 = st.columns(2)
            with col5:
                val = f"{float(g_aer):.1f}" if g_aer is not None else "—"
                st.metric("g_aer — CTL aeróbico", val, help="Carga crónica aeróbica (τ=42 d). Acumula solo los segundos por debajo de v_LT2 / HR@LT2.")
            with col6:
                val = f"{float(g_hii_):.1f}" if g_hii_ is not None else "—"
                st.metric("g_hii — CTL alta intensidad", val, help="Carga crónica de alta intensidad (τ=21 d). Acumula segundos por encima de v_LT2 / HR@LT2.")


    except Exception as e:
        st.error(f"Error computing form: {e}")


# ---------------------------------------------------------------------------
# Page: Form Trajectory
# ---------------------------------------------------------------------------

elif page == "Form Trajectory":
    st.title("Form Trajectory (Evolución de Índices)")
    st.write(f"Visualizando desde **{view_start}** hasta **{view_end}**")

    rows = _load_form_history(view_start, view_end)

    if not rows:
        st.info("No hay datos en este rango. Ejecuta 'Recalcular Forma' o sincroniza datos.")
    else:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        dates_str  = [r["date"] for r in rows]
        pi_vals    = [r["Pi"] for r in rows]
        pi_abs_vals = [r["Pi_abs"] if r["Pi_abs"] is not None else 0.0 for r in rows]
        fresh_vals = [r["freshness"] for r in rows]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.33, 0.33, 0.33],
            subplot_titles=(
                "Π (Tendencia) — Mejorando (+) vs Perdiendo forma / Sobreentrenando (−)",
                "Π_abs — Nivel de corredor (acumulado histórico)",
                "TSB — Frescura (Balance de carga)"
            ),
        )

        # --- Row 1: Π Relativo ---
        fig.add_trace(go.Scatter(
            x=dates_str, y=pi_vals,
            mode="lines",
            name="Π Relativo",
            line=dict(color="#5e16eb", width=2),
            hovertemplate="<b>%{x}</b><br>Π Relativo: <b>%{y:.3f}</b><extra></extra>",
        ), row=1, col=1)
        fig.add_hline(y=0, line=dict(color="red", dash="dash", width=1), row=1, col=1)

        # --- Row 2: Π Absoluto ---
        fig.add_trace(go.Scatter(
            x=dates_str, y=pi_abs_vals,
            mode="lines",
            name="Π Absoluto",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="<b>%{x}</b><br>Π Absoluto: <b>%{y:.1f}</b><extra></extra>",
        ), row=2, col=1)

        # --- Row 3: Freshness / TSB ---
        fig.add_trace(go.Scatter(
            x=dates_str, y=fresh_vals,
            mode="lines",
            name="TSB",
            line=dict(color="#4CAF50", width=1.5, dash="dot"),
            hovertemplate="<b>%{x}</b><br>TSB: <b>%{y:+.1f}</b><extra></extra>",
        ), row=3, col=1)
        fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1), row=3, col=1)

        _add_year_markers(fig, dates_str, n_rows=3)
        _add_race_markers(fig, _load_markers(), dates_str)
        fig.update_layout(height=800, showlegend=False, margin=dict(t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)



# ---------------------------------------------------------------------------
# Page: Trayectoria Histórica
# ---------------------------------------------------------------------------

elif page == "Trayectoria Histórica":
    st.title("Trayectoria Histórica")
    st.write(f"Visualizando desde **{view_start}** hasta **{view_end}**")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from kairos.config import HISTORY_START

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info("Ajusta el rango en la barra lateral para filtrar los datos.")
    with col_b:
        smooth_days = st.selectbox("Suavizado", [1, 3, 7, 14], index=2)

    rows = _load_form_history(view_start, view_end)

    if not rows:
        st.info("Sin datos en ese rango. Corre rebuild-form primero.")
    else:
        dates_str   = [r["date"] for r in rows]
        pi_vals     = [r["Pi"] for r in rows]
        pi_abs_vals = [r["Pi_abs"] if r["Pi_abs"] is not None else 0.0 for r in rows]
        fresh_vals  = [r["freshness"] for r in rows]
        g_vals      = [r["g"] for r in rows]
        h_vals      = [r["h"] for r in rows]
        g_aer_vals  = [r["g_aerobic"] for r in rows]
        g_hii_vals  = [r["g_hii"] for r in rows]

        import pandas as _pd
        def _smooth(vals, w):
            if w <= 1:
                return vals
            s = _pd.Series(vals, dtype=float).rolling(w, min_periods=1, center=True).mean()
            return s.tolist()

        pi_display    = _smooth(pi_vals, smooth_days)
        pi_abs_display = _smooth(pi_abs_vals, smooth_days)
        fresh_display = _smooth(fresh_vals, smooth_days)

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.25, 0.25, 0.25, 0.25],
            subplot_titles=(
                "Π (Tendencia) — Mejorando (+) vs Perdiendo forma / Sobreentrenando (−)",
                "Π_abs — Nivel de corredor (acumulado histórico)",
                "TSB — Frescura",
                "CTL / ATL — global (sólido) y por dominio aeróbico/HII (punteado)",
            ),
        )

        # --- Row 1: Π Relativo ---
        fig.add_trace(go.Scatter(
            x=dates_str, y=pi_display, mode="lines",
            name="Π (Relativo)",
            line=dict(color="#5e16eb", width=2),
            hovertemplate="<b>%{x}</b><br>Π Relativo: %{y:.3f}<extra></extra>"
        ), row=1, col=1)
        fig.add_hline(y=0, line=dict(color="red", dash="dash", width=1), row=1, col=1)
        
        # --- Row 2: Π Absoluto ---
        fig.add_trace(go.Scatter(
            x=dates_str, y=pi_abs_display, mode="lines",
            name="Π_abs (Absoluto)",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="<b>%{x}</b><br>Nivel: %{y:.0f}<extra></extra>"
        ), row=2, col=1)
        
        # --- Row 3: TSB / freshness ---
        fig.add_trace(go.Scatter(
            x=dates_str + dates_str[::-1],
            y=[max(v, 0) for v in fresh_display] + [0]*len(fresh_display),
            fill="toself", fillcolor="rgba(76,175,80,0.18)",
            line=dict(width=0), name="TSB positivo", hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=dates_str + dates_str[::-1],
            y=[min(v, 0) for v in fresh_display] + [0]*len(fresh_display),
            fill="toself", fillcolor="rgba(244,67,54,0.15)",
            line=dict(width=0), name="TSB negativo", hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=dates_str, y=fresh_display, mode="lines",
            name="TSB — Frescura",
            line=dict(color="#4CAF50", width=1.2, dash="dot"),
            customdata=fresh_vals,
            hovertemplate="<b>%{x}</b><br>TSB suavizado: <b>%{y:+.1f}</b>  (diario: %{customdata:+.1f})<extra></extra>",
        ), row=3, col=1)
        fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=0.8), row=3, col=1)

        # --- Row 4: CTL + ATL + dominios aeróbico/HII ---
        fig.add_trace(go.Scatter(
            x=dates_str, y=g_vals, mode="lines",
            name="g(t) CTL",
            line=dict(color="#1565C0", width=1.8),
            hovertemplate="<b>%{x}</b><br>CTL: <b>%{y:.1f}</b><extra></extra>",
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dates_str, y=h_vals, mode="lines",
            name="h(t) ATL",
            line=dict(color="#C62828", width=1.5, dash="dash"),
            hovertemplate="<b>%{x}</b><br>ATL: <b>%{y:.1f}</b><extra></extra>",
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dates_str, y=g_aer_vals, mode="lines",
            name="g_aer — CTL aeróbico",
            line=dict(color="#42A5F5", width=1.2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>g_aer: <b>%{y:.1f}</b><extra></extra>",
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dates_str, y=g_hii_vals, mode="lines",
            name="g_hii — CTL alta int.",
            line=dict(color="#EF9A9A", width=1.2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>g_hii: <b>%{y:.1f}</b><extra></extra>",
        ), row=4, col=1)

        _add_year_markers(fig, dates_str, n_rows=4)
        _add_race_markers(fig, _load_markers(), dates_str)
        fig.update_layout(
            height=900,
            hovermode="x unified",
            showlegend=True,
            xaxis=dict(showticklabels=False),
            xaxis2=dict(showticklabels=False),
            xaxis3=dict(showticklabels=False),
            xaxis4=dict(showticklabels=False),
            margin=dict(l=40, r=20, t=40, b=10),
        )
        fig.update_yaxes(title_text="Π rel", row=1, col=1)
        fig.update_yaxes(title_text="Π abs", row=2, col=1)
        fig.update_yaxes(title_text="TSB", row=3, col=1)
        fig.update_yaxes(title_text="TRIMP", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

        valid_pi = [v for v in pi_vals if v is not None]
        if valid_pi:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Π máximo", f"{max(valid_pi):.3f}")
            c2.metric("Π mínimo", f"{min(valid_pi):.3f}")
            c3.metric("CTL actual", f"{g_vals[-1]:.1f}")
            c4.metric("TSB actual", f"{fresh_vals[-1]:+.1f}")
            last_aer = next((v for v in reversed(g_aer_vals) if v is not None), None)
            last_hii = next((v for v in reversed(g_hii_vals) if v is not None), None)
            c5.metric("g_aer actual", f"{last_aer:.1f}" if last_aer is not None else "—")
            c6.metric("g_hii actual", f"{last_hii:.1f}" if last_hii is not None else "—")

