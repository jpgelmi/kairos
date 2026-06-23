# Correspondencia ecuaciones ↔ código — Kairós

Este documento mapea cada ecuación de `final.tex` a su implementación exacta en el código.

---

## Ecuación 1: Reserva de FC (HRr)

**final.tex §5.1 ec. (1)**:
```
x(s) = clamp((HR(s) − HR_rest) / (HR_max − HR_rest), 0, 1)
```

**Código**: `kairos/physio/trimp.py:banister_trimp()` líneas 57–60
```python
hrr = (hr - hrrest) / (hrmax - hrrest)
hrr = max(0.0, min(1.0, hrr))
```

---

## Ecuación 2: TRIMP de Banister

**final.tex §5.1 ec. (2)**:
```
L_Ban = (1/60) Σ_s x(s)·b₀·e^{b₁·x(s)},   b₀=0.64, b₁=1.92
```

**Código**: `kairos/physio/trimp.py:banister_trimp()` líneas 61–62
```python
total += hrr * 0.64 * np.exp(1.92 * hrr)
return total / 60.0
```

---

## Ecuación 3: Detección de HR_max con filtro de artefactos

**final.tex §5.1 ec. (3)**:
```
S = {HR_max^(i) : HR_max^(i) ≤ 1.35·HR_avg^(i)}
HR_max_hat = P₉₉(S)
```

**Código**: `kairos/physio/trimp.py:detect_hrmax()` líneas 88–105
```python
_SPIKE_RATIO = 1.35
vals = [float(r["max_hr"]) for r in rows
        if float(r["max_hr"]) <= float(r["avg_hr"]) * _SPIKE_RATIO]
return float(np.percentile(vals, 99))
```

---

## Ecuaciones 8–9: Separación de dominios aeróbico / alta intensidad

**final.tex §5.2 ec. (8)–(9)**:
```
L_aer = (1/60) Σ_{s: φ(s)<θ} x(s)·b₀·e^{b₁·x(s)}
L_hii = (1/60) Σ_{s: φ(s)≥θ} x(s)·b₀·e^{b₁·x(s)}
```
- Terreno llano: `(φ, θ) = (v(s), v_LT2)`
- Con desnivel: `(φ, θ) = (HR(s), HR@LT2)`

**Código**: `kairos/model/fitness_fatigue.py:compute_domain_trimp_from_records()`

Los umbrales `v_lt2` y `hr_lt2` se leen de `lactate_thresholds` mediante `get_threshold_at()`.
Si no hay umbral disponible para la fecha, la sesión no contribuye a los acumuladores de dominio.

**CLI para registrar umbrales**: `kairos add-threshold YYYY-MM-DD --v-lt2 N.NN --hr-lt2 NNN`

---

## Ecuaciones 5–6: PMC EWMA (CTL/ATL)

**final.tex §5.3 ec. (5)–(6)**:
```
g(t) = g(t−1)·e^{−1/τ_g} + L(t)·(1−e^{−1/τ_g}),   τ_g=42
h(t) = h(t−1)·e^{−1/τ_h} + L(t)·(1−e^{−1/τ_h}),   τ_h=7
```

**Código**: `kairos/model/fitness_fatigue.py:ewma_step()` líneas 60–62
```python
g = g_prev * _DG + load * (1 - _DG)   # _DG = exp(-1/42)
h = h_prev * _DH + load * (1 - _DH)   # _DH = exp(-1/7)
```

---

## Ecuación 7: TSB (frescura)

**final.tex §5.3 ec. (7)**:
```
TSB(t) = g(t−1) − h(t−1)
```

**Código**: `kairos/model/fitness_fatigue.py:freshness()` línea 66
```python
def freshness(g_prev, h_prev):
    return g_prev - h_prev
```

Importante: usa los valores del día ANTERIOR a la actualización EWMA, no los de hoy.

---

## Ecuaciones 10–13: Índice de forma relativo Π_rel

**final.tex §5.4 ec. (10)–(13)**:
```
r(j)    = (L(j) − g(j−1)) / τ_g                         [ec. 10]
r̄_7(t) = (1/7) Σ_{k=t−6}^{t} r(k)                      [ec. 11]
z_mom   = (r̄_7 − μ₉₀[r̄_7]) / max(σ₉₀[r̄_7], σ_min)   [ec. 12]
Π_rel(t) = tanh(z_mom / 1.5)                             [ec. 13]
```

**Código**: `kairos/model/fitness_fatigue.py:_calc_pi_rel()` y `rebuild_form()`

- `_RAMP_SMOOTH = 7` implementa el promedio de 7 días (ec. 11)
- `_PI_WINDOW = 90` implementa la ventana de 90 días para μ₉₀, σ₉₀ (ec. 12)
- `_SIGMA_FLOOR = 0.10` es σ_min (ec. 12)
- `_zsafe()` calcula el z-score con piso de varianza

---

## Ecuación 14: Odómetro Π_abs

**final.tex §5.5 ec. (14)**:
```
Π_abs(t) = Π_abs(t−1) + α·r̄_7(t)                              si día con sesión
Π_abs(t) = Π_abs(t−1)·[1−(1−e^{−(d−d₀)/τ_dt})·ρ]             si descanso, d>d₀
Π_abs(t) = Π_abs(t−1)                                           si descanso, d≤d₀
```

Parámetros: α=5.0, d₀=7, τ_dt=21, ρ=0.0025

**Código**: `kairos/model/fitness_fatigue.py:rebuild_form()` en el bucle principal
```python
if has_session:
    pi_abs += _PI_ABS_SCALE * ramp_today   # α·r̄_7(t), bidireccional
    pi_abs = max(_FLOOR, pi_abs)
else:
    consecutive_rest += 1
    if consecutive_rest > _DEAD_ZONE:
        decay_factor = 1.0 - math.exp(-(consecutive_rest - _DEAD_ZONE) / _TAU_DETRAIN)
        pi_abs -= decay_factor * _CHRONIC_RATE * max(0.0, pi_abs - _FLOOR)
```

**Nota importante**: la ganancia en días de entrenamiento es **bidireccional**: si `r̄_7(t) < 0`
(rampa negativa, carga crónica decayendo), `Π_abs` **baja**. Esto es consistente con
el texto del documento: "en un día de entrenamiento el nivel **baja** si la rampa es negativa".

---

## Ecuación 15: GCT drift

**final.tex §5.6 ec. (15)**:
```
drift_GCT = (med(GCT_20%_final) − med(GCT_20%_inicial)) / med(GCT_20%_inicial)
```

**Código**: `kairos/physio/dynamics.py:compute_gct_drift()`
```python
seg = max(1, n // 5)   # 20%
gct_init  = float(np.median(arr[:seg]))
gct_final = float(np.median(arr[-seg:]))
return (gct_final - gct_init) / gct_init
```

Para sesiones de intervalos: `gct_drift_from_segments()` compara primera vs. última
repetición de distancia comparable (±5%) o velocidad comparable (±5%).

---

## Ecuación 16: HRR₆₀ y τ_HRR

**final.tex §5.6 ec. (16)**:
```
HRR₆₀ = HR_pico − HR₆₀s
HR(t) = HR_rec + (HR_pico − HR_rec)·e^{−t/τ_HRR}
```
Solo si el esfuerzo alcanzó ≥ 0.85·HR_max.

**Código**: `kairos/physio/recovery.py:compute_hrr_transition()`
- `hrr60 = hr_peak - hr_at_60`
- `_fit_tau()` ajusta la curva exponencial (scipy.optimize.curve_fit, con fallback log-lineal)
- `HRR_MIN_PEAK_PCT = 0.85` en `kairos/config.py`

---

## Componentes adicionales (extensiones fuera del modelo de final.tex)

Estos módulos implementan funcionalidades útiles pero no forman parte del modelo
matemático especificado en `final.tex`:

| Módulo | Función |
|--------|---------|
| `physio/efficiency.py` | Factor de eficiencia aeróbica (GAP/FC) y decoupling |
| `physio/mmp.py` | Curva MMP y velocidad crítica (CS/D') desde datos de campo |
| `model/limiters.py` | Motor de recomendaciones build/peak (extensión del descriptor) |

---

## Decisiones de implementación

### HR_rest = 62 bpm

`final.tex §5.1` indica 62 bpm como prior inicial, citando que la media de sesión nunca
baja de ~76 bpm y por tanto no permite estimar el verdadero reposo. El valor se puede
sobrescribir en `config.py` con `HR_REST = <tu_valor>`.

### τ_h en acumuladores de dominio

El documento especifica `(τ_g, τ_h) = (42, 7)` para aeróbico y `(21, 7)` para HII.
Estos se implementan como constantes en `config.py` (`TAU_G_AEROBIC`, `TAU_H_AEROBIC`,
`TAU_G_HII`, `TAU_H_HII`) para facilitar futuros análisis de sensibilidad.

### Sesiones sin umbral de dominio

Si no hay test de lactato registrado para la fecha de una sesión, esa sesión no contribuye
a los acumuladores `g_aer`/`g_hii`. Esto implementa el principio de honestidad epistémica:
no se fabrica señal sobre información ausente.

### F_hat y peak_pred_date en form_state

La tabla `form_state` tiene columnas `F_hat` y `peak_pred_date` de una versión anterior
del modelo. `F_hat` no está definido en `final.tex` y nunca se escribe en la implementación
actual. `peak_pred_date` tampoco. Ambas columnas se mantienen en el esquema por compatibilidad
con la DB existente, pero están vacías (NULL).
