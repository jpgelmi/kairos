# Kairós — Descriptor de estado fisiológico para corredor de media y larga distancia

Kairós es un sistema de modelado fisiológico que describe el estado de un corredor de resistencia a partir de archivos de actividad (.fit). Es un **descriptor de estado**, no un predictor de rendimiento futuro.

La especificación matemática completa, la justificación fisiológica y las limitaciones del modelo se encuentran en [`final.tex`](./final.tex).

---

## Objetivo científico

El sistema responde a la pregunta: **"¿Cómo está el atleta hoy, respecto de su propia distribución histórica?"**

Tres principios guían su diseño (final.tex §2):
1. **Descriptor de estado, no predictor** — describe el estado actual, no pronostica marcas.
2. **Calibración intraindividual** — toda magnitud se estandariza contra la distribución histórica propia.
3. **Medir en lugar de modelar** — `v_LT2` y `HR@LT2` provienen de tests directos, no de inferencia continua.

---

## Salidas del modelo

| Símbolo | Descripción | Ecuación |
|---------|-------------|----------|
| `g(t)` | Carga crónica — CTL (*fitness*), τ=42 d | ec. 5 |
| `h(t)` | Carga aguda — ATL (*fatiga*), τ=7 d | ec. 6 |
| `TSB(t)` | Balance de estrés = g(t−1) − h(t−1) | ec. 7 |
| `g_aer`, `g_hii` | Acumuladores de dominio aeróbico / alta intensidad | ec. 8–9 |
| `Π_rel(t)` | *Momentum* de forma: ¿sube la aptitud? ∈ (−1, 1) | ec. 13 |
| `Π_abs(t)` | Odómetro de nivel durable acumulado | ec. 14 |
| `drift_GCT` | Deriva del tiempo de contacto con el suelo intrasesión | ec. 15 |
| `HRR₆₀`, `τ_HRR` | Recuperación cardíaca post-esfuerzo | ec. 16 |

---

## Descripción del modelo

### TRIMP de Banister (final.tex ec. 2–3)

Reserva de FC por segundo: `x(s) = clamp((HR(s)−HR_rest)/(HR_max−HR_rest), 0, 1)`

Carga de la sesión: `L = (1/60) Σ_s x(s)·0.64·exp(1.92·x(s))`

### Separación de dominios (final.tex ec. 8–9)

Cada segundo se clasifica en aeróbico o alta intensidad:
- En terreno llano: `φ=v(s)`, `θ=v_LT2` (velocidad en LT2 del test de escalón)
- Con desnivel: `φ=HR(s)`, `θ=HR@LT2` (FC en LT2)

Los umbrales provienen de la tabla `lactate_thresholds` (ver `kairos add-threshold`).

### PMC EWMA (final.tex ec. 5–7)

```
g(t) = g(t−1)·e^{−1/42} + L(t)·(1−e^{−1/42})   ← CTL (fitness, τ=42 d)
h(t) = h(t−1)·e^{−1/7}  + L(t)·(1−e^{−1/7})    ← ATL (fatiga, τ=7 d)
TSB(t) = g(t−1) − h(t−1)                          ← frescura
```

### Π_rel: momentum de forma (final.tex ec. 10–13)

```
r(j)    = (L(j) − g(j−1)) / 42           ← rampa diaria del CTL
r̄_7(t) = media de r en los últimos 7 días
z_mom   = (r̄_7 − μ_90[r̄_7]) / max(σ_90[r̄_7], 0.10)
Π_rel(t) = tanh(z_mom / 1.5) ∈ (−1, 1)
```

### Π_abs: odómetro de nivel durable (final.tex ec. 14)

```
Día con sesión : Π_abs(t−1) + 5·r̄_7(t)          ← bidireccional
Descanso d≤7   : Π_abs(t−1)                        ← zona muerta
Descanso d>7   : Π_abs(t−1)·[1−(1−e^{−(d−7)/21})·0.0025]  ← desentrenamiento bifásico
```

### Marcadores biomecánicos (final.tex ec. 15–16)

- **GCT drift**: `(median(GCT_final_20%) − median(GCT_init_20%)) / median(GCT_init_20%)`
- **HRR₆₀**: caída de FC en 60 s post-pico (solo si pico ≥ 0.85·HR_max)
- **τ_HRR**: constante de tiempo del descenso exponencial de la FC de recuperación

---

## Parámetros fijos (final.tex §4, Tabla 1)

| Parámetro | Valor | Fuente |
|-----------|-------|--------|
| τ_g | 42 d | Banister (1991) |
| τ_h | 7 d | Banister (1991) |
| τ_g^HII | 21 d | Diseño |
| b₀, b₁ | 0.64, 1.92 | Banister (1991) |
| α | 5.0 | Diseño |
| σ_min | 0.10 | Diseño |
| HR_rest prior | 62 bpm | Diseño |
| Filtro spike | 1.35 | Diseño |
| Umbral HRR | 0.85·HR_max | Stanley (2013) |
| ρ | 0.0025/d | Coyle (1984), Mujika (2000) |
| d₀ | 7 d | Mujika (2000) |
| τ_dt | 21 d | Diseño |

---

## Arquitectura del software

```
kairos/
├── config.py              — Parámetros fijos del modelo (final.tex Tabla 1)
├── db.py                  — Esquema SQLite y gestión de conexiones
├── cli.py                 — Interfaz de línea de comandos (typer)
├── ingest/
│   ├── fit_parser.py      — Parseo de archivos .fit (Garmin)
│   ├── garmin_sync.py     — Descarga desde Garmin Connect
│   ├── loaders.py         — Inserción en BD; pipeline de ingesta
│   └── quality.py         — Verificación de calidad de datos
├── physio/
│   ├── trimp.py           — TRIMP de Banister (ec. 2–3), HRmax/HRrest (ec. 4)
│   ├── dynamics.py        — GCT drift (ec. 15)
│   ├── recovery.py        — HRR₆₀ y τ_HRR (ec. 16)
│   ├── segmentation.py    — Segmentación work/recovery/warmup/cooldown
│   ├── efficiency.py      — Factor de eficiencia aeróbica [extensión, fuera del modelo]
│   └── mmp.py             — Curva MMP y velocidad crítica [extensión, fuera del modelo]
├── model/
│   ├── fitness_fatigue.py — PMC EWMA, Π_rel, Π_abs (ec. 5–14)
│   └── limiters.py        — Motor de recomendaciones build/peak [extensión]
└── tests/                 — Suite de 117 tests
```

---

## Flujo de ejecución

```
.fit files
   │
   ▼ ingest/fit_parser.py
parse_fit()
   │
   ├──► physio/trimp.py      banister_trimp()    → sessions.trimp
   ├──► model/fitness_fatigue  compute_domain_trimp_from_records()  → sessions.trimp_aerobic/hii
   ├──► physio/dynamics.py   compute_gct_drift() → running_dynamics.gct_drift_pct
   └──► physio/recovery.py   session_hrr()       → running_dynamics.hrr60/tau_hrr
                                    │
                             model/fitness_fatigue.rebuild_form()
                             (ec. 5–14)
                                    │
                             ┌──────▼──────────────────┐
                             │       form_state          │
                             │  g, h, TSB, Π_rel, Π_abs │
                             └──────────────────────────┘
                                    │
                             cli.py diagnose() → Recomendaciones
```

---

## Instalación

```bash
pipx install git+https://github.com/jpgelmi/kairos.git
```

> Requiere [pipx](https://pipx.pypa.io). Si no lo tienes: `brew install pipx` (macOS) o `pip install pipx`.

<details>
<summary>Alternativas</summary>

```bash
# uv (moderno)
uv tool install git+https://github.com/jpgelmi/kairos.git

# desarrollo local
git clone https://github.com/jpgelmi/kairos.git
cd kairos
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
</details>

---

## Flujo típico de uso

```bash
# 1. Setup inicial
kairos db-init

# 2. Descarga histórica desde Garmin Connect
kairos backfill --since 2023-01-01

# 3. Calcular TRIMP para todas las sesiones
kairos rebuild-trimp

# 4. Registrar tests de escalón de lactato (anclas para split aeróbico/HII)
kairos add-threshold 2024-03-15 --v-lt2 3.83 --hr-lt2 162
kairos add-threshold 2024-09-01 --v-lt2 3.95 --hr-lt2 165   # test posterior

# 5. Calcular split aeróbico/HII retroactivamente
kairos rebuild-domain-trimp

# 6. Reconstruir estado de forma completo
kairos rebuild-form

# 7. Consulta diaria
kairos form                        # CTL, ATL, TSB, Π_rel, Π_abs + recomendaciones
kairos state                       # última sesión: GCT drift, HRR60

# 8. Después de cada entrenamiento
kairos sync --days 2
kairos rebuild-trimp && kairos rebuild-domain-trimp && kairos rebuild-form

# Otros comandos
kairos snapshot 2024-06-01         # forma en una fecha pasada
kairos peak --top 5                # mejores 5 días del año
kairos form --race 2026-09-01 --event 5000   # modo peak
```

---

## Variables de entrada

| Variable | Unidad | Fuente |
|----------|--------|--------|
| `HR(s)` | bpm | Registros por segundo del .fit (banda pectoral u óptico) |
| `v(s)` | m/s | GPS/acelerómetro del .fit |
| `GCT(s)` | ms | Dinámica de carrera del .fit |
| `v_LT2` | m/s | Tests periódicos de escalón de lactato (`add-threshold`) |
| `HR@LT2` | bpm | Tests periódicos de escalón de lactato (`add-threshold`) |

---

## Limitaciones

1. **Un solo sujeto**: los parámetros de la literatura (b₀, b₁, τ_g, τ_h) no se estiman del individuo.
2. **Vigencia de umbrales**: el split de dominios usa el test más reciente; un cambio real de LT2 entre tests introduce sesgo transitorio.
3. **Estacionariedad local**: las ventanas de 42 y 90 días asumen que la distribución propia cambia lentamente.
4. **HR_rest**: se inicializa en 62 bpm; la media de sesión nunca captura el verdadero reposo.

Ver `final.tex §6–7` para el protocolo completo de análisis de sensibilidad.

---

## Tests

```bash
pytest kairos/tests/ -v
pytest kairos/tests/ --cov=kairos --cov-report=term-missing
```

---

## Estructura de directorios

```
kairos/
├── final.tex              — Especificación matemática completa del modelo
├── README.md              — Este archivo
├── pyproject.toml         — Configuración del paquete Python
├── .env.example           — Plantilla de credenciales (copiar a .env)
├── kairos/                — Código fuente
├── data/
│   └── race_markers.example.json  — Ejemplo de marcadores de carrera
└── docs/                  — Documentación técnica adicional
```

Los datos personales (`kairos.db`, `data/fit/`, `.env`) se guardan en `~/.kairos/` y nunca se versionan.
