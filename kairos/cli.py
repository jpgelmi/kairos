"""
Kairós CLI — all commands in one file, growing phase by phase.
Run with:  python -m kairos.cli <command>
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from kairos.config import KAIROS_HOME

load_dotenv(KAIROS_HOME / ".env")

_console = Console()

_LOGO = """\
  ██╗  ██╗ █████╗ ██╗██████╗  ██████╗ ███████╗
  ██║ ██╔╝██╔══██╗██║██╔══██╗██╔═══██╗██╔════╝
  █████╔╝ ███████║██║██████╔╝██║   ██║███████╗
  ██╔═██╗ ██╔══██║██║██╔══██╗██║   ██║╚════██║
  ██║  ██╗██║  ██║██║██║  ██║╚██████╔╝███████║
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝"""

app = typer.Typer(name="kairos", help="Kairós running performance system.")


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _console.print(_LOGO, style="bold cyan")
        _console.print()
        _console.print("  Hecho por Juan Pablo Gelmi  ·  [bold]n + 1[/bold]")
        _console.print()
        _console.print("  [dim]Disponible para conexión automática con Garmin Connect[/dim]")
        _console.print()
        typer.echo(ctx.get_help())


def _pace_str(v_ms: float) -> str:
    """m/s → 'M:SS /km' string."""
    if v_ms <= 0:
        return "—"
    pace_s = 1000.0 / v_ms
    return f"{int(pace_s // 60)}:{int(pace_s % 60):02d} /km"


def _log(verbose: bool) -> None:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s",
                        level=logging.DEBUG if verbose else logging.INFO)


def _progress() -> Progress:
    return Progress(
        TextColumn("  [progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# db-init
# ---------------------------------------------------------------------------

@app.command("db-init")
def db_init(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Crea o migra la base de datos SQLite."""
    _log(verbose)
    from kairos.db import init_db
    from kairos.config import DB_PATH
    init_db()
    _console.print(f"\n  [green]✓[/green]  Base de datos lista")
    _console.print(f"     [dim]{DB_PATH}[/dim]\n")


# ---------------------------------------------------------------------------
# sync  (Garmin Connect)
# ---------------------------------------------------------------------------

@app.command()
def sync(
    days: int = typer.Option(30, "--days", "-d"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    email: Optional[str] = typer.Option(None, envvar="GARMIN_EMAIL"),
    password: Optional[str] = typer.Option(None, envvar="GARMIN_PASSWORD", hide_input=True),
    tokenstore: Optional[str] = typer.Option(None, envvar="GARMIN_TOKENSTORE"),
    no_activities: bool = typer.Option(False, "--no-activities"),
    no_hrv: bool = typer.Option(False, "--no-hrv"),
    no_wellness: bool = typer.Option(False, "--no-wellness"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Descarga actividades, HRV y wellness desde Garmin Connect."""
    _log(verbose)
    from kairos.ingest.garmin_sync import sync_all
    start = date.fromisoformat(since) if since else date.today() - timedelta(days=days)
    end = date.fromisoformat(until) if until else date.today()

    with _console.status(f"  Sincronizando {start} → {end} …"):
        results = sync_all(since=start, until=end, email=email, password=password,
                           tokenstore=tokenstore, skip_activities=no_activities,
                           skip_hrv=no_hrv, skip_wellness=no_wellness)

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("tabla", style="dim", min_width=14)
    t.add_column("filas", justify="right", style="bold")

    total = 0
    for tabla, count in results.items():
        t.add_row(tabla, str(count))
        total += count

    t.add_row("", "")
    t.add_row("[dim]total[/dim]", f"[cyan]{total}[/cyan]")

    _console.print()
    _console.print(Panel(t, title=f"[bold]Sync  {start} → {end}[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# ingest-fit
# ---------------------------------------------------------------------------

@app.command("ingest-fit")
def ingest_fit(
    path: Path = typer.Argument(..., help=".fit file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Parsea un archivo .fit y guarda la sesión en la base de datos."""
    _log(verbose)
    from kairos.ingest.loaders import ingest_fit_file
    with _console.status(f"  Parseando {path.name} …"):
        result = ingest_fit_file(path)
    _console.print(f"\n  [green]✓[/green]  {result}\n")


# ---------------------------------------------------------------------------
# log-session
# ---------------------------------------------------------------------------

@app.command("log-session")
def log_session(
    date_: Optional[str] = typer.Option(None, "--date"),
    type_: str = typer.Option("easy", "--type", "-t"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Duración en minutos"),
    rpe: Optional[float] = typer.Option(None, "--rpe", min=0.0, max=10.0, help="RPE (0–10)"),
    notes: Optional[str] = typer.Option(None, "--notes"),
) -> None:
    """Registra una sesión manualmente (cinta u otras sin Garmin)."""
    from kairos.ingest.loaders import insert_session
    d = date_ or date.today().isoformat()
    sid = insert_session(d, type_, duration, rpe, notes=notes)

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("campo", style="dim", min_width=12)
    t.add_column("valor", style="bold")
    t.add_row("ID",       str(sid))
    t.add_row("Fecha",    d)
    t.add_row("Tipo",     type_)
    if duration is not None:
        t.add_row("Duración", f"{duration:.0f} min")
    if rpe is not None:
        t.add_row("RPE",      f"{rpe:.1f} / 10")
    if notes:
        t.add_row("Notas",    notes)

    _console.print()
    _console.print(Panel(t, title="[bold]Sesión registrada[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

@app.command()
def backfill(
    since: Optional[str] = typer.Option(None, "--since", help="Fecha inicio YYYY-MM-DD (default: 2023-01-01)"),
    until: Optional[str] = typer.Option(None, "--until", help="Fecha fin YYYY-MM-DD (default: hoy)"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignorar checkpoint guardado"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="Procesar .fit locales de este directorio primero"),
    no_activities: bool = typer.Option(False, "--no-activities"),
    no_hrv: bool = typer.Option(False, "--no-hrv"),
    no_wellness: bool = typer.Option(False, "--no-wellness"),
    email: Optional[str] = typer.Option(None, envvar="GARMIN_EMAIL"),
    password: Optional[str] = typer.Option(None, envvar="GARMIN_PASSWORD", hide_input=True),
    tokenstore: Optional[str] = typer.Option(None, envvar="GARMIN_TOKENSTORE"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Descarga todo el historial Garmin desde 2023-01-01 en chunks de 30 días (resumable)."""
    _log(verbose)
    from kairos.config import HISTORY_START
    from kairos.ingest.garmin_sync import run_backfill
    start = date.fromisoformat(since) if since else None
    end = date.fromisoformat(until) if until else date.today()
    label = str(start or HISTORY_START)

    with _console.status(f"  Backfill {label} → {end} …"):
        result = run_backfill(
            since=start, until=end,
            resume=(not no_resume),
            archive_dir=archive_dir,
            email=email, password=password, tokenstore=tokenstore,
            skip_activities=no_activities, skip_hrv=no_hrv, skip_wellness=no_wellness,
        )

    if result.get("status") == "already_complete":
        _console.print(f"\n  [green]✓[/green]  Ya completo  [dim](último chunk: {result['last_chunk']})[/dim]\n")
        return

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("tabla", style="dim", min_width=14)
    t.add_column("filas", justify="right", style="bold")
    for k, v in result.items():
        if k == "archive":
            t.add_row("archive", str(v))
        elif k != "status":
            t.add_row(k, str(v))

    _console.print()
    _console.print(Panel(t, title=f"[bold]Backfill  {label} → {end}[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# rebuild-form
# ---------------------------------------------------------------------------

@app.command("rebuild-form")
def rebuild_form_cmd(
    since: Optional[str] = typer.Option(None, "--since", help="Fecha inicio (default: 2023-01-01)"),
    until: Optional[str] = typer.Option(None, "--until", help="Fecha fin (default: hoy)"),
    window: int = typer.Option(90, "--window", "-w", help="Ventana en días para Π_rel"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Reconstruye form_state con spine completo desde cargas .fit."""
    _log(verbose)
    from kairos.config import HISTORY_START
    from kairos.model.fitness_fatigue import rebuild_form
    start = date.fromisoformat(since) if since else date.fromisoformat(HISTORY_START)
    end = date.fromisoformat(until) if until else date.today()

    with _console.status(f"  Reconstruyendo form_state  {start} → {end}  (ventana {window}d) …"):
        n = rebuild_form(start_date=start, end_date=end, window_days=window)

    _console.print(f"\n  [green]✓[/green]  {n} días escritos en form_state  "
                   f"[dim]({start} → {end})[/dim]\n")


# ---------------------------------------------------------------------------
# import-report
# ---------------------------------------------------------------------------

@app.command("import-report")
def import_report_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Chequeo de calidad de datos: gaps, outliers, cobertura de HRV."""
    _log(verbose)
    from kairos.ingest.quality import import_report

    with _console.status("  Analizando datos …"):
        report = import_report()

    cov = report["hrv_coverage"]
    n_issues = report["total_issues"]

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("campo", style="dim", min_width=18)
    t.add_column("valor", style="bold")

    t.add_row("[dim]COBERTURA[/dim]", "")
    t.add_row("HRV", f"{cov['total_hrv_days']} días  "
              f"[dim](banda {cov['fit_days']} · reloj {cov['watch_days']})[/dim]")
    t.add_row("Sesiones", f"{cov['total_sessions']} en {cov['total_session_days']} días")

    if report["session_outliers"]:
        t.add_row("", "")
        t.add_row("[dim]OUTLIERS[/dim]", "")
        for o in report["session_outliers"][:5]:
            t.add_row(o["date"], f"[yellow]{', '.join(o['issues'])}[/yellow]")

    if report["session_gaps"]:
        t.add_row("", "")
        t.add_row("[dim]GAPS[/dim]", "")
        for g in report["session_gaps"][:5]:
            t.add_row(f"{g['start']} → {g['end']}", f"[yellow]{g['days']} días[/yellow]")

    t.add_row("", "")
    issues_str = (f"[green]✓  Sin problemas[/green]" if n_issues == 0
                  else f"[yellow]⚠  {n_issues} issues[/yellow]")
    t.add_row("Total issues", issues_str)

    _console.print()
    _console.print(Panel(t, title="[bold]Reporte de calidad[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# rebuild-trimp
# ---------------------------------------------------------------------------

@app.command("rebuild-trimp")
def rebuild_trimp_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Recalcula TRIMP para todas las sesiones históricas desde archivos .fit."""
    _log(verbose)
    from kairos.config import DB_PATH, FIT_DIR
    from kairos.db import db_cursor, init_db
    from kairos.ingest.fit_parser import parse_fit
    from kairos.physio.trimp import compute_session_trimp, te_proxy_trimp, detect_hrmax
    from kairos.ingest.loaders import update_session_load
    import numpy as _np

    init_db()

    with db_cursor() as cur:
        rows = cur.execute(
            "SELECT id, fit_path, duration_min FROM sessions WHERE fit_path IS NOT NULL"
        ).fetchall()

    if not rows:
        _console.print("\n  [yellow]⚠[/yellow]  No hay sesiones con archivo .fit.\n")
        return

    max_hrs: list[float] = []
    fit_cache: dict[int, object] = {}

    with _progress() as prog:
        task1 = prog.add_task(f"Parseando {len(rows)} archivos .fit …", total=len(rows))
        for row in rows:
            try:
                fd = parse_fit(row["fit_path"])
                fit_cache[row["id"]] = fd
                mhr = fd.session_summary.get("max_heart_rate")
                if mhr is not None:
                    max_hrs.append(float(mhr))
            except Exception:
                pass
            prog.advance(task1)

    if max_hrs:
        from kairos.physio.trimp import _SPIKE_RATIO
        spike_filtered = []
        for row in rows:
            fd = fit_cache.get(row["id"])
            if fd is None:
                continue
            mhr = fd.session_summary.get("max_heart_rate")
            ahr = fd.session_summary.get("avg_heart_rate")
            if mhr is None:
                continue
            if ahr is not None and float(mhr) > float(ahr) * _SPIKE_RATIO:
                continue
            spike_filtered.append(float(mhr))
        hrmax = float(_np.percentile(spike_filtered, 99)) if spike_filtered else detect_hrmax()
        from kairos.ingest.garmin_sync import set_sync_state
        set_sync_state("hrmax_detected", str(hrmax))
    else:
        hrmax = detect_hrmax()

    from kairos.physio.trimp import detect_hrrest
    from kairos.config import LOAD_METRIC
    hrrest = detect_hrrest()

    counts: dict[str, int] = {"te_proxy": 0, "skipped": 0}
    with _progress() as prog:
        task2 = prog.add_task("Calculando TRIMP …", total=len(rows))
        for row in rows:
            fd = fit_cache.get(row["id"])
            if fd is None:
                counts["skipped"] += 1
                prog.advance(task2)
                continue
            hr_records = [float(r["heart_rate"]) for r in fd.records
                          if r.get("heart_rate") is not None]
            if hr_records:
                trimp_val, src = compute_session_trimp(hr_records, hrmax, hrrest=hrrest)
            else:
                te = fd.session_summary.get("total_training_effect")
                dur = row["duration_min"] or 0
                if te is not None and dur > 0:
                    trimp_val = te_proxy_trimp(float(te), float(dur))
                    src = "te_proxy"
                else:
                    counts["skipped"] += 1
                    prog.advance(task2)
                    continue
            avg_hr = float(_np.mean(hr_records)) if hr_records else None
            mhr = fd.session_summary.get("max_heart_rate")
            update_session_load(
                row["id"], trimp=trimp_val, load_source=src, avg_hr=avg_hr,
                max_hr=float(mhr) if mhr is not None else (float(max(hr_records)) if hr_records else None),
            )
            counts[src] = counts.get(src, 0) + 1
            prog.advance(task2)

    hr_sessions = sum(v for k, v in counts.items() if k not in ("te_proxy", "skipped"))

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("campo", style="dim", min_width=18)
    t.add_column("valor", style="bold", justify="right")
    t.add_row("[dim]PARÁMETROS[/dim]", "")
    t.add_row("HRmax",   f"{hrmax:.0f} bpm")
    t.add_row("HRrest",  f"{hrrest:.0f} bpm")
    t.add_row("Método",  LOAD_METRIC)
    t.add_row("", "")
    t.add_row("[dim]RESULTADOS[/dim]", "")
    t.add_row("HR TRIMP",  f"[green]{hr_sessions}[/green] sesiones")
    t.add_row("TE proxy",  f"{counts['te_proxy']} sesiones")
    t.add_row("Saltadas",  f"[dim]{counts['skipped']}[/dim] sesiones")

    _console.print()
    _console.print(Panel(t, title="[bold]rebuild-trimp[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# add-threshold  (anclas longitudinales episódicas — final.tex §4 y §5.2)
# ---------------------------------------------------------------------------

@app.command("add-threshold")
def add_threshold_cmd(
    date_: str = typer.Argument(..., help="Fecha del test YYYY-MM-DD"),
    v_lt2: Optional[float] = typer.Option(None, "--v-lt2", help="Velocidad en LT2 (m/s)"),
    hr_lt2: Optional[float] = typer.Option(None, "--hr-lt2", help="FC en LT2 (bpm)"),
    notes: Optional[str] = typer.Option(None, "--notes"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Registra un test de escalón de lactato (v_LT2 y/o HR@LT2)."""
    _log(verbose)
    if v_lt2 is None and hr_lt2 is None:
        _console.print("\n  [red]✗[/red]  Se requiere al menos --v-lt2 o --hr-lt2\n", err=True)
        raise typer.Exit(1)

    from kairos.db import db_cursor, init_db
    init_db()
    with db_cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO lactate_thresholds (date, v_lt2, hr_lt2, notes) "
            "VALUES (?,?,?,?)",
            (date_, v_lt2, hr_lt2, notes),
        )

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("campo", style="dim", min_width=12)
    t.add_column("valor", style="bold")
    t.add_row("Fecha",  date_)
    if v_lt2 is not None:
        t.add_row("v_LT2",  f"{v_lt2:.2f} m/s  [dim]({_pace_str(v_lt2)})[/dim]")
    if hr_lt2 is not None:
        t.add_row("HR@LT2", f"{hr_lt2:.0f} bpm")
    if notes:
        t.add_row("Notas",  notes)
    t.add_row("", "")
    t.add_row("[dim]siguiente paso[/dim]",
              "[dim]kairos rebuild-trimp && kairos rebuild-form[/dim]")

    _console.print()
    _console.print(Panel(t, title="[bold]Umbral de lactato registrado[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# rebuild-domain-trimp  (split aeróbico/HII retroactivo — final.tex ec. 8–9)
# ---------------------------------------------------------------------------

@app.command("rebuild-domain-trimp")
def rebuild_domain_trimp_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Recalcula el split aeróbico/HII para sesiones con .fit y umbral disponible."""
    _log(verbose)
    from kairos.db import db_cursor, init_db
    from kairos.ingest.fit_parser import parse_fit
    from kairos.physio.trimp import detect_hrmax, detect_hrrest
    from kairos.model.fitness_fatigue import get_threshold_at, compute_domain_trimp_from_records

    init_db()

    with db_cursor() as cur:
        rows = cur.execute(
            "SELECT id, date, fit_path FROM sessions WHERE fit_path IS NOT NULL"
        ).fetchall()

    if not rows:
        _console.print("\n  [yellow]⚠[/yellow]  No hay sesiones con archivo .fit.\n")
        return

    hrmax = detect_hrmax()
    hrrest = detect_hrrest()
    counts = {"split": 0, "no_threshold": 0, "no_hr": 0, "skipped": 0}

    with _progress() as prog:
        task = prog.add_task(f"Split aeróbico/HII  ({len(rows)} sesiones) …", total=len(rows))
        for row in rows:
            v_lt2, hr_lt2 = get_threshold_at(row["date"])
            if v_lt2 is None and hr_lt2 is None:
                counts["no_threshold"] += 1
                prog.advance(task)
                continue
            try:
                fd = parse_fit(row["fit_path"])
            except Exception:
                counts["skipped"] += 1
                prog.advance(task)
                continue
            l_aer, l_hii = compute_domain_trimp_from_records(
                fd.records, hrmax, hrrest, v_lt2, hr_lt2
            )
            if l_aer is None:
                counts["no_hr"] += 1
                prog.advance(task)
                continue
            with db_cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET trimp_aerobic=?, trimp_hii=? WHERE id=?",
                    (l_aer, l_hii, row["id"]),
                )
            counts["split"] += 1
            prog.advance(task)

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("campo", style="dim", min_width=18)
    t.add_column("valor", style="bold", justify="right")
    t.add_row("Split calculado",  f"[green]{counts['split']}[/green] sesiones")
    t.add_row("Sin umbral",       f"[dim]{counts['no_threshold']}[/dim] sesiones")
    t.add_row("Sin HR/GPS",       f"[dim]{counts['no_hr']}[/dim] sesiones")
    t.add_row("Error de parseo",  f"[dim]{counts['skipped']}[/dim] sesiones")
    t.add_row("", "")
    t.add_row("[dim]siguiente paso[/dim]", "[dim]kairos rebuild-form[/dim]")

    _console.print()
    _console.print(Panel(t, title="[bold]rebuild-domain-trimp[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# form  (PRIMARY output)
# ---------------------------------------------------------------------------

@app.command()
def form(
    date_: Optional[str] = typer.Option(None, "--date", help="YYYY-MM-DD (default: hoy)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Descriptor de estado de forma actual."""
    _log(verbose)
    from kairos.db import db_cursor

    d = date.fromisoformat(date_) if date_ else date.today()

    with db_cursor() as cur:
        row = cur.execute(
            "SELECT g, h, freshness, Pi, Pi_abs, g_aerobic, g_hii "
            "FROM form_state WHERE date = ?",
            (d.isoformat(),),
        ).fetchone()

    if row is None:
        _console.print(f"\n  [red]✗[/red]  Sin datos para {d}. Corre [bold]rebuild-form[/bold] primero.\n")
        raise typer.Exit(1)

    g      = float(row["g"])
    h      = float(row["h"])
    tsb    = float(row["freshness"])
    pi     = float(row["Pi"])        if row["Pi"]        is not None else None
    pi_abs = float(row["Pi_abs"])    if row["Pi_abs"]    is not None else None
    g_aer  = float(row["g_aerobic"]) if row["g_aerobic"] is not None else None
    g_hii_ = float(row["g_hii"])     if row["g_hii"]     is not None else None

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("metric", min_width=14)
    t.add_column("value",  justify="right", min_width=8, style="bold")
    t.add_column("label",  style="dim")

    tsb_style = "green" if tsb > 5 else "red" if tsb < -10 else "yellow"

    t.add_row("[dim]CARGA[/dim]", "", "")
    t.add_row("g(t)  CTL", f"{g:.1f}",                                          "carga crónica")
    t.add_row("h(t)  ATL", f"{h:.1f}",                                          "carga aguda")
    t.add_row("TSB",       f"[{tsb_style}]{tsb:+.1f}[/{tsb_style}]",           "frescura")

    if pi is not None or pi_abs is not None:
        t.add_row("", "", "")
        t.add_row("[dim]FORMA[/dim]", "", "")
        if pi is not None:
            pi_style = "green" if pi > 0.3 else "red" if pi < -0.3 else "yellow"
            pi_label = "↑ mejorando" if pi > 0.3 else "↓ perdiendo forma" if pi < -0.3 else "→ estable"
            t.add_row("Π_rel(t)", f"[{pi_style}]{pi:+.3f}[/{pi_style}]",      pi_label)
        if pi_abs is not None:
            t.add_row("Π_abs(t)", f"{pi_abs:.1f}",                             "nivel acumulado")

    if g_aer is not None or g_hii_ is not None:
        t.add_row("", "", "")
        t.add_row("[dim]DOMINIOS[/dim]", "", "")
        if g_aer is not None:
            t.add_row("g_aer", f"{g_aer:.1f}",                                 "CTL aeróbico")
        if g_hii_ is not None:
            t.add_row("g_hii", f"{g_hii_:.1f}",                                "CTL alta intensidad")

    _console.print()
    _console.print(Panel(t, title=f"[bold]Kairós — {d}[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# snapshot  (historical form on a past date)
# ---------------------------------------------------------------------------

@app.command()
def snapshot(
    date_: str = typer.Argument(..., help="Fecha YYYY-MM-DD"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Estado de forma en una fecha pasada exacta."""
    _log(verbose)
    from kairos.db import db_cursor

    try:
        d = date.fromisoformat(date_)
    except ValueError:
        _console.print(f"\n  [red]✗[/red]  Fecha inválida: '{date_}'  (formato: YYYY-MM-DD)\n")
        raise typer.Exit(1)
    if d > date.today():
        _console.print("\n  [red]✗[/red]  La fecha debe ser pasada o de hoy.\n")
        raise typer.Exit(1)

    with db_cursor() as cur:
        row = cur.execute(
            "SELECT g, h, freshness, Pi FROM form_state WHERE date=?",
            (d.isoformat(),),
        ).fetchone()

    if row is None:
        _console.print(f"\n  [red]✗[/red]  Sin datos para {d}. Corre [bold]rebuild-form[/bold] primero.\n")
        raise typer.Exit(1)

    with db_cursor() as cur:
        sessions = cur.execute(
            "SELECT date, type, duration_min, trimp FROM sessions "
            "WHERE date BETWEEN ? AND ? ORDER BY date",
            ((d - timedelta(days=3)).isoformat(), (d + timedelta(days=3)).isoformat()),
        ).fetchall()

    pi_stored = float(row["Pi"]) if row["Pi"] is not None else None
    tsb = float(row["freshness"])
    tsb_style = "green" if tsb > 5 else "red" if tsb < -10 else "yellow"

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("metric", min_width=14)
    t.add_column("value",  justify="right", min_width=8, style="bold")
    t.add_column("label",  style="dim")

    t.add_row("[dim]CARGA[/dim]", "", "")
    t.add_row("g(t)  CTL", f"{float(row['g']):.1f}",                       "carga crónica")
    t.add_row("h(t)  ATL", f"{float(row['h']):.1f}",                       "carga aguda")
    t.add_row("TSB",       f"[{tsb_style}]{tsb:+.1f}[/{tsb_style}]",      "frescura")
    if pi_stored is not None:
        pi_style = "green" if pi_stored > 0.3 else "red" if pi_stored < -0.3 else "yellow"
        pi_label = "↑ mejorando" if pi_stored > 0.3 else "↓ perdiendo forma" if pi_stored < -0.3 else "→ estable"
        t.add_row("", "", "")
        t.add_row("[dim]FORMA[/dim]", "", "")
        t.add_row("Π_rel(t)", f"[{pi_style}]{pi_stored:+.3f}[/{pi_style}]", pi_label)

    if sessions:
        t.add_row("", "", "")
        t.add_row("[dim]SESIONES ±3d[/dim]", "", "")
        for s in sessions:
            dur  = f"{s['duration_min']:.0f} min" if s["duration_min"] else "—"
            trimp = f"  TRIMP {s['trimp']:.0f}" if s["trimp"] else ""
            marker = "→ " if s["date"] == d.isoformat() else "  "
            t.add_row(f"{marker}{s['date']}", dur, f"{s['type']}{trimp}")

    _console.print()
    _console.print(Panel(t, title=f"[bold]Snapshot — {d}[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# peak  (best form within a date range)
# ---------------------------------------------------------------------------

@app.command()
def peak(
    desde: Optional[str] = typer.Argument(None, help="Fecha inicio YYYY-MM-DD (default: hace 1 año)"),
    hasta: Optional[str] = typer.Argument(None, help="Fecha fin YYYY-MM-DD (default: hoy)"),
    top: int = typer.Option(5, "--top", help="Cuántos picos mostrar"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Días de mejor forma (Π_rel máximo) dentro de un rango.

    Uso: kairos peak [DESDE] [HASTA] [--top N]
    """
    _log(verbose)
    from kairos.db import db_cursor

    d_hasta = date.fromisoformat(hasta) if hasta else date.today()
    d_desde = date.fromisoformat(desde) if desde else d_hasta.replace(year=d_hasta.year - 1)

    if d_desde >= d_hasta:
        _console.print("\n  [red]✗[/red]  --desde debe ser anterior a --hasta.\n")
        raise typer.Exit(1)

    with db_cursor() as cur:
        rows = cur.execute(
            "SELECT date, g, h, freshness, Pi FROM form_state "
            "WHERE date BETWEEN ? AND ? AND Pi IS NOT NULL ORDER BY Pi DESC LIMIT ?",
            (d_desde.isoformat(), d_hasta.isoformat(), top),
        ).fetchall()

    if not rows:
        _console.print(f"\n  [yellow]⚠[/yellow]  Sin datos entre {d_desde} y {d_hasta}. "
                       "Corre [bold]rebuild-form[/bold] primero.\n")
        raise typer.Exit(1)

    t = Table(box=None, show_header=True, padding=(0, 2), expand=False)
    t.add_column("#",      style="dim",    width=3,  justify="right")
    t.add_column("Fecha",  style="bold",   min_width=12)
    t.add_column("Π_rel",                  min_width=8,  justify="right")
    t.add_column("TSB",    style="dim",    min_width=7,  justify="right")
    t.add_column("CTL",    style="dim",    min_width=6,  justify="right")
    t.add_column("ATL",    style="dim",    min_width=6,  justify="right")

    for i, r in enumerate(rows, 1):
        pi_val = float(r["Pi"])
        pi_style = "green" if pi_val > 0.3 else "red" if pi_val < -0.3 else "yellow"
        tsb_val = float(r["freshness"])
        tsb_style = "green" if tsb_val > 5 else "red" if tsb_val < -10 else "default"
        t.add_row(
            str(i),
            r["date"],
            f"[{pi_style}]{pi_val:+.3f}[/{pi_style}]",
            f"[{tsb_style}]{tsb_val:+.1f}[/{tsb_style}]",
            f"{float(r['g']):.1f}",
            f"{float(r['h']):.1f}",
        )

    best = rows[0]
    footer = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    footer.add_column("k", style="dim", min_width=14)
    footer.add_column("v", style="bold")
    footer.add_row("Mejor día", f"{best['date']}  →  Π_rel = [green]{float(best['Pi']):+.3f}[/green]")

    from rich.console import Group
    _console.print()
    _console.print(Panel(
        Group(t, footer),
        title=f"[bold]Picos de forma  {d_desde} → {d_hasta}[/bold]",
        border_style="cyan", padding=(1, 2),
    ))
    _console.print()


# ---------------------------------------------------------------------------
# state  (SECONDARY — objective daily context)
# ---------------------------------------------------------------------------

@app.command()
def state(
    date_: Optional[str] = typer.Option(None, "--date", help="YYYY-MM-DD (default: hoy)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Estado objetivo de la última sesión: TRIMP, FC, GCT, HRR."""
    _log(verbose)
    from kairos.db import db_cursor

    d = date.fromisoformat(date_) if date_ else date.today()

    with db_cursor() as cur:
        sess = cur.execute(
            "SELECT s.id, s.date, s.type, s.duration_min, s.trimp, s.avg_hr, "
            "       rd.gct_mean_ms, rd.gct_drift_pct, rd.hrr60, rd.tau_hrr "
            "FROM sessions s LEFT JOIN running_dynamics rd ON rd.session_id = s.id "
            "WHERE s.date <= ? ORDER BY s.date DESC LIMIT 1",
            (d.isoformat(),),
        ).fetchone()

    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column("campo", style="dim", min_width=16)
    t.add_column("valor", style="bold")

    if sess is None:
        t.add_row("estado", "[dim]Sin sesiones registradas[/dim]")
    else:
        dur = f"{sess['duration_min']:.0f} min" if sess["duration_min"] else "—"
        t.add_row("Fecha",   f"{sess['date']}  [dim]({sess['type']} · {dur})[/dim]")
        if sess["trimp"] is not None:
            t.add_row("TRIMP",    f"{sess['trimp']:.1f}")
        if sess["avg_hr"] is not None:
            t.add_row("FC media", f"{sess['avg_hr']:.0f} bpm")
        if sess["gct_mean_ms"] is not None:
            t.add_row("GCT",      f"{sess['gct_mean_ms']:.0f} ms")
        if sess["gct_drift_pct"] is not None:
            drift = sess["gct_drift_pct"] * 100
            drift_style = "red" if abs(drift) > 5 else "default"
            t.add_row("GCT drift", f"[{drift_style}]{drift:+.1f} %[/{drift_style}]")
        if sess["hrr60"] is not None:
            t.add_row("HRR60",    f"{sess['hrr60']:.0f} bpm / 60 s")
        if sess["tau_hrr"] is not None:
            t.add_row("τ_HRR",    f"{sess['tau_hrr']:.0f} s")

    _console.print()
    _console.print(Panel(t, title=f"[bold]Estado — {d}[/bold]",
                         border_style="cyan", padding=(1, 2)))
    _console.print()


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

@app.command()
def dashboard() -> None:
    """Abre el dashboard Streamlit de Kairós en el navegador."""
    import subprocess
    import sys
    import importlib.util

    spec = importlib.util.find_spec("kairos.dashboard")
    if spec is None or spec.origin is None:
        _console.print("\n  [red]✗[/red]  No se encontró kairos.dashboard.\n")
        raise typer.Exit(1)

    streamlit = Path(sys.executable).parent / "streamlit"
    if not streamlit.exists():
        _console.print("\n  [red]✗[/red]  streamlit no está instalado en este entorno.\n")
        raise typer.Exit(1)

    _console.print("\n  Abriendo dashboard …  [dim](Ctrl+C para cerrar)[/dim]\n")
    subprocess.run([str(streamlit), "run", spec.origin])


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

@app.command("help")
def help_cmd() -> None:
    """Muestra todos los comandos agrupados por flujo de trabajo."""

    def _section(title: str, rows: list[tuple[str, str]]) -> None:
        _console.print(f"\n  [bold]{title}[/bold]")
        t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
        t.add_column("cmd",  style="cyan",  no_wrap=True, min_width=46)
        t.add_column("desc", style="dim")
        for cmd, desc in rows:
            t.add_row(cmd, desc)
        _console.print(t)

    _console.print()
    _console.rule("[bold cyan]Kairós — Comandos[/bold cyan]", style="cyan")

    _section("1 · SETUP", [
        ("kairos db-init",                                    "Crea/migra la base de datos SQLite."),
    ])
    _section("2 · INGESTIÓN", [
        ("kairos sync [--days 30]",                           "Descarga los últimos N días desde Garmin Connect."),
        ("kairos backfill [--since YYYY-MM-DD]",              "Historial completo desde 2023-01-01 (resumable)."),
        ("kairos ingest-fit <ruta.fit>",                      "Parsea un .fit local y lo carga a la BD."),
        ("kairos log-session --type easy --duration 45",      "Registra una sesión manualmente."),
    ])
    _section("3 · PROCESAMIENTO", [
        ("kairos rebuild-trimp",                              "Recalcula TRIMP desde .fit."),
        ("kairos add-threshold YYYY-MM-DD --v-lt2 N.NN",     "Registra test de escalón de lactato."),
        ("kairos rebuild-domain-trimp",                       "Split aeróbico/HII retroactivo."),
        ("kairos rebuild-form [--window 90]",                 "Reconstruye CTL, ATL, TSB, Π_rel, Π_abs."),
        ("kairos import-report",                              "Calidad de datos: gaps y outliers."),
    ])
    _section("4 · ANÁLISIS", [
        ("kairos form [--date YYYY-MM-DD]",                   "Estado de forma del día."),
        ("kairos state [--date YYYY-MM-DD]",                  "Última sesión: TRIMP, FC, GCT, HRR."),
        ("kairos snapshot <YYYY-MM-DD>",                      "Forma exacta en fecha pasada."),
        ("kairos peak [--desde …] [--hasta …] [--top 5]",    "Días de mejor Π_rel en un rango."),
        ("kairos dashboard",                                   "Abre la GUI en el navegador."),
    ])

    _console.print()
    _console.rule(style="cyan dim")
    _console.print("\n  [bold]FLUJO DIARIO[/bold]")
    _console.print("  [cyan]kairos sync && kairos rebuild-trimp && kairos rebuild-form && kairos form[/cyan]\n")


if __name__ == "__main__":
    app()
