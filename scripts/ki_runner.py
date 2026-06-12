"""Master-Entry-Point fuer RedditOrakel v3.5 -- KI/Training-Launcher.

Modi:
  --mode=train  Volle Pipeline 1->10 headless sequenziell ausfuehren
  --mode=menu   Klassisches interaktives 12-Punkte-Menue
  --mode=auto   Modus-Auswahl-TUI (Default wenn nichts angegeben)

Aufruf typischerweise ueber ki-run-and-train.bat / .ps1 im Projekt-Root.
Fuer den normalen App-Start (Dashboard + Backend) bitte start.bat verwenden.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from scripts._warnings import silence_known_warnings  # noqa: E402

silence_known_warnings()

import argparse  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

if os.name == "nt":
    os.system("chcp 65001 >nul 2>&1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _con: "Console | None" = Console(force_terminal=True)
    _RICH = True
except ImportError:
    _con = None
    _RICH = False


# ---------------------------------------------------------------------------
# Pipeline-Definition
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    label: str
    script: str | None
    args: list[str]
    inline_code: str | None
    critical: bool


def _build_steps() -> list[PipelineStep]:
    from scripts._ml_pipeline import inline_optuna, inline_pagerank, inline_pymc

    return [
        PipelineStep("WC-Daten synchronisieren",  "sync_from_api.py",          [],                  None,                       True),
        PipelineStep("WM-Daten aufbauen",         "rebuild_wm_data.py",        [],                  None,                       True),
        PipelineStep("Match-Configs erzeugen",    "generate_match_configs.py", [],                  None,                       True),
        PipelineStep("Team-Daten aktualisieren",  "update_yaml_team_data.py",  [],                  None,                       False),
        PipelineStep("Elo neu berechnen",         "refresh_elo.py",            [],                  None,                       False),
        PipelineStep("xG-Modell trainieren",      "train_xg_predictor.py",     [],                  None,                       False),
        PipelineStep("LightGBM trainieren",       "train_xg_predictor.py",     ["--model", "lgbm"], None,                       False),
        PipelineStep("Optuna Gewichts-Tuning",    None,                        [],                  inline_optuna(trials=100),  False),
        PipelineStep("PyMC Bayes-Posterior",      None,                        [],                  inline_pymc(draws=2000),    False),
        PipelineStep("PageRank aufbauen",         None,                        [],                  inline_pagerank(),          False),
    ]


def _run_step(step: PipelineStep) -> int:
    if step.script:
        cmd = [PYTHON, str(ROOT / "scripts" / step.script), *step.args]
    elif step.inline_code:
        cmd = [PYTHON, "-c", step.inline_code]
    else:
        return 1
    try:
        return subprocess.run(cmd, cwd=str(ROOT)).returncode
    except KeyboardInterrupt:
        print("\n[Abgebrochen]")
        return 130
    except FileNotFoundError as exc:
        print(f"[Fehler] {exc}")
        return 1


# ---------------------------------------------------------------------------
# Mode: TRAIN
# ---------------------------------------------------------------------------

def train_pipeline() -> int:
    steps = _build_steps()

    if _RICH and _con is not None:
        _con.print()
        _con.print(Panel(
            "[bold cyan]TRAIN-Modus[/bold cyan]  -  Volle Pipeline 1->10\n"
            "[dim]Daten-Sync + ML-Training, headless, Default-Parameter[/dim]",
            border_style="cyan", padding=(0, 2),
        ))
    else:
        print("\n  === TRAIN-Modus: Volle Pipeline 1->10 ===\n")

    results: list[tuple[str, int, float]] = []
    aborted = False

    for idx, step in enumerate(steps, start=1):
        header = f"[{idx:>2}/{len(steps)}] {step.label}"
        if _RICH and _con is not None:
            _con.rule(f"[cyan]{header}[/cyan]")
        else:
            print(f"\n--- {header} ---")

        t0 = time.time()
        rc = _run_step(step)
        elapsed = time.time() - t0
        results.append((step.label, rc, elapsed))

        if rc != 0 and step.critical:
            if _RICH and _con is not None:
                _con.print(f"[red]Kritischer Schritt fehlgeschlagen (Exit {rc}).[/red] Pipeline abgebrochen.")
            else:
                print(f"  KRITISCHER FEHLER (Exit {rc}). Pipeline abgebrochen.")
            aborted = True
            break
        elif rc != 0:
            if _RICH and _con is not None:
                _con.print(f"[yellow]Schritt fehlgeschlagen (Exit {rc}), Pipeline laeuft weiter.[/yellow]")
            else:
                print(f"  WARNUNG: Exit {rc}, weiter mit naechstem Schritt.")

    _render_summary(results, aborted)
    return 1 if aborted else 0


def _render_summary(results: list[tuple[str, int, float]], aborted: bool) -> None:
    if _RICH and _con is not None:
        t = Table(title="Train-Pipeline Zusammenfassung", show_lines=False)
        t.add_column("Schritt", style="bold")
        t.add_column("Status")
        t.add_column("Dauer", justify="right")
        for label, rc, elapsed in results:
            status = "[green]ok[/green]" if rc == 0 else f"[red]fail ({rc})[/red]"
            t.add_row(label, status, f"{elapsed:.1f}s")
        _con.print()
        _con.print(t)
        if aborted:
            _con.print("[bold red]Pipeline abgebrochen.[/bold red]")
        else:
            _con.print("[bold green]Pipeline abgeschlossen.[/bold green]")
    else:
        print("\n=== Zusammenfassung ===")
        for label, rc, elapsed in results:
            tag = "OK  " if rc == 0 else f"FAIL"
            print(f"  [{tag}] {label} ({elapsed:.1f}s)")
        print("ABGEBROCHEN." if aborted else "FERTIG.")


# ---------------------------------------------------------------------------
# Mode: MENU  (klassisches 12-Punkte-Menue)
# ---------------------------------------------------------------------------

def run_menu() -> int:
    from scripts import menu  # lazy import
    menu.main()
    return 0


# ---------------------------------------------------------------------------
# Mode: AUTO  (TUI-Modus-Auswahl)
# ---------------------------------------------------------------------------

def _mode_picker() -> str | None:
    if _RICH and _con is not None:
        _con.print()
        _con.print(Panel(
            "[bold cyan]RedditOrakel v3.5[/bold cyan]  -  KI/Training-Launcher\n"
            "[dim]18-Faktor-Ensemble  ·  Dixon-Coles/NegBin/GLM  ·  NVIDIA-LLM[/dim]\n"
            "[yellow]Fuer App-Start (Dashboard + Backend): start.bat[/yellow]",
            border_style="cyan", padding=(0, 2),
        ))
        t = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        t.add_column("Nr", style="bold cyan", width=4, no_wrap=True)
        t.add_column("Modus", style="bold", width=14)
        t.add_column("Beschreibung", style="dim")
        t.add_row("1", "TRAIN", "Volle Pipeline 1->10 automatisch (Daten + ML, headless)")
        t.add_row("2", "MENU",  "Klassisches interaktives 12-Punkte-Menue (Einzeloperationen)")
        t.add_row("0", "Beenden", "")
        _con.print(Panel(t, title="[bold]Was moechtest du tun?[/bold]", border_style="blue", padding=(1, 2)))
    else:
        print("\n  RedditOrakel v3.5 -- KI/Training-Launcher")
        print("  =========================================")
        print("    1  TRAIN  Volle Pipeline 1->10 automatisch")
        print("    2  MENU   Interaktives 12-Punkte-Menue")
        print("    0  Beenden")
        print("\n  Fuer App-Start (Dashboard + Backend): start.bat")

    try:
        raw = input("\n  Auswahl [0-2]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    return {"1": "train", "2": "menu", "0": None}.get(raw, "invalid")


def auto_mode() -> int:
    while True:
        choice = _mode_picker()
        if choice is None:
            return 0
        if choice == "invalid":
            print("  Ungueltige Eingabe.")
            continue
        if choice == "train":
            return train_pipeline()
        if choice == "menu":
            return run_menu()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ki-run-and-train",
        description="RedditOrakel v3.5 KI/Training-Launcher (App-Start: start.bat)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "menu", "auto"],
        default="auto",
        help="Modus direkt waehlen (sonst TUI-Auswahl)",
    )
    args = parser.parse_args(argv)

    if args.mode == "train":
        return train_pipeline()
    if args.mode == "menu":
        return run_menu()
    return auto_mode()


if __name__ == "__main__":
    sys.exit(main())
