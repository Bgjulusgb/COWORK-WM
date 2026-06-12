"""Interaktives Verwaltungsmenue fuer RedditOrakel v3.4.

Starten via:
    ki-run-and-train.bat --mode=menu   (Doppelklick aus dem Projektordner)
    ki-run-and-train.ps1 -Mode menu    (PowerShell)
    python -m scripts.ki_runner --mode=menu   (direkt, venv muss aktiv sein)
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from scripts._warnings import silence_known_warnings  # noqa: E402

silence_known_warnings()

import os  # noqa: E402
import subprocess  # noqa: E402

from scripts._ml_pipeline import inline_optuna, inline_pagerank, inline_pymc  # noqa: E402

if os.name == "nt":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
PYTHON = sys.executable

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
    _con = Console(force_terminal=True)
    _RICH = True
except ImportError:
    _con = None
    _RICH = False


def _banner() -> None:
    if _RICH:
        _con.print()
        _con.print(Panel(
            "[bold cyan]RedditOrakel v3.3[/bold cyan]  —  WM 2026 Vorhersage-System\n"
            "[dim]18-Faktor-Ensemble · Dixon-Coles/NegBin/GLM · NVIDIA-LLM · Optuna · PyMC[/dim]",
            border_style="cyan",
            padding=(0, 2),
        ))
    else:
        print("\n  RedditOrakel v3.3 — WM 2026 Vorhersage-System")
        print("  18-Faktor-Ensemble · Dixon-Coles/NegBin/GLM · NVIDIA-LLM")
        print("  ============================================================")


def _menu_table() -> None:
    if _RICH:
        t = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        t.add_column("Nr", style="bold cyan", width=4, no_wrap=True)
        t.add_column("Aktion", style="bold", width=30)
        t.add_column("Beschreibung", style="dim")

        t.add_row("", "[underline]Daten & Konfiguration[/underline]", "")
        t.add_row("1", "WC-Daten synchronisieren", "worldcup26.ir -> wm2026_data.json")
        t.add_row("2", "WM-Daten aufbauen", "wm2026_data.json aus API-Snapshot")
        t.add_row("3", "Match-Configs erzeugen", "YAML-Configs für alle 72 Spiele")
        t.add_row("4", "Team-Daten aktualisieren", "FIFA-Ranking + Form in YAMLs patchen")
        t.add_row("5", "Elo neu berechnen", "Team-Elo aus Match-History (Dry-Run)")

        t.add_row("", "[underline]ML & Wissenschaft[/underline]", "")
        t.add_row("6", "xG-Modell trainieren", "XGBoost xG-Prädiktor (Submenu)")
        t.add_row("7", "LightGBM trainieren", "Zweiter ML-Head für Blend-Faktor")
        t.add_row("8", "Optuna Gewichts-Tuning", "Bayesian Search über Faktor-Gewichte")
        t.add_row("9", "PyMC Bayes-Posterior", "MCMC-Posterior der Faktor-Gewichte")
        t.add_row("10", "PageRank aufbauen", "Netzwerk-Stärke aus Match-Graph")

        t.add_row("", "[underline]System[/underline]", "")
        t.add_row("11", "Connector-Status prüfen", "HEAD-Ping aller Live-Endpoints")
        t.add_row("12", ".env bearbeiten", ".env im Standard-Editor öffnen")
        t.add_row("")
        t.add_row("0", "Beenden", "")
        _con.print(Panel(t, title="[bold]Verwaltungsmenü[/bold]", border_style="blue", padding=(1, 2)))
    else:
        print("\n  ┌─ Verwaltungsmenü ─────────────────────────────────────────────────┐")
        print("  │                                                                    │")
        print("  │  Daten & Konfiguration                                             │")
        print("  │   1  WC-Daten synchronisieren    (worldcup26.ir)                   │")
        print("  │   2  WM-Daten aufbauen           (aus API-Snapshot)                │")
        print("  │   3  Match-Configs erzeugen      (72 YAML-Dateien)                 │")
        print("  │   4  Team-Daten aktualisieren    (FIFA-Ranking + Form)             │")
        print("  │   5  Elo neu berechnen           (Dry-Run)                         │")
        print("  │                                                                    │")
        print("  │  ML & Wissenschaft                                                 │")
        print("  │   6  xG-Modell trainieren        (XGBoost)                         │")
        print("  │   7  LightGBM trainieren         (zweiter ML-Head)                 │")
        print("  │   8  Optuna Gewichts-Tuning      (Bayesian Search)                 │")
        print("  │   9  PyMC Bayes-Posterior        (MCMC)                            │")
        print("  │  10  PageRank aufbauen           (Netzwerk-Stärke)                 │")
        print("  │                                                                    │")
        print("  │  System                                                            │")
        print("  │  11  Connector-Status prüfen                                       │")
        print("  │  12  .env bearbeiten                                               │")
        print("  │   0  Beenden                                                       │")
        print("  └────────────────────────────────────────────────────────────────────┘")


def _run(script: str, extra_args: list[str] | None = None) -> int:
    cmd = [PYTHON, str(ROOT / "scripts" / script)] + (extra_args or [])
    try:
        result = subprocess.run(cmd, cwd=str(ROOT))
        return result.returncode
    except KeyboardInterrupt:
        print("\n[Abgebrochen]")
        return 130
    except FileNotFoundError as exc:
        print(f"[Fehler] Script nicht gefunden: {exc}")
        return 1


def _run_module(module: str, extra_args: list[str] | None = None) -> int:
    cmd = [PYTHON, "-m", module] + (extra_args or [])
    try:
        result = subprocess.run(cmd, cwd=str(ROOT))
        return result.returncode
    except KeyboardInterrupt:
        print("\n[Abgebrochen]")
        return 130
    except FileNotFoundError as exc:
        print(f"[Fehler] Modul nicht gefunden: {exc}")
        return 1


def _run_inline(code: str) -> int:
    cmd = [PYTHON, "-c", code]
    try:
        result = subprocess.run(cmd, cwd=str(ROOT))
        return result.returncode
    except KeyboardInterrupt:
        print("\n[Abgebrochen]")
        return 130


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _warn_missing(module: str, pip_name: str) -> bool:
    if _can_import(module):
        return True
    msg = f"  Modul '{module}' fehlt. Installiere mit: pip install {pip_name}"
    if _RICH:
        _con.print(f"  [red]Modul '{module}' fehlt.[/red] Installiere mit: [bold]pip install {pip_name}[/bold]")
    else:
        print(msg)
    return False


def _pause() -> None:
    try:
        input("\n  [ENTER] — Zurück zum Menü ...")
    except (KeyboardInterrupt, EOFError):
        pass


def _ask_int(prompt: str, default: int) -> int:
    try:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return int(raw) if raw else default
    except (KeyboardInterrupt, EOFError, ValueError):
        return default


# ── Handlers ────────────────────────────────────────────────────────

def _training_submenu() -> None:
    if _RICH:
        t = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        t.add_column("Nr", style="bold cyan", width=4)
        t.add_column("Option", style="bold", width=28)
        t.add_column("Beschreibung", style="dim")
        t.add_row("1", "Standard (Online)", "openfootball WC 2010/14/18/22 + auto data/training/")
        t.add_row("2", "Mit eigenem Datenordner", "Zusätzliche JSON/CSV-Dateien aus eigenem Ordner")
        t.add_row("3", "Nur lokale Dateien", "Kein Netzwerk — nur data/training/ oder eigener Pfad")
        t.add_row("0", "Zurück", "")
        _con.print(Panel(t, title="[bold]ML-Training — Datenstrategie[/bold]", border_style="cyan", padding=(1, 2)))
    else:
        print("\n  Training-Optionen:")
        print("    1  Standard (Online)")
        print("    2  Mit eigenem Datenordner")
        print("    3  Nur lokale Dateien (Offline)")
        print("    0  Zurück")

    try:
        choice = input("\n  Auswahl [0-3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if choice == "0":
        return
    elif choice == "1":
        _run("train_xg_predictor.py")
    elif choice == "2":
        try:
            path = input("  Pfad zum Datenordner: ").strip().strip('"').strip("'")
        except (KeyboardInterrupt, EOFError):
            return
        if path:
            _run("train_xg_predictor.py", ["--data-dir", path])
        else:
            _run("train_xg_predictor.py")
    elif choice == "3":
        try:
            path = input("  Pfad zum Datenordner [Enter = data/training]: ").strip().strip('"').strip("'")
        except (KeyboardInterrupt, EOFError):
            return
        args = ["--local-only"]
        if path:
            args += ["--data-dir", path]
        else:
            default = ROOT / "data" / "training"
            if not default.is_dir():
                if _RICH:
                    _con.print(f"  [yellow]Hinweis:[/yellow] {default} existiert nicht — wird erstellt.")
                else:
                    print(f"  Hinweis: {default} existiert nicht — wird erstellt.")
                default.mkdir(parents=True, exist_ok=True)
            args += ["--data-dir", str(default)]
        _run("train_xg_predictor.py", args)
    else:
        print("  Ungültige Eingabe.")

    _pause()


def _handle_lightgbm() -> None:
    if _RICH:
        _con.rule("[cyan]LightGBM trainieren[/cyan]")
    else:
        print("\n  --- LightGBM trainieren ---")

    if not _warn_missing("lightgbm", "lightgbm"):
        return

    print("  LightGBM-Training benötigt vorbereitete Feature-Daten.")
    print("  Das Training wird über den gleichen Datenpfad wie XGBoost gestartet.")
    print()
    _run("train_xg_predictor.py", ["--model", "lgbm"])


def _handle_optuna() -> None:
    if _RICH:
        _con.rule("[cyan]Optuna Gewichts-Tuning[/cyan]")
    else:
        print("\n  --- Optuna Gewichts-Tuning ---")

    if not _warn_missing("optuna", "optuna"):
        return

    trials = _ask_int("Anzahl Trials", 100)
    print(f"  Starte Optuna mit {trials} Trials (TPE-Sampler) ...")
    print("  Ergebnis wird in models_ml/artifacts/tuned_weights.yaml gespeichert.")
    print()

    _run_inline(inline_optuna(trials=trials))


def _handle_pymc() -> None:
    if _RICH:
        _con.rule("[cyan]PyMC Bayes-Posterior[/cyan]")
    else:
        print("\n  --- PyMC Bayes-Posterior ---")

    if not _warn_missing("pymc", "pymc"):
        return
    if not _warn_missing("arviz", "arviz"):
        return

    draws = _ask_int("Anzahl MCMC-Draws", 2000)
    tune = _ask_int("Tuning-Schritte", 1000)
    print(f"  Starte PyMC NUTS mit {draws} Draws / {tune} Tuning ...")
    print("  (Das kann einige Minuten dauern)")
    print("  Ergebnis wird in models_ml/artifacts/bayes_weights.json gespeichert.")
    print()

    _run_inline(inline_pymc(draws=draws, tune=tune))


def _handle_pagerank() -> None:
    if _RICH:
        _con.rule("[cyan]PageRank-Netzwerkstärke aufbauen[/cyan]")
    else:
        print("\n  --- PageRank aufbauen ---")

    print("  Lese Match-History und berechne PageRank-Scores ...")
    print("  Ergebnis wird in models_ml/artifacts/network_strength.json gespeichert.")
    print()

    _run_inline(inline_pagerank())


def _handle_connector_check() -> None:
    if _RICH:
        _con.rule("[cyan]Connector-Status[/cyan]")
    else:
        print("\n  --- Connector-Status ---")

    _run_module("scripts.launcher", ["--check", "--non-interactive"])


def _handle_edit_env() -> None:
    if _RICH:
        _con.rule("[cyan].env bearbeiten[/cyan]")
    else:
        print("\n  --- .env bearbeiten ---")

    if not ENV_PATH.exists():
        print(f"  [Hinweis] {ENV_PATH} existiert nicht.")
        print("  Starte zuerst start.bat um eine .env zu erzeugen.")
        return

    print(f"  Öffne: {ENV_PATH}")
    try:
        if os.name == "nt":
            os.startfile(str(ENV_PATH))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(ENV_PATH)])
        else:
            subprocess.run(["xdg-open", str(ENV_PATH)])
    except Exception as exc:
        print(f"  [Fehler] Konnte .env nicht öffnen: {exc}")
        print(f"  Manuell öffnen: {ENV_PATH}")


# ── Main Loop ───────────────────────────────────────────────────────

def main() -> None:
    _banner()

    while True:
        _menu_table()

        try:
            choice = input("\n  Auswahl [0-12]: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "0":
            break
        elif choice == "1":
            if _RICH:
                _con.rule("[cyan]WC-Daten synchronisieren[/cyan]")
            _run("sync_from_api.py")
        elif choice == "2":
            if _RICH:
                _con.rule("[cyan]WM-Daten aufbauen[/cyan]")
            _run("rebuild_wm_data.py")
        elif choice == "3":
            if _RICH:
                _con.rule("[cyan]Match-Configs erzeugen[/cyan]")
            _run("generate_match_configs.py")
        elif choice == "4":
            if _RICH:
                _con.rule("[cyan]Team-Daten aktualisieren[/cyan]")
            _run("update_yaml_team_data.py")
        elif choice == "5":
            if _RICH:
                _con.rule("[cyan]Elo neu berechnen[/cyan]")
            _run("refresh_elo.py")
        elif choice == "6":
            _training_submenu()
            continue
        elif choice == "7":
            _handle_lightgbm()
        elif choice == "8":
            _handle_optuna()
        elif choice == "9":
            _handle_pymc()
        elif choice == "10":
            _handle_pagerank()
        elif choice == "11":
            _handle_connector_check()
        elif choice == "12":
            _handle_edit_env()
        else:
            print("  Ungültige Eingabe.")
            continue

        _pause()

    if _RICH:
        _con.print("\n  [dim]Auf Wiedersehen![/dim]\n")
    else:
        print("\n  Auf Wiedersehen!\n")


if __name__ == "__main__":
    main()
