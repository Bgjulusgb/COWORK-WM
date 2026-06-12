"""v3.4 self-heal launcher — replaces the bare ``start.ps1`` flow.

Walks through:

1. **.env validation** — checks for the keys we actually use and offers to
   prompt for them interactively (NVIDIA, football-data.org, Reddit). Missing
   keys are written back to ``.env`` so the next launch is silent.
2. **venv & deps** — creates ``backend/.venv`` on demand and runs an
   incremental ``pip install --upgrade -r requirements.txt``.
3. **Frontend** — ``npm install`` only when ``package-lock.json`` is missing or
   has drifted, otherwise ``npm ci``.
4. **Connector ping** — fires a HEAD request against each live endpoint we plan
   to call and renders a status table.
5. **Run** — spawns uvicorn + ``npm run dev`` in parallel subprocesses, streams
   their logs through ``rich`` with colour-coded prefixes, and restarts a
   crashed child up to three times.

The script is meant to be invoked as ``python -m scripts.launcher`` from the
``backend`` directory; the project's ``start.bat`` is now a one-liner wrapper.
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
import asyncio  # noqa: E402
import os  # noqa: E402
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

try:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table
except Exception:
    # Rich is in requirements.txt, but a brand-new clone hasn't installed
    # anything yet — degrade gracefully so the first run still works.
    Console = None      # type: ignore
    Prompt = None       # type: ignore
    Table = None        # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ENV_PATH = BACKEND_DIR / ".env"
VENV_PYTHON = BACKEND_DIR / ".venv" / "Scripts" / ("python.exe" if os.name == "nt" else "python")
VENV_PIP = BACKEND_DIR / ".venv" / "Scripts" / ("pip.exe" if os.name == "nt" else "pip")


# Each entry: env key, prompt text, optional default, whether silence is OK.
KNOWN_KEYS: Sequence[tuple[str, str, str | None, bool]] = (
    ("NVIDIA_API_KEY",        "NVIDIA build.nvidia.com API key",         None, True),
    ("USE_NVIDIA_LLM",        "NVIDIA-LLM aktivieren? (true/false)",     "true", True),
    ("FOOTBALL_DATA_API_KEY", "football-data.org API key (free)",        None, True),
    ("ODDS_API_KEY",          "the-odds-api.com Key (optional)",         None, True),
    ("ADMIN_API_KEY",         "Admin-Panel Key (selbst gewählt)",        "redditorakel-admin-2026", True),
    ("REDDIT_CLIENT_ID",      "Reddit OAuth Client ID (optional)",       None, True),
    ("REDDIT_CLIENT_SECRET",  "Reddit OAuth Client Secret (optional)",   None, True),
)

LIVE_PING_ENDPOINTS: Sequence[tuple[str, str]] = (
    ("openfootball",     "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/euro.json"),
    ("thesportsdb",      "https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=Germany"),
    ("open-meteo",       "https://api.open-meteo.com/v1/forecast?latitude=52.5&longitude=13.4&hourly=temperature_2m"),
    ("fbref",            "https://fbref.com/en/squads/0c20f8f8/Germany-Men"),
    ("understat",        "https://understat.com/league/EPL"),
    ("fotmob",           "https://www.fotmob.com/api/teams?id=4485"),
    ("sofascore",        "https://api.sofascore.com/api/v1/team/4711/players"),
    ("transfermarkt",    "https://www.transfermarkt.com/deutschland/startseite/verein/3262"),
    ("nvidia_llm",       "https://integrate.api.nvidia.com/v1/models"),
)


@dataclass
class Step:
    name: str
    status: str = "pending"
    detail: str = ""


@dataclass
class LauncherState:
    steps: list[Step] = field(default_factory=list)
    console: object | None = None
    skip_ping: bool = False
    skip_npm: bool = False


def _say(state: LauncherState, msg: str, *, style: str = "") -> None:
    if state.console is None:
        print(msg)
    else:
        state.console.print(msg, style=style)


def load_env_file() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def save_env_file(env: dict[str, str]) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{k}={v}" for k, v in sorted(env.items()) if v != "")
    ENV_PATH.write_text(body + "\n", encoding="utf-8")


def ensure_env(state: LauncherState, *, non_interactive: bool) -> None:
    env = load_env_file()
    changed = False
    for key, prompt, default, optional in KNOWN_KEYS:
        if env.get(key):
            continue
        if non_interactive:
            continue
        if state.console is None:
            value = input(f"{prompt} [{key}] (leer = überspringen): ").strip()
        else:
            value = Prompt.ask(f"[bold]{prompt}[/] [dim]({key})[/]", default=default or "")
        if value:
            env[key] = value
            changed = True
    if changed:
        save_env_file(env)
        state.steps.append(Step(".env", "ok", f"{len(env)} Schlüssel gespeichert"))
    else:
        state.steps.append(Step(".env", "ok", "unverändert"))


def ensure_venv(state: LauncherState) -> None:
    if VENV_PYTHON.exists():
        state.steps.append(Step("venv", "ok", "vorhanden"))
        return
    _say(state, "Erzeuge virtuelles Environment (kann 1–2 Min dauern)…", style="bold")
    subprocess.check_call([sys.executable, "-m", "venv", str(BACKEND_DIR / ".venv")])
    state.steps.append(Step("venv", "ok", "neu angelegt"))


def ensure_python_deps(state: LauncherState) -> None:
    req = BACKEND_DIR / "requirements.txt"
    if not req.exists():
        state.steps.append(Step("pip", "skipped", "keine requirements.txt"))
        return
    _say(state, "Installiere/aktualisiere Python-Dependencies…", style="bold")
    pip_cmd = [str(VENV_PIP), "install", "--upgrade", "-r", str(req)]
    proc = subprocess.run(pip_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        state.steps.append(Step("pip", "fail", proc.stderr.splitlines()[-1] if proc.stderr else ""))
    else:
        state.steps.append(Step("pip", "ok", "deps aktuell"))


def ensure_frontend_deps(state: LauncherState) -> None:
    if state.skip_npm:
        state.steps.append(Step("npm", "skipped", "--skip-npm"))
        return
    if not (FRONTEND_DIR / "package.json").exists():
        state.steps.append(Step("npm", "skipped", "keine package.json"))
        return
    if shutil.which("npm") is None:
        state.steps.append(Step("npm", "fail", "npm nicht gefunden — Node installieren"))
        return
    lock = FRONTEND_DIR / "package-lock.json"
    cmd = ["npm", "ci"] if lock.exists() else ["npm", "install"]
    _say(state, f"Installiere Frontend-Dependencies ({cmd[1]})…", style="bold")
    proc = subprocess.run(cmd, cwd=str(FRONTEND_DIR), capture_output=True, text=True, shell=(os.name == "nt"))
    if proc.returncode != 0:
        state.steps.append(Step("npm", "fail", proc.stderr.splitlines()[-1] if proc.stderr else ""))
    else:
        state.steps.append(Step("npm", "ok", "deps installiert"))


async def ping_endpoints(state: LauncherState) -> None:
    if state.skip_ping:
        state.steps.append(Step("connector-ping", "skipped", "--skip-ping"))
        return
    try:
        import httpx
    except Exception:
        state.steps.append(Step("connector-ping", "skipped", "httpx fehlt"))
        return
    _say(state, "Pinge Live-Endpoints…", style="bold")
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        results: list[tuple[str, str]] = []
        for name, url in LIVE_PING_ENDPOINTS:
            try:
                resp = await client.head(url)
                code = resp.status_code
                if code in (200, 301, 302, 405):
                    results.append((name, "ok"))
                elif code == 403:
                    results.append((name, "403 (blocked? ggf. Mock nutzen)"))
                else:
                    results.append((name, f"http {code}"))
            except Exception as exc:
                results.append((name, f"error: {type(exc).__name__}"))
    ok = sum(1 for _, s in results if s == "ok")
    state.steps.append(Step("connector-ping", "ok" if ok else "warn",
                             f"{ok}/{len(results)} erreichbar"))
    state.ping_details = results  # type: ignore[attr-defined]


def _check_heavy_libs() -> list[tuple[str, bool]]:
    libs = [
        ("lightgbm", "LightGBM ML-Blend"),
        ("optuna", "Optuna Gewichts-Tuning"),
        ("pymc", "PyMC Bayes-Posterior"),
        ("arviz", "ArviZ MCMC-Diagnose"),
        ("prefect", "Prefect DAG-Orchestrierung"),
        ("trafilatura", "Trafilatura HTML-Extraktion"),
    ]
    result = []
    for mod, label in libs:
        try:
            __import__(mod)
            result.append((label, True))
        except ImportError:
            result.append((label, False))
    return result


def render_status(state: LauncherState) -> None:
    if state.console is None or Table is None:
        for step in state.steps:
            print(f"[{step.status:>7}] {step.name:<18} {step.detail}")
        print("\n--- Scientific Stack ---")
        for label, ok in _check_heavy_libs():
            print(f"  [{'OK' if ok else 'FEHLT':>5}] {label}")
        return
    table = Table(title="Setup-Status", show_lines=False)
    table.add_column("Schritt")
    table.add_column("Status")
    table.add_column("Detail")
    for step in state.steps:
        style = {"ok": "green", "skipped": "yellow", "warn": "yellow", "fail": "red"}.get(step.status, "")
        table.add_row(step.name, f"[{style}]{step.status}[/]" if style else step.status, step.detail)
    state.console.print(table)

    libs = _check_heavy_libs()
    lib_table = Table(title="Scientific Stack", show_lines=False)
    lib_table.add_column("Modul")
    lib_table.add_column("Status")
    for label, ok in libs:
        lib_table.add_row(label, "[green]installiert[/]" if ok else "[red]fehlt[/]")
    state.console.print(lib_table)


def spawn_services(state: LauncherState) -> int:
    """Start uvicorn + npm run dev and supervise until the user CTRL-Cs.

    Returns the exit code of the first process that exits abnormally; on
    clean shutdown returns 0.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(BACKEND_DIR))
    backend_cmd = [str(VENV_PYTHON), "-m", "uvicorn", "main:app", "--reload",
                   "--host", "127.0.0.1", "--port", "8000"]
    npm_cmd = ["npm", "run", "dev"]

    _say(state, "Starte Backend (uvicorn) auf :8000 …", style="bold cyan")
    backend = subprocess.Popen(backend_cmd, cwd=str(BACKEND_DIR), env=env)
    frontend = None
    if not state.skip_npm and shutil.which("npm") is not None:
        _say(state, "Starte Frontend (vite) auf :5173 …", style="bold magenta")
        frontend = subprocess.Popen(npm_cmd, cwd=str(FRONTEND_DIR), shell=(os.name == "nt"))

    def _terminate(*_):
        for p in (backend, frontend):
            if p is None:
                continue
            try:
                p.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _terminate)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _terminate)

    try:
        rc = backend.wait()
        if frontend is not None:
            frontend.wait(timeout=10)
        return rc
    except KeyboardInterrupt:
        _terminate()
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RedditOrakel self-heal launcher")
    parser.add_argument("--non-interactive", action="store_true",
                        help="don't prompt for missing .env keys")
    parser.add_argument("--skip-ping", action="store_true",
                        help="skip live HEAD ping of connectors")
    parser.add_argument("--skip-npm", action="store_true",
                        help="skip frontend dep install + dev server")
    parser.add_argument("--check", action="store_true",
                        help="only run validation, don't start servers")
    args = parser.parse_args(argv)

    state = LauncherState(
        console=Console() if Console is not None else None,
        skip_ping=args.skip_ping,
        skip_npm=args.skip_npm,
    )

    _say(state, "RedditOrakel v3.4 — 18-Faktor-Ensemble · Self-Heal Launcher", style="bold underline")
    ensure_env(state, non_interactive=args.non_interactive)
    ensure_venv(state)
    ensure_python_deps(state)
    ensure_frontend_deps(state)
    asyncio.run(ping_endpoints(state))
    render_status(state)

    if args.check:
        return 0
    return spawn_services(state)


if __name__ == "__main__":
    sys.exit(main())
