"""``wm2026 doctor`` — Dependency-, Pipeline- und Schema-Self-Check.

Eindeutige Diagnose, **ohne** dass der Hook im Hintergrund laufen muss. Ruft
jede Phase einzeln auf und meldet sie als Tabelle (✅/⚠️/❌) inklusive
Hilfe-Befehl. Exit-Codes:

* ``0`` — alles in Ordnung
* ``1`` — Core-Dep fehlt (Pipeline ist nicht lauffaehig)
* ``2`` — Pipeline laeuft, aber liefert ungueltiges Schema / Smoke schlaegt fehl

Designziel: ein Befehl, ein Block, **keine** Token-Krise. Maximal ~300 Tokens
Output.
"""
from __future__ import annotations

import importlib.util as _util
import json as _json
import sys
from typing import Any


_CORE = ("numpy", "scipy", "httpx", "pydantic", "yaml", "structlog")
_EXTRAS = {
    "matplotlib": "viz",
    "sklearn":    "stats",
    "statsmodels": "stats",
    "vaderSentiment": "sentiment",
    "textblob":   "sentiment",
    "pytest":     "test",
    "optuna":     "tune",
}
_PIPELINE_MODULES = (
    "wm2026.pipeline", "wm2026.markets", "wm2026.edge", "wm2026.report",
    "wm2026.report_html", "wm2026.context", "wm2026.summary",
    "analysis.factor_ensemble", "analysis.match_predictor",
    "analysis.calibration", "models_ml.poisson_goals",
    "factors.registry", "data_sources.orchestrator", "config.settings",
)
# Free-OSS migration — the three live API keys the project policy supports.
# Doctor reports their presence but never the values; missing keys downgrade
# their connector silently (project contract), so this is informational only.
_API_KEYS = (
    ("NVIDIA_API_KEY",        "nvidia_api_key",        "build.nvidia.com — LLM aspect sentiment"),
    ("ODDS_API_KEY",          "odds_api_key",          "the-odds-api.com — live bookmaker odds"),
    ("FOOTBALL_DATA_API_KEY", "football_data_api_key", "football-data.org — fixture cross-check"),
)

# Per-connector dry-run: would it go LIVE with the current Settings? Each row is
# (display_name, use_mock_<flag>, key_required_attr_or_None, note).
_CONNECTORS = (
    ("openfootball",  "use_mock_openfootball",  None,                    "GitHub raw JSON — no auth"),
    ("thesportsdb",   "use_mock_thesportsdb",   None,                    "free public API"),
    ("openligadb",    "use_mock_openligadb",    None,                    "public API"),
    ("wikidata",      "use_mock_wikidata",      None,                    "SPARQL endpoint"),
    ("weather",       "use_mock_weather",       None,                    "open-meteo — no auth"),
    ("rss_news",      "use_mock_rss",           None,                    "BBC/Guardian/ESPN feeds"),
    ("clubelo",       "use_mock_clubelo",       None,                    "Elo ratings (free)"),
    ("football_data", "use_mock_football_data", "football_data_api_key", "needs FOOTBALL_DATA_API_KEY"),
    ("fbref",         "use_mock_fbref",         None,                    "scraper — ToS risk"),
    ("understat",     "use_mock_understat",     None,                    "scraper — ToS risk"),
    ("fotmob",        "use_mock_fotmob",        None,                    "unofficial API"),
    ("sofascore",     "use_mock_sofascore",     None,                    "unofficial API"),
    ("transfermarkt", "use_mock_transfermarkt", None,                    "scraper — polite rate-limit"),
    ("odds_api",      "use_mock_odds_api",      "odds_api_key",          "needs ODDS_API_KEY"),
    ("reddit",        "use_mock_crawler",       None,                    "old.reddit.com/.json — no auth"),
    ("nvidia_llm",    None,                     "nvidia_api_key",        "key-gated; use_nvidia_llm default on"),
)


def _connector_status(settings_obj) -> tuple[list[str], int, int]:
    """Each connector reports ``live``, ``mock`` (toggle on) or ``mock-no-key``."""
    rows: list[str] = []
    live = 0
    for name, mock_flag, key_attr, note in _CONNECTORS:
        if mock_flag is not None and getattr(settings_obj, mock_flag, False):
            rows.append(_row("⚪", name, f"mock — toggle {mock_flag.upper()}=true"))
            continue
        if key_attr is not None and not getattr(settings_obj, key_attr, ""):
            rows.append(_row("⚠️", name, f"mock — no {key_attr.upper()}"))
            continue
        rows.append(_row("✅", name, f"live — {note}"))
        live += 1
    return rows, live, len(_CONNECTORS)
# Minimaler Schema-Vertrag — wenn eines dieser Felder fehlt, ist der Report
# nicht "current" (Schema-Bump hat eine Stelle nicht erreicht).
_SCHEMA_REQUIRED = (
    "schema_version", "match_id", "mode", "lambda_home", "lambda_away",
    "markets", "derived_markets", "edge_table",
    "best_value_cons", "ensemble_confidence", "factors_used", "factors_total",
    "warnings", "claude_tasks",
)


def _row(status: str, name: str, note: str = "") -> str:
    return f"  {status}  {name:30}  {note}"


def _check_imports(prefix: str, names) -> tuple[list[str], int, int]:
    rows: list[str] = []
    missing = 0
    for name in names:
        spec = _util.find_spec(name)
        if spec is None:
            rows.append(_row("❌", name, "import failed"))
            missing += 1
        else:
            rows.append(_row("✅", name, ""))
    return rows, missing, len(names)


def _import_extras() -> tuple[list[str], int, int]:
    rows: list[str] = []
    missing = 0
    for name, group in _EXTRAS.items():
        spec = _util.find_spec(name)
        if spec is None:
            rows.append(_row("⚠️", name, f"missing → `pip install '.[{group}]'`"))
            missing += 1
        else:
            rows.append(_row("✅", name, f"[{group}]"))
    return rows, missing, len(_EXTRAS)


def _quiet_logs() -> None:
    """Silence structlog + stdlib logging so the smoke output is clean."""
    import logging
    logging.disable(logging.CRITICAL)
    try:
        import structlog
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL + 1)
        )
    except Exception:
        pass


def _smoke_predict(*, silence_stdout: bool = False) -> tuple[bool, str, dict[str, Any] | None]:
    """Run a tiny mock predict and return the parsed JSON for schema validation.

    ``silence_stdout`` redirects stdout to ``/dev/null`` for the duration of
    the pipeline call — required for the ``--json`` path so a single clean
    JSON document ends up on the caller's stdout (structlog handlers can
    otherwise be configured to emit there).
    """
    try:
        import asyncio
        import contextlib
        import io
        import os
        _quiet_logs()
        from wm2026.context import synth_config
        from wm2026.pipeline import run_prediction
        from wm2026.report import build_report

        cfg = synth_config(home_team="DocA", away_team="DocB",
                           home_xg=1.4, away_xg=1.3)

        def _do():
            return asyncio.run(run_prediction(
                cfg, mode="mock", bootstrap_n=32,
                odds_1x2=[2.10, 3.40, 3.20],
            ))

        if silence_stdout:
            with open(os.devnull, "w") as devnull, \
                    contextlib.redirect_stdout(devnull), \
                    contextlib.redirect_stderr(io.StringIO()):
                result = _do()
        else:
            result = _do()
        js = build_report(result)["json"]
        return True, "ok", js
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def run(*, verbose: bool = False) -> int:
    """Print the diagnostic block; return the exit code."""
    _quiet_logs()
    print("# wm2026 doctor")
    print(f"python {sys.version.split()[0]}  ·  platform {sys.platform}")
    print()
    print("## Core deps (Pipeline-pflichtig)")
    core_rows, core_missing, core_total = _check_imports("core", _CORE)
    print("\n".join(core_rows))
    if core_missing:
        print(f"\n❌ {core_missing}/{core_total} core deps missing — "
              f"`pip install -r requirements.txt` first.")
        return 1
    print(f"\n✅ {core_total}/{core_total} core deps ok")
    print()

    print("## Optional extras (sauberer Degrade ohne)")
    extra_rows, extra_missing, extra_total = _import_extras()
    print("\n".join(extra_rows))
    note = (f"\n→ {extra_total - extra_missing}/{extra_total} extras present"
            + (" (fehlende sind optional)" if extra_missing else ""))
    print(note)
    print()

    print("## Pipeline modules")
    pipe_rows, pipe_missing, pipe_total = _check_imports("pipe", _PIPELINE_MODULES)
    print("\n".join(pipe_rows))
    if pipe_missing:
        print(f"\n❌ {pipe_missing}/{pipe_total} pipeline modules failed to import.")
        return 2
    print(f"\n✅ {pipe_total}/{pipe_total} pipeline modules import")
    print()

    # Free-OSS migration — show which of the three live keys are configured.
    # Connectors still fall back to mock when a key is missing, so this is
    # informational; no key is a hard error.
    from config.settings import settings as _s
    print("## API keys (free / free-tier — Reddit & Twitter not needed)")
    key_present = 0
    for env_var, attr, note in _API_KEYS:
        value = getattr(_s, attr, "") or ""
        if value:
            print(_row("✅", env_var, f"set — {note}"))
            key_present += 1
        else:
            print(_row("⚠️", env_var, f"unset → connector stays mock — {note}"))
    print(f"\n→ {key_present}/{len(_API_KEYS)} live keys configured "
          f"(missing keys degrade gracefully)")
    print()

    # Connector dry-run — same settings the orchestrator will read. ✅ = live,
    # ⚪ = mock by toggle, ⚠️ = mock because key missing. Nothing here is a
    # hard error; the pipeline keeps running on mock fallback either way.
    print("## Connectors (would-go-live with current settings)")
    conn_rows, conn_live, conn_total = _connector_status(_s)
    print("\n".join(conn_rows))
    print(f"\n→ {conn_live}/{conn_total} connectors would go live")
    print()

    print("## Schema smoke (mock predict, bootstrap_n=32)")
    ok, msg, js = _smoke_predict(silence_stdout=False)
    if not ok or js is None:
        print(_row("❌", "predict + build_report", msg))
        return 2
    missing_keys = [k for k in _SCHEMA_REQUIRED if k not in js]
    if missing_keys:
        print(_row("❌", "schema check", f"missing: {', '.join(missing_keys)}"))
        return 2
    s = sum(js["markets"]["1x2"][k] for k in ("home", "draw", "away"))
    if abs(s - 1.0) > 0.01:
        print(_row("❌", "1X2 sum", f"= {s:.4f} (expected 1.000 ± 0.01)"))
        return 2
    print(_row("✅", "predict + build_report",
               f"schema {js.get('schema_version')}, mode {js['mode']}, "
               f"factors {js['factors_used']}/{js['factors_total']}"))
    print(_row("✅", "schema fields", f"all {len(_SCHEMA_REQUIRED)} required keys present"))
    print(_row("✅", "1X2 ΣP", f"= {s:.4f}"))

    if verbose:
        # Token-sparsam: nur ein Sanity-Diff der edge-table fuer den User.
        print()
        print("## verbose: edge_table head")
        for r in (js.get("edge_table") or [])[:4]:
            print(f"  {r.get('market'):<12} {r.get('selection'):<8} "
                  f"edge={r.get('edge_pct')}  p5={r.get('edge_pct_cons')}  "
                  f"action={r.get('action')}")

    print()
    print("🎉 doctor: all checks passed — pipeline is ready.")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser("wm2026 doctor")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--json", action="store_true",
                   help="emit a compact JSON status instead of human text")
    args = p.parse_args(argv)

    if args.json:
        # Headless / CI-Pfad.
        core_rows, core_missing, core_total = _check_imports("core", _CORE)
        extra_rows, extra_missing, extra_total = _import_extras()
        pipe_rows, pipe_missing, pipe_total = _check_imports("pipe", _PIPELINE_MODULES)
        # Read the connector/key status BEFORE the smoke predict: the smoke runs
        # the pipeline in mock mode, which calls apply_runtime_profile("mock")
        # and flips every USE_MOCK_* flag to True on the global settings — so
        # reading the status afterwards would always report 0 live connectors.
        from config.settings import settings as _s
        keys = {env: bool(getattr(_s, attr, "")) for env, attr, _ in _API_KEYS}
        _, conn_live, conn_total = _connector_status(_s)
        ok, msg, js = (True, "skipped", None)
        if core_missing == 0 and pipe_missing == 0:
            ok, msg, js = _smoke_predict(silence_stdout=True)
        status = {
            "core_missing": core_missing,
            "extras_missing": extra_missing,
            "pipeline_missing": pipe_missing,
            "smoke_ok": bool(ok),
            "smoke_msg": msg,
            "schema_version": (js or {}).get("schema_version"),
            "api_keys_present": keys,
            "connectors_live": conn_live,
            "connectors_total": conn_total,
        }
        print(_json.dumps(status, indent=2, ensure_ascii=False))
        if core_missing:
            return 1
        if pipe_missing or not ok:
            return 2
        return 0
    return run(verbose=args.verbose)


__all__ = ["run", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
