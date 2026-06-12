"""Train the optional xG regressor (EXTEND-05) — leakage-free + temporally validated.

Walks the match history in chronological order and, for each match, snapshots
features computed **only from prior matches** (no look-ahead):

    elo_delta            walk-forward World-Football-Elo before kickoff
    home/away_avg_xg     rolling mean goals scored (last 10)
    home/away_xg_conceded rolling mean goals conceded (last 10)
    home/away_form_pts   points from the last 5 results
    h2h_score            (home_wins - away_wins) over prior meetings

Target = (home, away) goals.  Evaluated with expanding-window temporal CV.

Data sources (highest to lowest priority):
  1. openfootball (WC 2010/2014/2018/2022 via GitHub) — automatic
  2. --data-dir PATH   directory of local .json / .csv match files

Usage:
    cd backend && python scripts/train_xg_predictor.py
    cd backend && python scripts/train_xg_predictor.py --data-dir data/training
    cd backend && python scripts/train_xg_predictor.py --local-only --data-dir data/training
"""
from __future__ import annotations

import argparse
import asyncio
import csv as csv_mod
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from data_sources.openfootball import OpenfootballConnector, _parse_worldcup_json  # noqa: E402
from data_sources.schemas import HistoricalMatch  # noqa: E402
from data_sources.team_codes import CODE_TO_NAMES, preferred_name, to_code  # noqa: E402
from factors.elo_update import update_match  # noqa: E402
from models_ml.xg_predictor import _FEATURE_ORDER  # noqa: E402

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _con = Console()
    def _p(msg: str, **kw): _con.print(msg, **kw)
    def _h(msg: str): _con.rule(f"[bold cyan]{msg}[/bold cyan]")
    _RICH = True
except ImportError:
    _con = None
    def _p(msg: str, **kw): print(msg)
    def _h(msg: str): print(f"\n── {msg} ──")
    _RICH = False

_ARTIFACT = ROOT / "models_ml" / "artifacts" / "xg_predictor.json"
_POINTS = {"W": 3, "D": 1, "L": 0}
_MIN_PRIOR = 1   # require at least 1 prior match so rolling deques have real data.
# 3 excluded all group-stage games (each team's first 3 WC matches), leaving
# only ~30 knockout rows and producing a useless home MAE of 2.1 goals.
# At 1, matchday 2+ games qualify; missing priors fall back to the 1.3-goal
# default in _mean() — Ridge down-weights those noisy rows automatically.
_DEFAULT_ELO = 1500.0


# ── Local file helpers ──────────────────────────────────────────────────────

def _parse_date_str(value: Any, default_year: int = 2020) -> datetime:
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value[:10], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return datetime(default_year, 6, 1, tzinfo=timezone.utc)


def _make_match(home: str, away: str, date: str, hs: int, as_: int,
                source: str, tier: int = 2) -> HistoricalMatch | None:
    hc = to_code(home)
    ac = to_code(away)
    if not hc or not ac:
        return None
    return HistoricalMatch(
        source=source,
        tournament=source,
        competition_tier=tier,
        home_code=hc,
        away_code=ac,
        home_name=home,
        away_name=away,
        kickoff_utc=_parse_date_str(date),
        home_score=hs,
        away_score=as_,
        is_finished=True,
    )


def _load_local_json(path: Path, seen: set) -> list[HistoricalMatch]:
    """Accept openfootball JSON OR simple array [{date,home,away,home_score,away_score}]."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    source = f"local:{path.stem}"
    out: list[HistoricalMatch] = []

    # ── Simple array format ──
    if isinstance(raw, list):
        for item in raw:
            try:
                m = _make_match(
                    str(item.get("home") or item.get("home_team") or ""),
                    str(item.get("away") or item.get("away_team") or ""),
                    str(item.get("date") or item.get("kickoff") or ""),
                    int(item.get("home_score") or item.get("score1") or 0),
                    int(item.get("away_score") or item.get("score2") or 0),
                    source=source,
                    tier=int(item.get("competition_tier", 2)),
                )
                if m is None:
                    continue
                key = (m.kickoff_utc.date(), m.home_code, m.away_code, m.home_score, m.away_score)
                if key not in seen:
                    seen.add(key)
                    out.append(m)
            except (TypeError, ValueError):
                continue
        return out

    # ── openfootball dict format ──
    tournament_name = raw.get("name") or path.stem
    year_hint = 2020
    if isinstance(tournament_name, str):
        for y in range(2000, 2030):
            if str(y) in tournament_name:
                year_hint = y
                break
    parsed = _parse_worldcup_json(raw, year_hint)
    for m in parsed:
        if m.home_score is None:
            continue
        m = m.model_copy(update={"source": source, "tournament": tournament_name})
        key = (m.kickoff_utc.date(), m.home_code, m.away_code, m.home_score, m.away_score)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _load_local_csv(path: Path, seen: set) -> list[HistoricalMatch]:
    """CSV format: date,home,away,home_score,away_score[,competition_tier]
    Lines starting with # and the header row are skipped automatically."""
    source = f"local:{path.stem}"
    out: list[HistoricalMatch] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv_mod.reader(r for r in f if r.strip() and not r.startswith("#")):
            if not row or row[0].lower().strip() in ("date", "kickoff", "datum"):
                continue
            try:
                date_str, home, away = row[0].strip(), row[1].strip(), row[2].strip()
                hs, as_ = int(row[3].strip()), int(row[4].strip())
                tier = int(row[5].strip()) if len(row) > 5 else 2
                m = _make_match(home, away, date_str, hs, as_, source=source, tier=tier)
                if m is None:
                    continue
                key = (m.kickoff_utc.date(), m.home_code, m.away_code, m.home_score, m.away_score)
                if key not in seen:
                    seen.add(key)
                    out.append(m)
            except (ValueError, IndexError):
                continue
    return out


def _load_local_files(data_dir: Path, seen: set) -> list[HistoricalMatch]:
    out: list[HistoricalMatch] = []
    any_found = False
    for p in sorted(data_dir.glob("**/*.json")):
        any_found = True
        try:
            loaded = _load_local_json(p, seen)
            _p(f"  [green]✓[/green] {p.name}: {len(loaded)} neue Spiele" if _RICH else
               f"  OK  {p.name}: {len(loaded)} neue Spiele")
            out.extend(loaded)
        except Exception as exc:
            _p(f"  [yellow]![/yellow] {p.name}: {exc}" if _RICH else f"  WARN {p.name}: {exc}")
    for p in sorted(data_dir.glob("**/*.csv")):
        any_found = True
        try:
            loaded = _load_local_csv(p, seen)
            _p(f"  [green]✓[/green] {p.name}: {len(loaded)} neue Spiele" if _RICH else
               f"  OK  {p.name}: {len(loaded)} neue Spiele")
            out.extend(loaded)
        except Exception as exc:
            _p(f"  [yellow]![/yellow] {p.name}: {exc}" if _RICH else f"  WARN {p.name}: {exc}")
    if not any_found:
        _p(f"  (Keine JSON/CSV-Dateien in [italic]{data_dir}[/italic] gefunden.)" if _RICH else
           f"  (Keine Dateien in {data_dir})")
    return out


# ── Network history ─────────────────────────────────────────────────────────

async def _fetch_network() -> tuple[list[HistoricalMatch], set]:
    conn = OpenfootballConnector()
    try:
        results = await asyncio.gather(
            *(conn.get_historical_results(c) for c in CODE_TO_NAMES)
        )
    finally:
        from data_sources.base import BaseConnector
        await BaseConnector.close_all()
    seen: set = set()
    matches: list[HistoricalMatch] = []
    for res in results:
        for m in res.data or []:
            key = (m.kickoff_utc.date(), m.home_code, m.away_code, m.home_score, m.away_score)
            if key not in seen and m.home_score is not None:
                seen.add(key)
                matches.append(m)
    return matches, seen


async def _history(data_dir: Path | None, local_only: bool) -> list[HistoricalMatch]:
    _h("Daten laden")
    matches: list[HistoricalMatch] = []
    seen: set = set()

    if not local_only:
        if settings.use_mock_openfootball:
            _p("[yellow]  MOCK-Modus aktiv[/yellow] — Setze USE_MOCK_OPENFOOTBALL=false für echte WC-Daten." if _RICH else
               "  MOCK-Modus aktiv — USE_MOCK_OPENFOOTBALL=false setzen für echte Daten.")
        _p("  Netzwerk: openfootball WC 2010/2014/2018/2022 ...")
        matches, seen = await _fetch_network()
        by_year: dict[str, int] = defaultdict(int)
        for m in matches:
            by_year[m.tournament or "?"] += 1
        for t, n in sorted(by_year.items()):
            _p(f"    {t}: {n} Spiele")
        _p(f"  [bold]Netzwerk gesamt: {len(matches)} Spiele[/bold]" if _RICH else
           f"  Netzwerk gesamt: {len(matches)} Spiele")
    else:
        _p("[yellow]  --local-only: Netzwerk übersprungen[/yellow]" if _RICH else
           "  --local-only: Netzwerk übersprungen")

    if data_dir:
        _p(f"\n  Lokale Dateien aus [italic]{data_dir}[/italic] ..." if _RICH else
           f"\n  Lokale Dateien: {data_dir} ...")
        local = _load_local_files(data_dir, seen)
        matches.extend(local)
        _p(f"  Lokale Dateien gesamt: {len(local)} neue Spiele")

    default_dir = ROOT / "data" / "training"
    if not data_dir and default_dir.is_dir():
        _p(f"\n  Auto-scan: [italic]{default_dir}[/italic] ..." if _RICH else
           f"\n  Auto-scan {default_dir} ...")
        local = _load_local_files(default_dir, seen)
        matches.extend(local)
        if local:
            _p(f"  {len(local)} neue Spiele aus {default_dir.name}")

    matches.sort(key=lambda m: m.kickoff_utc)
    _p(f"\n  [bold green]Gesamt: {len(matches)} Spiele für das Training[/bold green]" if _RICH else
       f"\n  Gesamt: {len(matches)} Spiele")
    return matches


# ── Dataset builder ─────────────────────────────────────────────────────────

def _build_dataset(matches: list):
    """Walk forward, emit (features, home_goals, away_goals) with no leakage."""
    elo: dict[str, float] = defaultdict(lambda: _DEFAULT_ELO)
    results: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    scored: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    conceded: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    pair_wins: dict[tuple, int] = defaultdict(int)
    pair_draws: dict[frozenset, int] = defaultdict(int)

    def _mean(dq, default):
        return sum(dq) / len(dq) if dq else default

    rows, y_home, y_away = [], [], []
    for m in matches:
        h, a = m.home_code, m.away_code
        hs, as_ = int(m.home_score), int(m.away_score)

        if len(results[h]) >= _MIN_PRIOR and len(results[a]) >= _MIN_PRIOR:
            hw = pair_wins[(h, a)]
            aw = pair_wins[(a, h)]
            dr = pair_draws[frozenset({h, a})]
            n = hw + aw + dr
            feat = {k: 0.0 for k in _FEATURE_ORDER}
            feat["elo_delta"] = elo[h] - elo[a]
            feat["home_avg_xg"] = _mean(scored[h], 1.3)
            feat["away_avg_xg"] = _mean(scored[a], 1.3)
            feat["home_avg_xg_conceded"] = _mean(conceded[h], 1.3)
            feat["away_avg_xg_conceded"] = _mean(conceded[a], 1.3)
            feat["home_form_pts"] = sum(_POINTS[r] for r in results[h])
            feat["away_form_pts"] = sum(_POINTS[r] for r in results[a])
            feat["h2h_score"] = (hw - aw) / n if n else 0.0
            rows.append([feat[k] for k in _FEATURE_ORDER])
            y_home.append(float(hs))
            y_away.append(float(as_))

        # Update state AFTER snapshotting
        elo[h], elo[a] = update_match(elo[h], elo[a], hs, as_, tier=getattr(m, "competition_tier", 4))
        hr = "W" if hs > as_ else ("D" if hs == as_ else "L")
        results[h].append(hr)
        results[a].append({"W": "L", "L": "W", "D": "D"}[hr])
        scored[h].append(hs); conceded[h].append(as_)
        scored[a].append(as_); conceded[a].append(hs)
        if hs > as_:
            pair_wins[(h, a)] += 1
        elif hs < as_:
            pair_wins[(a, h)] += 1
        else:
            pair_draws[frozenset({h, a})] += 1

    return rows, y_home, y_away


# ── Cross-validation ─────────────────────────────────────────────────────────

def _temporal_cv(np, Ridge, X, y, folds=4):
    n = len(y)
    if n < folds * 4:
        return None
    fold = n // (folds + 1)
    errs = []
    for k in range(1, folds + 1):
        tr_end = fold * k
        te_end = fold * (k + 1)
        model = Ridge(alpha=1.0).fit(X[:tr_end], y[:tr_end])
        pred = np.clip(model.predict(X[tr_end:te_end]), 0, None)
        errs.append(float(np.mean(np.abs(pred - y[tr_end:te_end]))))
    return sum(errs) / len(errs)


# ── Output helpers ───────────────────────────────────────────────────────────

def _print_results(artifact: dict, mae_home: float | None, mae_away: float | None) -> None:
    _h("Ergebnis")

    if _RICH:
        # MAE summary
        good_h = mae_home is not None and mae_home < 1.0
        good_a = mae_away is not None and mae_away < 1.0
        color_h = "green" if good_h else ("yellow" if mae_home and mae_home < 1.3 else "red")
        color_a = "green" if good_a else ("yellow" if mae_away and mae_away < 1.3 else "red")
        mae_h_str = f"[{color_h}]{mae_home:.3f}[/{color_h}]" if mae_home is not None else "—"
        mae_a_str = f"[{color_a}]{mae_away:.3f}[/{color_a}]" if mae_away is not None else "—"
        _con.print(f"\n  Temporal-CV MAE  Heim: {mae_h_str}  Gast: {mae_a_str}  (Ziel: < 1.0 Tore)")

        # Feature importance table
        t = Table(title="Top-Feature-Gewichte", show_header=True, header_style="bold")
        t.add_column("Feature", style="cyan")
        t.add_column("Heim", justify="right")
        t.add_column("Gast", justify="right")
        home_c = artifact.get("home_coeffs", {})
        away_c = artifact.get("away_coeffs", {})
        ranked = sorted(home_c.keys(), key=lambda k: abs(home_c.get(k, 0)), reverse=True)
        for feat in ranked[:8]:
            hv = home_c.get(feat, 0.0)
            av = away_c.get(feat, 0.0)
            t.add_row(
                feat,
                f"[green]+{hv:.4f}[/green]" if hv >= 0 else f"[red]{hv:.4f}[/red]",
                f"[green]+{av:.4f}[/green]" if av >= 0 else f"[red]{av:.4f}[/red]",
            )
        _con.print(t)

        # Activation recommendation
        if mae_home is not None and mae_away is not None:
            avg_mae = (mae_home + mae_away) / 2
            if avg_mae < 0.80:
                rec = "[bold green]✓ Gut — aktivieren mit FACTOR_WEIGHT_ML=0.07[/bold green]"
            elif avg_mae < 1.00:
                rec = "[green]✓ Akzeptabel — aktivieren mit FACTOR_WEIGHT_ML=0.04[/green]"
            elif avg_mae < 1.30:
                rec = "[yellow]~ Schwach — FACTOR_WEIGHT_ML=0 empfohlen, mehr Daten sammeln[/yellow]"
            else:
                rec = "[red]✗ Schlecht — nicht aktivieren, zu wenige Trainingsdaten[/red]"
            _con.print(f"\n  Empfehlung: {rec}")
    else:
        mae_h_str = f"{mae_home:.3f}" if mae_home is not None else "n/a"
        mae_a_str = f"{mae_away:.3f}" if mae_away is not None else "n/a"
        print(f"  Temporal-CV MAE  home={mae_h_str} goals  away={mae_a_str} goals")
        print(f"\n  Top-Features (Heim-Koeffizienten):")
        home_c = artifact.get("home_coeffs", {})
        for feat, val in sorted(home_c.items(), key=lambda x: abs(x[1]), reverse=True)[:6]:
            sign = "+" if val >= 0 else ""
            print(f"    {feat:<30} {sign}{val:.4f}")
        if mae_home is not None and mae_away is not None:
            avg_mae = (mae_home + mae_away) / 2
            if avg_mae < 1.0:
                print("\n  Empfehlung: Akzeptabel — FACTOR_WEIGHT_ML=0.05 setzen")
            else:
                print("\n  Empfehlung: Nicht aktivieren — zu wenig Trainingsdaten")


# ── Main ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trainiert den optionalen xG-Regressionsterm für das WM-Vorhersagemodell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python scripts/train_xg_predictor.py
  python scripts/train_xg_predictor.py --data-dir data/training
  python scripts/train_xg_predictor.py --local-only --data-dir data/training

Eigene Daten:
  Erstelle backend/data/training/ und lege dort .json oder .csv Dateien ab.

  CSV-Format (Kopfzeile optional):
    date,home,away,home_score,away_score[,competition_tier]
    2023-06-20,Argentina,Brazil,1,0,2

  JSON-Format (einfaches Array):
    [{"date":"2023-06-20","home":"Argentina","away":"Brazil","home_score":1,"away_score":0}]

  JSON-Format (openfootball):
    {"name":"CONMEBOL 2023","rounds":[{"matches":[...]}]}
""",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Ordner mit lokalen .json/.csv Dateien (wird zusätzlich zu openfootball geladen)",
    )
    p.add_argument(
        "--local-only", action="store_true",
        help="Nur lokale Dateien verwenden, kein Netzwerkzugriff",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import numpy as np
        from sklearn.linear_model import Ridge
    except ImportError:
        _p("[red]scikit-learn nicht installiert. Ausführen: pip install scikit-learn numpy[/red]" if _RICH else
           "scikit-learn nicht installiert. Ausführen: pip install scikit-learn numpy")
        return

    if _RICH:
        _con.print(Panel("[bold cyan]RedditOrakel — ML-Training[/bold cyan]\nxG-Prädiktor (Ridge Regression)",
                         border_style="cyan"))
    else:
        print("\n  RedditOrakel — ML-Training\n")

    matches = asyncio.run(_history(args.data_dir, args.local_only))

    _h("Datensatz aufbauen")
    rows, y_home, y_away = _build_dataset(matches)
    _p(f"  Leakage-freie Trainingszeilen: [bold]{len(rows)}[/bold] (aus {len(matches)} Spielen)" if _RICH else
       f"  Rows: {len(rows)} (from {len(matches)} matches)")

    if len(rows) < 30:
        _p(
            f"\n  [red]Zu wenige Zeilen ({len(rows)}) — mindestens 30 benötigt.[/red]\n"
            f"  Tipps:\n"
            f"    • Setze USE_MOCK_OPENFOOTBALL=false in .env für echte WC-Daten\n"
            f"    • Lege eigene Matchdaten in [italic]{ROOT}/data/training/[/italic]\n"
            f"    • Verwende --data-dir für einen anderen Ordner"
            if _RICH else
            f"Zu wenige Rows ({len(rows)}). USE_MOCK_OPENFOOTBALL=false setzen oder --data-dir angeben."
        )
        return

    _h("Training & Validierung")
    X = np.array(rows)
    yh, ya = np.array(y_home), np.array(y_away)

    mae_home = _temporal_cv(np, Ridge, X, yh)
    mae_away = _temporal_cv(np, Ridge, X, ya)
    if mae_home is not None:
        _p(f"  Temporal-CV MAE  Heim={mae_home:.3f} Tore  Gast={mae_away:.3f} Tore")

    home_model = Ridge(alpha=1.0).fit(X, yh)
    away_model = Ridge(alpha=1.0).fit(X, ya)

    artifact = {
        "model": "ridge",
        "trained_on": len(rows),
        "cv_mae_home": round(mae_home, 4) if mae_home is not None else None,
        "cv_mae_away": round(mae_away, 4) if mae_away is not None else None,
        "home_coeffs": {k: float(c) for k, c in zip(_FEATURE_ORDER, home_model.coef_)},
        "away_coeffs": {k: float(c) for k, c in zip(_FEATURE_ORDER, away_model.coef_)},
        "home_intercept": float(home_model.intercept_),
        "away_intercept": float(away_model.intercept_),
    }
    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    _p(f"\n  [green]✓ Artifact geschrieben:[/green] {_ARTIFACT}" if _RICH else
       f"Wrote {_ARTIFACT}")

    _print_results(artifact, mae_home, mae_away)

    _p("\n  [dim]Setze FACTOR_WEIGHT_ML > 0 in .env um MlBlendFactor zu aktivieren.[/dim]\n" if _RICH else
       "Set FACTOR_WEIGHT_ML > 0 to activate the MlBlendFactor.")


if __name__ == "__main__":
    main()
