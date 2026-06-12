#!/usr/bin/env python3
"""debug.py — exercise every workflow function with MOCK data, in one shot.

    python debug.py            # full run, offline (no keys, no network)
    python debug.py -q         # only the summary + any failures

A fast, dependency-light smoke harness that complements the pytest suites: it
calls each pure function and the full 8-phase pipeline with deterministic mock
inputs, prints ✅/❌ per check with a short result preview, and exits non-zero if
anything fails. Use it to eyeball every module's output at a glance or to bisect
a regression without spinning up pytest.

Everything here is mock/offline by design — it is the verification counterpart to
the live workflow (`wm2026 predict --mode live`).
"""
from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

QUIET = "-q" in sys.argv[1:]
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    """Run ``fn`` (a 0-arg callable), record PASS/FAIL + a short preview."""
    try:
        preview = fn()
        _RESULTS.append((name, True, "" if preview is None else str(preview)[:88]))
        if not QUIET:
            print(f"  ✅ {name:<46} {('' if preview is None else str(preview)[:88])}")
    except Exception as exc:                       # noqa: BLE001 - harness
        _RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  ❌ {name:<46} {type(exc).__name__}: {exc}")
        if not QUIET:
            traceback.print_exc()


def section(title: str) -> None:
    if not QUIET:
        print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


# ── mock builders ─────────────────────────────────────────────────────────────
def _matrix(home_xg: float = 1.7, away_xg: float = 1.1):
    from models_ml.poisson_goals import DixonColesPoisson
    return DixonColesPoisson(rho=0.1, max_goals=6).predict_matrix(home_xg, away_xg)


def _mock_signal(name, h, a, *, weight=0.2, conf=0.7, available=True, kind="tilt"):
    from factors.base import FactorSignal
    return FactorSignal(name=name, home_strength=h, away_strength=a, weight=weight,
                        confidence=conf, available=available, source="mock", kind=kind)


def _mock_signals():
    return [
        _mock_signal("elo_strength", 1.15, 0.9, weight=0.30),
        _mock_signal("form", 1.05, 0.98, weight=0.20),
        _mock_signal("head_to_head", 1.02, 1.0, weight=0.15),
        _mock_signal("goal_efficiency", 1.08, 0.95, weight=0.15),
        _mock_signal("weather", 0.97, 0.97, weight=0.05, kind="global"),
        _mock_signal("sentiment", 1.0, 1.0, weight=0.10, available=False),
    ]


def _mock_rows():
    """MatchPrediction-shaped rows for backtesting / calibration."""
    spec = [
        (0.70, 0.20, 0.10, 2, 0), (0.62, 0.22, 0.16, 1, 0),
        (0.55, 0.25, 0.20, 0, 0), (0.40, 0.35, 0.25, 1, 1),
        (0.30, 0.30, 0.40, 0, 2), (0.46, 0.30, 0.24, 0, 1),
        (0.60, 0.25, 0.15, 3, 1), (0.50, 0.25, 0.25, 1, 2),
        (0.36, 0.40, 0.24, 1, 1), (0.74, 0.16, 0.10, 2, 0),
    ]
    return [SimpleNamespace(home_win_prob=h, draw_prob=d, away_win_prob=a,
                            actual_home_score=hs, actual_away_score=as_)
            for (h, d, a, hs, as_) in spec]


# ── 1 · derived markets ───────────────────────────────────────────────────────
def run_markets() -> None:
    section("wm2026.markets")
    from wm2026 import markets as mk
    M = _matrix()
    p = mk.one_x_two(M)
    check("one_x_two", lambda: {k: round(v, 3) for k, v in p.items()})
    check("double_chance", lambda: {k: round(v, 3) for k, v in mk.double_chance(p["home"], p["draw"], p["away"]).items()})
    check("draw_no_bet", lambda: {k: round(v, 3) for k, v in mk.draw_no_bet(p["home"], p["draw"], p["away"]).items()})
    check("asian_handicap(-0.5)", lambda: {k: round(v, 3) for k, v in mk.asian_handicap(M, -0.5).items()})
    check("asian_handicap(-0.75 quarter)", lambda: round(mk.asian_handicap(M, -0.75)["push"], 3))
    check("total_over_under(2.5)", lambda: round(mk.total_over_under(M, 2.5)["over"], 3))
    check("total_over_under(2.75 quarter)", lambda: round(mk.total_over_under(M, 2.75)["over"], 3))
    check("team_total(home,1.5)", lambda: round(mk.team_total(M, "home", 1.5)["over"], 3))
    check("clean_sheet", lambda: {k: round(v, 3) for k, v in mk.clean_sheet(M).items()})
    check("win_to_nil", lambda: {k: round(v, 3) for k, v in mk.win_to_nil(M).items()})
    check("odd_even_goals", lambda: {k: round(v, 3) for k, v in mk.odd_even_goals(M).items()})
    check("winning_margin (Σ=1)", lambda: round(sum(mk.winning_margin(M).values()), 6))
    check("multi_goal_bands (Σ=1)", lambda: round(sum(mk.multi_goal_bands(M).values()), 6))
    check("exact_total_goals (Σ=1)", lambda: round(sum(mk.exact_total_goals(M).values()), 6))
    check("first_goal", lambda: {k: round(v, 3) for k, v in mk.first_goal(M, 1.7, 1.1).items()})
    check("ht_ft (Σ=1)", lambda: round(sum(mk.ht_ft(1.7, 1.1).values()), 6))
    check("derive_all (keys)", lambda: len(mk.derive_all(M).keys()))


# ── 2 · edge / Kelly ──────────────────────────────────────────────────────────
def run_edge() -> None:
    section("wm2026.edge")
    from wm2026 import edge
    check("parse_odds", lambda: edge.parse_odds("2.10/3.40/3.20"))
    check("devig", lambda: [round(x, 3) for x in edge.devig([2.10, 3.40, 3.20])[0]])
    check("kelly_fraction", lambda: round(edge.kelly_fraction(0.6, 2.0), 3))
    check("stake_band", lambda: edge.stake_band(0.12))
    row = edge.evaluate_line("1X2", "Home", 0.55, 2.10, 0.44, model_p_lower=0.48)
    check("evaluate_line (+conservative)", lambda: (row.edge_pct, row.edge_pct_cons))
    ci = {"home_win": [0.48, 0.55, 0.62], "over_25": [0.42, 0.5, 0.58], "btts": [0.44, 0.5, 0.56],
          "draw": [0.18, 0.22, 0.26], "away_win": [0.18, 0.23, 0.28]}
    rows = edge.compute_edges({"home_win": 0.55, "draw": 0.22, "away_win": 0.23, "over_25": 0.5, "btts": 0.5},
                              odds_1x2=[2.10, 3.40, 3.20], odds_ou25=[1.95, 1.95],
                              odds_btts=[1.9, 1.9], odds_dc=[1.28, 1.30, 1.55], ci=ci)
    check("compute_edges (n rows)", lambda: len(rows))
    ah = {"line": -0.5, "home_win": 0.6, "push": 0.0, "away_win": 0.4,
          "home_prob_nopush": 0.6, "away_prob_nopush": 0.4}
    check("evaluate_asian_handicap", lambda: edge.evaluate_asian_handicap(ah, home_odd=1.8, away_odd=2.1)[0]["edge_pct"])
    check("best_value_pick", lambda: (edge.best_value_pick(rows) or {}).get("selection"))


# ── 3 · backtesting metrics ───────────────────────────────────────────────────
def run_backtesting() -> None:
    section("analysis.backtesting")
    from analysis.backtesting import _rps, compute
    check("_rps (perfect=0)", lambda: _rps((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    check("_rps (worst=1)", lambda: _rps((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    rep = compute(_mock_rows())
    check("compute (Brier/RPS/acc)", lambda: (round(rep.brier, 3), round(rep.rps, 3), round(rep.accuracy, 3)))


# ── 4 · calibration ───────────────────────────────────────────────────────────
def run_calibration() -> None:
    section("analysis.calibration (sklearn-free)")
    from analysis import calibration as calib
    check("_pav (monotone)", lambda: calib._pav([0.9, 0.1, 0.8, 0.2]))
    curve = calib._isotonic_pav_curve([0.2] * 10 + [0.8] * 10, [1] + [0] * 9 + [1] * 6 + [0] * 4)
    check("_isotonic_pav_curve.transform(0.8)", lambda: round(curve.transform(0.8), 3))
    pc = calib._platt_newton([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], [0, 0, 0, 1, 0, 1, 1, 1])
    check("_platt_newton (a>0)", lambda: round(pc.a, 3))
    check("PlattCurve.transform", lambda: round(calib.PlattCurve(1.0, 0.0).transform(0.5), 3))
    check("market_anchor(w=0.5)", lambda: {k: round(v, 3) for k, v in calib.market_anchor(0.5, 0.25, 0.25, (0.4, 0.3, 0.3), 0.5).items()})
    # fit + apply without touching the repo artifacts dir.
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    calib._ARTIFACT_DIR, calib._ISOTONIC_PATH, calib._PLATT_PATH = tmp, tmp / "i.json", tmp / "p.json"
    iso, platt = calib.fit_calibrators(_mock_rows())
    check("fit_calibrators (iso knots)", lambda: len(iso.curves["home"].x))
    check("apply (renorm Σ=1)", lambda: round(sum(calib.apply(iso, 0.5, 0.3, 0.2).values()), 6))
    check("transform_intervals", lambda: list(calib.transform_intervals(
        {"blended": {"home_win": [0.4, 0.5, 0.6], "draw": [0.3, 0.3, 0.3], "away_win": [0.1, 0.2, 0.3]}}, iso)["blended"].keys()))


# ── 5 · goal models ───────────────────────────────────────────────────────────
def run_models() -> None:
    section("models_ml.poisson_goals")
    from models_ml import poisson_goals as pg
    for name in ("poisson", "negbin", "glm_poisson", "bivariate"):
        m = pg.build_goal_model(name)
        check(f"build_goal_model({name}).markets", lambda m=m: round(m.markets(m.predict_matrix(1.6, 1.1))["home_win"], 3))
    models = pg.build_all_goal_models()
    per = {n: m.markets(m.predict_matrix(1.6, 1.1)) for n, m in models.items()}
    check("blend_markets", lambda: round(pg.blend_markets(per)["home_win"], 3))
    check("blend_score_matrix Σ=1", lambda: round(float(pg.blend_score_matrix(models, 1.6, 1.1).sum()), 6))
    check("bootstrap_markets (n=40)", lambda: [round(x, 3) for x in pg.bootstrap_markets(models["poisson"], 1.6, 1.1, n=40)["home_win"]])
    check("BivariatePoisson lifts draws", lambda: round(float(pg.BivariatePoisson(lambda3=0.2).predict_matrix(1.5, 1.3).trace()), 3))


# ── 6 · ensemble + predictor ──────────────────────────────────────────────────
def run_predictor() -> None:
    section("analysis.factor_ensemble + match_predictor")
    from analysis.factor_ensemble import FactorEnsemble
    from analysis.match_predictor import MatchPredictor, PredictionInput
    ens = FactorEnsemble().combine(_mock_signals())
    check("FactorEnsemble.combine (λ_home_mult)", lambda: round(ens.lambda_home_multiplier, 3))
    pi = PredictionInput(home_elo=1800, away_elo=1650, home_avg_xg=1.6, away_avg_xg=1.1,
                         home_avg_xg_conceded=1.0, away_avg_xg_conceded=1.4,
                         home_form_pts=10, away_form_pts=6, home_sentiment=0.1, away_sentiment=-0.05)
    out = MatchPredictor().predict(pi)
    check("MatchPredictor.predict (1X2 Σ=1)", lambda: round(out.home_win_prob + out.draw_prob + out.away_win_prob, 4))
    pred, ens2 = MatchPredictor(bootstrap_n=40).predict_from_signals(_mock_signals(), 1.6, 1.1)
    check("predict_from_signals (λ_home)", lambda: round(pred.home_xg, 3))
    check("predict_from_signals (per_model)", lambda: sorted(pred.features["per_model"].keys()))


# ── 7 · context + full pipeline + report ──────────────────────────────────────
def run_pipeline() -> None:
    section("wm2026.context + pipeline + report")
    from wm2026.context import apply_runtime_profile, build_context, synth_config
    from wm2026.pipeline import run_prediction
    from wm2026.report import build_report
    cfg = synth_config(home_team="Germany", away_team="Brazil", stage="QF",
                       home_xg=1.7, away_xg=1.3, odds_1x2="2.40/3.20/2.90")
    check("synth_config", lambda: cfg["match"]["id"])
    check("apply_runtime_profile('mock')", lambda: apply_runtime_profile("mock"))
    check("build_context", lambda: build_context(cfg).match_id)
    result = asyncio.run(run_prediction(
        cfg, mode="mock", bootstrap_n=64,
        odds_1x2=[2.40, 3.20, 2.90], odds_ou25=[1.85, 1.95],
        odds_dc=[1.28, 1.30, 1.55], odds_ah=(-0.5, 1.95, 1.95), calibrate="market"))
    check("run_prediction (warnings n)", lambda: len(result["warnings"]))
    check("run_prediction (derived_markets)", lambda: sorted(result["derived_markets"].keys()))
    check("run_prediction (calibration)", lambda: result["calibration"]["method"])
    check("run_prediction (claude_tasks)", lambda: len(result.get("claude_tasks", [])))
    report = build_report(result)
    check("build_report (json keys)", lambda: len(report["json"]))
    check("build_report (markdown len)", lambda: f"{len(report['markdown'])} chars")
    check("run_prediction (overrides_applied)", lambda: result.get("overrides_applied"))
    from wm2026.report_html import build_html
    check("build_html (self-contained)", lambda: f"{len(build_html(result, report['json']))} chars HTML")


def run_advanced() -> None:
    section("analysis.xg_estimator + wm2026.tournament")
    from datetime import datetime, timezone
    from analysis.xg_estimator import estimate_strengths, lambdas_for_fixture
    now = datetime.now(timezone.utc)
    games = [SimpleNamespace(home_code=h, away_code=a, home_score=hs, away_score=as_,
                             kickoff_utc=now, competition_tier=1)
             for h, a, hs, as_ in [("A", "B", 2, 0), ("C", "A", 1, 1), ("B", "C", 0, 2),
                                   ("A", "C", 3, 1), ("B", "A", 0, 1), ("C", "B", 2, 2),
                                   ("A", "B", 1, 0), ("C", "A", 0, 0)]]
    s = estimate_strengths(games, ref_time=now, xi=0.0, rho0=0.0, min_matches=6)
    check("estimate_strengths (sum-to-zero)",
          lambda: round(sum(s.attack.values()), 6) if s else "None")
    check("lambdas_for_fixture", lambda: tuple(round(x, 2) for x in lambdas_for_fixture(s, "A", "B")) if s else "None")
    from models_ml.poisson_goals import build_all_goal_models
    from wm2026.tournament import simulate_tournament
    groups = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    res = simulate_tournament(groups, lam_provider=lambda h, a: (1.4, 1.2),
                              models=build_all_goal_models(), n_sims=100, seed=0, n_best_thirds=0)
    check("simulate_tournament (Σtitle=1)", lambda: round(sum(res.title_prob.values()), 4))
    check("simulate_tournament (Σadvance=4)", lambda: round(sum(res.advance_prob.values()), 4))


def main() -> int:
    print("🔧 debug.py — mock-data smoke harness for the WM 2026 workflow\n")
    for runner in (run_markets, run_edge, run_backtesting, run_calibration,
                   run_models, run_predictor, run_pipeline, run_advanced):
        try:
            runner()
        except Exception as exc:                   # noqa: BLE001 - keep going
            _RESULTS.append((runner.__name__, False, f"section crashed: {exc}"))
            print(f"  ❌ {runner.__name__} section crashed: {exc}")
            traceback.print_exc()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = [n for n, ok, _ in _RESULTS if not ok]
    print(f"\n{'═' * 64}\nSUMMARY: {passed}/{len(_RESULTS)} checks passed")
    if failed:
        print("FAILED:\n  - " + "\n  - ".join(failed))
    else:
        print("🎉 all functions OK on mock data")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
