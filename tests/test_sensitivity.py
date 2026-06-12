"""Phase 5 — Sensitivity-Check + Edge-auf-kalibriert + v3-Template (bare pytest)."""
from __future__ import annotations

import asyncio

from models_ml.poisson_goals import build_goal_model, resolve_blend_weights
from wm2026.context import overrides_template, synth_config
from wm2026.pipeline import run_prediction
from wm2026.sensitivity import sensitivity_check


def _models():
    names = ["poisson", "negbin", "glm_poisson"]
    return {n: build_goal_model(n) for n in names}


def _row(market, sel, odd):
    return {"market": market, "selection": sel, "decimal_odd": odd}


# ── 5.1 sensitivity_check ────────────────────────────────────────────────────

def test_zero_perturbation_matches_point_edge():
    models = _models()
    rows = [_row("1X2", "Home", 2.0)]
    s0 = sensitivity_check(models, 1.5, 1.2, rows, rel=0.0, steps=(0.0,))
    sel = s0["selections"]["1X2/Home"]
    # rel=0 ⇒ ein Szenario, min == max == Punkt-Edge
    assert s0["n_scenarios"] == 1
    assert sel["edge_min_pct"] == sel["edge_max_pct"]
    assert sel["robust_pct"] in (0.0, 1.0)


def test_fragile_edge_detected():
    models = _models()
    # Quote so gewaehlt, dass die Edge nahe 0 liegt → ±15 % λ kippt das Vorzeichen.
    # P(home win | 1.4 vs 1.2) ≈ 0.44 ⇒ odd ≈ 1/0.44 ≈ 2.27 ist der Kipp-Punkt.
    rows = [_row("1X2", "Home", 2.27)]
    s = sensitivity_check(models, 1.4, 1.2, rows, rel=0.15)
    sel = s["selections"]["1X2/Home"]
    assert 0.0 < sel["robust_pct"] < 1.0
    assert "1X2/Home" in s["fragile"]
    assert sel["edge_min_pct"] < 0 < sel["edge_max_pct"]


def test_strong_edge_is_robust():
    models = _models()
    rows = [_row("1X2", "Home", 4.0)]   # massiv ueberteuert → immer positiv
    s = sensitivity_check(models, 1.8, 0.8, rows, rel=0.15)
    sel = s["selections"]["1X2/Home"]
    assert sel["robust_pct"] == 1.0
    assert s["fragile"] == []


def test_unpriced_and_unknown_rows_ignored():
    models = _models()
    rows = [{"market": "1X2", "selection": "Home", "decimal_odd": None},
            {"market": "HT/FT", "selection": "H/H", "decimal_odd": 5.0}]
    s = sensitivity_check(models, 1.4, 1.2, rows)
    assert s["selections"] == {} and s["fragile"] == []


def test_functional_consistency_complements():
    # Over + Under = 1, BTTS Yes + No = 1 in jedem Szenario (Linearitaet der Matrix)
    models = _models()
    rows = [_row("O/U 2.5", "Over 2.5", 2.0), _row("O/U 2.5", "Under 2.5", 2.0),
            _row("BTTS", "Yes", 2.0), _row("BTTS", "No", 2.0)]
    s = sensitivity_check(models, 1.4, 1.2, rows, rel=0.0, steps=(0.0,))
    e = {k: v["edge_min_pct"] for k, v in s["selections"].items()}
    # edge = 2p−1 (in %) ⇒ Summe der Komplement-Edges = 0
    assert abs(e["O/U 2.5/Over 2.5"] + e["O/U 2.5/Under 2.5"]) < 0.02
    assert abs(e["BTTS/Yes"] + e["BTTS/No"]) < 0.02


# ── Pipeline-Integration (opt-in, Default-Stabilitaet) ───────────────────────

def _run(**kw):
    cfg = synth_config(home_team="A", away_team="B", odds_1x2="2.00/3.20/4.80")
    return asyncio.run(run_prediction(
        cfg, mode="mock", odds_1x2=[2.00, 3.20, 4.80], calibrate="market", **kw))


def test_default_off_no_sensitivity_field_payload():
    r = _run()
    assert r["sensitivity"] is None          # Default off → unveraendert


def test_sensitivity_flag_populates_result_and_warns_on_fragile_pick():
    r = _run(sensitivity=True)
    s = r["sensitivity"]
    assert s and s["n_scenarios"] == 9 and s["selections"]
    bv = r.get("best_value_cons")
    if bv:
        key = f"{bv['market']}/{bv['selection']}"
        sel = s["selections"].get(key)
        if sel and sel["robust_pct"] < 1.0:
            assert any("NOT robust" in w for w in r["warnings"])


def test_edge_on_calibrated_shrinks_1x2_divergence():
    raw = _run()
    cal = _run(edge_on_calibrated=True)
    def _p(res, sel):
        return next(r["model_p"] for r in res["edges"]
                    if r["market"] == "1X2" and r["selection"] == sel)
    def _fair(res, sel):
        return next(r["fair_p"] for r in res["edges"]
                    if r["market"] == "1X2" and r["selection"] == sel)
    for sel in ("Home", "Draw", "Away"):
        # kalibrierte p liegen naeher am vig-freien Markt als rohe p (Anker w=0.5)
        assert abs(_p(cal, sel) - _fair(cal, sel)) <= abs(_p(raw, sel) - _fair(raw, sel)) + 1e-9
    # O/U & BTTS bleiben unberuehrt (nur 1X2 wird kalibriert)
    assert raw["prediction"].over_25 == cal["prediction"].over_25


# ── 5.3 overrides_template v3 ────────────────────────────────────────────────

def test_overrides_template_scaffolds_v3_blocks():
    tpl = overrides_template(synth_config(home_team="A", away_team="B"))
    assert "teams" in tpl and "match" in tpl and "_research_log" in tpl
    for side in ("home", "away"):
        assert set(tpl["teams"][side]) >= {"elo", "fifa_rank", "last5_results"}
    assert set(tpl["match"]) >= {"kickoff_utc", "venue"}
    # Altform bleibt erhalten (additiv)
    assert "xg" in tpl and "elo" in tpl and "weather" in tpl


# ── Phase 6 — Orchestrator-Fallback, Markt-Baseline, Report-Sektionen ────────

def test_client_init_failure_degrades_to_mock_not_error():
    """6.1: Exceptions bei der httpx-Client-Konstruktion (z.B. SOCKS ohne
    socksio) muessen im Konnektor-Fallback landen — Slice wird mock, nie error."""
    from data_sources.base import BaseConnector
    from data_sources.openfootball import OpenfootballConnector
    from config.settings import settings

    async def _boom(self):
        raise RuntimeError("Using SOCKS proxy, but the 'socksio' package is not installed.")

    orig_client, orig_flag = BaseConnector._get_client, settings.use_mock_openfootball
    orig_local = settings.openfootball_local_clone
    try:
        BaseConnector._get_client = _boom
        object.__setattr__(settings, "use_mock_openfootball", False)
        object.__setattr__(settings, "openfootball_local_clone", "")
        res = asyncio.run(OpenfootballConnector().get_head_to_head("CAN", "BIH"))
        assert res.mode == "mock"          # degradiert, statt error/raise
        assert res.data is not None
    finally:
        BaseConnector._get_client = orig_client
        object.__setattr__(settings, "use_mock_openfootball", orig_flag)
        object.__setattr__(settings, "openfootball_local_clone", orig_local)


def test_market_implied_lambdas_recovers_synthetic_market():
    """6.2: Inversion-Recovery — Markt aus bekannten λ erzeugen, λ zurueckfinden."""
    import numpy as np
    from models_ml.poisson_goals import blend_score_matrix
    from wm2026.market_baseline import market_implied_lambdas

    models = _models()
    true_lh, true_la = 1.62, 1.08
    M = np.asarray(blend_score_matrix(models, true_lh, true_la))
    n = M.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(M.shape[1]))
    fair = (float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum()))
    over = float(M[idx >= 3].sum())

    lh, la, diag = market_implied_lambdas(models, fair, over)
    assert diag["converged"]
    assert abs(lh - true_lh) < 0.02 and abs(la - true_la) < 0.02


def test_baseline_market_changes_base_xg_source():
    """6.2: --baseline market setzt base_xg_source und bleibt im sane-Range."""
    cfg = synth_config(home_team="A", away_team="B", odds_1x2="2.00/3.20/4.80")
    r = asyncio.run(run_prediction(cfg, mode="mock", odds_1x2=[2.00, 3.20, 4.80],
                                   odds_ou25=[1.95, 1.65], baseline="market"))
    assert r["base_xg_source"] == "market-implied"
    assert 0.3 <= r["base_home_xg"] <= 4.0 and 0.3 <= r["base_away_xg"] <= 4.0
    # Default bleibt yaml
    r2 = asyncio.run(run_prediction(cfg, mode="mock", odds_1x2=[2.00, 3.20, 4.80]))
    assert r2["base_xg_source"] == "yaml"


def test_markdown_contains_sensitivity_and_research_log():
    """6.3: Markdown-Report rendert Sensitivity-Tabelle + Research-Log."""
    from wm2026.report import build_report
    cfg = synth_config(home_team="A", away_team="B", odds_1x2="2.00/3.20/4.80")
    r = asyncio.run(run_prediction(
        cfg, mode="mock", odds_1x2=[2.00, 3.20, 4.80], sensitivity=True,
        overrides={"_research_log": [{"slice": "elo", "value": "1788/1595",
                                      "url": "https://eloratings.net",
                                      "fetched_at": "2026-06-12"}]},
    ))
    md = build_report(r)["markdown"]
    assert "Sensitivity" in md and "Research-Log" in md
    assert "eloratings.net" in md
