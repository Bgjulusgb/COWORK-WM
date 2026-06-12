"""Phase 4 — Backend-Härtung: DC-Bootstrap-Invariante, Blend-Weight-Resolver,
opt-in BivariatePoisson im Blend, bootstrap_blend_metrics, Bankroll-Staking,
Per-Quelle-Toggles. Bare pytest (asyncio.run), mock-only, kein Netz."""
from __future__ import annotations

import asyncio

import numpy as np

from models_ml.poisson_goals import (
    BLEND_WEIGHTS_WITH_BIVARIATE,
    DEFAULT_BLEND_WEIGHTS,
    MODEL_NAMES,
    bootstrap_blend_metrics,
    bootstrap_markets,
    build_all_goal_models,
    build_goal_model,
    resolve_blend_weights,
)
from wm2026.context import synth_config
from wm2026.pipeline import run_prediction


def _run(cfg, **kw):
    return asyncio.run(run_prediction(cfg, mode="mock", bootstrap_n=64, **kw))


# ── 4.1 DC-Quantile im Bootstrap ──────────────────────────────────────────────
def test_bootstrap_emits_dc_keys_with_exact_complement_invariant():
    model = build_goal_model("poisson")
    rng = np.random.default_rng(7)
    ci = bootstrap_markets(model, 1.5, 1.2, n=200, rng=rng)
    for key in ("dc_1x", "dc_12", "dc_x2"):
        p5, p50, p95 = ci[key]
        assert 0.0 <= p5 <= p50 <= p95 <= 1.0
    # Pro Sample gilt dc_12 ≡ 1 − draw ⇒ die Quantile spiegeln sich exakt:
    # p5(dc_12) = 1 − p95(draw) und p95(dc_12) = 1 − p5(draw).
    assert abs(ci["dc_12"][0] - (1.0 - ci["draw"][2])) < 1e-12
    assert abs(ci["dc_12"][2] - (1.0 - ci["draw"][0])) < 1e-12
    # Und die DC-Mediane liegen über jeder Einzel-Komponente (Summe zweier
    # nichtnegativer Anteile).
    assert ci["dc_1x"][1] >= ci["home_win"][1]
    assert ci["dc_1x"][1] >= ci["draw"][1]


# ── 4.2 bootstrap_blend_metrics ───────────────────────────────────────────────
def test_bootstrap_blend_metrics_quantiles_are_ordered_and_tight_at_zero_sigma():
    models = build_all_goal_models()
    from wm2026 import markets as markets_mod

    def home_np(M):
        return markets_mod.asian_handicap(M, -0.5)["home_prob_nopush"]

    rng = np.random.default_rng(11)
    ci = bootstrap_blend_metrics(models, 1.6, 1.1, {"home_np": home_np},
                                 n=150, xg_sigma=0.15, rng=rng)
    p5, p50, p95 = ci["home_np"]
    assert 0.0 <= p5 <= p50 <= p95 <= 1.0
    # σ→0 ⇒ alle Samples identisch ⇒ Quantile kollabieren auf den Punktwert.
    tight = bootstrap_blend_metrics(models, 1.6, 1.1, {"home_np": home_np},
                                    n=20, xg_sigma=1e-9,
                                    rng=np.random.default_rng(1))["home_np"]
    assert abs(tight[0] - tight[2]) < 1e-6


def test_bootstrap_blend_metrics_skips_broken_metric():
    models = build_all_goal_models()

    def boom(M):  # eine kaputte Metrik darf die anderen nicht mitreißen
        raise ValueError("broken metric")

    ci = bootstrap_blend_metrics(
        models, 1.4, 1.3,
        {"ok": lambda M: float(M.sum()), "broken": boom},
        n=10, rng=np.random.default_rng(3),
    )
    assert "ok" in ci and "broken" not in ci


# ── 2.2 Blend-Weight-Resolver + Bivariate opt-in ──────────────────────────────
def test_resolve_blend_weights_default_trio_matches_legacy_ratios():
    w = resolve_blend_weights(MODEL_NAMES)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    for name in MODEL_NAMES:
        assert abs(w[name] - DEFAULT_BLEND_WEIGHTS[name]) < 1e-12


def test_resolve_blend_weights_with_bivariate_renormalises():
    names = list(MODEL_NAMES) + ["bivariate"]
    w = resolve_blend_weights(names)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["bivariate"] > 0.0
    # Verhältnis poisson:negbin bleibt das der Bivariate-Tabelle.
    expect = (BLEND_WEIGHTS_WITH_BIVARIATE["poisson"]
              / BLEND_WEIGHTS_WITH_BIVARIATE["negbin"])
    assert abs(w["poisson"] / w["negbin"] - expect) < 1e-9


def test_resolve_blend_weights_unknown_only_falls_back_uniform():
    w = resolve_blend_weights(["mystery_a", "mystery_b"])
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert abs(w["mystery_a"] - 0.5) < 1e-12


def test_build_all_goal_models_bivariate_opt_in():
    trio = build_all_goal_models()
    assert set(trio) == set(MODEL_NAMES)
    quad = build_all_goal_models(include_bivariate=True)
    assert set(quad) == set(MODEL_NAMES) | {"bivariate"}
    # Bivariate-Matrix ist eine echte Verteilung bei denselben λ.
    M = quad["bivariate"].predict_matrix(1.5, 1.2)
    assert abs(float(M.sum()) - 1.0) < 1e-9


def test_pipeline_with_bivariate_enabled_stays_valid():
    from config.settings import settings as S
    object.__setattr__(S, "include_bivariate", True)
    try:
        result = _run(synth_config(home_team="A", away_team="B",
                                   home_xg=1.5, away_xg=1.2))
        out = result["prediction"]
        s = out.home_win_prob + out.draw_prob + out.away_win_prob
        assert abs(s - 1.0) < 0.01
        feats = out.features
        assert "bivariate" in feats["per_model"]
        assert abs(sum(feats["blend_weights"].values()) - 1.0) < 1e-9
        assert feats["blend_weights"]["bivariate"] > 0.0
    finally:
        object.__setattr__(S, "include_bivariate", False)


def test_default_pipeline_unchanged_without_bivariate():
    result = _run(synth_config(home_team="A", away_team="B"))
    feats = result["prediction"].features
    assert set(feats["per_model"]) == set(MODEL_NAMES)
    assert feats["blend_weights"] == {
        n: DEFAULT_BLEND_WEIGHTS[n] / sum(DEFAULT_BLEND_WEIGHTS.values())
        for n in MODEL_NAMES
    }


# ── 4.1/4.2 E2E: konservative Spalten in der Edge-Tabelle ─────────────────────
def test_pipeline_dc_and_ah_rows_carry_conservative_columns():
    cfg = synth_config(home_team="Brazil", away_team="Serbia",
                       home_xg=1.9, away_xg=0.9)
    result = _run(cfg, odds_dc=[1.20, 1.25, 1.60], odds_ah=(-0.5, 1.95, 1.95))
    dc_rows = [r for r in result["edges"] if r["market"] == "Double Chance"]
    ah_rows = [r for r in result["edges"] if r["market"].startswith("AH ")]
    assert dc_rows and ah_rows
    assert all(r["edge_pct_cons"] is not None for r in dc_rows)
    assert all(r["edge_pct_cons"] is not None for r in ah_rows)
    # Konservativ liegt nie über dem Punktschätzer.
    for r in dc_rows + ah_rows:
        assert r["edge_pct_cons"] <= r["edge_pct"] + 1e-9


# ── 4.3 best_value_cons im Result/Report ──────────────────────────────────────
def test_result_carries_best_value_cons_field():
    cfg = synth_config(home_team="France", away_team="Senegal")
    result = _run(cfg, odds_1x2=[1.70, 3.60, 4.50])
    assert "best_value_cons" in result
    bvc = result["best_value_cons"]
    if bvc is not None:
        assert bvc["edge_pct_cons"] > 0.0


# ── 4.4 Bankroll-Staking ──────────────────────────────────────────────────────
def test_bankroll_annotates_stakes_and_respects_p5_discipline():
    cfg = synth_config(home_team="France", away_team="Senegal")
    result = _run(cfg, odds_1x2=[1.70, 3.60, 4.50], bankroll=1000.0)
    assert result["bankroll"] == 1000.0
    priced = [r for r in result["edges"] if r.get("decimal_odd")]
    assert priced
    for r in priced:
        assert "stake_half_kelly" in r and "stake_cons" in r
        # ½-Kelly-auf-p5-Disziplin: kein konservativer Einsatz ohne positive p5-Edge.
        if not (r.get("edge_pct_cons") or 0) > 0:
            assert r["stake_cons"] == 0.0
        if r.get("half_kelly_pct"):
            assert abs(r["stake_half_kelly"]
                       - round(1000.0 * r["half_kelly_pct"] / 100.0, 2)) < 1e-9


def test_no_bankroll_means_no_stake_fields():
    cfg = synth_config(home_team="France", away_team="Senegal")
    result = _run(cfg, odds_1x2=[1.70, 3.60, 4.50])
    assert result["bankroll"] is None
    assert all("stake_half_kelly" not in r for r in result["edges"])


# ── 4.6 Per-Quelle-Toggles (CLI) ──────────────────────────────────────────────
def test_cli_source_toggle_parsing_and_env(monkeypatch):
    import argparse
    from wm2026.cli import _parse_source_list, _seed_source_toggles

    assert _parse_source_list("weather, clubelo", "--live-sources") == ["weather", "clubelo"]
    assert _parse_source_list(None, "--live-sources") == []
    try:
        _parse_source_list("weather,nosuch", "--live-sources")
        raise AssertionError("expected SystemExit for unknown source")
    except SystemExit:
        pass

    for k in list(__import__("os").environ):
        if k.startswith("USE_MOCK_"):
            monkeypatch.delenv(k, raising=False)
    args = argparse.Namespace(mode="live", live_sources="weather,clubelo",
                              mock_sources=None)
    _seed_source_toggles(args)
    import os
    assert os.environ["USE_MOCK_WEATHER"] == "false"
    assert os.environ["USE_MOCK_CLUBELO"] == "false"
    assert os.environ["USE_MOCK_TRANSFERMARKT"] == "true"
    # --live-sources zwingt mock → live um.
    args2 = argparse.Namespace(mode="mock", live_sources="weather",
                               mock_sources=None)
    _seed_source_toggles(args2)
    assert args2.mode == "live"


def test_cli_parser_accepts_new_flags():
    from wm2026.cli import build_parser
    p = build_parser()
    args = p.parse_args([
        "predict", "--home", "A", "--away", "B", "--mode", "mock",
        "--bankroll", "500", "--live-sources", "weather",
        "--mock-sources", "rss",
    ])
    assert args.bankroll == 500.0
    assert args.live_sources == "weather"
    assert args.mock_sources == "rss"
