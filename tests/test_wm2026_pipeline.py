"""Smoke + invariant tests for the wm2026 workflow (mock mode, no network).

Deliberately self-contained: uses ``asyncio.run`` instead of pytest-asyncio so
the suite runs with a bare ``pytest`` + the core requirements.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from wm2026.context import synth_config
from wm2026.edge import compute_edges, devig, kelly_fraction, parse_odds
from wm2026.pipeline import run_prediction
from wm2026.report import build_report

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "config" / "matches" / "group_a" / "cze_vs_rsa.yaml"


def _run(cfg, **kw):
    return asyncio.run(run_prediction(cfg, mode="mock", bootstrap_n=64, **kw))


# ── edge math (pure) ──────────────────────────────────────────────────────────
def test_parse_odds_variants():
    assert parse_odds("2.10/3.40/3.20") == [2.10, 3.40, 3.20]
    assert parse_odds("1.85, 1.95") == [1.85, 1.95]
    assert parse_odds("") is None
    assert parse_odds(None) is None


def test_devig_normalises_to_one():
    fair, overround = devig([2.10, 3.40, 3.20])
    assert overround > 1.0                      # book always has a margin
    assert abs(sum(fair) - 1.0) < 1e-9          # fair probs sum to 1


def test_kelly_zero_when_no_edge():
    # p * odd = 1 → no edge → no stake
    assert kelly_fraction(0.5, 2.0) == 0.0
    assert kelly_fraction(0.6, 2.0) > 0.0       # genuine edge → positive stake


def test_compute_edges_flags_value():
    rows = compute_edges({"home_win": 0.55, "draw": 0.22, "away_win": 0.23,
                          "over_25": 0.5, "btts": 0.5},
                         odds_1x2=[2.10, 3.40, 3.20])
    home = next(r for r in rows if r["selection"] == "Home")
    assert home["edge_pct"] is not None and home["edge_pct"] > 10.0
    assert home["action"] == "sanity-check"


# ── full pipeline (mock) ──────────────────────────────────────────────────────
def test_pipeline_runs_on_sample_config():
    cfg = __import__("yaml").safe_load(SAMPLE.read_text(encoding="utf-8"))
    result = _run(cfg, odds_1x2=[2.10, 3.40, 3.20])
    out = result["prediction"]
    s = out.home_win_prob + out.draw_prob + out.away_win_prob
    assert abs(s - 1.0) < 0.01                  # 1X2 is a proper distribution
    assert 0.3 <= out.home_xg <= 4.0            # λ stays physical
    assert 0.3 <= out.away_xg <= 4.0
    assert result["best_value"] is not None     # +EV home line with these odds


def test_pipeline_synth_no_cli_odds_no_fake_edge():
    """No fake data → no fake edge. In mock mode the odds connector returns
    mock-seeded odds, but the pipeline deliberately does NOT adopt them (they
    are uncorrelated with the model and would manufacture nonsense edges). With
    no --odds passed, there is no value pick — the honest outcome."""
    cfg = synth_config(home_team="Germany", away_team="Brazil",
                       home_xg=1.7, away_xg=1.6)
    result = _run(cfg)
    assert result["best_value"] is None         # no real odds → no value pick
    assert result["prediction"].over_25 >= 0.0


def test_pipeline_adopts_genuinely_live_odds(monkeypatch):
    """When the odds connector returns a *live* result, the pipeline adopts it
    as the edge-table default (so a configured ODDS_API_KEY needs no --odds)."""
    from data_sources.base import FetchResult
    from data_sources.odds_api import OddsApiConnector

    async def _live_odds(self, *a, **kw):
        return FetchResult(
            {"1x2": [2.10, 3.40, 3.20], "ou_2_5": [1.85, 1.95]},
            "live", None, "odds_api",
        )

    monkeypatch.setattr(OddsApiConnector, "get_odds", _live_odds)
    cfg = synth_config(home_team="Germany", away_team="Brazil",
                       home_xg=1.7, away_xg=1.6)
    result = _run(cfg)
    priced = [r for r in result["edges"]
              if r["market"] == "1X2" and r.get("decimal_odd") is not None]
    assert len(priced) == 3                      # live odds populated the table


def test_report_json_schema_and_markdown():
    cfg = synth_config(home_team="France", away_team="Senegal",
                       odds_1x2="1.70/3.60/4.50")
    result = _run(cfg, odds_1x2=[1.70, 3.60, 4.50])
    report = build_report(result)
    js = report["json"]
    # JSON is serialisable and carries the required top-level keys.
    json.dumps(js)
    for key in ("match_id", "model_version", "markets", "per_model",
                "confidence_intervals", "ensemble_confidence",
                "factors_used", "factors_total", "edge_table", "derived_markets"):
        assert key in js, f"missing {key}"
    assert set(js["per_model"]) == {"poisson", "negbin", "glm_poisson"}
    assert "# 🏆 WM 2026" in report["markdown"]
    assert "Edge Table" in report["markdown"]
    assert "Derived markets" in report["markdown"]


def test_derived_markets_present_and_consistent():
    cfg = synth_config(home_team="Spain", away_team="Morocco",
                       home_xg=1.6, away_xg=1.1)
    result = _run(cfg)
    dm = result["derived_markets"]
    for key in ("double_chance", "draw_no_bet", "asian_handicap", "totals",
                "clean_sheet", "win_to_nil", "odd_even"):
        assert key in dm
    # Double chance 1X must equal home_win + draw of the headline 1X2.
    out = result["prediction"]
    assert abs(dm["double_chance"]["1X"] - (out.home_win_prob + out.draw_prob)) < 1e-6
    # Totals at 2.5 must match the headline over_25.
    o25 = next(t for t in dm["totals"] if abs(t["line"] - 2.5) < 1e-9)
    assert abs(o25["over"] - out.over_25) < 1e-6


def test_double_chance_and_asian_handicap_edges():
    cfg = synth_config(home_team="Brazil", away_team="Serbia",
                       home_xg=1.9, away_xg=0.9)
    result = _run(cfg, odds_dc=[1.20, 1.25, 1.60],
                  odds_ah=(-0.5, 1.95, 1.95))
    markets = {r["market"] for r in result["edges"]}
    assert "Double Chance" in markets
    assert any(m.startswith("AH ") for m in markets)


def test_market_calibration_mode_anchors_to_odds():
    cfg = synth_config(home_team="Italy", away_team="Ghana",
                       home_xg=1.6, away_xg=1.0)
    result = _run(cfg, odds_1x2=[1.80, 3.50, 4.50], calibrate="market")
    cal = result["calibration"]
    assert cal["applied"] is True
    assert cal["method"] == "market-anchor"
    c = cal["calibrated"]
    assert abs(c["home_win"] + c["draw"] + c["away_win"] - 1.0) < 1e-6


def test_calibrate_none_disables():
    cfg = synth_config(home_team="Italy", away_team="Ghana")
    result = _run(cfg, odds_1x2=[1.80, 3.50, 4.50], calibrate="none")
    assert result["calibration"]["applied"] is False


def test_claude_tasks_flags_live_gaps_and_missing_odds():
    from wm2026.pipeline import _claude_tasks
    cfg = synth_config(home_team="Spain", away_team="Japan")
    prov = {
        "xg_home": {"mode": "mock"}, "xg_away": {"mode": "live"},
        "weather": {"mode": "error"},
        "lineup_home": {"mode": "live"}, "lineup_away": {"mode": "live"},
    }
    tasks = _claude_tasks(prov, cfg, mode="live", has_odds=False)
    cats = {t["category"] for t in tasks}
    assert {"xg", "weather", "odds"} <= cats     # one mock slice / error / no odds → tasks
    assert "lineup" not in cats                  # both live → no task
    assert tasks[0]["priority"] == "high"        # sorted, odds/xg first
    # Mock mode has no live gaps by design; odds still requested when absent.
    assert _claude_tasks(prov, cfg, mode="mock", has_odds=True) == []


def test_base_xg_source_is_yaml_by_default():
    # Default-stability contract: the MLE estimator is gated off, so base xG comes
    # from the YAML path and the default numeric output is unchanged.
    result = _run(synth_config(home_team="A", away_team="B", home_xg=1.5, away_xg=1.2))
    assert result["base_xg_source"] == "yaml"


def test_mle_xg_enabled_does_not_crash():
    from config.settings import settings as S
    object.__setattr__(S, "use_mle_xg", True)
    try:
        result = _run(synth_config(home_team="Brazil", away_team="Serbia"))
        # fits on the mock history or falls back to YAML — never raises.
        assert result["base_xg_source"] in ("mle", "yaml")
        assert 0.3 <= result["prediction"].home_xg <= 4.0
    finally:
        object.__setattr__(S, "use_mle_xg", False)


def test_report_renders_cowork_assignment_when_odds_missing():
    """No --odds and only mock-sourced odds (not adopted) → the report asks
    Claude to research the bookmaker odds."""
    cfg = synth_config(home_team="Spain", away_team="Japan")
    result = _run(cfg)                            # mock, no real odds → odds task fires
    assert any(t["category"] == "odds" for t in result["claude_tasks"])
    report = build_report(result)
    assert "claude_tasks" in report["json"]
    assert "Cowork-Auftrag" in report["markdown"]


def test_factor_unavailability_renormalises():
    # No external data + no sentiment → some factors go unavailable, but the
    # ensemble must still return a proper distribution (re-normalisation).
    cfg = synth_config(home_team="Japan", away_team="Croatia")
    result = _run(cfg)
    used = result["prediction"].features["active_factors"]
    skipped = result["prediction"].features["skipped_factors"]
    assert len(used) >= 5                        # core factors always fire
    assert len(skipped) >= 1                     # sentiment/weather neutralise
