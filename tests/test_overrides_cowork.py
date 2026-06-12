"""Cowork v3 — overrides schema (research-fixture skill) + market-divergence guard.

Bare pytest (no pytest-asyncio): async paths use ``asyncio.run()`` directly,
matching ``tests/test_wm2026_pipeline.py``.
"""
from __future__ import annotations

import asyncio

from wm2026.context import apply_overrides, build_context, synth_config
from wm2026.pipeline import _validate, run_prediction


def _ctx():
    return build_context(synth_config(home_team="A", away_team="B"))


# ── apply_overrides: research-fixture-Skill-Schema (Cowork v3) ───────────────

def test_overrides_teams_block_maps_to_yaml_fields():
    ctx = _ctx()
    applied = apply_overrides(ctx, {
        "teams": {
            "home": {"elo": 1788, "fifa_rank": 30, "avg_xg_season": 1.2,
                     "last5_results": ["D", "W", "D", "D", "W"]},
            "away": {"elo": 1595, "fifa_rank": 64, "avg_xg_conceded": 0.8},
        },
    })
    home = ctx.config["teams"]["home"]
    away = ctx.config["teams"]["away"]
    assert home["elo_rating"] == 1788.0 and away["elo_rating"] == 1595.0
    assert home["world_ranking"] == 30 and away["world_ranking"] == 64
    assert home["avg_xg_season"] == 1.2 and away["avg_xg_conceded"] == 0.8
    assert home["form_last5"] == ["D", "W", "D", "D", "W"]
    assert "teams.home.elo_rating" in applied
    assert "teams.away.world_ranking" in applied
    # xG override stamps research provenance (clears the all-mock warning)
    assert ctx.provenance["xg_home"]["mode"] == "research"


def test_overrides_match_block_sets_kickoff_and_venue():
    ctx = _ctx()
    applied = apply_overrides(ctx, {
        "match": {"kickoff_utc": "2026-06-12T19:00:00Z", "venue": "BMO Field, Toronto"},
    })
    assert ctx.config["match"]["kickoff_utc"] == "2026-06-12T19:00:00Z"
    assert ctx.config["match"]["venue"] == "BMO Field, Toronto"
    assert set(applied) == {"match.kickoff_utc", "match.venue"}


def test_overrides_weather_aliases():
    ctx = _ctx()
    applied = apply_overrides(ctx, {
        "weather": {"temp_c": 24, "wind_kph": 15, "precip_mm": 1.0, "humidity": 76},
    })
    assert "weather" in applied
    assert ctx.weather.wind_kmh == 15
    assert ctx.weather.precipitation_mm == 1.0
    assert ctx.weather.humidity_pct == 76
    assert ctx.weather.temp_c == 24


def test_overrides_legacy_schema_still_works():
    ctx = _ctx()
    applied = apply_overrides(ctx, {
        "xg": {"home": {"avg_xg_season": 1.5}},
        "elo": {"home": 1800, "away": 1600},
    })
    assert ctx.config["teams"]["home"]["avg_xg_season"] == 1.5
    assert ctx.config["teams"]["home"]["elo_rating"] == 1800.0
    assert "xg.home" in applied and "elo.home" in applied


def test_overrides_empty_and_unknown_keys_ignored():
    ctx = _ctx()
    assert apply_overrides(ctx, None) == []
    assert apply_overrides(ctx, {}) == []
    assert apply_overrides(ctx, {"_research_log": [], "_notes": "x"}) == []


# ── Phase 7: market-divergence guard ─────────────────────────────────────────

class _Out:
    home_xg = 1.5
    away_xg = 1.2
    home_win_prob = 0.35
    draw_prob = 0.24
    away_win_prob = 0.41


class _Ens:
    confidence = 0.8


def test_validate_warns_on_market_divergence():
    warnings = _validate(_Out(), _Ens(), [], {"x": {"mode": "live"}},
                         market_implied=(0.49, 0.31, 0.20))
    div = [w for w in warnings if "model-market divergence" in w]
    assert len(div) == 2  # home Δ14pp + away Δ21pp; draw Δ7pp stays silent
    assert any("away" in w for w in div) and any("home" in w for w in div)


def test_validate_silent_when_model_tracks_market():
    warnings = _validate(_Out(), _Ens(), [], {"x": {"mode": "live"}},
                         market_implied=(0.36, 0.25, 0.39))
    assert not [w for w in warnings if "divergence" in w]


def test_validate_backwards_compatible_without_market():
    # additive Signatur: alter Aufruf ohne market_implied bleibt gültig
    warnings = _validate(_Out(), _Ens(), [], {"x": {"mode": "live"}})
    assert not [w for w in warnings if "divergence" in w]


# ── end-to-end: overrides + research_log reach the report dict ───────────────

def test_run_prediction_carries_overrides_and_research_log():
    cfg = synth_config(home_team="A", away_team="B",
                       odds_1x2="2.00/3.20/4.80")
    result = asyncio.run(run_prediction(
        cfg, mode="mock",
        overrides={
            "teams": {"home": {"elo": 1788}, "away": {"elo": 1595}},
            "_research_log": [{"slice": "elo", "value": "1788/1595",
                               "url": "https://eloratings.net", "fetched_at": "2026-06-12"}],
        },
    ))
    assert "teams.home.elo_rating" in result["overrides_applied"]
    assert result["research_log"] and result["research_log"][0]["slice"] == "elo"
