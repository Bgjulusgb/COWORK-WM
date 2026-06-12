"""Tests for Cowork v2 — apply_overrides + the overrides template."""
from __future__ import annotations

import asyncio

from wm2026.context import (
    apply_overrides,
    build_context,
    overrides_template,
    synth_config,
)
from wm2026.pipeline import run_prediction


def _run(cfg, **kw):
    return asyncio.run(run_prediction(cfg, mode="mock", bootstrap_n=0, **kw))


def test_apply_overrides_writes_cfg_slices_and_provenance():
    cfg = synth_config(home_team="A", away_team="B", home_xg=1.4, away_xg=1.3)
    ctx = build_context(cfg)
    applied = apply_overrides(ctx, {
        "xg": {"home": {"avg_xg_season": 2.4, "avg_xg_conceded": 0.7}},
        "elo": {"home": 1900},
        "weather": {"temp_c": 31, "wind_kmh": 10},
        "sentiment": {"sample_size": 300, "home_sentiment": 0.3},
    })
    assert {"xg.home", "elo.home", "weather", "sentiment"} <= set(applied)
    assert cfg["teams"]["home"]["avg_xg_season"] == 2.4
    assert cfg["teams"]["home"]["elo_rating"] == 1900.0
    assert ctx.provenance["xg_home"]["mode"] == "research"
    assert ctx.weather is not None and ctx.weather.temp_c == 31
    assert ctx.sentiment_payload["sample_size"] == 300


def test_apply_overrides_none_is_noop():
    cfg = synth_config(home_team="A", away_team="B")
    ctx = build_context(cfg)
    assert apply_overrides(ctx, None) == []
    assert apply_overrides(ctx, {}) == []


def test_overrides_shift_lambda_and_clear_mock_warning():
    base = _run(synth_config(home_team="A", away_team="B", home_xg=1.4, away_xg=1.3))
    over = _run(
        synth_config(home_team="A", away_team="B", home_xg=1.4, away_xg=1.3),
        overrides={"xg": {"home": {"avg_xg_season": 2.7, "avg_xg_conceded": 0.6}}},
    )
    assert over["prediction"].home_xg > base["prediction"].home_xg + 0.2
    assert over["overrides_applied"]                      # non-empty list
    # the research stamp makes ≥1 slice non-mock → "all sources mock" warning clears
    assert any("all data sources are mock" in w for w in base["warnings"])
    assert not any("all data sources are mock" in w for w in over["warnings"])


def test_overrides_template_has_every_category():
    t = overrides_template(synth_config(home_team="France", away_team="Senegal"))
    assert {"xg", "elo", "weather", "sentiment"} <= set(t.keys())
    assert t["xg"]["home"]["avg_xg_season"] is None      # blank to fill
    assert "France vs Senegal" in t["_fixture"]
