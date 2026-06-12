"""Tests fuer models_ml.poisson_goals.bootstrap_markets — K2 Coverage."""
from __future__ import annotations

import numpy as np
import pytest

from models_ml.poisson_goals import (
    DEFAULT_BLEND_WEIGHTS,
    MODEL_NAMES,
    bootstrap_markets,
    build_all_goal_models,
    build_goal_model,
)


def test_bootstrap_markets_returns_p5_p50_p95_per_market():
    model = build_goal_model("poisson")
    out = bootstrap_markets(model, 1.4, 1.2, n=50, rng=np.random.default_rng(7))
    for key in ("home_win", "draw", "away_win", "over_15", "over_25", "over_35", "btts"):
        assert key in out
        p5, p50, p95 = out[key]
        assert 0.0 <= p5 <= p50 <= p95 <= 1.0, f"{key} not ordered/in-range: {out[key]}"


def test_bootstrap_markets_is_deterministic_with_seeded_rng():
    model = build_goal_model("poisson")
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    out1 = bootstrap_markets(model, 1.4, 1.2, n=80, rng=rng1)
    out2 = bootstrap_markets(model, 1.4, 1.2, n=80, rng=rng2)
    assert out1 == out2


def test_bootstrap_markets_narrows_with_smaller_sigma():
    model = build_goal_model("poisson")
    wide = bootstrap_markets(
        model, 1.4, 1.2, n=200, xg_sigma=0.30, rng=np.random.default_rng(1)
    )
    tight = bootstrap_markets(
        model, 1.4, 1.2, n=200, xg_sigma=0.05, rng=np.random.default_rng(1)
    )
    wide_band = wide["home_win"][2] - wide["home_win"][0]
    tight_band = tight["home_win"][2] - tight["home_win"][0]
    assert tight_band < wide_band, (tight_band, wide_band)


def test_build_all_goal_models_yields_three():
    models = build_all_goal_models()
    assert set(models.keys()) == set(MODEL_NAMES)
    assert set(MODEL_NAMES) == {"poisson", "negbin", "glm_poisson"}


def test_blend_weights_sum_to_one():
    total = sum(DEFAULT_BLEND_WEIGHTS.values())
    assert total == pytest.approx(1.0, abs=1e-6)
