"""Tests for Phase-4 model-parameter tuning to RPS (analysis.weight_optimizer)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from analysis.weight_optimizer import (
    _MODEL_PARAM_RANGES,
    normalise_blend,
    rps_objective_from_results,
    synthetic_rps_objective,
)


def _rows():
    return [
        SimpleNamespace(actual_home_score=2, actual_away_score=0, home_xg=2.0, away_xg=0.8),
        SimpleNamespace(actual_home_score=3, actual_away_score=1, home_xg=1.9, away_xg=0.9),
        SimpleNamespace(actual_home_score=0, actual_away_score=1, home_xg=0.9, away_xg=1.7),
        SimpleNamespace(actual_home_score=1, actual_away_score=2, home_xg=0.8, away_xg=1.6),
    ]


def test_normalise_blend_sums_to_one():
    w = normalise_blend({"blend_poisson": 0.4, "blend_negbin": 0.2, "blend_glm_poisson": 0.2})
    assert set(w) == {"poisson", "negbin", "glm_poisson"}
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_rps_objective_rewards_sharp_correct_predictions():
    rows = _rows()
    # sharp rule: back the higher-λ side hard (always correct on these rows)
    sharp = rps_objective_from_results(
        rows, predict_fn=lambda p, r: ((0.7, 0.2, 0.1) if r.home_xg > r.away_xg else (0.1, 0.2, 0.7)))
    uniform = rps_objective_from_results(rows, predict_fn=lambda p, r: (1 / 3, 1 / 3, 1 / 3))
    assert sharp({}) < uniform({})
    assert sharp({}) < 0.1


def test_synthetic_objective_has_minimum_at_centres():
    obj = synthetic_rps_objective([0.0])
    centres = {k: (lo + hi) / 2 for k, (lo, hi) in _MODEL_PARAM_RANGES.items()}
    off = {k: hi for k, (lo, hi) in _MODEL_PARAM_RANGES.items()}
    assert obj(centres) < obj(off)


def test_tune_model_params_approaches_optimum():
    pytest.importorskip("optuna")
    from analysis.weight_optimizer import tune_model_params
    res = tune_model_params(synthetic_rps_objective([0.0]), n_trials=60)
    centres = {k: (lo + hi) / 2 for k, (lo, hi) in _MODEL_PARAM_RANGES.items()}
    # TPE should get reasonably close to the bowl's centre on each axis.
    for k, v in res.best_params.items():
        span = _MODEL_PARAM_RANGES[k][1] - _MODEL_PARAM_RANGES[k][0]
        assert abs(v - centres[k]) < 0.35 * span
