"""Tests for the Karlis-Ntzoufras bivariate Poisson goal model.

Core deps only (numpy/scipy). Verifies the three defining properties:
preserved marginal means, positive goal correlation (more draws than the
independent product), and the λ₃→0 limit recovering independent Poisson.
"""
from __future__ import annotations

import numpy as np

from models_ml.poisson_goals import (
    BivariatePoisson,
    DixonColesPoisson,
    build_goal_model,
)


def _marginal_means(matrix: np.ndarray) -> tuple[float, float]:
    n = matrix.shape[0]
    idx = np.arange(n)
    home_mean = float((matrix.sum(axis=1) * idx).sum())
    away_mean = float((matrix.sum(axis=0) * idx).sum())
    return home_mean, away_mean


def test_matrix_is_a_distribution():
    M = BivariatePoisson(lambda3=0.15).predict_matrix(1.7, 1.1)
    assert abs(M.sum() - 1.0) < 1e-9
    assert (M >= 0).all()


def test_preserves_marginal_means():
    # λ₃ is split out of the supplied xG, so the marginal means must come back
    # to (home_xg, away_xg) up to the truncation at max_goals.
    home_xg, away_xg = 1.8, 1.2
    M = BivariatePoisson(lambda3=0.2, max_goals=10).predict_matrix(home_xg, away_xg)
    hm, am = _marginal_means(M)
    assert abs(hm - home_xg) < 0.03
    assert abs(am - away_xg) < 0.03


def test_positive_covariance_lifts_draws():
    # With the same marginal means, the shared λ₃ component must increase the
    # probability mass on the diagonal (draws) vs the independent Poisson.
    home_xg, away_xg = 1.5, 1.3
    indep = DixonColesPoisson(rho=0.0, max_goals=8).predict_matrix(home_xg, away_xg)
    bipois = BivariatePoisson(lambda3=0.25, max_goals=8).predict_matrix(home_xg, away_xg)
    assert float(np.trace(bipois)) > float(np.trace(indep))


def test_lambda3_zero_recovers_independent_poisson():
    home_xg, away_xg = 1.6, 0.9
    indep = DixonColesPoisson(rho=0.0, max_goals=8).predict_matrix(home_xg, away_xg)
    bipois = BivariatePoisson(lambda3=0.0, max_goals=8).predict_matrix(home_xg, away_xg)
    assert np.allclose(indep, bipois, atol=1e-9)


def test_factory_builds_bivariate():
    m = build_goal_model("bivariate")
    assert isinstance(m, BivariatePoisson)
    # markets() is inherited and must yield a proper 1X2 distribution.
    mk = m.markets(m.predict_matrix(1.4, 1.1))
    assert abs(mk["home_win"] + mk["draw"] + mk["away_win"] - 1.0) < 1e-9
