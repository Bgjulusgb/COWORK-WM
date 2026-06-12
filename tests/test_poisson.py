import numpy as np

from models_ml.poisson_goals import DixonColesPoisson


def test_matrix_sums_to_one():
    dc = DixonColesPoisson(rho=0.1, max_goals=6)
    matrix = dc.predict_matrix(1.5, 1.0)
    assert abs(matrix.sum() - 1.0) < 1e-6


def test_markets_probabilities_in_range():
    dc = DixonColesPoisson()
    matrix = dc.predict_matrix(1.8, 1.2)
    m = dc.markets(matrix)
    for key in ("home_win", "draw", "away_win", "over_25", "btts"):
        assert 0.0 <= m[key] <= 1.0


def test_1x2_sums_to_one():
    dc = DixonColesPoisson()
    matrix = dc.predict_matrix(1.5, 1.5)
    m = dc.markets(matrix)
    total = m["home_win"] + m["draw"] + m["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_higher_home_xg_increases_home_win():
    dc = DixonColesPoisson()
    low = dc.markets(dc.predict_matrix(0.8, 1.5))["home_win"]
    high = dc.markets(dc.predict_matrix(2.5, 0.8))["home_win"]
    assert high > low


def test_dixon_coles_correction_matches_formula():
    """Standard Dixon-Coles: down-weights 0-0/1-1, up-weights 1-0/0-1."""
    dc = DixonColesPoisson(rho=0.1)
    mu, lam = 1.2, 1.1
    assert dc._correction(0, 0, mu, lam) == 1 - mu * lam * 0.1
    assert dc._correction(0, 1, mu, lam) == 1 + mu * 0.1
    assert dc._correction(1, 0, mu, lam) == 1 + lam * 0.1
    assert dc._correction(1, 1, mu, lam) == 1 - 0.1
    assert dc._correction(2, 2, mu, lam) == 1.0  # no correction above (1,1)
