"""Tests for the negative-binomial goal model + factory."""
import numpy as np

from analysis.match_predictor import MatchPredictor
from factors.base import FactorSignal
from models_ml.poisson_goals import (
    DixonColesPoisson,
    NegativeBinomialDixonColes,
    build_goal_model,
)


def test_match_predictor_uses_negbin_end_to_end():
    """The goal_model flag must thread through MatchPredictor into the matrix."""
    sig = FactorSignal(name="elo", home_strength=1.2, away_strength=0.8,
                       weight=0.3, confidence=0.85, available=True, source="test")
    predictor = MatchPredictor(goal_model="negbin", negbin_size=4.0, max_goals=8)
    assert isinstance(predictor.poisson, NegativeBinomialDixonColes)
    pred, _ = predictor.predict_from_signals([sig], base_home_xg=1.5, base_away_xg=1.2)
    total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
    assert 0.98 <= total <= 1.02
    assert 0.0 <= pred.btts <= 1.0


def test_negbin_matrix_sums_to_one():
    m = NegativeBinomialDixonColes(rho=0.1, max_goals=8, size=8.0).predict_matrix(1.5, 1.2)
    assert abs(m.sum() - 1.0) < 1e-6


def test_negbin_1x2_in_range():
    model = NegativeBinomialDixonColes(max_goals=8)
    markets = model.markets(model.predict_matrix(1.6, 1.1))
    total = markets["home_win"] + markets["draw"] + markets["away_win"]
    assert abs(total - 1.0) < 0.02
    assert 0.0 <= markets["btts"] <= 1.0


def _total_goal_moments(matrix: np.ndarray) -> tuple[float, float]:
    """Mean and variance of the total-goals distribution implied by the matrix."""
    n = matrix.shape[0]
    totals = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            totals[i + j] += matrix[i][j]
    ks = np.arange(len(totals))
    mean = float((ks * totals).sum())
    var = float(((ks - mean) ** 2 * totals).sum())
    return mean, var


def test_negbin_more_dispersed_than_poisson():
    """Over-dispersion ⇒ higher total-goals variance at (roughly) equal mean."""
    xg = (1.8, 1.6)
    pois = DixonColesPoisson(max_goals=12)
    nb = NegativeBinomialDixonColes(max_goals=12, size=3.0)  # strong dispersion
    mean_p, var_p = _total_goal_moments(pois.predict_matrix(*xg))
    mean_nb, var_nb = _total_goal_moments(nb.predict_matrix(*xg))
    assert var_nb > var_p                      # the defining property
    assert abs(mean_nb - mean_p) < 0.2         # mean broadly preserved


def test_factory_defaults_to_poisson():
    assert isinstance(build_goal_model("poisson"), DixonColesPoisson)
    assert not isinstance(build_goal_model("poisson"), NegativeBinomialDixonColes)
    assert isinstance(build_goal_model("negbin"), NegativeBinomialDixonColes)
    # Unknown model string falls back to Poisson rather than raising.
    assert isinstance(build_goal_model("garbage"), DixonColesPoisson)


def test_negbin_converges_to_poisson_at_high_size():
    xg = (1.4, 1.2)
    pois = DixonColesPoisson(max_goals=8).predict_matrix(*xg)
    nb = NegativeBinomialDixonColes(max_goals=8, size=500.0).predict_matrix(*xg)
    # Large dispersion parameter ⇒ NB ≈ Poisson.
    assert np.max(np.abs(pois - nb)) < 0.01
