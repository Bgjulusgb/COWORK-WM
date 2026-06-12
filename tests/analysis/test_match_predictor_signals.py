"""Tests for MatchPredictor.predict_from_signals."""
import pytest

from analysis.match_predictor import MatchPredictor
from factors.base import FactorSignal


def _signal(name="elo", home=1.2, away=0.8, weight=0.3, conf=0.85, available=True):
    return FactorSignal(
        name=name,
        home_strength=home,
        away_strength=away,
        weight=weight,
        confidence=conf,
        available=available,
        source="test",
    )


def test_predict_from_signals_outputs_valid_probabilities():
    pred, ensemble = MatchPredictor().predict_from_signals(
        signals=[_signal()],
        base_home_xg=1.4,
        base_away_xg=1.2,
    )
    total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
    assert 0.99 <= total <= 1.01
    assert 0.0 <= pred.btts <= 1.0
    assert 0.0 <= pred.confidence <= 1.0
    assert ensemble.lambda_home_multiplier == pytest.approx(1.2)


def test_predict_from_signals_is_deterministic():
    p1, _ = MatchPredictor().predict_from_signals(
        [_signal()], base_home_xg=1.4, base_away_xg=1.2
    )
    p2, _ = MatchPredictor().predict_from_signals(
        [_signal()], base_home_xg=1.4, base_away_xg=1.2
    )
    assert p1.home_win_prob == p2.home_win_prob
    assert p1.home_xg == p2.home_xg


def test_market_prior_pulls_probabilities_toward_book():
    # Model alone leans heavily home; book says 50/30/20. The blend halves the gap.
    pred_no_prior, _ = MatchPredictor().predict_from_signals(
        [_signal(home=1.5, away=0.5)], base_home_xg=1.6, base_away_xg=1.2
    )
    pred_with_prior, _ = MatchPredictor().predict_from_signals(
        [_signal(home=1.5, away=0.5)],
        base_home_xg=1.6,
        base_away_xg=1.2,
        market_prior=(0.5, 0.3, 0.2),
    )
    # Book home prob is lower → blended home_win_prob must be below the no-prior one.
    assert pred_with_prior.home_win_prob < pred_no_prior.home_win_prob


def test_all_unavailable_signals_fall_back_to_base_xg():
    pred, ensemble = MatchPredictor().predict_from_signals(
        signals=[_signal(available=False)],
        base_home_xg=1.4,
        base_away_xg=1.2,
    )
    # No multiplier applied — Poisson should treat base_xg as λ directly.
    assert pred.home_xg == pytest.approx(1.4)
    assert pred.away_xg == pytest.approx(1.2)
    assert ensemble.confidence == 0.0
    assert pred.confidence == 0.0


def test_feature_payload_reports_active_factors():
    pred, _ = MatchPredictor().predict_from_signals(
        signals=[
            _signal(name="elo"),
            _signal(name="form", available=False),
        ],
        base_home_xg=1.4,
        base_away_xg=1.2,
    )
    assert pred.features["active_factors"] == ["elo"]
    assert pred.features["skipped_factors"] == ["form"]
    assert pred.features["market_used"] is False
