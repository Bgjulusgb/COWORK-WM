"""Tests for the Ranked Probability Score added to analysis.backtesting."""
from __future__ import annotations

from types import SimpleNamespace

from analysis.backtesting import _rps, compute


def _row(hp, dp, ap, hs, as_):
    return SimpleNamespace(
        home_win_prob=hp, draw_prob=dp, away_win_prob=ap,
        actual_home_score=hs, actual_away_score=as_,
    )


def test_rps_perfect_is_zero():
    assert _rps((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 0.0
    assert _rps((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) == 0.0


def test_rps_worst_case_is_one():
    # All mass on Home, Away actually happens → maximal ordinal distance.
    assert abs(_rps((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)) - 1.0) < 1e-12


def test_rps_orders_by_ordinal_distance():
    # Predict Home; a Draw should be penalised less than an Away (Draw is the
    # adjacent category, Away is the far one). This is the whole point of RPS
    # over Brier — Brier would score both misses identically on the wrong cell.
    p = (1.0, 0.0, 0.0)
    draw_miss = _rps(p, (0.0, 1.0, 0.0))
    away_miss = _rps(p, (0.0, 0.0, 1.0))
    assert draw_miss < away_miss
    assert abs(draw_miss - 0.5) < 1e-12


def test_compute_reports_rps_between_zero_and_one():
    rows = [
        _row(0.6, 0.25, 0.15, 2, 0),   # home predicted, home happened
        _row(0.2, 0.3, 0.5, 0, 1),     # away predicted, away happened
        _row(0.4, 0.35, 0.25, 1, 1),   # draw happened
    ]
    report = compute(rows)
    assert report.n_evaluated == 3
    assert 0.0 <= report.rps <= 1.0
    # A sharper, well-calibrated set must beat the uniform baseline (rps=1/3).
    assert report.rps < 1.0 / 3.0
