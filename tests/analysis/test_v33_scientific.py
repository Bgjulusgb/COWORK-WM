"""Smoke tests for the new scientific-stack modules (network_strength,
weight_optimizer's synthetic objective). Each module's heavy dependency
(PyMC, LightGBM) is *optional* so we only smoke-test what doesn't require it
in the default test environment.
"""
from datetime import datetime, timezone

from analysis.network_strength import build_network_pagerank


def test_network_pagerank_orders_winners_above_losers():
    matches = [
        {"home": "STRONG", "away": "WEAK", "home_goals": 3, "away_goals": 0,
         "kickoff": datetime(2025, 6, 1, tzinfo=timezone.utc)},
        {"home": "WEAK", "away": "STRONG", "home_goals": 0, "away_goals": 2,
         "kickoff": datetime(2025, 7, 1, tzinfo=timezone.utc)},
        {"home": "STRONG", "away": "MID", "home_goals": 2, "away_goals": 1,
         "kickoff": datetime(2025, 8, 1, tzinfo=timezone.utc)},
    ]
    scores = build_network_pagerank(matches, fifa_rank={"STRONG": 5, "MID": 30, "WEAK": 90})
    # STRONG must rank above WEAK in the resulting strength scoring.
    assert scores.get("STRONG", 0) > scores.get("WEAK", 0)


def test_weight_optimizer_synthetic_objective_is_minimised_at_centres():
    from analysis.weight_optimizer import _PRIOR_RANGES, synthetic_brier_objective

    obj = synthetic_brier_objective(targets=[1.0])
    centres = {k: (lo + hi) / 2 for k, (lo, hi) in _PRIOR_RANGES.items()}
    perturbed = {k: v + 0.05 for k, v in centres.items()}
    # The objective should be smaller at the centres than at a perturbed point.
    assert obj(centres) < obj(perturbed)
