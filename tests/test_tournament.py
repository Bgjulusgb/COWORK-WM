"""Tests for the tournament Monte-Carlo engine (wm2026.tournament)."""
from __future__ import annotations

from models_ml.poisson_goals import build_all_goal_models
from wm2026.tournament import (
    _bracket_seeds,
    _largest_pow2,
    simulate_tournament,
)

MODELS = build_all_goal_models()
GROUPS = {g: [f"{g}{i}" for i in range(1, 5)] for g in "ABCD"}     # 4 groups × 4 = 16 teams


def _lam(strength):
    def provider(home, away):
        return (max(0.2, strength[home]), max(0.2, strength[away]))
    return provider


def _flat_strength(value=1.3):
    return {t: value for ts in GROUPS.values() for t in ts}


def test_bracket_and_pow2_helpers():
    assert _largest_pow2(12) == 8 and _largest_pow2(32) == 32 and _largest_pow2(7) == 4
    assert sorted(_bracket_seeds(8)) == list(range(1, 9))     # a permutation of 1..8
    assert _bracket_seeds(4) == [1, 4, 2, 3]                  # 1v4 | 2v3, 1&2 split halves
    # WC-2026 default: 12 groups → 8 best thirds (24 + 8 = 32-team knockout)
    base = 24
    assert _largest_pow2(base + 12) - base == 8


def test_conservation_and_determinism():
    res1 = simulate_tournament(GROUPS, lam_provider=_lam(_flat_strength()), models=MODELS,
                               n_sims=300, seed=0, n_best_thirds=0)
    res2 = simulate_tournament(GROUPS, lam_provider=_lam(_flat_strength()), models=MODELS,
                               n_sims=300, seed=0, n_best_thirds=0)
    assert res1.title_prob == res2.title_prob          # determinism (same seed)
    # exactly 8 advance / 2 finalists / 1 champion every sim
    assert abs(sum(res1.advance_prob.values()) - 8.0) < 1e-9
    assert abs(sum(res1.final_prob.values()) - 2.0) < 1e-9
    assert abs(sum(res1.title_prob.values()) - 1.0) < 1e-9


def test_stronger_team_dominates():
    strength = _flat_strength(1.1)
    strength["A1"] = 2.6                                # one clearly stronger team
    res = simulate_tournament(GROUPS, lam_provider=_lam(strength), models=MODELS,
                              n_sims=400, seed=1, n_best_thirds=0)
    assert res.advance_prob["A1"] > 0.85               # walks out of its group (Poisson variance)
    assert res.title_prob["A1"] == max(res.title_prob.values())
    assert res.title_prob["A1"] > 0.25
    assert res.ranked("title_prob")[0][0] == "A1"
