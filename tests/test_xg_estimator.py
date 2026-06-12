"""Tests for the Dixon-Coles MLE λ-estimator (analysis.xg_estimator).

Deterministic (fixed seed), bare pytest, core deps. Recovery-from-synthetic uses
tolerance assertions (never exact equality — BLAS builds differ).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from analysis.xg_estimator import (
    _collect_games,
    estimate_base_xg,
    estimate_strengths,
    lambdas_for_fixture,
)

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _synthetic(seed=0, n=600, T=12):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(T)]
    attack = rng.normal(0, 0.35, T); attack -= attack.mean()
    defence = rng.normal(0, 0.35, T); defence -= defence.mean()
    home_adv, mu = 0.25, math.log(1.35)
    games = []
    for _ in range(n):
        i, j = rng.choice(T, 2, replace=False)
        lh = math.exp(mu + home_adv + attack[i] - defence[j])
        la = math.exp(mu + attack[j] - defence[i])
        games.append(SimpleNamespace(
            home_code=teams[i], away_code=teams[j],
            home_score=int(rng.poisson(lh)), away_score=int(rng.poisson(la)),
            kickoff_utc=NOW - timedelta(days=int(rng.integers(0, 30))),
            competition_tier=1))
    return games, teams, attack, defence, home_adv, mu


def test_recovers_known_strengths_and_lambdas():
    games, teams, attack, defence, home_adv, mu = _synthetic()
    s = estimate_strengths(games, ref_time=NOW, xi=0.0, rho0=0.0, min_matches=6)
    assert s is not None and s.converged
    assert abs(sum(s.attack.values())) < 1e-6        # sum-to-zero identifiability
    assert abs(sum(s.defence.values())) < 1e-6
    assert s.home_adv > 0.1                           # positive home advantage recovered
    # held-out fixture λ recovered within tolerance
    i, j = 0, 5
    lh_true = math.exp(mu + home_adv + attack[i] - defence[j])
    la_true = math.exp(mu + attack[j] - defence[i])
    lh, la = lambdas_for_fixture(s, teams[i], teams[j])
    assert abs(lh - lh_true) < 0.25 and abs(la - la_true) < 0.25
    # strongest attacker ranks above the weakest in the fitted attack
    assert s.attack[teams[int(np.argmax(attack))]] > s.attack[teams[int(np.argmin(attack))]]


def test_fallback_on_sparse_history():
    games = [SimpleNamespace(home_code="A", away_code="B", home_score=1, away_score=0,
                             kickoff_utc=NOW, competition_tier=1) for _ in range(3)]
    assert estimate_strengths(games, ref_time=NOW, min_matches=6) is None


def test_fallback_on_non_identifiable():
    # Two teams only ever playing each other → < 2 distinct opponents each.
    games = [SimpleNamespace(home_code="A", away_code="B", home_score=h, away_score=a,
                             kickoff_utc=NOW - timedelta(days=d), competition_tier=1)
             for d, (h, a) in enumerate([(1, 0), (2, 1), (0, 0), (3, 1), (1, 1), (2, 2), (0, 1)])]
    assert estimate_strengths(games, ref_time=NOW, min_matches=6) is None


def test_estimate_base_xg_disabled_by_default():
    ctx = SimpleNamespace(historical_matches_home=[], historical_matches_away=[],
                          head_to_head=[], kickoff_utc=NOW)
    res, diag = estimate_base_xg(ctx, "A", "B", settings=SimpleNamespace())
    assert res is None and diag["source"] == "yaml"


def test_estimate_base_xg_enabled_returns_lambdas():
    games, teams, *_ = _synthetic()
    ctx = SimpleNamespace(historical_matches_home=games, historical_matches_away=[],
                          head_to_head=[], kickoff_utc=NOW)
    settings = SimpleNamespace(use_mle_xg=True, mle_time_decay_xi=0.0,
                               mle_min_matches=6, mle_max_iter=200)
    res, diag = estimate_base_xg(ctx, teams[0], teams[1], settings=settings)
    assert res is not None and diag["source"] == "mle"
    assert 0.2 <= res[0] <= 4.5 and 0.2 <= res[1] <= 4.5


def test_time_decay_downweights_old_matches():
    games = [
        SimpleNamespace(home_code="A", away_code="B", home_score=1, away_score=0,
                        kickoff_utc=NOW, competition_tier=1),
        SimpleNamespace(home_code="A", away_code="B", home_score=1, away_score=0,
                        kickoff_utc=NOW - timedelta(days=365), competition_tier=1),
    ]
    weights = sorted(g[4] for g in _collect_games(games, NOW, xi=0.0065))
    assert weights[0] < 0.2 < weights[1]      # 1-year-old match heavily down-weighted
