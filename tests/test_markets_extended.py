"""Tests for the Phase-1 extended markets in wm2026.markets.

Pure math, core deps only. Covers the load-bearing invariants: distributions
sum to 1, the exact-total mean equals λ_home+λ_away, winning margins reconcile
with 1X2, first-goal splits by the λ ratio, and the HT/FT grid convolves back to
the full-match 1X2.
"""
from __future__ import annotations

import numpy as np

from models_ml.poisson_goals import DixonColesPoisson
from wm2026 import markets as mk

# Independent Poisson (rho=0) so HT⊛FT convolution is exact for the reconciliation.
LAM_H, LAM_A = 1.7, 1.1
INDEP = DixonColesPoisson(rho=0.0, max_goals=8).predict_matrix(LAM_H, LAM_A)
REAL = DixonColesPoisson(rho=0.1, max_goals=6).predict_matrix(1.6, 1.0)


def test_exact_total_goals_is_a_distribution_with_right_mean():
    dist = mk.exact_total_goals(INDEP)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    mean = sum(k * p for k, p in dist.items())
    lh, la = mk._lambdas_from_matrix(INDEP)
    # mean of the total = Σ(i+j)·M = λ_home + λ_away exactly (same matrix marginals)
    assert abs(mean - (lh + la)) < 1e-9


def test_winning_margin_sums_to_one_and_reconciles_with_1x2():
    wm = mk.winning_margin(REAL)
    assert abs(sum(wm.values()) - 1.0) < 1e-9
    p = mk.one_x_two(REAL)
    assert abs((wm["home_by_1"] + wm["home_by_2plus"]) - p["home"]) < 1e-9
    assert abs((wm["away_by_1"] + wm["away_by_2plus"]) - p["away"]) < 1e-9
    assert abs(wm["draw"] - p["draw"]) < 1e-9


def test_multi_goal_bands_sum_to_one():
    bands = mk.multi_goal_bands(REAL)
    assert set(bands) == {"0-1", "2-3", "4-6", "7+"}
    assert abs(sum(bands.values()) - 1.0) < 1e-9


def test_first_goal_sums_to_one_and_splits_by_lambda():
    fg = mk.first_goal(REAL, 1.6, 1.0)
    assert abs(fg["home"] + fg["away"] + fg["none"] - 1.0) < 1e-9
    assert abs(fg["none"] - float(REAL[0, 0])) < 1e-12
    # home:away first-goal odds equal the λ ratio
    assert abs(fg["home"] / fg["away"] - 1.6 / 1.0) < 1e-9


def test_ht_ft_grid_sums_to_one_and_convolves_to_fulltime_1x2():
    models = {"poisson": DixonColesPoisson(rho=0.0, max_goals=8)}
    grid = mk.ht_ft(LAM_H, LAM_A, ht_share=0.45, models=models, weights={"poisson": 1.0})
    assert len(grid) == 9
    assert abs(sum(grid.values()) - 1.0) < 1e-9
    # Sum HT outcomes → full-time 1X2 must match the full-match matrix (Poisson
    # convolution is exact; small truncation tolerance).
    ft_home = grid["H/H"] + grid["D/H"] + grid["A/H"]
    ft_draw = grid["H/D"] + grid["D/D"] + grid["A/D"]
    ft_away = grid["H/A"] + grid["D/A"] + grid["A/A"]
    p = mk.one_x_two(INDEP)
    assert abs(ft_home - p["home"]) < 0.02
    assert abs(ft_draw - p["draw"]) < 0.02
    assert abs(ft_away - p["away"]) < 0.02


def test_ht_ft_lead_can_only_be_held_or_lost_sensibly():
    grid = mk.ht_ft(2.2, 0.6)        # strong home favourite
    # Leading at HT and winning FT should be the single biggest joint cell.
    assert grid["H/H"] == max(grid.values())


def test_derive_all_includes_new_markets():
    out = mk.derive_all(REAL, lam_home=1.6, lam_away=1.0)
    for key in ("winning_margin", "multi_goal_bands", "exact_total_goals",
                "first_goal", "ht_ft"):
        assert key in out
    assert abs(sum(out["ht_ft"].values()) - 1.0) < 1e-9
