"""Conservative (bootstrap-p5) staking + Asian-handicap edge rows."""
from __future__ import annotations

from wm2026.edge import (
    best_value_cons_pick,
    compute_edges,
    devig,
    evaluate_asian_handicap,
    evaluate_line,
)


def test_conservative_edge_is_below_point_estimate():
    row = evaluate_line("1X2", "Home", 0.55, 2.10, 0.44, model_p_lower=0.48)
    # p50 edge = 0.55*2.10-1 = 0.155; p5 edge = 0.48*2.10-1 = 0.008
    assert row.edge_pct is not None and row.edge_pct_cons is not None
    assert row.edge_pct_cons < row.edge_pct
    assert row.half_kelly_cons is not None and row.half_kelly_cons <= row.half_kelly_pct


def test_conservative_fields_none_without_ci():
    row = evaluate_line("1X2", "Home", 0.55, 2.10, 0.44)
    assert row.model_p_lower is None
    assert row.edge_pct_cons is None
    assert row.half_kelly_cons is None


def test_compute_edges_threads_ci_including_complements():
    ci = {
        "home_win": [0.48, 0.55, 0.62],
        "draw": [0.18, 0.22, 0.26],
        "away_win": [0.18, 0.23, 0.28],
        "over_25": [0.42, 0.50, 0.58],
        "btts": [0.44, 0.50, 0.56],
    }
    rows = compute_edges(
        {"home_win": 0.55, "draw": 0.22, "away_win": 0.23,
         "over_25": 0.50, "btts": 0.50},
        odds_1x2=[2.10, 3.40, 3.20], odds_ou25=[1.95, 1.95],
        odds_btts=[1.90, 1.90], ci=ci,
    )
    home = next(r for r in rows if r["selection"] == "Home")
    assert home["model_p_lower"] == 0.48
    # Under 2.5 uses the complement lower bound: 1 - p95(over) = 1 - 0.58 = 0.42.
    under = next(r for r in rows if r["selection"] == "Under 2.5")
    assert abs(under["model_p_lower"] - 0.42) < 1e-9


def test_double_chance_rows_emitted_with_odds():
    rows = compute_edges(
        {"home_win": 0.55, "draw": 0.22, "away_win": 0.23,
         "over_25": 0.5, "btts": 0.5},
        odds_dc=[1.25, 1.30, 1.55],
    )
    dc = [r for r in rows if r["market"] == "Double Chance"]
    assert {r["selection"] for r in dc} == {"1X", "12", "X2"}
    one_x = next(r for r in dc if r["selection"] == "1X")
    assert abs(one_x["model_p"] - 0.77) < 1e-9     # 0.55 + 0.22


def test_asian_handicap_edge_rows():
    ah = {"line": -0.5, "home_win": 0.60, "push": 0.0, "away_win": 0.40,
          "home_prob_nopush": 0.60, "away_prob_nopush": 0.40}
    rows = evaluate_asian_handicap(ah, home_odd=1.80, away_odd=2.10)
    assert len(rows) == 2
    home = next(r for r in rows if r["selection"] == "Home")
    # EV = 0.60*1.80 + 0 - 1 = 0.08 → 8%
    assert abs(home["edge_pct"] - 8.0) < 1e-6
    assert home["market"] == "AH -0.5"
    # model-only when no odds
    none_rows = evaluate_asian_handicap(ah)
    assert all(r["decimal_odd"] is None for r in none_rows)


# ── Phase 4 — DC conservative columns + AH p5 + best_value_cons ───────────────
def test_double_chance_gets_conservative_columns_from_dc_ci():
    """4.1 — DC rows must carry the p5 guard once the bootstrap supplies dc_*."""
    ci = {
        "home_win": [0.48, 0.55, 0.62],
        "draw": [0.18, 0.22, 0.26],
        "away_win": [0.18, 0.23, 0.28],
        "dc_1x": [0.70, 0.77, 0.84],
        "dc_12": [0.74, 0.78, 0.82],
        "dc_x2": [0.38, 0.45, 0.52],
    }
    rows = compute_edges(
        {"home_win": 0.55, "draw": 0.22, "away_win": 0.23,
         "over_25": 0.5, "btts": 0.5},
        odds_dc=[1.25, 1.30, 1.55], ci=ci,
    )
    dc = {r["selection"]: r for r in rows if r["market"] == "Double Chance"}
    assert dc["1X"]["model_p_lower"] == 0.70
    assert dc["12"]["model_p_lower"] == 0.74
    assert dc["X2"]["model_p_lower"] == 0.38
    # 1X @ 1.25 with p5 0.70: cons edge = 0.70*1.25-1 = -12.5% — below p50 edge.
    assert dc["1X"]["edge_pct_cons"] is not None
    assert dc["1X"]["edge_pct_cons"] < dc["1X"]["edge_pct"]


def test_double_chance_cons_none_without_dc_keys():
    """Old-style CI (no dc_*) must not crash and leaves the cons columns None."""
    ci = {"home_win": [0.48, 0.55, 0.62], "draw": [0.18, 0.22, 0.26],
          "away_win": [0.18, 0.23, 0.28]}
    rows = compute_edges(
        {"home_win": 0.55, "draw": 0.22, "away_win": 0.23,
         "over_25": 0.5, "btts": 0.5},
        odds_dc=[1.25, 1.30, 1.55], ci=ci,
    )
    one_x = next(r for r in rows if r["selection"] == "1X")
    assert one_x["edge_pct_cons"] is None


def test_asian_handicap_conservative_columns():
    """4.2 — AH rows accept bootstrap p5 of the no-push probabilities."""
    ah = {"line": -0.5, "home_win": 0.60, "push": 0.0, "away_win": 0.40,
          "home_prob_nopush": 0.60, "away_prob_nopush": 0.40}
    rows = evaluate_asian_handicap(
        ah, home_odd=1.80, away_odd=2.10,
        home_p_lower=0.52, away_p_lower=0.33,
    )
    home = next(r for r in rows if r["selection"] == "Home")
    away = next(r for r in rows if r["selection"] == "Away")
    # cons edge = p5 * odd - 1 (same desk approximation as the point estimate)
    assert abs(home["edge_pct_cons"] - (0.52 * 1.80 - 1.0) * 100.0) < 1e-6
    assert abs(away["edge_pct_cons"] - (0.33 * 2.10 - 1.0) * 100.0) < 1e-6
    assert home["edge_pct_cons"] < home["edge_pct"]
    assert home["model_p_lower"] == 0.52


def test_best_value_cons_pick_prefers_p5_survivor():
    """4.3 — the honest pick maximises the conservative edge, not the raw one."""
    rows = [
        # Big raw edge but collapses at p5 (the classic sanity-check trap).
        {"market": "1X2", "selection": "Away", "edge_pct": 13.0, "edge_pct_cons": -15.0},
        # Modest raw edge that survives the lower bound.
        {"market": "Double Chance", "selection": "12", "edge_pct": 9.9, "edge_pct_cons": 5.5},
        {"market": "BTTS", "selection": "Yes", "edge_pct": 4.0, "edge_pct_cons": 1.2},
        # No CI at all → not eligible.
        {"market": "AH -0.5", "selection": "Home", "edge_pct": 20.0, "edge_pct_cons": None},
    ]
    pick = best_value_cons_pick(rows)
    assert pick is not None
    assert (pick["market"], pick["selection"]) == ("Double Chance", "12")


def test_best_value_cons_pick_none_when_nothing_survives():
    rows = [
        {"market": "1X2", "selection": "Home", "edge_pct": 8.0, "edge_pct_cons": -2.0},
        {"market": "BTTS", "selection": "No", "edge_pct": 3.0, "edge_pct_cons": None},
    ]
    assert best_value_cons_pick(rows) is None
    assert best_value_cons_pick([]) is None


# ── Malformed-odds hardening (regression guard for the IndexError class) ──────
def test_devig_returns_empty_on_all_invalid_odds():
    """All-zero / negative input → ([], 0.0) — no IndexError downstream."""
    assert devig([0.0, 0.0, 0.0]) == ([], 0.0)
    assert devig([-1.0, 0.0, 1.0]) == ([], 0.0)
    assert devig([]) == ([], 0.0)


def test_devig_drops_invalid_entries_keeps_valid_ones():
    """Partial input is allowed; callers must check len() before indexing."""
    fair, overround = devig([2.0, 0.0, 3.0])
    assert len(fair) == 2
    assert overround > 0
    assert abs(sum(fair) - 1.0) < 1e-9


def test_compute_edges_survives_all_zero_odds():
    """A typo like --odds "0/0/0" must not crash the edge table —
    every 1X2 row falls through to model-only (decimal_odd=None, edge=None)."""
    rows = compute_edges(
        {"home_win": 0.55, "draw": 0.22, "away_win": 0.23,
         "over_25": 0.50, "btts": 0.50},
        odds_1x2=[0.0, 0.0, 0.0],
        odds_ou25=[-1.0, 0.5],
    )
    one_x_two = [r for r in rows if r["market"] == "1X2"]
    assert len(one_x_two) == 3
    for r in one_x_two:
        assert r["fair_p"] is None
        assert r["decimal_odd"] is None
        assert r["edge_pct"] is None
