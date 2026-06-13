"""Unit tests for wm2026.markets — derived markets from the score matrix.

Pure math, core deps only (numpy). Covers the Asian-handicap quarter-line
settlement (the trickiest piece) with a hand-computed matrix plus the
"quarter = average of its two neighbours" identity on a realistic Dixon-Coles
matrix.
"""
from __future__ import annotations

import numpy as np

from models_ml.poisson_goals import DixonColesPoisson
from wm2026 import markets as mk

# Hand matrix: M[i][j] = P(home i, away j). Already sums to 1.0.
#   home win  = [1,0]               = 0.3
#   draw      = [0,0] + [1,1]       = 0.1 + 0.4 = 0.5
#   away win  = [0,1]               = 0.2
HAND = [[0.1, 0.2], [0.3, 0.4]]

# A realistic, asymmetric matrix for the property checks.
REAL = DixonColesPoisson(rho=0.1, max_goals=6).predict_matrix(1.7, 1.0)


def test_one_x_two_matches_dixon_coles_markets():
    got = mk.one_x_two(REAL)
    ref = DixonColesPoisson(max_goals=6).markets(REAL)
    assert abs(got["home"] - ref["home_win"]) < 1e-12
    assert abs(got["draw"] - ref["draw"]) < 1e-12
    assert abs(got["away"] - ref["away_win"]) < 1e-12
    assert abs(sum(got.values()) - 1.0) < 1e-9


def test_double_chance_and_dnb():
    p = mk.one_x_two(HAND)
    dc = mk.double_chance(p["home"], p["draw"], p["away"])
    assert abs(dc["1X"] - 0.8) < 1e-9       # 0.3 + 0.5
    assert abs(dc["12"] - 0.5) < 1e-9       # 0.3 + 0.2
    assert abs(dc["X2"] - 0.7) < 1e-9       # 0.5 + 0.2
    dnb = mk.draw_no_bet(p["home"], p["draw"], p["away"])
    assert abs(dnb["home"] + dnb["away"] - 1.0) < 1e-12
    assert abs(dnb["home"] - 0.3 / 0.5) < 1e-9


def test_asian_handicap_zero_equals_1x2_with_push_on_draw():
    ah0 = mk.asian_handicap(HAND, 0.0)
    assert abs(ah0["home_win"] - 0.3) < 1e-9
    assert abs(ah0["push"] - 0.5) < 1e-9     # the draw mass pushes
    assert abs(ah0["away_win"] - 0.2) < 1e-9
    # no-push prob equals draw-no-bet home
    p = mk.one_x_two(HAND)
    dnb = mk.draw_no_bet(p["home"], p["draw"], p["away"])
    assert abs(ah0["home_prob_nopush"] - dnb["home"]) < 1e-9


def test_asian_handicap_half_line_has_no_push():
    ah = mk.asian_handicap(HAND, -0.5)
    assert ah["push"] < 1e-12
    assert abs(ah["home_win"] - 0.3) < 1e-9   # only the [1,0] win survives -0.5
    assert abs(ah["away_win"] - 0.7) < 1e-9


def test_asian_handicap_quarter_is_average_of_neighbours():
    # AH(-0.75) must be the average of AH(-0.5) and AH(-1.0); AH(-0.25) the
    # average of AH(0) and AH(-0.5). This is the defining property of quarter
    # lines and the strongest correctness check for the settlement table.
    for q, lo, hi in [(-0.75, -1.0, -0.5), (-0.25, -0.5, 0.0),
                      (0.25, 0.0, 0.5), (0.75, 0.5, 1.0)]:
        aq = mk.asian_handicap(REAL, q)
        a_lo = mk.asian_handicap(REAL, lo)
        a_hi = mk.asian_handicap(REAL, hi)
        assert abs(aq["home_win"] - 0.5 * (a_lo["home_win"] + a_hi["home_win"])) < 1e-9
        assert abs(aq["push"] - 0.5 * (a_lo["push"] + a_hi["push"])) < 1e-9


def test_asian_handicap_shares_sum_to_one():
    for ln in (-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.5, 1.25):
        ah = mk.asian_handicap(REAL, ln)
        assert abs(ah["home_win"] + ah["push"] + ah["away_win"] - 1.0) < 1e-9


def test_total_over_under_matches_existing_markets():
    ref = DixonColesPoisson(max_goals=6).markets(REAL)
    for line, key in [(0.5, "over_05"), (1.5, "over_15"),
                      (2.5, "over_25"), (3.5, "over_35")]:
        ou = mk.total_over_under(REAL, line)
        assert abs(ou["over"] - ref[key]) < 1e-9
        assert ou["push"] < 1e-12             # half line → no push
        assert abs(ou["over"] + ou["under"] - 1.0) < 1e-9


def test_total_quarter_line_is_average_of_neighbours():
    # O/U 2.75 splits into 2.5 and 3.0.
    q = mk.total_over_under(REAL, 2.75)
    lo = mk.total_over_under(REAL, 2.5)
    hi = mk.total_over_under(REAL, 3.0)
    assert abs(q["over"] - 0.5 * (lo["over"] + hi["over"])) < 1e-9


def test_clean_sheet_win_to_nil_and_parity():
    cs = mk.clean_sheet(HAND)
    assert abs(cs["home"] - (0.1 + 0.3)) < 1e-9   # away scores 0 → col 0
    assert abs(cs["away"] - (0.1 + 0.2)) < 1e-9   # home scores 0 → row 0
    wtn = mk.win_to_nil(HAND)
    assert abs(wtn["home"] - 0.3) < 1e-9          # [1,0] → home 1, away 0
    assert abs(wtn["away"] - 0.2) < 1e-9          # [0,1] → away 1, home 0
    # (REAL) win-to-nil must never exceed the clean-sheet probability.
    cs_r, wtn_r = mk.clean_sheet(REAL), mk.win_to_nil(REAL)
    assert wtn_r["home"] <= cs_r["home"] + 1e-12
    oe = mk.odd_even_goals(REAL)
    assert abs(oe["odd"] + oe["even"] - 1.0) < 1e-9


def test_team_total_sums_to_one():
    for side in ("home", "away"):
        tt = mk.team_total(REAL, side, 1.5)
        assert abs(tt["over"] + tt["push"] + tt["under"] - 1.0) < 1e-9


def test_derive_all_shape():
    out = mk.derive_all(REAL)
    for key in ("double_chance", "draw_no_bet", "asian_handicap", "totals",
                "team_total_home", "team_total_away", "clean_sheet",
                "win_to_nil", "odd_even"):
        assert key in out
    assert len(out["asian_handicap"]) >= 5
    # seeded 1X2 must flow into double chance
    seeded = mk.derive_all(REAL, p1x2=(0.6, 0.25, 0.15))
    assert abs(seeded["double_chance"]["1X"] - 0.85) < 1e-9


# ── Count markets (corners, cards) ────────────────────────────────────────────
def test_over_under_count_sums_to_one():
    """over + under is always a clean partition for an integer threshold."""
    pmf = mk.over_under_count(5.0, 4.5)
    assert abs(pmf["over"] + pmf["under"] - 1.0) < 1e-9
    assert 0.0 <= pmf["over"] <= 1.0


def test_over_under_count_zero_lambda():
    """λ = 0 means count is deterministic 0 — over any positive line is 0."""
    p = mk.over_under_count(0.0, 0.5)
    assert p["over"] == 0.0
    assert p["under"] == 1.0


def test_corners_market_lambda_sum_property():
    """Total λ is the per-team sum (independence of two Poissons)."""
    out = mk.corners_market(5.5, 4.5)
    assert abs(out["lambda_total"] - 10.0) < 1e-9
    assert {"lambda_home", "lambda_away", "lambda_total", "lines"} == set(out)
    # default 4 lines (8.5, 9.5, 10.5, 11.5)
    assert len(out["lines"]) == 4
    # Higher line ⇒ lower P(over).
    p_over_85 = next(l["over"] for l in out["lines"] if l["line"] == 8.5)
    p_over_115 = next(l["over"] for l in out["lines"] if l["line"] == 11.5)
    assert p_over_85 > p_over_115


def test_cards_market_default_lines():
    out = mk.cards_market(4.2)
    assert len(out["lines"]) == 4
    assert out["lambda_total"] == 4.2
    # over + under partitions on each line.
    for line in out["lines"]:
        assert abs(line["over"] + line["under"] - 1.0) < 1e-9


def test_derive_all_emits_corners_only_when_lambdas_provided():
    """Default call: no corners/cards keys. With lambdas: both keys present."""
    plain = mk.derive_all(REAL)
    assert "corners" not in plain and "cards" not in plain
    enriched = mk.derive_all(
        REAL, corners_lam_home=5.0, corners_lam_away=4.5,
        cards_lam_total=4.2,
    )
    assert "corners" in enriched and "cards" in enriched
    assert enriched["corners"]["lambda_total"] == 9.5
