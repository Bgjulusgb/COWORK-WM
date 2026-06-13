"""Tests for the bracket renderer (wm2026.tournament_viz)."""
from __future__ import annotations

from wm2026.tournament import TournamentResult
from wm2026.tournament_viz import render_bracket


def _result(title=None, final=None, advance=None) -> TournamentResult:
    return TournamentResult(
        advance_prob=advance or {},
        final_prob=final or {},
        title_prob=title or {},
        n_sims=1000,
    )


def test_bracket_orders_teams_by_descending_probability():
    res = _result(
        title={"ARG": 0.18, "BRA": 0.15, "FRA": 0.13},
        final={"ARG": 0.32, "BRA": 0.28, "FRA": 0.25, "ESP": 0.19},
        advance={"ARG": 0.95, "BRA": 0.93, "FRA": 0.90, "ESP": 0.88, "DEU": 0.85},
    )
    names = {"ARG": "Argentinien", "BRA": "Brasilien", "FRA": "Frankreich",
             "ESP": "Spanien", "DEU": "Deutschland"}
    out = render_bracket(res, names)
    # The three pyramid headers must appear
    assert "WELTMEISTER" in out
    assert "FINALE" in out
    assert "ACHTELFINALE" in out
    # Top team in each tier is Argentinien (highest p in every map)
    arg_first = out.index("Argentinien")
    bra_first = out.index("Brasilien")
    assert arg_first < bra_first
    # 1000 sims headline carries through
    assert "1000" in out


def test_bracket_drops_zero_probability_teams_per_tier():
    res = _result(
        title={"ARG": 0.20, "ZZZ": 0.0},
        final={"ARG": 0.30, "ZZZ": 0.0},
        advance={"ARG": 0.95, "ZZZ": 0.10},
    )
    names = {"ARG": "Argentinien", "ZZZ": "ZeroLand"}
    out = render_bracket(res, names)
    # ZeroLand only shows in the advance tier (the only positive one).
    occurrences = out.count("ZeroLand")
    assert occurrences == 1


def test_bracket_handles_empty_result():
    out = render_bracket(_result(), names={})
    assert "WELTMEISTER" in out
    assert "keine Wahrscheinlichkeit" in out
