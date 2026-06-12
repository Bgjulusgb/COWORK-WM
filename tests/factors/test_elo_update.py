"""Tests for the World-Football-Elo K-update utility."""
import pytest

from data_sources.schemas import HistoricalMatch
from datetime import datetime, timezone
from factors.elo_update import expected_score, recompute_from_history, update_match


def test_expected_score_symmetry():
    assert expected_score(1800, 1800) == pytest.approx(0.5)
    assert expected_score(2000, 1800) > 0.5
    # Home + away expected scores sum to 1.
    assert expected_score(2000, 1800) + expected_score(1800, 2000) == pytest.approx(1.0)


def test_update_is_zero_sum():
    nh, na = update_match(1800, 1800, 2, 0, tier=1)
    assert nh > 1800 and na < 1800
    assert (nh - 1800) == pytest.approx(1800 - na)


def test_underdog_win_moves_more_than_favourite_win():
    # Favourite (2000) beating minnow (1500) gains little; the reverse gains a lot.
    fav_gain = update_match(2000, 1500, 1, 0, tier=2)[0] - 2000
    dog_gain = update_match(1500, 2000, 1, 0, tier=2)[0] - 1500
    assert dog_gain > fav_gain


def test_blowout_multiplier_amplifies_change():
    narrow = update_match(1800, 1800, 1, 0, tier=1)[0] - 1800
    rout = update_match(1800, 1800, 5, 0, tier=1)[0] - 1800
    assert rout > narrow


def _m(h, a, hs, as_, tier=1):
    return HistoricalMatch(source="t", competition_tier=tier, home_code=h, away_code=a,
                           home_name=h, away_name=a,
                           kickoff_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
                           home_score=hs, away_score=as_)


def test_recompute_from_history_tracks_winner():
    matches = [_m("AAA", "BBB", 3, 0), _m("AAA", "BBB", 2, 1), _m("BBB", "AAA", 0, 1)]
    out = recompute_from_history({"AAA": 1500, "BBB": 1500}, matches)
    assert out["AAA"] > 1500 > out["BBB"]
