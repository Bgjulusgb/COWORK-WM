"""Deterministic mock for the the-odds-api connector.

Used in test/CI/offline mode and as the graceful fallback when the live
endpoint 401/429s. The numbers are derived from the FIFA-code pair so two
calls for the same fixture always return the same odds — the mock-path tests
can rely on exact values.
"""
from __future__ import annotations

import hashlib
from typing import Any

# Vig built into the mock so devig() exercises the same code path as live.
_OVERROUND = 1.05


def _seed(home_code: str, away_code: str) -> int:
    raw = f"{home_code.upper()}|{away_code.upper()}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _spread(seed: int, lo: float, hi: float) -> float:
    """Stable pseudo-random in [lo, hi]."""
    return lo + (hi - lo) * ((seed % 1009) / 1008.0)


def _three_way(seed: int, lean: float) -> tuple[float, float, float]:
    """Build a 1X2 triple whose Σ(1/odd) == _OVERROUND (bookie has the edge)."""
    base_home = 0.42 - 0.18 * lean
    base_draw = 0.26
    base_away = 1.0 - base_home - base_draw
    # ``decimal_odd = 1 / (true_prob * overround)`` — Σ(implied) = overround.
    return (
        round(1.0 / (base_home * _OVERROUND), 2),
        round(1.0 / (base_draw * _OVERROUND), 2),
        round(1.0 / (base_away * _OVERROUND), 2),
    )


def _two_way(p_over: float) -> tuple[float, float]:
    """Two-way book at the same overround as the 1X2."""
    p_under = 1.0 - p_over
    return (
        round(1.0 / (p_over * _OVERROUND), 2),
        round(1.0 / (p_under * _OVERROUND), 2),
    )


def odds_for(home_code: str, away_code: str) -> dict[str, list[float]]:
    """Return the same shape the live connector emits."""
    seed = _seed(home_code, away_code)
    lean = (_spread(seed, 0.0, 1.0) - 0.5) * 2.0      # [-1, 1]

    h, d, a = _three_way(seed, lean)
    over, under = _two_way(_spread(seed >> 4, 0.50, 0.60))
    yes, no = _two_way(_spread(seed >> 8, 0.50, 0.62))

    return {
        "1x2": [h, d, a],
        "ou_2_5": [over, under],
        "btts": [yes, no],
    }


def envelope_for(home_code: str, away_code: str) -> dict[str, Any]:
    """Return a payload shaped like the live API would, for tests that need
    the full envelope (e.g. parser tests)."""
    odds = odds_for(home_code, away_code)
    return {
        "id": f"mock-{home_code.lower()}-{away_code.lower()}",
        "sport_key": "soccer_fifa_world_cup",
        "commence_time": "2026-06-18T18:00:00Z",
        "home_team": home_code,
        "away_team": away_code,
        "bookmakers": [{
            "key": "mock_book",
            "title": "MockBook",
            "last_update": "2026-06-12T12:00:00Z",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": home_code, "price": odds["1x2"][0]},
                    {"name": "Draw", "price": odds["1x2"][1]},
                    {"name": away_code, "price": odds["1x2"][2]},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": odds["ou_2_5"][0], "point": 2.5},
                    {"name": "Under", "price": odds["ou_2_5"][1], "point": 2.5},
                ]},
                {"key": "btts", "outcomes": [
                    {"name": "Yes", "price": odds["btts"][0]},
                    {"name": "No", "price": odds["btts"][1]},
                ]},
            ],
        }],
    }


__all__ = ["odds_for", "envelope_for"]
