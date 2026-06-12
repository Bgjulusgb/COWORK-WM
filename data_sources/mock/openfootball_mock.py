"""Synthetic match history + head-to-head, used when openfootball is mocked.

Generates plausible finished matches (scores follow each team's ranking-based
strength) so FormFactor / GoalEfficiencyFactor / HeadToHeadFactor have data to
chew on without any network access.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from data_sources.mock import team_strength
from data_sources.schemas import HistoricalMatch
from data_sources.team_codes import CODE_TO_NAMES, preferred_name

_ALL_CODES = list(CODE_TO_NAMES.keys())
_EPOCH = datetime(2026, 5, 1, tzinfo=timezone.utc)


def _simulate_score(rng: random.Random, attacker: str, defender: str) -> int:
    diff = team_strength(attacker) - team_strength(defender)
    expected = 1.25 + 0.55 * diff
    goals = round(rng.gauss(expected, 0.9))
    return max(0, min(6, goals))


def _make_match(rng: random.Random, code: str, opp: str, when: datetime, code_at_home: bool) -> HistoricalMatch:
    home, away = (code, opp) if code_at_home else (opp, code)
    home_goals = _simulate_score(rng, home, away)
    away_goals = _simulate_score(rng, away, home)
    tier = rng.choice([1, 2, 2, 4])  # WC final, qualifiers (x2), friendly
    return HistoricalMatch(
        source="mock",
        tournament="Mock International",
        competition_tier=tier,
        home_code=home,
        away_code=away,
        home_name=preferred_name(home),
        away_name=preferred_name(away),
        kickoff_utc=when,
        home_score=home_goals,
        away_score=away_goals,
        is_finished=True,
    )


def historical_results(code: str, n: int = 12) -> list[HistoricalMatch]:
    """Deterministic last-n results for a team, newest first."""
    code = code.upper()
    rng = random.Random(f"hist::{code}")
    pool = [c for c in _ALL_CODES if c != code] or _ALL_CODES
    out: list[HistoricalMatch] = []
    for i in range(n):
        opp = rng.choice(pool)
        when = _EPOCH - timedelta(days=18 * i + rng.randint(0, 6))
        out.append(_make_match(rng, code, opp, when, code_at_home=(i % 2 == 0)))
    return out


def head_to_head(home: str, away: str, n: int = 6) -> list[HistoricalMatch]:
    """Deterministic past meetings between two teams, newest first."""
    home, away = home.upper(), away.upper()
    # Stable regardless of argument order so HOME_AWAY and AWAY_HOME agree.
    seed = "h2h::" + "::".join(sorted([home, away]))
    rng = random.Random(seed)
    out: list[HistoricalMatch] = []
    for i in range(n):
        when = _EPOCH - timedelta(days=210 * i + rng.randint(0, 40))
        out.append(_make_match(rng, home, away, when, code_at_home=(i % 2 == 0)))
    return out


__all__ = ["historical_results", "head_to_head"]
