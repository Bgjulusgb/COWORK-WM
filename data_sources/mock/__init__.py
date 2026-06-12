"""Deterministic synthetic data for offline / CI runs (USE_MOCK_*=true).

Every mock is seeded from the team code so repeated runs return identical data
— important for stable tests. Strength is derived from the real FIFA ranking
table so mock results still look plausible (stronger teams win more).
"""
from __future__ import annotations

from scripts.team_real_data import get_world_ranking


def team_strength(code: str) -> float:
    """Map a FIFA ranking (lower = better) to a 0.55..2.1 strength proxy."""
    rank = get_world_ranking(code)
    return max(0.55, min(2.1, 2.25 - rank * 0.016))
