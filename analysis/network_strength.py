"""PageRank-over-results network — builds the offline artifact for
:mod:`factors.network_strength`.

Approach (Bryan & Leise 2006, Massey 1997):

    * Build a directed graph: an edge from loser → winner per match.
    * Edge weight = 1 + goal_difference, multiplied by a recency decay
      exp(-Δt / τ), τ = 730 days (≈2 years).
    * Add a small personalised vector seeded from FIFA ranking (lower
      rank = higher prior strength) so newly-ranked sides aren't NaN.
    * Run :func:`networkx.pagerank` and persist the score per team to
      ``models_ml/artifacts/network_strength.json``.

This module exposes both the builder (called by ``scripts/refresh_network``)
and a tiny pure-python ranker used in the unit tests so the test suite can
verify the factor wiring without depending on ``networkx``.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import structlog

from config.settings import settings

log = structlog.get_logger("analysis.network_strength")

_ARTIFACT = settings.base_dir / "models_ml" / "artifacts" / "network_strength.json"
_DECAY_TAU_DAYS = 730.0


def _recency_weight(kickoff: datetime, now: datetime) -> float:
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - kickoff).total_seconds() / 86400.0)
    return math.exp(-days / _DECAY_TAU_DAYS)


def build_network_pagerank(
    matches: Iterable[dict],
    fifa_rank: Mapping[str, int] | None = None,
    *,
    artifact_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, float]:
    """Compute PageRank from a list of finished matches.

    ``matches`` items: ``{"home": CODE, "away": CODE, "home_goals": int,
    "away_goals": int, "kickoff": datetime}``.
    """
    try:
        import networkx as nx
    except Exception:
        # In tests / lightweight installs without networkx we fall back to
        # the pure-python tally below — same shape, smaller numbers.
        log.debug("networkx_missing_fallback")
        return _fallback_score(matches, fifa_rank, now)

    G = nx.DiGraph()
    now = now or datetime.now(timezone.utc)
    for m in matches:
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        if hg == ag:
            continue
        winner, loser = (m["home"], m["away"]) if hg > ag else (m["away"], m["home"])
        weight = (1 + abs(hg - ag)) * _recency_weight(m["kickoff"], now)
        if G.has_edge(loser, winner):
            G[loser][winner]["weight"] += weight
        else:
            G.add_edge(loser, winner, weight=weight)

    personalisation = None
    if fifa_rank:
        # Lower rank = higher prior weight. Normalise so it sums to 1.
        raw = {code: 1.0 / (1 + r) for code, r in fifa_rank.items() if code in G}
        total = sum(raw.values())
        if total > 0:
            personalisation = {k: v / total for k, v in raw.items()}

    scores = nx.pagerank(G, alpha=0.85, personalization=personalisation, weight="weight")
    path = artifact_path or _ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    return scores


def _fallback_score(
    matches: Iterable[dict],
    fifa_rank: Mapping[str, int] | None,
    now: datetime | None,
) -> dict[str, float]:
    tally: dict[str, float] = {}
    now = now or datetime.now(timezone.utc)
    for m in matches:
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        if hg == ag:
            continue
        winner = m["home"] if hg > ag else m["away"]
        tally[winner] = tally.get(winner, 0.0) + (1 + abs(hg - ag)) * _recency_weight(m["kickoff"], now)
    if fifa_rank:
        for code, r in fifa_rank.items():
            tally[code] = tally.get(code, 0.0) + 0.1 / (1 + r)
    total = sum(tally.values()) or 1.0
    return {k: v / total for k, v in tally.items()}


__all__ = ["build_network_pagerank"]
