"""EXTEND-04: player availability (injuries / suspensions).

Static JSON snapshot for now (`scripts/player_availability.json`). When a
data feed (Transfermarkt scraper, ESPN injury API) is hooked up later, the
update script populates the same file and downstream code is unchanged.

xG impact: per-team aggregated impact score in [0, 1]. We translate that
into an xG multiplier in [0.85, 1.0] — even a fully decimated squad gets a
floor because the predictor's base xG already encodes squad quality.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

_SNAPSHOT = Path(__file__).resolve().parents[1] / "scripts" / "player_availability.json"

_CACHED: dict | None = None


def _load() -> dict:
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    if not _SNAPSHOT.exists():
        _CACHED = {}
        return _CACHED
    try:
        _CACHED = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        _CACHED = {}
    return _CACHED


def impact_for(team_code: str) -> float:
    """Return the impact score in [0, 1] (higher = more important players out)."""
    data = _load()
    rec = data.get(team_code.upper())
    if not rec:
        return 0.0
    try:
        return max(0.0, min(1.0, float(rec.get("impact", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def xg_multiplier(team_code: str) -> float:
    impact = impact_for(team_code)
    return 1.0 - 0.15 * impact


def missing_players(team_code: str) -> list[dict]:
    rec = _load().get(team_code.upper())
    if not rec:
        return []
    return list(rec.get("players", []))
