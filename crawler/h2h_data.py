"""EXTEND-03: head-to-head record over the last N meetings.

Backed by a static snapshot (`h2h_snapshot.json`) so the system works offline.
When `football-data.org` credentials are configured a live lookup can replace
this lookup transparently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger("crawler.h2h_data")

_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "h2h_snapshot.json"
_DATA: Optional[dict] = None


def _load() -> dict:
    global _DATA
    if _DATA is not None:
        return _DATA
    if not _SNAPSHOT_PATH.exists():
        _DATA = {}
        return _DATA
    try:
        _DATA = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("h2h_snapshot_load_failed", error=str(exc))
        _DATA = {}
    return _DATA


def lookup(home_code: str, away_code: str) -> Optional[dict]:
    """Return {'home_wins': int, 'draws': int, 'away_wins': int, 'avg_goals': float}
    or None when no record is available."""
    data = _load()
    key = f"{home_code.upper()}_{away_code.upper()}"
    rec = data.get(key)
    if rec is not None:
        return {
            "home_wins": int(rec.get("home_wins", 0)),
            "draws": int(rec.get("draws", 0)),
            "away_wins": int(rec.get("away_wins", 0)),
            "avg_goals": float(rec.get("avg_goals", 0.0)),
        }
    # Reverse lookup with flipped home/away.
    rev = data.get(f"{away_code.upper()}_{home_code.upper()}")
    if rev is None:
        return None
    return {
        "home_wins": int(rev.get("away_wins", 0)),
        "draws": int(rev.get("draws", 0)),
        "away_wins": int(rev.get("home_wins", 0)),
        "avg_goals": float(rev.get("avg_goals", 0.0)),
    }
