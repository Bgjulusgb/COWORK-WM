"""Recompute team Elo ratings from match history and (optionally) write them
back into every match YAML's `teams.*.elo_rating`.

Mechanism: seed from the ratings already in the YAMLs, fold the openfootball
history chronologically through the World-Football-Elo K-update
(factors.elo_update.recompute_from_history), then patch the YAMLs.

Safe by default: prints the deltas (dry-run). Pass --apply to write. The data
source respects USE_MOCK_OPENFOOTBALL — in mock mode it uses the deterministic
recent (2026) mock history, which is a better illustration than the live
2018/2022-only feed. Re-runnable; only elo_rating is touched.

Usage:
    cd backend && python scripts/refresh_elo.py            # dry-run
    cd backend && python scripts/refresh_elo.py --apply     # write YAMLs
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_sources.openfootball import OpenfootballConnector  # noqa: E402
from data_sources.team_codes import CODE_TO_NAMES  # noqa: E402
from factors.elo_update import recompute_from_history  # noqa: E402

MATCHES_DIR = ROOT / "config" / "matches"


def _seed_ratings() -> dict[str, float]:
    """Current elo_rating per code, read from the YAMLs (first occurrence)."""
    ratings: dict[str, float] = {}
    for path in MATCHES_DIR.rglob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for side in ("home", "away"):
            team = (data.get("teams") or {}).get(side) or {}
            code, elo = team.get("code"), team.get("elo_rating")
            if code and isinstance(elo, (int, float)) and code not in ratings:
                ratings[code] = float(elo)
    return ratings


async def _collect_history() -> list:
    """Deduplicated, oldest-first match history across all 48 teams."""
    conn = OpenfootballConnector()
    try:
        results = await asyncio.gather(
            *(conn.get_historical_results(code) for code in CODE_TO_NAMES)
        )
    finally:
        from data_sources.base import BaseConnector
        await BaseConnector.close_all()

    seen: set[tuple] = set()
    matches: list = []
    for res in results:
        for m in res.data or []:
            key = (m.kickoff_utc.date(), m.home_code, m.away_code, m.home_score, m.away_score)
            if key in seen:
                continue
            seen.add(key)
            matches.append(m)
    matches.sort(key=lambda m: m.kickoff_utc)  # oldest first for the fold
    return matches


def _patch_yaml(path: Path, updated: dict[str, float]) -> bool:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "teams" not in data:
        return False
    changed = False
    for side in ("home", "away"):
        team = data["teams"].get(side)
        if not isinstance(team, dict):
            continue
        code = team.get("code")
        if code in updated:
            new_elo = round(updated[code])
            if team.get("elo_rating") != new_elo:
                team["elo_rating"] = new_elo
                changed = True
    if changed:
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return changed


async def main() -> None:
    apply = "--apply" in sys.argv
    seed = _seed_ratings()
    history = await _collect_history()
    updated = recompute_from_history(seed, history)

    deltas = sorted(
        ((c, seed.get(c, 0.0), updated[c]) for c in updated),
        key=lambda t: abs(t[2] - t[1]), reverse=True,
    )
    print(f"History matches folded: {len(history)} | teams: {len(updated)}")
    print(f"{'CODE':<5}{'old':>8}{'new':>8}{'delta':>8}")
    for code, old, new in deltas[:20]:
        print(f"{code:<5}{old:>8.0f}{round(new):>8}{new - old:>+8.0f}")

    if not apply:
        print("\n(dry-run) — re-run with --apply to write the YAMLs.")
        return
    touched = sum(_patch_yaml(p, updated) for p in sorted(MATCHES_DIR.rglob("*.yaml")))
    print(f"\nApplied: patched {touched} match YAMLs.")


if __name__ == "__main__":
    asyncio.run(main())
