"""Sync WM 2026 match data from worldcup26.ir API into wm2026_data.json.

Usage:
    cd backend && python -m scripts.sync_from_api
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.wc2026_api import fetch_games, fetch_groups, fetch_teams, fetch_stadiums


async def main():
    print("Fetching data from worldcup26.ir ...")
    games, groups, teams, stadiums = await asyncio.gather(
        fetch_games(), fetch_groups(), fetch_teams(), fetch_stadiums()
    )

    print(f"  Games:    {len(games)}")
    print(f"  Groups:   {len(groups)}")
    print(f"  Teams:    {len(teams)}")
    print(f"  Stadiums: {len(stadiums)}")

    out = {
        "source": "worldcup26.ir",
        "games": games,
        "groups": groups,
        "teams": teams,
        "stadiums": stadiums,
    }

    out_path = Path(__file__).resolve().parent / "wc2026_api_data.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
