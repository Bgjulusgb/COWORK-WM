"""Patch the form_last5 / world_ranking placeholders in every match YAML.

Reads scripts/team_real_data.py and applies the real values in-place. Safe
to re-run — only the two fields are touched, other content (subreddit lists,
slang dicts, prediction config) is preserved.

Usage:
    cd backend && python scripts/update_yaml_team_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.team_real_data import get_form, get_world_ranking  # noqa: E402


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or "teams" not in data:
        return False

    changed = False
    for side in ("home", "away"):
        team = data["teams"].get(side)
        if not isinstance(team, dict):
            continue
        code = team.get("code")
        if not code:
            continue
        new_form = get_form(code)
        new_rank = get_world_ranking(code)
        if team.get("form_last5") != new_form:
            team["form_last5"] = new_form
            changed = True
        if team.get("world_ranking") != new_rank:
            team["world_ranking"] = new_rank
            changed = True

    if changed:
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return changed


def main() -> None:
    root = ROOT / "config" / "matches"
    files = sorted(root.rglob("*.yaml"))
    touched = 0
    for f in files:
        if patch_file(f):
            touched += 1
    print(f"Patched {touched} of {len(files)} match YAMLs.")


if __name__ == "__main__":
    main()
