"""Guard: the committed example report under docs/examples/ stays current.

Trips when the JSON schema is bumped without regenerating the example (so the
docs never silently go stale). Regenerate with:

    python -m wm2026.cli predict --mode mock \
      --match config/matches/group_a/cze_vs_rsa.yaml --odds "2.10/3.40/3.20" \
      --calibrate market --format html --charts --out docs/examples
"""
from __future__ import annotations

import json
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


def test_example_json_parses_and_is_current():
    d = json.loads((EXAMPLES / "example.json").read_text(encoding="utf-8"))
    assert d["schema_version"] == "1.3"          # bump → regenerate the example
    for key in ("fixture", "markets", "derived_markets", "edge_table",
                "calibration", "claude_tasks",
                "best_value", "best_value_cons", "bankroll"):
        assert key in d, f"example.json missing {key}"
    # the new Phase-1 markets must be present in the committed example
    for mk in ("ht_ft", "first_goal", "winning_margin", "exact_total_goals"):
        assert mk in d["derived_markets"]


def test_example_artifacts_exist():
    for name in ("example_report.md", "example_report.html", "example.json",
                 "example_tornado.png", "example_heatmap.png", "tournament.md",
                 "example.summary.md"):
        assert (EXAMPLES / name).exists(), f"missing docs/examples/{name}"
    # the HTML report is self-contained (embeds the charts, no external assets)
    html = (EXAMPLES / "example_report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html


def test_example_summary_renders_required_blocks():
    s = (EXAMPLES / "example.summary.md").read_text(encoding="utf-8")
    assert "## λ + CI" in s
    assert "## Recommendation" in s
    # Token budget contract: the on-disk summary stays small.
    assert len(s) < 2500, f"example summary grew to {len(s)} chars"
