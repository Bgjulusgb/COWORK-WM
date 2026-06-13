"""Tests for the self-contained HTML report (wm2026.report_html)."""
from __future__ import annotations

import asyncio

from wm2026.context import synth_config
from wm2026.pipeline import run_prediction
from wm2026.report import build_report
from wm2026.report_html import build_html


def _result():
    cfg = synth_config(home_team="France", away_team="Senegal", odds_1x2="1.70/3.60/4.50")
    return asyncio.run(run_prediction(cfg, mode="mock", bootstrap_n=32,
                                      odds_1x2=[1.70, 3.60, 4.50]))


def test_build_html_self_contained_with_content():
    result = _result()
    js = build_report(result)["json"]
    doc = build_html(result, js)
    assert "<html" in doc and doc.strip().endswith("</html>")
    assert "France" in doc and "Senegal" in doc
    assert "Edge Table" in doc
    assert "Derived markets" in doc
    assert "HT" in doc and "FT" in doc          # the new HT/FT grid renders
    # self-contained: no external http(s) asset references
    assert "http://" not in doc and "https://" not in doc


def test_build_html_without_matplotlib_renders(monkeypatch):
    result = _result()
    js = build_report(result)["json"]
    import wm2026.viz as viz
    monkeypatch.setattr(viz, "chart_b64", lambda r: {"tornado": None, "heatmap": None})
    doc = build_html(result, js)
    assert "<html" in doc
    assert "data:image/png" not in doc          # no embedded charts, still valid HTML
    assert "not betting advice" in doc


def test_build_html_shows_cowork_when_gaps():
    # mock + no real odds (the mock-sourced odds are not adopted) → an odds
    # Cowork task fires → the HTML Cowork block appears.
    cfg = synth_config(home_team="Spain", away_team="Japan")
    result = asyncio.run(run_prediction(cfg, mode="mock", bootstrap_n=0))
    js = build_report(result)["json"]
    doc = build_html(result, js)
    assert "Cowork-Auftrag" in doc
