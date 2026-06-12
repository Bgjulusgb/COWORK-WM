"""Phase 8F — a self-contained HTML report.

``build_html(result, js)`` returns a single portable HTML document (inline CSS,
no external assets) mirroring the Markdown report: a hero header, executive
summary cards, Claude's Cowork assignment, the edge table (incl. conservative
p5 columns), the full derived-markets board, the per-model blend, embedded
charts (base64 PNGs via :mod:`wm2026.viz`, ASCII fallback when matplotlib is
absent), and a data-provenance strip. Pure string assembly — the CLI writes the
file. Degrades to core deps only.
"""
from __future__ import annotations

import html
from typing import Any

from wm2026 import MODEL_VERSION


def _pct(p: float | None) -> str:
    return "—" if p is None else f"{100.0 * p:.1f}%"


def _num(x: Any) -> str:
    return "—" if x is None else str(x)


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else "—"))


_CSS = """
:root{--bg:#0f1419;--card:#1b232d;--ink:#e6edf3;--mut:#9aa7b4;--line:#2b3744;
--home:#3fb950;--away:#f85149;--accent:#58a6ff;--warn:#d29922;--ok:#2ea043}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 60px}
.hero{background:linear-gradient(135deg,#1b232d,#161b22);border:1px solid var(--line);
border-radius:16px;padding:24px 26px;margin-bottom:20px}
.hero h1{margin:0 0 6px;font-size:26px}.hero .meta{color:var(--mut);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.big{font-size:22px;font-weight:700}
.bar{height:10px;border-radius:6px;background:#0d1117;overflow:hidden;display:flex;margin-top:8px}
.bar i{display:block;height:100%}.bar .h{background:var(--home)}.bar .d{background:#6e7681}.bar .a{background:var(--away)}
h2{font-size:17px;margin:26px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 4px}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}thead th{color:var(--mut);font-weight:600}
tr:hover td{background:#222c38}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600}
.pill.s{background:#15301c;color:var(--ok)}.pill.w{background:#3a2d10;color:var(--warn)}
.pos{color:var(--home)}.neg{color:var(--away)}
.cowork{background:#15203a;border:1px solid #284070;border-radius:12px;padding:14px 16px;margin:14px 0}
.cowork li{margin:6px 0}.cowork code{background:#0d1117;padding:1px 5px;border-radius:5px}
img.chart{max-width:100%;border:1px solid var(--line);border-radius:10px;background:#fff;margin:8px 0}
pre{background:#0d1117;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;font-size:12px}
.foot{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
"""


def _edge_table(js: dict[str, Any]) -> str:
    rows = []
    for r in js.get("edge_table", []):
        edge = r.get("edge_pct")
        cls = "pos" if (edge is not None and edge > 0) else ("neg" if edge is not None else "")
        action = r.get("action", "")
        pill = "s" if action in ("standard", "small") else ("w" if action == "sanity-check" else "")
        rows.append(
            f"<tr><td>{_esc(r['market'])}</td><td>{_esc(r['selection'])}</td>"
            f"<td>{_pct(r['model_p'])}</td><td>{_pct(r.get('fair_p'))}</td>"
            f"<td>{_num(r.get('decimal_odd'))}</td>"
            f"<td class='{cls}'>{_num(edge)}</td><td>{_num(r.get('edge_pct_cons'))}</td>"
            f"<td>{_num(r.get('half_kelly_pct'))}</td><td>{_num(r.get('half_kelly_cons'))}</td>"
            f"<td><span class='pill {pill}'>{_esc(action)}</span></td></tr>"
        )
    if not rows:
        return ""
    head = ("<tr><th>Market</th><th>Selection</th><th>Model P</th><th>Fair P</th>"
            "<th>Odd</th><th>Edge %</th><th>Edge% p5</th><th>½K %</th><th>½K p5</th><th>Action</th></tr>")
    return f"<h2>Edge Table</h2><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _kv_table(title: str, pairs: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>" for k, v in pairs)
    return f"<h3>{_esc(title)}</h3><table><tbody>{body}</tbody></table>"


def _derived(js: dict[str, Any]) -> str:
    dm = js.get("derived_markets") or {}
    if not dm:
        return ""
    out = ["<h2>Derived markets</h2><div class='grid'>"]
    dc, dnb = dm.get("double_chance", {}), dm.get("draw_no_bet", {})
    if dc:
        out.append("<div class='card'>" + _kv_table("Double Chance / DNB", [
            ("1X", _pct(dc.get("1X"))), ("12", _pct(dc.get("12"))), ("X2", _pct(dc.get("X2"))),
            ("DNB home", _pct(dnb.get("home"))), ("DNB away", _pct(dnb.get("away"))),
        ]) + "</div>")
    wm = dm.get("winning_margin", {})
    if wm:
        out.append("<div class='card'>" + _kv_table("Winning margin", [
            ("Home +1", _pct(wm.get("home_by_1"))), ("Home +2", _pct(wm.get("home_by_2plus"))),
            ("Draw", _pct(wm.get("draw"))),
            ("Away +1", _pct(wm.get("away_by_1"))), ("Away +2", _pct(wm.get("away_by_2plus"))),
        ]) + "</div>")
    fg, oe, cs, wtn = (dm.get("first_goal", {}), dm.get("odd_even", {}),
                       dm.get("clean_sheet", {}), dm.get("win_to_nil", {}))
    out.append("<div class='card'>" + _kv_table("Goals & sheets", [
        ("First goal H/A/none", f"{_pct(fg.get('home'))} / {_pct(fg.get('away'))} / {_pct(fg.get('none'))}"),
        ("Goals odd/even", f"{_pct(oe.get('odd'))} / {_pct(oe.get('even'))}"),
        ("Clean sheet H/A", f"{_pct(cs.get('home'))} / {_pct(cs.get('away'))}"),
        ("Win-to-nil H/A", f"{_pct(wtn.get('home'))} / {_pct(wtn.get('away'))}"),
    ]) + "</div>")
    out.append("</div>")

    ah = [r for r in dm.get("asian_handicap", [])
          if r.get("line") in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)]
    if ah:
        rows = "".join(
            f"<tr><td>{('+' if r['line']>=0 else '')}{r['line']:g}</td>"
            f"<td>{_pct(r.get('home_prob_nopush'))}</td><td>{_pct(r.get('push'))}</td>"
            f"<td>{_pct(r.get('away_prob_nopush'))}</td></tr>" for r in ah)
        out.append("<h3>Asian handicap (no-push prob)</h3>"
                   f"<table><thead><tr><th>Line</th><th>Home</th><th>Push</th><th>Away</th></tr></thead>"
                   f"<tbody>{rows}</tbody></table>")
    htft = dm.get("ht_ft", {})
    if htft:
        labels = {"H": "Home", "D": "Draw", "A": "Away"}
        body = "".join(
            f"<tr><td>{labels[ht]}</td>" +
            "".join(f"<td>{_pct(htft.get(f'{ht}/{ft}'))}</td>" for ft in "HDA") + "</tr>"
            for ht in "HDA")
        out.append("<h3>HT / FT (row = halftime, col = full-time)</h3>"
                   "<table><thead><tr><th>HT＼FT</th><th>Home</th><th>Draw</th><th>Away</th></tr></thead>"
                   f"<tbody>{body}</tbody></table>")
    return "".join(out)


def _cowork(result: dict[str, Any]) -> str:
    tasks = result.get("claude_tasks") or []
    if not tasks:
        return ""
    items = "".join(
        f"<li><b>[{_esc(t['priority'])}]</b> {_esc(t['task'])}<br>"
        f"<small>→ einspeisen via: <code>{_esc(t['fill_via'])}</code></small></li>"
        for t in tasks)
    return ("<div class='cowork'><h2 style='border:none;margin-top:0'>🤝 Cowork-Auftrag (live data gaps)</h2>"
            "<p>Recherchiere diese Werte (Web Search) und füttere sie via "
            "<code>--overrides-json</code> / <code>--odds*</code> zurück.</p>"
            f"<ul>{items}</ul></div>")


def _charts(result: dict[str, Any], home: str, away: str,
            *, external_prefix: str | None = None) -> str:
    """Render the charts block.

    With ``external_prefix`` (e.g. ``"wm2026_groupa_kor_vs_cze"``) the HTML
    references on-disk PNG siblings (``<prefix>_tornado.png`` /
    ``<prefix>_heatmap.png``) instead of inlining the base64 — drops the
    HTML payload from ~95 KB to ~10 KB at the price of two extra files.
    """
    blocks = []
    if external_prefix:
        for name, title in (("tornado", "Factor Tornado"),
                            ("heatmap", "Score Probability Matrix")):
            blocks.append(
                f"<h3>{title}</h3>"
                f"<img class='chart' alt='{title}' loading='lazy' "
                f"src='{external_prefix}_{name}.png'>")
        return "<h2>Charts</h2>" + "".join(blocks)
    try:
        from wm2026.viz import chart_b64
        imgs = chart_b64(result)
    except Exception:
        imgs = {"tornado": None, "heatmap": None}
    for name, title in (("tornado", "Factor Tornado"),
                        ("heatmap", "Score Probability Matrix")):
        b64 = imgs.get(name)
        if b64:
            blocks.append(f"<h3>{title}</h3><img class='chart' alt='{title}' "
                          f"src='data:image/png;base64,{b64}'>")
    if blocks:
        return "<h2>Charts</h2>" + "".join(blocks)
    return ""   # ASCII fallback already in the markdown view; keep HTML lean


def build_html(result: dict[str, Any], js: dict[str, Any], *,
               external_charts_prefix: str | None = None) -> str:
    fx = js.get("fixture", {})
    home = _esc(fx.get("home") or "Home")
    away = _esc(fx.get("away") or "Away")
    p = js["markets"]["1x2"]
    out = result["prediction"]
    conf = js.get("ensemble_confidence", 0.0)
    bv = js.get("best_value")
    cal = js.get("calibration", {})

    parts = [f"<!doctype html><html lang='de'><head><meta charset='utf-8'>",
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>",
             f"<title>WM 2026 — {home} vs {away}</title><style>{_CSS}</style></head><body><div class='wrap'>"]

    meta = " · ".join(_esc(x) for x in [fx.get("stage"), fx.get("kickoff_utc"), fx.get("venue")] if x)
    parts.append(
        f"<div class='hero'><h1>🏆 {home} <span style='color:#6e7681'>vs</span> {away}</h1>"
        f"<div class='meta'>{meta}{' · ' if meta else ''}mode <b>{_esc(js.get('mode'))}</b> · "
        f"model {_esc(js.get('model_version', MODEL_VERSION))} · {_esc(js.get('predicted_at'))}</div></div>")

    # summary cards
    parts.append("<div class='grid'>")
    parts.append(
        f"<div class='card'><h3>1X2</h3>"
        f"<div class='big'>{home} {_pct(p['home'])}</div>"
        f"<div class='bar'><i class='h' style='width:{100*p['home']:.0f}%'></i>"
        f"<i class='d' style='width:{100*p['draw']:.0f}%'></i>"
        f"<i class='a' style='width:{100*p['away']:.0f}%'></i></div>"
        f"<div class='meta'>Draw {_pct(p['draw'])} · {away} {_pct(p['away'])}</div></div>")
    parts.append(
        f"<div class='card'><h3>Expected goals (λ)</h3>"
        f"<div class='big'>{out.home_xg:.2f} – {out.away_xg:.2f}</div>"
        f"<div class='meta'>O2.5 {_pct(out.over_25)} · BTTS {_pct(out.btts)}</div></div>")
    gauge = "🟢" if conf >= 0.66 else ("🟡" if conf >= 0.5 else "🔴")
    parts.append(
        f"<div class='card'><h3>Confidence</h3><div class='big'>{gauge} {conf:.2f}</div>"
        f"<div class='meta'>{js.get('factors_used')}/{js.get('factors_total')} factors live</div></div>")
    if bv:
        parts.append(
            f"<div class='card'><h3>Value pick</h3>"
            f"<div class='big'>{_esc(bv['selection'])} @ {_esc(bv['decimal_odd'])}</div>"
            f"<div class='meta'>{_esc(bv['market'])} · edge {bv['edge_pct']}% · ½K {bv['half_kelly_pct']}%</div></div>")
    if cal.get("applied") and cal.get("calibrated"):
        c = cal["calibrated"]
        parts.append(
            f"<div class='card'><h3>Calibration ({_esc(cal.get('method'))})</h3>"
            f"<div class='meta'>{home} {_pct(c.get('home_win'))} · Draw {_pct(c.get('draw'))} · "
            f"{away} {_pct(c.get('away_win'))}</div></div>")
    parts.append("</div>")

    parts.append(_cowork(result))
    parts.append(_charts(result, home, away,
                         external_prefix=external_charts_prefix))
    parts.append(_edge_table(js))
    parts.append(_derived(js))

    # per-model blend
    pm = js.get("per_model", {})
    if pm:
        rows = "".join(
            f"<tr><td>{_esc(name)}</td><td>{_pct(mk.get('home_win'))}</td><td>{_pct(mk.get('draw'))}</td>"
            f"<td>{_pct(mk.get('away_win'))}</td><td>{_pct(mk.get('over_25'))}</td><td>{_pct(mk.get('btts'))}</td></tr>"
            for name, mk in pm.items())
        parts.append("<h2>Goal-model blend</h2><table><thead><tr><th>Model</th><th>Home</th>"
                     "<th>Draw</th><th>Away</th><th>O2.5</th><th>BTTS</th></tr></thead>"
                     f"<tbody>{rows}</tbody></table>")

    prov = js.get("data_sources", {})
    modes: dict[str, int] = {}
    for v in prov.values():
        if isinstance(v, dict):
            modes[v.get("mode", "?")] = modes.get(v.get("mode", "?"), 0) + 1
    mode_str = " · ".join(f"{m} ×{n}" for m, n in sorted(modes.items()))
    parts.append(
        f"<div class='foot'>Data sources: {mode_str or '—'}<br>"
        "Generated by the WM 2026 workflow — <b>not betting advice</b>. "
        "Mock mode is illustrative.</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


__all__ = ["build_html"]
