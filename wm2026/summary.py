"""Token-sparsame Briefing-Synthese aus dem ``wm2026`` JSON-Report.

Der ganze Report ist ~4 k Tokens — fuer eine Cowork-Sitzung, in der eine Skill
die Headline zusammenfasst, reichen meist die **Top-Picks + Lambdas + Edges + p5**.
``summarise()`` extrahiert genau das. Pure Python; liest nur das schema_version-
1.3-JSON, kein Pipeline-Import noetig.

Designziele:

* **deterministisch** — selbes JSON ⇒ exakt selber Text (kein Floating-Format
  driftet zwischen Runs).
* **<500 Tokens** — Caller (`wm2026 summary` CLI, `read-report` Skill, der
  `wm-quant-analyst`-Agent) koennen das ohne Token-Budget-Sorge lesen.
* **degradiert sauber** — fehlende Felder fallen in "—" zurueck, kein KeyError.
"""
from __future__ import annotations

from typing import Any


def _pct(x: Any) -> str:
    if not isinstance(x, (int, float)):
        return "  —  "
    return f"{100.0 * x:5.1f}%"


def _signed(x: Any, suffix: str = "%") -> str:
    if not isinstance(x, (int, float)):
        return "   —  "
    return f"{x:+6.2f}{suffix}"


def _lam(d: dict[str, Any] | None) -> str:
    """Format a lambda triple (``{p5,p50,p95}``) or a bare float."""
    if isinstance(d, dict):
        p5, p50, p95 = d.get("p5"), d.get("p50"), d.get("p95")
        if all(isinstance(v, (int, float)) for v in (p5, p50, p95)):
            return f"{p50:.2f} [p5 {p5:.2f} / p95 {p95:.2f}]"
        if isinstance(p50, (int, float)):
            return f"{p50:.2f}"
    if isinstance(d, (int, float)):
        return f"{d:.2f}"
    return "—"


def _ci_triple(ci: dict[str, Any] | None, key: str) -> str:
    if not isinstance(ci, dict):
        return "—"
    triple = ci.get(key)
    if not isinstance(triple, (list, tuple)) or len(triple) < 3:
        return "—"
    p5, p50, p95 = triple[0], triple[1], triple[2]
    return f"{100*p50:5.1f}% [p5 {100*p5:5.1f}% / p95 {100*p95:5.1f}%]"


def _gauge(conf: float | None) -> str:
    if not isinstance(conf, (int, float)):
        return "—"
    if conf >= 0.66:
        return f"🟢 {conf:.2f}"
    if conf >= 0.5:
        return f"🟡 {conf:.2f}"
    return f"🔴 {conf:.2f}"


def _pick_1x2(js: dict[str, Any]) -> tuple[float, float, float]:
    """Calibrated 1X2 if Phase-5 ran, else raw markets."""
    cal = (js.get("calibration") or {}).get("calibrated") or {}
    if cal:
        return (float(cal.get("home_win", 0.0)),
                float(cal.get("draw", 0.0)),
                float(cal.get("away_win", 0.0)))
    m = (js.get("markets") or {}).get("1x2") or {}
    return (float(m.get("home", 0.0)),
            float(m.get("draw", 0.0)),
            float(m.get("away", 0.0)))


def summarise(js: dict[str, Any], *, top_edges: int = 5) -> str:
    """One-page betting briefing for the JSON report.

    ``top_edges`` caps the edge table (sorted by conservative p5 edge desc).
    """
    fx = js.get("fixture") or {}
    home = fx.get("home") or "Home"
    away = fx.get("away") or "Away"
    stage = fx.get("stage") or "—"
    kick = fx.get("kickoff_utc") or fx.get("kickoff") or "—"
    mode = js.get("mode", "?")

    L: list[str] = []
    L.append(f"# 🎯 {home} vs {away} · {stage} · {kick}")
    L.append(f"mode `{mode}` · conf {_gauge(js.get('ensemble_confidence'))} · "
             f"factors {js.get('factors_used', '?')}/{js.get('factors_total', '?')}")
    L.append("")

    # Lambdas + Bootstrap-CIs
    L.append("## λ + CI (blended bootstrap)")
    L.append(f"- **{home}**: λ = {_lam(js.get('lambda_home'))}")
    L.append(f"- **{away}**: λ = {_lam(js.get('lambda_away'))}")

    h, d, a = _pick_1x2(js)
    label = "calibrated" if (js.get("calibration") or {}).get("calibrated") else "raw"
    L.append(f"- **1X2 ({label})**: H {_pct(h)} · D {_pct(d)} · A {_pct(a)}")
    ci = ((js.get("confidence_intervals") or {}).get("blended")) or {}
    if ci:
        L.append(f"  - CI home_win: {_ci_triple(ci, 'home_win')}")
        L.append(f"  - CI draw:     {_ci_triple(ci, 'draw')}")
        L.append(f"  - CI away_win: {_ci_triple(ci, 'away_win')}")

    mk = js.get("markets") or {}
    ou = (mk.get("over_under") or {})
    # The headline over_under block has both shapes in the wild — the
    # compact path (just {over, under}) and the full report's
    # {over_05, over_15, over_25, over_35}. Handle both.
    o25 = ou.get("over") if isinstance(ou.get("over"), (int, float)) else ou.get("over_25")
    u25 = (1.0 - o25) if isinstance(o25, (int, float)) else ou.get("under")
    L.append(f"- **O/U 2.5**: Over {_pct(o25)} · "
             f"Under {_pct(u25)}  ·  CI O2.5: {_ci_triple(ci, 'over_25')}")
    btts = mk.get("btts") or {}
    L.append(f"- **BTTS**:    Yes  {_pct(btts.get('yes'))} · "
             f"No    {_pct(btts.get('no'))}  ·  CI BTTS: {_ci_triple(ci, 'btts')}")
    L.append("")

    # Edge-Tabelle (sortiert nach konservativer p5-Edge)
    rows = list(js.get("edge_table") or [])
    priced = [r for r in rows if r.get("decimal_odd") is not None]
    if priced:
        priced.sort(key=lambda r: (r.get("edge_pct_cons") if r.get("edge_pct_cons") is not None else -1e9),
                    reverse=True)
        priced = priced[: max(1, top_edges)]
        L.append(f"## Edges (top {len(priced)} by p5)")
        L.append("```")
        L.append(f"{'market':<14}{'sel':<10}{'odd':>6}{'edge%':>9}{'p5%':>9}{'½K':>6}{'p5K':>6}  action")
        for r in priced:
            stake_cons = r.get("stake_cons")
            stake = f"  stake_p5={stake_cons}" if isinstance(stake_cons, (int, float)) else ""
            L.append(f"{(r.get('market') or '')[:14]:<14}"
                     f"{(r.get('selection') or '')[:10]:<10}"
                     f"{(r.get('decimal_odd') or 0):>6.2f}"
                     f"{_signed(r.get('edge_pct'))}{_signed(r.get('edge_pct_cons'))}"
                     f"{_signed(r.get('half_kelly_pct'), suffix='')}"
                     f"{_signed(r.get('half_kelly_cons'), suffix='')}"
                     f"  {r.get('action', '—')}{stake}")
        L.append("```")
        L.append("")

    # Ehrliche Empfehlung
    bvc = js.get("best_value_cons")
    bv = js.get("best_value")
    L.append("## Recommendation")
    if bvc:
        bankroll = js.get("bankroll")
        stake = (f" · stake **{bvc.get('stake_cons', '–')}** of {bankroll}"
                 if bankroll and bvc.get("stake_cons") is not None else "")
        L.append(f"- ✅ **{bvc.get('market')} · {bvc.get('selection')}** @ "
                 f"{bvc.get('decimal_odd')} — p5-edge **{bvc.get('edge_pct_cons')}%**, "
                 f"½-Kelly(p5) {bvc.get('half_kelly_cons')}%{stake}")
    else:
        L.append("- ❌ **Pass** — no edge survives the bootstrap lower bound (p5).")
    if bv and (not bvc or (bv.get("selection") != bvc.get("selection"))):
        L.append(f"- ⚠️ Sanity-check candidate (raw edge): {bv.get('market')} · "
                 f"{bv.get('selection')} @ {bv.get('decimal_odd')} — "
                 f"edge **{bv.get('edge_pct')}%**, p5 **{bv.get('edge_pct_cons')}%** "
                 f"({bv.get('action')})")
    L.append("")

    # Cowork-Status
    tasks = list(js.get("claude_tasks") or [])
    warnings = list(js.get("warnings") or [])
    if tasks:
        L.append(f"## 🤝 Cowork open ({len(tasks)} tasks)")
        for t in tasks[:5]:
            L.append(f"- [{t.get('priority')}] {t.get('task')}  → "
                     f"`{t.get('fill_via')}`")
        if len(tasks) > 5:
            L.append(f"  …and {len(tasks)-5} more.")
        L.append("")
    if warnings:
        L.append("## ⚠️ Warnings")
        for w in warnings[:5]:
            L.append(f"- {w}")
        L.append("")

    L.append("> Forschung/Bildung — **keine Wett-Empfehlung**. "
             "Mock = illustrativ. ½-Kelly auf p5, niemals > 2 % Bankroll.")
    return "\n".join(L)


__all__ = ["summarise"]
