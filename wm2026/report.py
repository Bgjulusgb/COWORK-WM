"""Phase 8 — Output. Turn the raw ``result`` dict into a structured JSON object
and a human-readable Markdown report.

``build_report(result)`` returns ``{"json": <dict>, "markdown": <str>}``:

* **json**     — copy-paste-able, DB-ready, follows the master prompt's schema.
* **markdown** — executive summary, factor tornado, score heatmap, edge table.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from wm2026 import MODEL_VERSION


# ── small formatting helpers ──────────────────────────────────────────────────
def _pct(p: float | None) -> str:
    return "—" if p is None else f"{100.0 * p:5.1f}%"


def _iso(dt: Any) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _score_label(item: dict[str, Any]) -> str:
    return f"{int(item.get('home', 0))}-{int(item.get('away', 0))}"


def _confidence_gauge(conf: float, warnings: list[str]) -> str:
    if any("mock" in w for w in warnings):
        return "🟡 (mock data — illustrative)"
    if conf >= 0.66:
        return "🟢 high"
    if conf >= 0.5:
        return "🟡 medium"
    return "🔴 low"


# ── JSON assembly (Phase 8A) ──────────────────────────────────────────────────
def build_json(result: dict[str, Any]) -> dict[str, Any]:
    out = result["prediction"]
    ensemble = result["ensemble"]
    cfg = result.get("config", {})
    match = cfg.get("match", {})
    teams = cfg.get("teams", {})
    feats = out.features or {}

    signals_payload = []
    eff = {
        s["name"]: s.get("effective_weight", 0.0)
        for s in ensemble.breakdown_payload.get("signals", [])
    }
    for s in result["signals"]:
        signals_payload.append({
            "name": s.name,
            "home_strength": round(s.home_strength, 4),
            "away_strength": round(s.away_strength, 4),
            "weight": round(s.weight, 4),
            "effective_weight": round(eff.get(s.name, 0.0), 4),
            "confidence": round(s.confidence, 3),
            "available": s.available,
            "kind": s.kind,
            "source": s.source,
        })

    n_used = sum(1 for s in result["signals"] if s.available)

    over = {
        "over_05": round(out.features.get("per_model", {})
                   .get(feats.get("goal_model_primary", "poisson"), {})
                   .get("over_05", 0.0), 4) if feats.get("per_model") else None,
        "over_15": round(out.over_15, 4),
        "over_25": round(out.over_25, 4),
        "over_35": round(out.over_35, 4),
    }

    top5 = [
        {"score": _score_label(t), "p": round(float(t.get("probability", 0.0)), 4)}
        for t in (out.top_scores or [])
    ]

    return {
        "schema_version": "1.3",
        "match_id": match.get("id") or out.__dict__.get("match_id", "wm2026_match"),
        "model_version": MODEL_VERSION,
        "predicted_at": _iso(result.get("started_at")),
        "mode": result.get("mode", "mock"),
        "fixture": {
            "home": (teams.get("home", {}) or {}).get("name"),
            "away": (teams.get("away", {}) or {}).get("name"),
            "stage": match.get("phase") or match.get("stage"),
            "kickoff_utc": match.get("kickoff_utc"),
            "venue": match.get("venue"),
        },
        "lambda_home": result["lambda_home_ci"],
        "lambda_away": result["lambda_away_ci"],
        "xg": {"home": round(out.home_xg, 3), "away": round(out.away_xg, 3)},
        "markets": {
            "1x2": {
                "home": round(out.home_win_prob, 4),
                "draw": round(out.draw_prob, 4),
                "away": round(out.away_win_prob, 4),
            },
            "over_under": over,
            "btts": {"yes": round(out.btts, 4), "no": round(1.0 - out.btts, 4)},
            "correct_score_top5": top5,
            "recommended_bet": out.recommended_bet,
            "bet_probability": (round(out.bet_probability, 4)
                                if out.bet_probability is not None else None),
        },
        "derived_markets": _round_derived(result.get("derived_markets", {})),
        "per_model": _round_per_model(feats.get("per_model", {})),
        "confidence_intervals": feats.get("confidence_intervals", {}),
        "ensemble_confidence": round(ensemble.confidence, 4),
        "factors_used": n_used,
        "factors_total": len(result["signals"]),
        "factors": signals_payload,
        "calibration": result.get("calibration", {}),
        "edge_table": result.get("edges", []),
        "best_value": result.get("best_value"),
        # Phase 4 (schema 1.3, additiv): der "ehrliche" Pick — höchste Edge,
        # die auch auf der konservativen Bootstrap-Untergrenze (p5) positiv
        # bleibt. None ⇒ kein Markt überlebt die p5-Disziplin (Pass).
        "best_value_cons": result.get("best_value_cons"),
        "bankroll": result.get("bankroll"),
        "warnings": result.get("warnings", []),
        "claude_tasks": result.get("claude_tasks", []),
        "data_sources": result.get("provenance", {}),
        # Aggregated provenance — how much of the data is genuinely live vs.
        # mock fallback. Lets the user see at a glance whether a `mode: live`
        # run is substantively live or mostly fail-soft-mocked.
        "data_provenance_summary": result.get("provenance_summary", {}),
    }


def _round_per_model(per_model: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulky top_scores arrays; keep the scalar markets per model."""
    out: dict[str, Any] = {}
    for name, mk in (per_model or {}).items():
        out[name] = {
            k: round(v, 4) for k, v in mk.items()
            if isinstance(v, (int, float))
        }
    return out


def _round_derived(d: Any) -> Any:
    """Recursively round the derived-markets payload for the JSON output."""
    if isinstance(d, float):
        return round(d, 4)
    if isinstance(d, dict):
        return {k: _round_derived(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_round_derived(v) for v in d]
    return d


# ── Compact mode (Token-Budget) ───────────────────────────────────────────────
_COMPACT_FACTOR_KEYS = {"name", "home_strength", "away_strength",
                        "weight", "effective_weight", "confidence",
                        "available", "kind"}
# Headline-CIs muessen erhalten bleiben; per-Modell-CIs sind kompakt-redundant.
_COMPACT_CI_PRUNE_MODELS = ("poisson", "negbin", "glm_poisson", "bivariate")
# AH-Linien fuer die Token-Schmal-Variante (5 statt 13).
_COMPACT_AH_LINES = {-1.0, -0.5, 0.0, 0.5, 1.0}


def _compress_provenance(prov: dict[str, Any]) -> dict[str, Any]:
    """Replace per-slice dicts with the bare mode token (``"live"`` / ``"mock"`` / ...).

    The full provenance can carry the entire upstream payload (FBref last-10,
    Reddit posts, weather raw JSON) — useful for debugging, not for a Cowork
    briefing. Downstream skills only consume "which slices are live, which
    are mock" — keep exactly that. Anything an agent legitimately needs more
    of, it can re-fetch from the uncompressed JSON via ``--out`` (no compact).
    """
    out: dict[str, Any] = {}
    for slice_name, raw in (prov or {}).items():
        if isinstance(raw, dict):
            out[slice_name] = raw.get("mode", "?")
        else:
            out[slice_name] = raw
    return out


def _compress_factors(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip raw_data; drop unavailable factors entirely (they carry no signal)."""
    out: list[dict[str, Any]] = []
    for f in factors or []:
        if not f.get("available"):
            continue
        out.append({k: f.get(k) for k in _COMPACT_FACTOR_KEYS if k in f})
    return out


def _compress_confidence_intervals(ci: dict[str, Any]) -> dict[str, Any]:
    """Keep the *blended* CIs (this is what edge math consumes); drop per-model."""
    if not isinstance(ci, dict):
        return ci
    blended = ci.get("blended")
    if blended is None:
        # Old runs without bootstrap_n>0 — just return as-is.
        return ci
    return {"blended": blended}


def _compress_derived(dm: dict[str, Any], *,
                      ah_lines: set[float] | None = None) -> dict[str, Any]:
    """Limit AH and exact_total_goals payloads to a small headline set."""
    if not isinstance(dm, dict):
        return dm
    out = dict(dm)
    lines = ah_lines if ah_lines else _COMPACT_AH_LINES
    ah = out.get("asian_handicap")
    if isinstance(ah, list):
        out["asian_handicap"] = [r for r in ah
                                 if isinstance(r, dict) and r.get("line") in lines]
    # exact_total_goals: keep totals 0..6, drop the long tail.
    etg = out.get("exact_total_goals")
    if isinstance(etg, dict):
        try:
            out["exact_total_goals"] = {
                k: v for k, v in etg.items()
                if isinstance(k, (str, int)) and int(str(k)) <= 6
            }
        except Exception:
            pass
    return out


def _compress_edge_table(rows: list[dict[str, Any]],
                         *, top_n: int = 12) -> list[dict[str, Any]]:
    """Sort by |edge| desc and cap; drop None-only rows (no odds)."""
    priced = [r for r in (rows or []) if r.get("decimal_odd") is not None]
    priced.sort(key=lambda r: abs(r.get("edge_pct") or 0.0), reverse=True)
    return priced[: max(1, top_n)]


def compact(js: dict[str, Any], *,
            ah_lines: set[float] | None = None,
            top_edges: int = 12) -> dict[str, Any]:
    """Return a token-budget-friendly copy of the JSON report.

    Drops debug-only blobs while keeping the *betting-relevant* shape
    (lambdas, blended CIs, headline markets, edge table, conservative pick).
    Typical size reduction: ~3 950 → ~1 700 tokens.

    Concretely: ``factors`` keeps only the available ones with eight scalar
    fields each; ``confidence_intervals`` keeps just the blended block;
    ``derived_markets`` drops the AH long tail (default keeps the 5 main
    lines); ``edge_table`` is sorted by ``|edge|`` and capped; ``data_sources``
    collapses each slice's dict to the bare mode token; ``per_model`` and
    ``correct_score_top5`` are dropped entirely. A ``"compact": true`` flag
    marks the payload so downstream skills know what to expect.
    """
    out = dict(js)
    out["factors"] = _compress_factors(js.get("factors") or [])
    out["confidence_intervals"] = _compress_confidence_intervals(
        js.get("confidence_intervals") or {})
    out["derived_markets"] = _compress_derived(
        js.get("derived_markets") or {}, ah_lines=ah_lines)
    out["edge_table"] = _compress_edge_table(
        js.get("edge_table") or [], top_n=top_edges)
    out["data_sources"] = _compress_provenance(js.get("data_sources") or {})
    out.pop("per_model", None)
    # top-5 correct scores live in markets — they're nice for the HTML, not for
    # a JSON consumer that already has the score matrix.
    if isinstance(out.get("markets"), dict):
        out["markets"] = {k: v for k, v in out["markets"].items()
                          if k != "correct_score_top5"}
    out["compact"] = True
    return out


# ── Markdown assembly (Phase 8B-E) ────────────────────────────────────────────
def build_markdown(result: dict[str, Any], js: dict[str, Any]) -> str:
    out = result["prediction"]
    ensemble = result["ensemble"]
    fx = js["fixture"]
    L: list[str] = []

    home = fx.get("home") or "Home"
    away = fx.get("away") or "Away"
    L.append(f"# 🏆 WM 2026 — {home} vs {away}")
    L.append("")
    meta = " · ".join(
        str(x) for x in [fx.get("stage"), fx.get("kickoff_utc"), fx.get("venue")] if x
    )
    if meta:
        L.append(f"*{meta}*")
    L.append(f"*Mode: `{js['mode']}` · model `{js['model_version']}` · "
             f"predicted {js['predicted_at']}*")
    L.append("")

    # B) Executive summary -----------------------------------------------------
    L.append("## Executive Summary")
    p1x2 = js["markets"]["1x2"]
    fav = max([("home", p1x2["home"]), ("draw", p1x2["draw"]), ("away", p1x2["away"])],
              key=lambda t: t[1])
    fav_label = {"home": home, "draw": "Draw", "away": away}[fav[0]]
    L.append(f"- **Most likely 1X2:** {fav_label} ({_pct(fav[1])})  ·  "
             f"{home} {_pct(p1x2['home'])} / Draw {_pct(p1x2['draw'])} / {away} {_pct(p1x2['away'])}")
    L.append(f"- **Expected goals (λ):** {home} {out.home_xg:.2f} — {away} {out.away_xg:.2f}  ·  "
             f"O2.5 {_pct(out.over_25)} · BTTS {_pct(out.btts)}")
    top3 = _top_factors(result["signals"], ensemble, 3)
    if top3:
        L.append("- **Top-3 driving factors:** " + "; ".join(top3))
    bv = js.get("best_value")
    if bv:
        L.append(f"- **Value pick:** {bv['market']} — {bv['selection']} "
                 f"@ {bv['decimal_odd']} → edge **{bv['edge_pct']}%**, "
                 f"half-Kelly {bv['half_kelly_pct']}% ({bv['action']})")
    else:
        L.append("- **Value pick:** none (no odds supplied or edge < 2%)")
    bvc = js.get("best_value_cons")
    if bvc:
        L.append(f"- **Conservative pick (p5-survivor):** {bvc['market']} — "
                 f"{bvc['selection']} @ {bvc['decimal_odd']} → p5-edge "
                 f"**{bvc['edge_pct_cons']}%**, ½-Kelly(p5) {bvc['half_kelly_cons']}%")
    elif bv:
        L.append("- **Conservative pick (p5-survivor):** none — no edge survives "
                 "the bootstrap lower bound (honest call: pass)")
    L.append(f"- **Confidence:** {_confidence_gauge(ensemble.confidence, result['warnings'])} "
             f"(ensemble {ensemble.confidence:.2f}, {js['factors_used']}/{js['factors_total']} factors live)")
    if js["calibration"].get("applied"):
        cal = js["calibration"].get("calibrated") or {}
        method = js["calibration"].get("method", "")
        if cal:
            L.append(f"- **Calibration ({method}):** {home} {_pct(cal.get('home_win'))} / "
                     f"Draw {_pct(cal.get('draw'))} / {away} {_pct(cal.get('away_win'))}")
    else:
        L.append("- **Calibration:** raw probabilities (no historical artifact — see note in JSON)")
    L.append("")

    if result["warnings"]:
        L.append("## ⚠️ Validation warnings")
        for w in result["warnings"]:
            L.append(f"- {w}")
        L.append("")

    # Claude's essential Cowork assignment — the live-data gaps to research.
    tasks = result.get("claude_tasks") or []
    if tasks:
        L.append("## 🤝 Claude — Cowork-Auftrag (live data gaps)")
        L.append("*Diese Werte konnten **nicht** automatisch geholt werden. "
                 "Recherchiere sie (Web Search), trage `(value, source, fetched_at)` "
                 "nach und füttere sie zurück — sonst bleibt die Prediction "
                 "mock-degradiert (illustrativ).*")
        for i, t in enumerate(tasks, 1):
            L.append(f"{i}. **[{t['priority']}]** {t['task']}")
            L.append(f"   → einspeisen via: `{t['fill_via']}`")
        L.append("")

    # C) Factor tornado --------------------------------------------------------
    L.append("## Factor Tornado (home ◀ favour ▶ away)")
    L.append("```")
    L.extend(_tornado(result["signals"], ensemble))
    L.append("```")
    L.append("")

    # D) Score heatmap ---------------------------------------------------------
    matrix = result.get("score_matrix") or []
    if matrix:
        L.append(f"## Score Probability Matrix (rows = {home}, cols = {away})")
        L.append("```")
        L.extend(_heatmap(matrix, home, away))
        L.append("```")
        L.append("")
    if js["markets"]["correct_score_top5"]:
        tops = "  ".join(f"{t['score']} ({_pct(t['p'])})"
                         for t in js["markets"]["correct_score_top5"])
        L.append(f"**Top-5 correct scores:** {tops}")
        L.append("")

    # E) Edge table ------------------------------------------------------------
    def _num(x: Any) -> str:
        return "—" if x is None else str(x)

    L.append("## Edge Table (Phase 6)")
    L.append("*`(p5)` columns are the conservative edge / half-Kelly on the "
             "bootstrap 5th-percentile — value that survives the model's own uncertainty.*")
    L.append("| Market | Selection | Model P | Fair P | Odd | Edge % | Edge% (p5) | ½-Kelly % | ½K (p5) | Action |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in js["edge_table"]:
        L.append(
            f"| {r['market']} | {r['selection']} | {_pct(r['model_p'])} | "
            f"{_pct(r['fair_p'])} | {_num(r['decimal_odd'])} | "
            f"{_num(r['edge_pct'])} | {_num(r.get('edge_pct_cons'))} | "
            f"{_num(r['half_kelly_pct'])} | {_num(r.get('half_kelly_cons'))} | {r['action']} |"
        )
    L.append("")

    # E2) Derived markets (Phase-1 math upgrade) -------------------------------
    L.extend(_derived_section(js.get("derived_markets", {}), home, away))

    # Per-model + data provenance ---------------------------------------------
    L.append("## Goal-model blend")
    pm = js.get("per_model", {})
    if pm:
        L.append("| Model | Home | Draw | Away | O2.5 | BTTS |")
        L.append("|---|---|---|---|---|---|")
        for name, mk in pm.items():
            L.append(f"| {name} | {_pct(mk.get('home_win'))} | {_pct(mk.get('draw'))} | "
                     f"{_pct(mk.get('away_win'))} | {_pct(mk.get('over_25'))} | {_pct(mk.get('btts'))} |")
        L.append("")
    L.append("## Data sources (provenance)")
    prov = js.get("data_sources", {})
    if prov:
        modes: dict[str, int] = {}
        for p in prov.values():
            if isinstance(p, dict):
                modes[p.get("mode", "?")] = modes.get(p.get("mode", "?"), 0) + 1
        L.append("- Mode counts: " + ", ".join(f"`{m}`×{n}" for m, n in sorted(modes.items())))
    L.append("")
    L.append("---")
    L.append("*Generated by the WM 2026 workflow — not betting advice. "
             "Mock mode is illustrative; use `--mode live` with API keys for real data.*")
    return "\n".join(L)


def _derived_section(dm: dict[str, Any], home: str, away: str) -> list[str]:
    """Markdown block for the derived markets (Double Chance, DNB, AH, totals …)."""
    if not dm:
        return []
    L: list[str] = ["## Derived markets (Phase-1 math)"]
    dc = dm.get("double_chance", {})
    dnb = dm.get("draw_no_bet", {})
    if dc or dnb:
        L.append(
            f"- **Double Chance:** 1X {_pct(dc.get('1X'))} · 12 {_pct(dc.get('12'))} "
            f"· X2 {_pct(dc.get('X2'))}   ·   **Draw-No-Bet:** "
            f"{home} {_pct(dnb.get('home'))} / {away} {_pct(dnb.get('away'))}"
        )
    cs = dm.get("clean_sheet", {})
    wtn = dm.get("win_to_nil", {})
    oe = dm.get("odd_even", {})
    if cs or wtn or oe:
        L.append(
            f"- **Clean sheet:** {home} {_pct(cs.get('home'))} / {away} {_pct(cs.get('away'))}"
            f"   ·   **Win-to-nil:** {home} {_pct(wtn.get('home'))} / {away} {_pct(wtn.get('away'))}"
            f"   ·   **Goals:** odd {_pct(oe.get('odd'))} / even {_pct(oe.get('even'))}"
        )
    L.append("")

    ah = dm.get("asian_handicap", [])
    main = [r for r in ah if r.get("line") in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)]
    if main:
        L.append(f"**Asian handicap** (home line · no-push probability):")
        L.append(f"| Line | {home} | Push | {away} |")
        L.append("|---|---|---|---|")
        for r in main:
            ln = r.get("line", 0.0)
            sign = "+" if ln >= 0 else ""
            L.append(f"| {sign}{ln:g} | {_pct(r.get('home_prob_nopush'))} | "
                     f"{_pct(r.get('push'))} | {_pct(r.get('away_prob_nopush'))} |")
        L.append("")

    totals = dm.get("totals", [])
    if totals:
        L.append("**Alternative totals:**")
        L.append("| Line | Over | Under |")
        L.append("|---|---|---|")
        for r in totals:
            L.append(f"| {r.get('line'):g} | {_pct(r.get('over'))} | {_pct(r.get('under'))} |")
        L.append("")

    wm = dm.get("winning_margin", {})
    fg = dm.get("first_goal", {})
    bands = dm.get("multi_goal_bands", {})
    if wm:
        L.append(f"- **Winning margin:** {home} +1 {_pct(wm.get('home_by_1'))} / +2 "
                 f"{_pct(wm.get('home_by_2plus'))} · Draw {_pct(wm.get('draw'))} · "
                 f"{away} +1 {_pct(wm.get('away_by_1'))} / +2 {_pct(wm.get('away_by_2plus'))}")
    if fg:
        L.append(f"- **First goal:** {home} {_pct(fg.get('home'))} · "
                 f"{away} {_pct(fg.get('away'))} · none {_pct(fg.get('none'))}")
    if bands:
        L.append("- **Total-goals bands:** " + " · ".join(
            f"{k} {_pct(v)}" for k, v in bands.items()))
    if wm or fg or bands:
        L.append("")

    htft = dm.get("ht_ft", {})
    if htft:
        L.append("**HT/FT** (rows = halftime, cols = full-time · H/D/A):")
        L.append(f"| HT＼FT | {home[:3]} | Draw | {away[:3]} |")
        L.append("|---|---|---|---|")
        labels = {"H": home[:3], "D": "Draw", "A": away[:3]}
        for ht in "HDA":
            cells = " | ".join(_pct(htft.get(f"{ht}/{ft}")) for ft in "HDA")
            L.append(f"| {labels[ht]} | {cells} |")
        L.append("")
    return L


def _top_factors(signals: list, ensemble: Any, k: int) -> list[str]:
    """The k factors with the largest weighted home/away divergence."""
    eff = {s["name"]: s.get("effective_weight", 0.0)
           for s in ensemble.breakdown_payload.get("signals", [])}
    scored = []
    for s in signals:
        if not s.available or s.kind == "global":
            continue
        impact = eff.get(s.name, 0.0) * abs(s.home_strength - s.away_strength)
        if impact <= 1e-9:
            continue
        lean = "home" if s.home_strength >= s.away_strength else "away"
        scored.append((impact, f"{s.name} → {lean}"))
    scored.sort(reverse=True)
    return [label for _, label in scored[:k]]


def _tornado(signals: list, ensemble: Any) -> list[str]:
    eff = {s["name"]: s.get("effective_weight", 0.0)
           for s in ensemble.breakdown_payload.get("signals", [])}
    rows = []
    for s in signals:
        impact = eff.get(s.name, 0.0) * (s.home_strength - s.away_strength)
        rows.append((abs(impact), impact, s))
    rows.sort(key=lambda r: r[0], reverse=True)
    lines = []
    width = 18
    for _, impact, s in rows:
        if not s.available:
            bar = "·" * width + "|" + "·" * width
            lines.append(f"{s.name:18} {bar}  (n/a)")
            continue
        mag = max(-1.0, min(1.0, impact * 8.0))   # scale for visibility
        n = int(round(abs(mag) * width))
        if impact >= 0:                            # home favour → left
            bar = " " * (width - n) + "█" * n + "|" + " " * width
        else:                                      # away favour → right
            bar = " " * width + "|" + "█" * n + " " * (width - n)
        lines.append(f"{s.name:18} {bar}  {impact:+.3f}")
    lines.append(f"{'':18} {'home ':>18}<┘└>{' away':<18}")
    return lines


def _heatmap(matrix: list[list[float]], home: str, away: str) -> list[str]:
    n = min(7, len(matrix))
    blocks = " ░▒▓█"
    hi = max((matrix[i][j] for i in range(n) for j in range(n)), default=0.0) or 1.0
    lines = ["      " + "".join(f"{j:>5}" for j in range(n)) + f"   ({away} →)"]
    for i in range(n):
        cells = []
        for j in range(n):
            v = matrix[i][j]
            idx = min(len(blocks) - 1, int(v / hi * (len(blocks) - 1)))
            cells.append(f"{blocks[idx]}{100*v:4.1f}")
        lines.append(f"  {i:>2}  " + "".join(cells))
    lines.append(f"({home} ↓)")
    return lines


# ── public entry ──────────────────────────────────────────────────────────────
def build_report(result: dict[str, Any]) -> dict[str, Any]:
    js = build_json(result)
    md = build_markdown(result, js)
    return {"json": js, "markdown": md}


__all__ = ["build_report", "build_json", "build_markdown"]
