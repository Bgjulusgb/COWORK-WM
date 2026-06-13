---
name: read-report
description: Parse a wm2026 prediction report (JSON / Markdown / HTML) and produce a structured, human-readable briefing — headline 1X2, lambdas with CIs, full derived markets board, edge table with p5 conservative column, warnings, and the cowork-task status. Use when interpreting an existing report, comparing two runs, or building the final user-facing summary.
---

# Read-Report — JSON-Report in UI-fertige Briefing-Form bringen

Die Pipeline schreibt drei Artefakte nach `reports/<match_id>.{json,md,html}`.
Das **JSON** ist die maschinenlesbare Wahrheit — alles andere ist Rendering.

## 1. Report-Schema-Übersicht (schema_version 1.3)

```python
{
  "schema_version": "1.3",
  "match_id": "wm2026_groupa_kor_vs_cze",
  "match": { "home": "...", "away": "...", "stage": "...", "kickoff": "..." },
  "mode": "live|mock",
  "factors_used": 18, "factors_total": 20,

  # Phase 4 output — lambdas come as quantile dicts (p5/p50/p95)
  "xg":           {"home": 1.40, "away": 1.30},
  "lambda_home":  {"p5": 0.98, "p50": 1.30, "p95": 1.62},
  "lambda_away":  {"p5": 0.97, "p50": 1.29, "p95": 1.60},
  "per_model":    {"poisson": {...}, "negbin": {...}, "glm_poisson": {...}, "blended": {...}},

  # Phase 4 — bootstrap quantiles per market (1X2 / O/U / BTTS)
  "confidence_intervals": {
    "poisson":     {...},
    "negbin":      {...},
    "glm_poisson": {...},
    "blended": {
      "home_win":  [p5, p50, p95],
      "draw":      [...], "away_win": [...],
      "over_2_5":  [...], "btts_yes": [...]
    }
  },
  "ensemble_confidence": 0.73,   # 0..1 — calibration quality + factor coverage

  # Phase 4/6 — headline markets
  "markets": {
    "1x2":         {"home": 0.36, "draw": 0.27, "away": 0.36},
    "over_under":  {"line": 2.5, "over": 0.51, "under": 0.49},
    "btts":        {"yes": 0.53, "no": 0.47}
  },

  # Phase 6 — derived markets board (linear functionals of the blended score matrix)
  "derived_markets": {
    "double_chance":  {"1x": 0.63, "12": 0.72, "x2": 0.64},
    "draw_no_bet":    {"home": 0.49, "away": 0.51},
    "asian_handicap": [
      {"line": -0.5, "home": 0.39, "away": 0.61},
      {"line": -0.25, "home": 0.42, "away": 0.58},
      {"line":  0.0, "home": 0.49, "away": 0.51}
    ],
    "team_totals":     {...},
    "clean_sheet":     {"home": 0.32, "away": 0.30},
    "win_to_nil":      {"home": 0.16, "away": 0.15},
    "odd_even_goals":  {"odd": 0.51, "even": 0.49},
    "winning_margin":  [...],
    "multi_goal_bands": {"0-1": 0.34, "2-3": 0.46, "4+": 0.20},
    "exact_total_goals": {"0": ..., "1": ..., "2": ..., ...},
    "first_goal":      {"home": 0.48, "away": 0.42, "none": 0.10},
    "ht_ft":           {...}
  },

  # Phase 5
  "calibration": {"mode": "market", "ssh": 0.018, "shrink": 0.42},

  # Phase 6 — edge table (only if odds given)
  "edge_table": [
    {
      "market": "1x2", "selection": "home", "decimal_odd": 2.60,
      "model_p": 0.365, "fair_p": 0.378,
      "edge_pct": +5.3, "edge_pct_cons": -2.1,         # p5 conservative
      "half_kelly_pct": 1.2, "half_kelly_cons": 0.0,
      # Phase 4 — when --bankroll is set:
      "stake_half_kelly": 12.00, "stake_cons": 0.00,
      "action": "PASS"
    }, ...
  ],
  "best_value": {...},        # max p50 edge (often a sanity-check candidate)
  "best_value_cons": {...},   # Phase 4 — max p5 edge: the honest pick (or null)
  "bankroll": 1000.0,         # Phase 4 — echoed back from --bankroll

  # Phase 7
  "warnings": ["mock = illustrative, not live", ...],
  "claude_tasks": [
    {"priority": "P0", "task": "fetch live odds", "fill_via": "--odds H/D/A"}
  ]
}
```

## 2. Briefing-Vorlage — was du dem User lieferst

### Block 1 — Executive Summary (3–5 Zeilen)
```
🎯 <HOME> vs <AWAY>  ·  <STAGE>  ·  <Kickoff>  ·  mode: <live|mock>
λ_home = X.XX [p5 X.XX / p95 X.XX]   λ_away = X.XX [p5 X.XX / p95 X.XX]
1X2 (kalibriert): HOME XX.X %  ·  Draw XX.X %  ·  AWAY XX.X %
Konfidenz: <Ampel> (ensemble_confidence X.XX  ·  factors X/Y)
Pick: <Favorit | Pick'em>     Stake-Level: <Pass | Token | Standard>
```

### Block 2 — Headline-Märkte (Tabelle)
```
| Markt   | Sel  | Modell p   | Markt-fair p | Quote | Edge   | (p5)    | ½-Kelly | Action |
|---------|------|-----------|--------------|-------|--------|---------|---------|--------|
| 1X2     | Home | 36.5 %    | 38.5 %       | 2.60  | -5.1%  | -12.3 % | 0.0 %   | PASS   |
| O/U 2.5 | Over | 50.7 %    | 49.0 %       | 2.05  | +3.0%  | -20.3 % | 0.0 %   | PASS   |
| BTTS    | Yes  | 52.5 %    | 51.2 %       | 1.81  | -4.9%  | -14.0 % | 0.0 %   | PASS   |
```

### Block 3 — Derived Markets Board
- **Double Chance** 1X / 12 / X2
- **Draw-No-Bet** Home / Away
- **Asian Handicap** — Tabelle aller Linien (-0.5, -0.25, 0.0, +0.25, +0.5)
  inkl. Viertel-Linien (push-Anteil ausweisen)
- **Team Totals** (Over 1.5 home/away, etc.)
- **Clean Sheet** Home / Away
- **Win-to-Nil** Home / Away
- **Winning Margin** 1-Tor / 2-Tore / 3+
- **Exact Totals** Top-5 (P(2 goals), P(3 goals), …)
- **HT/FT** 9 Felder (HH/HD/HA/DH/DD/DA/AH/AD/AA)
- **First Goal** Home / Away / None

### Block 4 — Edge-Analyse mit p5
**Regel:** Eine Edge zählt nur, wenn sie auch **konservativ (p5)** positiv ist.
- ✅ p5-Edge > 0 → **echter Wert**, Kelly-Stake ausweisen
- ⚠️ p5-Edge < 0 aber p50-Edge > +10 % → **Sanity-Check ausgeben**:
  „warum würde der Markt das verpassen?" — meist Datenlage zu dünn
- ❌ p50-Edge < 0 → **Pass**, nicht weiter diskutieren

### Block 5 — Warnings & Cowork-Status
- `warnings` 1:1 zitieren (besonders „mock = illustrative")
- `claude_tasks`: was offen blieb, was wurde nachgereicht
- **Konfidenz-Ampel**:
  - 🟢 ensemble_confidence ≥ 0.75 + ≥18/20 factors + 0 critical warnings
  - 🟡 0.55–0.75 oder 14–17/20 factors
  - 🔴 < 0.55 oder ≤ 13/20 factors oder live-data-gap kritisch

### Block 6 — Disclaimer (immer)
> Forschung/Bildung, **keine Wett-Empfehlung**. Mock-Vorhersagen sind illustrativ.

## 3. Praktische Parser-Snippets

### Erste Wahl: `wm2026 summary`
Liefert dasselbe in ~400 Tokens statt 4 000 — siehe Skill `inspect-data`:
```bash
python -m wm2026.cli summary reports/<match_id>.json          # oder .json.gz
```

### Komplette Übersicht (falls Skript-Pipeline)
```bash
python3 - <<'PY'
import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "reports/wm2026_match.json"))
lh, la = d["lambda_home"], d["lambda_away"]
print(f"mode={d['mode']} · factors={d['factors_used']}/{d['factors_total']} · conf={d.get('ensemble_confidence', '–'):.2f}")
print(f"lambda_home: p50={lh['p50']:.2f} [p5 {lh['p5']:.2f} / p95 {lh['p95']:.2f}]")
print(f"lambda_away: p50={la['p50']:.2f} [p5 {la['p5']:.2f} / p95 {la['p95']:.2f}]")
print(f"xg={d['xg']}")
ci = d.get("confidence_intervals", {}).get("blended", {})
for k, v in ci.items():
    if isinstance(v, list) and len(v) == 3:
        print(f"  {k:10}: [p5 {v[0]:.3f} / p50 {v[1]:.3f} / p95 {v[2]:.3f}]")
print("\n=== EDGE TABLE ===")
edges = sorted(d.get("edge_table", []),
               key=lambda x: -(x.get("edge_pct_cons") if x.get("edge_pct_cons") is not None else -1e9))
for r in edges:
    print(f"{r['market']:10} {r['selection']:8} p={r['model_p']:.3f}  q={r['decimal_odd']:.2f}  "
          f"edge={r['edge_pct']:+.2f}%  p5={(r.get('edge_pct_cons') or 0):+.2f}%  -> {r['action']}")
print("\n=== WARNINGS ===")
for w in d.get("warnings", []): print(" -", w)
print("\n=== CLAUDE_TASKS (offen) ===")
for t in d.get("claude_tasks", []): print(f" [{t['priority']}] {t['task']}  -> {t['fill_via']}")
PY
```

### Derived markets (volles Board)
```bash
python3 -c "
import json, sys
d = json.load(open('reports/<match_id>.json'))
import pprint; pprint.pprint(d['derived_markets'])
"
```

### Zwei Reports vergleichen (z.B. vor/nach Overrides)
```bash
python3 - <<'PY'
import json
a = json.load(open("reports/before.json"))
b = json.load(open("reports/after.json"))
for k in ("lambda_home", "lambda_away", "ensemble_confidence", "factors_used"):
    print(f"{k:22} {a.get(k):>6}  →  {b.get(k):>6}")
print("\nclaude_tasks:", len(a.get("claude_tasks", [])), "→", len(b.get("claude_tasks", [])))
PY
```

## 4. HTML-Report öffnen (UI-Ausgabe)

```bash
ls reports/*.html
# Im Browser öffnen — eingebettete Charts, eingebettete Heatmap, alle Märkte
```

Wenn `--charts` aktiviert + matplotlib installiert (Hook macht das):
- `<match_id>_tornado.png` — Faktor-Tornado (welche Faktoren ziehen wohin)
- `<match_id>_heatmap.png` — Score-Heatmap P(Endstand)

## 5. Antwort-Stil

- **Keine λ ohne CI.** Immer `λ = X [p5 / p95]`.
- **Edges immer mit p5-Spalte zitieren.**
- **Disclaimer am Ende.**
- Wenn der Report `"mode": "mock"` zeigt, **explizit** sagen
  „illustrativ — kein Live-Anker".
