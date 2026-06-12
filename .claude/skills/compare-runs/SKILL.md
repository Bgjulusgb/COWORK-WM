---
name: compare-runs
description: A/B comparison of two wm2026 prediction reports — quantify how much an override pack, a calibration mode, the bivariate-Poisson opt-in, or a different bankroll moved the lambdas, the 1X2, the markets, and the edge table. Use after a re-run with --overrides-json, after switching --calibrate market vs auto, or after toggling INCLUDE_BIVARIATE/LAMBDA_AGGREGATION to verify the delta is sensible.
---

# Compare-Runs — zwei Reports diff-en wie ein Quant

Die Pipeline ist deterministisch (gleicher Mock + gleiche Seeds ⇒ identische
Outputs). Sobald sich etwas ändert — Overrides, Calibration-Mode, ein Toggle —
ist der nüchternste Sanity-Check ein **diff zweier JSON-Reports**, gewichtet
nach Wettrelevanz: λ, 1X2, Märkte, Edge-Tabelle, ehrlicher Pick.

## 1. Wann lohnt sich das?

| Trigger | Was sehen? |
|---|---|
| Vor/nach **`--overrides-json`** (Cowork-Loop) | Lambdas, 1X2, `factors_used`, verbleibende `claude_tasks` |
| Vor/nach **`--calibrate market`** vs `auto` vs `none` | nur `calibration`+1X2, Lambdas konstant |
| Vor/nach **`INCLUDE_BIVARIATE=true`** | `per_model`, Score-Heatmap-Korrelation, BTTS, draw-Anteil |
| Vor/nach **`LAMBDA_AGGREGATION=geom`** | λ-Mult, danach automatisch alles abwärts |
| Vor/nach **`--bankroll`** | `stake_half_kelly`/`stake_cons` neu, sonst nichts |
| Vor/nach **Quoten-Update** (Closing-Line-Bewegung) | nur `edge_table` + `best_value*` |

## 2. Standard-Diff

```bash
python3 - <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

def pct(x):
    return f"{x*100:6.2f}%" if isinstance(x, (int, float)) else "  –   "

def diff_pct(av, bv):
    if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
        return "    –   "
    delta = (bv - av) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:5.2f}pp"

print(f"{'':22}{'A':>14}{'B':>14}{'Δ':>12}")
print(f"{'mode':22}{a['mode']:>14}{b['mode']:>14}")
print(f"{'schema_version':22}{a['schema_version']:>14}{b['schema_version']:>14}")
print(f"{'factors_used':22}{a['factors_used']:>14}{b['factors_used']:>14}"
      f"{b['factors_used']-a['factors_used']:>+12}")
print(f"{'ensemble_confidence':22}{a['ensemble_confidence']:>14.3f}{b['ensemble_confidence']:>14.3f}"
      f"{b['ensemble_confidence']-a['ensemble_confidence']:>+12.3f}")
print()
print("--- Lambdas (p50) ---")
for side in ("lambda_home", "lambda_away"):
    av = a[side]["p50"]; bv = b[side]["p50"]
    print(f"{side:22}{av:>14.3f}{bv:>14.3f}{bv-av:>+12.3f}")
print()
print("--- 1X2 (calibrated if present, else raw) ---")
def pick_1x2(d):
    cal = (d.get("calibration") or {}).get("calibrated")
    if cal:
        return cal["home_win"], cal["draw"], cal["away_win"]
    m = d["markets"]["1x2"]
    return m["home"], m["draw"], m["away"]
ah, dh, awh = pick_1x2(a); bh, bd, bawh = pick_1x2(b)
for lbl, av, bv in [("home", ah, bh), ("draw", dh, bd), ("away", awh, bawh)]:
    print(f"{lbl:22}{pct(av)}{pct(bv)}{diff_pct(av, bv):>12}")
print()
print("--- Headline markets ---")
def m(d, k1, k2):
    return d["markets"].get(k1, {}).get(k2)
for label, k in [("over_25", ("over_under", "over")), ("btts_yes", ("btts", "yes"))]:
    av = m(a, *k); bv = m(b, *k)
    print(f"{label:22}{pct(av)}{pct(bv)}{diff_pct(av, bv):>12}")
print()
print("--- best_value_cons (the honest pick) ---")
def fmt(bv):
    if bv is None: return "–"
    return f"{bv['market']}·{bv['selection']} @{bv['decimal_odd']} p5={bv['edge_pct_cons']:+.1f}%"
print("  A:", fmt(a.get("best_value_cons")))
print("  B:", fmt(b.get("best_value_cons")))
print()
print("--- claude_tasks remaining ---")
print(f"  A: {len(a.get('claude_tasks', []))}   B: {len(b.get('claude_tasks', []))}")
PY
reports/before.json reports/after.json
```

## 3. Edge-Tabellen-Diff (nur Märkte mit Quote)

```bash
python3 - <<'PY'
import json
a = {(r["market"], r["selection"]): r for r in json.load(open("reports/before.json"))["edge_table"]}
b = {(r["market"], r["selection"]): r for r in json.load(open("reports/after.json"))["edge_table"]}
keys = sorted(set(a) | set(b))
print(f"{'market':<14}{'sel':<10}{'edge A':>9}{'edge B':>9}{'Δ':>7}  "
      f"{'p5 A':>8}{'p5 B':>8}{'Δp5':>7}  action A → B")
for k in keys:
    ra, rb = a.get(k), b.get(k)
    if not (ra and rb): continue
    ea, eb = ra.get("edge_pct"), rb.get("edge_pct")
    ca, cb = ra.get("edge_pct_cons"), rb.get("edge_pct_cons")
    if ea is None and eb is None: continue
    de = (eb - ea) if (ea is not None and eb is not None) else None
    dc = (cb - ca) if (ca is not None and cb is not None) else None
    print(f"{k[0]:<14}{k[1]:<10}"
          f"{(f'{ea:+.1f}' if ea is not None else '–'):>9}"
          f"{(f'{eb:+.1f}' if eb is not None else '–'):>9}"
          f"{(f'{de:+.1f}' if de is not None else '–'):>7}"
          f"  {(f'{ca:+.1f}' if ca is not None else '–'):>8}"
          f"{(f'{cb:+.1f}' if cb is not None else '–'):>8}"
          f"{(f'{dc:+.1f}' if dc is not None else '–'):>7}"
          f"  {ra['action']} → {rb['action']}")
PY
```

## 4. Interpretieren

| Beobachtung | Bedeutung |
|---|---|
| `Δλ_home > 0.20` | Override / Toggle hat den Heim-xG-Anker bewegt — gross genug, dass das ganze Markt-Board mitgeht. |
| `Δhome_win > +3 pp` ohne `Δλ` | Calibration hat gezogen (Market-Anker > raw). Übliche Bewegung bei Pick'em-Spielen. |
| `Δp5 > Δp50` einer Edge | Bootstrap ist enger geworden → mehr Konfidenz (faktoren_used gestiegen?). Gut. |
| `best_value_cons` wird **neu** zu None | Closing-Line aufgeholt; Pass. |
| `claude_tasks` von 3 → 0 | Cowork-Loop sauber geschlossen → nicht-mock-degradiert. |
| `ensemble_confidence > +0.10` | Mehr Faktoren live, mehr Agreement — Stake-Level darf eine Stufe rauf. |

## 5. Was du dem User schreibst

```
=== Δ <HOME> vs <AWAY>  (A → B)  ===
Trigger: <Overrides eingespeist | Calibration mode | Bivariate on | Bankroll 1000>

λ:           home  X.XX → X.XX (ΔY.YY)   away X.XX → X.XX (ΔY.YY)
1X2:         home  X.X% → X.X% (Δ+Y pp)  draw …  away …
Konfidenz:   X.XX → X.XX  ·  factors X → Y  ·  tasks Z → 0
Pick (p5):   <vorher> → <nachher>

Wirkung: <ein Satz pro signifikante Bewegung — mit Mathematik-Begründung,
nicht "es ist halt anders">
```
