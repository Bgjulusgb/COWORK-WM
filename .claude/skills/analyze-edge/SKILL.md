---
name: analyze-edge
description: Deep edge / value-betting analysis on an existing prediction report — apply the conservative p5 bootstrap rule, sanity-check edges > 10%, compute Kelly stake from the bankroll, surface the truly tradeable picks (or honest "no value, pass") across 1X2, totals, BTTS, Asian Handicap, Double Chance. Use after `predict-match` when the user asks "where's the value?", "which bet is real?", or "is this edge honest?".
---

# Analyze-Edge — von der Quote zur ehrlichen Empfehlung

Diese Skill konzentriert sich **ausschließlich** auf den Wett-Wert. Sie geht
davon aus, dass ein Report (JSON) bereits existiert (siehe Skill `predict-match`).

## 1. Edge-Mathematik — was bedeutet was

### Vig (Buchmacher-Marge) entfernen
Für ein Drei-Wege-Markt (1X2):
```
implied_p_i = 1 / decimal_odd_i
overround   = Σ implied_p_i           (typisch 1.05 – 1.10 → 5-10% Marge)
fair_p_i    = implied_p_i / overround
```
Bei Zwei-Wege-Märkten (Over/Under, BTTS) analog.

### Edge
```
edge = model_p - fair_p
edge_pct = edge / fair_p * 100        (relative Marge zum fair_p)
```

### Konservative Edge (p5)
Aus dem Bootstrap-Vektor `confidence_intervals.blended.<market>` nehmen wir
den **p5-Quantil** statt p50:
```
edge_p5 = model_p5 - fair_p          (für direkte Selections wie Home/Over/Yes)
edge_p5 = (1 - model_p95) - fair_p   (für Komplemente wie Away/Under/No)
```
**Begründung:** wenn die Edge auch unter Pessimismus-Annahme positiv bleibt,
überlebt sie die eigene Modell-Unsicherheit.

### Half-Kelly-Stake
```
b = decimal_odd - 1               # Netto-Quote
f_full = (model_p * b - (1 - model_p)) / b
f_half = max(0, 0.5 * f_full)     # half-Kelly: weniger Drawdown-Risiko
```
**Konservativer Kelly:** dasselbe mit `model_p5` statt `model_p`.

## 2. Die 5-Schritte-Edge-Pipeline

### Schritt 1 — Report laden und Edge-Tabelle scannen
```bash
python3 - <<'PY'
import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "reports/<match_id>.json"))
print(f"mode={d['mode']}  conf={d['ensemble_confidence']:.2f}  factors={d['factors_used']}/{d['factors_total']}  bankroll={d.get('bankroll')}")
print(f"{'market':<14}{'sel':<10}{'p_mod':>7}{'odd':>6}{'edge':>9}{'p5_edge':>10}{'½K p5':>7}{'stake p5':>10}  action")
for r in sorted(d.get("edge_table", []),
                key=lambda x: -(x.get("edge_pct_cons") if x.get("edge_pct_cons") is not None else -99)):
    e = r.get("edge_pct"); ec = r.get("edge_pct_cons"); hkc = r.get("half_kelly_cons")
    sc = r.get("stake_cons")
    print(f"{r['market']:<14}{r['selection']:<10}{r['model_p']:>7.3f}"
          f"{(r.get('decimal_odd') or 0):>6.2f}"
          f"{(f'{e:+.1f}%' if e is not None else '–'):>9}"
          f"{(f'{ec:+.1f}%' if ec is not None else '–'):>10}"
          f"{(f'{hkc:.1f}%' if hkc is not None else '–'):>7}"
          f"{(f'{sc:.2f}' if sc is not None else '–'):>10}  {r['action']}")
print()
print("best_value (p50 — for transparency):     ", d.get("best_value"))
print("best_value_cons (p5 — the honest pick):  ", d.get("best_value_cons"))
PY
```

> **Phase-4-Update:** auch **Double-Chance** und **Asian-Handicap**-Zeilen
> haben jetzt eine `edge_pct_cons`-Spalte und damit denselben p5-Guard wie
> 1X2/O-U/BTTS. `best_value_cons` zeigt direkt den ehrlichen Pick (None ⇒ Pass).
> Mit `--bankroll 1000` enthält jede Zeile zusätzlich `stake_half_kelly`
> (p50) und `stake_cons` (p5, = 0 wenn die konservative Edge ≤ 0).

### Schritt 2 — Mode-Check
- Wenn `"mode": "mock"` → **STOPP**: Edges sind illustrativ. Briefing
  mit klarem Hinweis „kein Live-Anker → keine echte Wett-Empfehlung möglich".
  Diese Pipeline-Output **nie** als Tipp ausspielen.

### Schritt 3 — Drei-Stufen-Filter
Für jede Zeile der Edge-Tabelle:

| Stufe | Bedingung | Was tun |
|---|---|---|
| **F1: Sanity** | `edge_pct > 10` | **Sanity-Check** ausführen (siehe unten) |
| **F2: Konservativ** | `edge_pct_cons > 0` | Echter Wert — Kelly ausweisen |
| **F3: Konfidenz** | `ensemble_confidence > 0.65` UND `factors_used >= 16/20` | Stake-Level erhöhen |

### Schritt 4 — Sanity-Check für „große" Edges (>10 %)
Wenn Modell-Edge > 10 %, **immer** prüfen:
1. **Liquidität:** Ist der Markt am Markt der scharfe (Pinnacle/Asian
   bookies)? Oder ein Square-Buch mit 8% Vig?
2. **Modell-Plausibilität:** λ-Vorteil > 0.4 Tore? Bei Group-Stage
   typisch < 0.3 — sonst Datenlage suspekt.
3. **Closing-Line-Bewegung:** Quote in den letzten 24h gefallen?
   (Effizient-Markt-Hypothese: scharfe Buchhalter würden bewegen.)
4. **Datenlage:** `factors_used / factors_total` < 14/20 ⇒ degradiert →
   große Edge ist meist Daten-Artefakt.

Falls keiner der Punkte den großen Edge erklärt: **explizit kennzeichnen**
„Sanity-Check ausgelöst, aber kein Erklär-Konflikt — Edge gilt **mit Vorbehalt**."

### Schritt 5 — Empfehlung formulieren
**Stake-Level** ergibt sich aus dem Filter-Stack:

| Filter | Stake-Level |
|---|---|
| F1 fail (edge > 10 % unerklärt) ODER F2 fail (p5 < 0) | **Pass** |
| F1 ok + F2 ok + F3 fail (Konfidenz mittel) | **Token-Bet** (¼ × ½-Kelly) |
| F1 ok + F2 ok + F3 ok | **Standard-Bet** (½-Kelly auf p5) |

**Niemals** Voll-Kelly empfehlen (zu volatil). **Niemals** mehr als 2 %
des Bankrolls auf eine einzige Selection.

## 3. Markt-spezifische Caveats

### 1X2 / Double Chance
- Vig auf Draw-Selections oft höher → Edge < +3 % praktisch nicht
  überlebbar.
- Markt-Anker-Kalibrierung (`--calibrate market`) zieht Modell-1X2 schon
  zur fair_p — Restedge nach Kalibrierung ist „echter" als roher Edge.

### Over/Under 2.5
- Bewege auch zu Linien 2.0, 3.0 — alternative Totals sind im
  `derived_markets.exact_total_goals` enthalten und oft mit schlechteren
  Quoten = besseren Edges.

### BTTS
- Hohe Korrelation mit O/U 2.5 → wenn beide grün, vorsichtig mit
  Doppel-Stake (Bankroll-Konzentration).

### Asian Handicap (inkl. Viertel)
- Quarter-Linien (`-0.25`, `+0.25`, etc.) sind **die meist
  unterbewerteten Märkte**. Push-Anteil im Modell explizit ausweisen.
- Half-Push-Settlement: bei `-0.25` Sieg mit 1 Tor = halber Gewinn
  + halber Push.
- Use case: bei knappem Favoriten oft besser als 1X2.

### BTTS-No / Under / Away mit Home-Wertet
Komplement-Selections: p5-Edge benutzt `1 - p95` für die Modell-Wahrscheinlichkeit.

## 4. Multi-Market-Sanity

Wenn drei Edges grün leuchten — **prüfe Korrelation**:
- Home-Sieg + Over 2.5 + BTTS-Yes sind **stark korreliert**. Drei separate
  Stakes = effektiv überkonzentriert. Eher als Akkumulator (mit den
  korrelations-bedingten Quoten-Aufschlag-Verlust) bewerten.
- Independent: Home-Sieg + Under 1.5 (negativ korreliert → echte
  Diversifizierung).

## 5. Antwort-Vorlage

```
=== Edge-Analyse <HOME> vs <AWAY> ===
mode: <live|mock>   conf: X.XX   factors: X/20

✅ Echte Werte (p5-Edge > 0):
  1.  <Markt> <Sel> @ <quote>  ·  p_model=X.XX  ·  p5-edge=+X.X%  ·  ½-Kelly p5: X.X%
      → <Token | Standard>-Bet empfohlen
  2.  …

⚠️ Sanity-Check ausgelöst (p50-Edge > 10 %, p5 ≤ 0):
  1.  <Markt> <Sel> — p50=+XX %, p5=−Y %
      Erklärung: <Datenlage / Vig / Liquidität / …>  →  PASS

❌ Pass (p50- und p5-Edge negativ): N Selections, alle unter fair_p.

📊 Empfohlene Stake-Größe (% vom Bankroll, half-Kelly auf p5):
    [Summe der Standard-Bets ausgewiesen]

⚠️ Disclaimer: Forschungsprojekt, keine Wett-Empfehlung. Mock-Daten illustrativ.
```

## 6. Edge-Decay über Zeit (wenn nicht sofort gespielt)

Quoten bewegen sich. Wenn zwischen Recherche und Anstoß > 4 h:
- Neu fetchen, Pipeline erneut fahren, Edge-Tabelle vergleichen.
- Wenn neue Edge < 50 % der alten → Closing-Line ist schärfer geworden →
  **Pass**.
