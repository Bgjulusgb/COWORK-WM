---
name: backtest-results
description: Score saved JSON reports against actual match outcomes — Brier/RPS/LogLoss for 1X2 plus hit-rate of best_value vs best_value_cons and naïve half-Kelly ROI. Use when the user asks "how good were my last predictions?", "did the p5 filter actually prevent bad bets?", "ROI of the last X tips?", or "validate the conservative rule".
---

# Backtest-Results — Realität gegen Modell prüfen

`predict-match` schreibt JSON-Reports nach `reports/<match_id>.json`. Mit
diesem Skill validierst du im Nachhinein, ob die p5-Konservativ-Regel die
schlechten Picks wirklich aussortiert und ob die Brier/RPS-Werte das Vertrauen
in das Modell rechtfertigen.

## 1. Vorbereitung

Du brauchst:
1. Ein Verzeichnis mit JSON-Reports (`reports/`) — wird automatisch befüllt,
   sobald du `wm2026 predict --out reports/` mit `--bankroll` läufst.
2. Eine Ground-Truth-CSV mit `match_id,home_score,away_score`.

CSV-Beispiel:
```csv
match_id,home_score,away_score
wm2026_groupa_cze_vs_rsa,2,1
wm2026_groupa_arg_vs_jpn,3,0
wm2026_groupb_ger_vs_bra,1,2
```

> **Tipp:** Die `match_id` siehst du im Header jedes Reports oder via
> `wm2026 summary reports/<datei>.json`.

## 2. Backtest fahren

```bash
python -m wm2026.cli backtest \
  --reports reports/ \
  --truth data/results.csv \
  --bankroll 1000 \
  --format markdown
```

Beispiel-Output:
```
# 📊 WM-2026 Backtest — Reports vs. Reality

20 matches evaluated · 0 reports without matching truth row · bankroll 1000

## Aggregate accuracy

| metric | value |
|---|---|
| Brier (lower=better) | **0.187** |
| RPS (lower=better) | **0.142** |
| LogLoss (lower=better) | **0.812** |
| 1X2 best-class hit-rate | **0.55** |

## Pick hit-rate (raw vs. conservative)

| pick | attempts | hits | hit-rate | PnL | ROI |
|---|---|---|---|---|---|
| best_value (raw max edge) | 18 | 9 | 0.50 | +124.50 | 12.45% |
| best_value_cons (p5 survivor) | 11 | 8 | 0.73 | +156.20 | 15.62% |
```

## 3. Interpretation

- **Brier < 0.20** ⇒ das Modell ist kalibrierter als ein Coin-flip
  (uniformer Baseline ≈ 0.222 für 1/3-1/3-1/3).
- **RPS < 0.17** ⇒ ordinal scharf; gute Bookies liegen bei ~0.18–0.20.
- **cons_hit_rate > raw_hit_rate** ⇒ die p5-Disziplin filtert thin-data-Picks
  raus — sie erfüllt ihr Versprechen.
- **cons_roi > raw_roi** ⇒ weniger Picks, aber sauberere Edges; half-Kelly auf
  p5-Stakes übersteigt die naive Voll-Kelly-Variante. **Das ist das eigentliche
  Empfehlungsschild.**

## 4. JSON-Modus für Weiterverarbeitung

```bash
python -m wm2026.cli backtest --reports reports/ --truth data/results.csv \
  --format json > backtest.json
```

Das JSON enthält zusätzlich `per_match[]` mit allen Einzel-Werten — Input für
Tableau / pandas / Diagramme.

## 5. Wann es sich NICHT lohnt

- n < 20 Reports ⇒ statistische Signifikanz fehlt; Bereiche überlappen.
- Reports aus verschiedenen Pipeline-Versionen ⇒ Apfel/Birnen, p5-Definition
  hat sich vielleicht zwischen Versionen geändert.
- Ground-Truth aus Test-Daten / Mock-Smoke-Tests ⇒ irreführend (alle "Real-
  Welt"-Aussagen sind dann Lügen).

## 6. Was du dem User reportest

```
Backtest auf 30 Gruppen-Spielen:
- Brier  0.187  (Baseline 0.222 → Modell ist kalibriert)
- RPS    0.142  (top-decile)
- best_value (raw):   18/30, ROI -2.3 %
- best_value_cons:    11/30, ROI +15.6 %

→ Die p5-Konservativ-Regel hat aus einem leichten Verlust einen klaren
  Gewinn gemacht. Sie filtert ~40 % der raw-Picks (die mit thin data) und
  behält die mit echter Edge. Empfehlung: weiter half-Kelly auf p5 fahren.
```

## 7. Verify

```bash
pytest tests/test_backtest_cli.py -q     # 6 Tests, ~1s
python -m wm2026.cli backtest --reports reports/ --truth data/results.csv \
  --format markdown                       # End-to-end Smoke
```
