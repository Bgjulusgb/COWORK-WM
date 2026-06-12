---
name: calibrate-offline
description: Fit calibration artifacts (isotonic + Platt for 1X2 / Over-Under / BTTS) from a historical CSV so that --calibrate auto applies a real-history-tuned calibration to every future prediction. Use when the user has a prior-tournament CSV (WC2022, EURO2024, Copa) or asks "why is my 1X2 not calibrated?".
---

# Calibrate-Offline — fitted Artefakt vs. Markt-Anker

Die Pipeline kalibriert auf zwei Wegen:
- **`--calibrate market`** (pro Spiel): ankert 1X2 an die vig-freie Quote.
  Funktioniert **ohne Historie** — perfekt im Cowork-Flow.
- **`--calibrate auto`** (über die Historie): nutzt ein **fitted Artefakt**
  (isotonic / Platt) das du **einmal offline** baust. Dieses Skill macht das.

## 1. Wann brauchst du das fitted Artefakt?

| Szenario | Kalibrierung |
|---|---|
| Du recherchierst Live-Quoten pro Spiel | `--calibrate market` reicht völlig |
| Du hast keine Live-Quoten | `--calibrate auto` (sonst raw) |
| Du willst Brier-Score quantifizieren | beides — Artefakt + Markt-Anker |
| Forschung / Reproducibility | Artefakt (fixierte Kurve) |

## 2. Die CSV — Format

`history.csv` braucht 5 Spalten:
```csv
home_win_prob,draw_prob,away_win_prob,home_score,away_score
0.52,0.27,0.21,2,1
0.41,0.28,0.31,1,1
0.18,0.24,0.58,0,3
...
```

**Quellen für Prior-Sets:**
- **WC 2022** (64 Matches) — Closing-Quoten via oddsportal.com Archiv
- **EURO 2024** (51 Matches) — official UEFA odds
- **Copa America 2024** (32 Matches)
- **Gold Cup 2023** (31 Matches)
- **Conmebol/AFC/CAF/UEFA Qualifiers 2023-2026** — riesig, aber lärmiger

Insgesamt typisch **150–200 Spiele** → genug für stabile Isotonic-Kurve.

## 3. Fit-Lauf

```bash
# Default-Pfad
python scripts/fit_calibration_offline.py history.csv

# Custom Output-Verzeichnis
python scripts/fit_calibration_offline.py history.csv \
  --out analysis/calibration_artifacts/

# Mit Brier-vorher/nachher Diagnose
python scripts/fit_calibration_offline.py history.csv --verbose
```

Output:
```
fit_calibration_offline.py history.csv
─────────────────────────────────────────
read 152 rows
isotonic-PAV  ↔ 1X2  · Brier raw 0.2014  →  calibrated 0.1872  (Δ-7.1%)
isotonic-PAV  ↔ O/U2.5 · Brier raw 0.2381  →  calibrated 0.2123  (Δ-10.8%)
Platt (Newton) ↔ BTTS  · Brier raw 0.2417  →  calibrated 0.2298  (Δ-4.9%)

wrote:
  analysis/calibration_artifacts/iso_1x2.json
  analysis/calibration_artifacts/iso_ou25.json
  analysis/calibration_artifacts/platt_btts.json

Use:  python -m wm2026.cli predict --calibrate auto …
```

## 4. Pure-Python — kein sklearn nötig

Die Fit-Routinen in `analysis/calibration.py` benutzen:
- **PAV (Pool-Adjacent-Violators)** für isotonische Kurven — ohne sklearn,
  reine numpy-Schleife
- **Newton-Raphson** für Platt-Skalierung — IRLS-Style mit scipy

→ Das Artefakt ist ein **JSON** (knotenweise Kurve + Sigmoid-Parameter), kein
Pickle. Reproduzierbar, einsehbar, klein (~5 KB total).

## 5. Verify nach dem Fit

```bash
pytest tests/test_calibration_offline.py -q
# erwartet: 7 passed (isotonic monotonie, Platt sigmoid, brier_improvement, …)
```

Und ein realer Vergleich:
```bash
# Mit Markt-Anker
python -m wm2026.cli predict --match config/matches/group_a/cze_vs_rsa.yaml \
  --odds "2.10/3.40/3.20" --calibrate market --format json | \
  python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('market: ', d['markets']['1x2'])"

# Mit fitted Artefakt
python -m wm2026.cli predict --match config/matches/group_a/cze_vs_rsa.yaml \
  --odds "2.10/3.40/3.20" --calibrate auto --format json | \
  python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('auto:   ', d['markets']['1x2'])"
```

Beide Modi sollten ähnliche, **aber nicht identische** Outputs liefern — das
fitted Artefakt zieht zur historischen Verteilung, der Markt-Anker zur
aktuellen Quote.

## 6. Konfidenz-Boost

Wenn das Artefakt vorhanden ist, steigt `ensemble_confidence` typisch um +0.10
in den meisten Predictions, weil Phase 5 nicht mehr „leer läuft".

## 7. Häufige Fehler

| Fehler | Ursache | Fix |
|---|---|---|
| `KeyError: home_win_prob` | falsche CSV-Header | exakt `home_win_prob,draw_prob,away_win_prob,home_score,away_score` |
| Brier wird *schlechter* nach Fit | Overfit auf Mini-Datensatz | mindestens 80 Zeilen verwenden; Cross-Validation in Script (`--cv 5`) |
| `--calibrate auto` ignoriert Artefakt | Pfad falsch | Default-Pfad: `analysis/calibration_artifacts/`; per `--cal-dir` überschreiben |
| Isotonic-Kurve flach | Daten zu klein (< 50) | mehr Historie sammeln oder bei `market` bleiben |

## 8. Forschungs-Wert

Mit dem Artefakt kannst du **echte Brier/LogLoss/RPS-Backtests** fahren:
```bash
pytest tests/test_backtesting_rps.py -q
```

Und `analysis/backtesting.py` zeigt dir, ob deine Faktor-Gewichte
RPS-optimal sind — siehe Skill `tune-models`.
