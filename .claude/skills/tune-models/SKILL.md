---
name: tune-models
description: Offline RPS-based tuning of goal-model blend weights (Dixon-Coles / NegBin / GLM-Poisson / BiPoisson) and Dixon-Coles ρ via Optuna. Use when the user asks to optimize the model, improve RPS/Brier score, or get the best blend weights from a historical backtest set.
---

# Tune-Models — RPS-optimale Blend-Gewichte finden

Standard-Blend ist 0.4/0.3/0.3 (statt 0.4/0.3/0.2/0.1) und ρ=0.10 — manuelle
Defaults, nicht datengetrieben. Mit einer Backtest-CSV kannst du beide via
Optuna optimieren.

## 1. Vorbereitung

Braucht: **Optuna** (im `[tune]` Extra; vom Hook nicht standardmäßig
installiert):
```bash
pip install '.[tune]'           # falls noch nicht
```

Und eine Backtest-CSV (gleiches Format wie für `calibrate-offline`):
```csv
home_win_prob,draw_prob,away_win_prob,home_score,away_score
…
```

## 2. Tuning-Lauf

```bash
# Default: 200 Trials, RPS als Loss
python scripts/tune_models_offline.py history.csv

# Ausführlicher
python scripts/tune_models_offline.py history.csv \
  --trials 500 \
  --metric rps           # rps | brier | logloss
  --bootstrap 200        # Bootstrap für Konfidenz auf RPS-Schätzung
  --out config/runtime_blend.json
```

Output:
```
trial 0  · RPS 0.2017
trial 50 · RPS 0.1952  (best)
trial 99 · RPS 0.1947
...
trial 500 · RPS 0.1922  (best @ 487)

Best params:
  blend = {dixon_coles: 0.34, negbin: 0.18, glm_poisson: 0.28, bivariate: 0.20}
  rho   = 0.087
  
  RPS  0.1922  (Δ -4.7 % vs. default)
  Brier 0.1734  (Δ -3.1 %)

→ wrote config/runtime_blend.json
   Use:  env BLEND_WEIGHTS=config/runtime_blend.json python -m wm2026.cli predict …
   Oder permanent: cp config/runtime_blend.json config/blend_weights.json
```

## 3. Was getuned wird

| Parameter | Range | Wirkung |
|---|---|---|
| `blend.dixon_coles` | [0.05, 0.6] | Anteil am Score-Matrix-Stack |
| `blend.negbin` | [0.05, 0.5] | Über-Dispersion (NegBin-Size fix bei 8) |
| `blend.glm_poisson` | [0.05, 0.5] | Exakter GLM (braucht statsmodels) |
| `blend.bivariate` | [0.0, 0.4] | λ₃-Korrelation (Karlis-Ntzoufras) |
| `rho` (Dixon-Coles) | [0.0, 0.3] | Score-Korrektur für 0:0/1:0/0:1/1:1 |

**Constraint:** `Σ blend.* = 1.0` (Optuna ProbabilitySimplex).

## 4. Loss-Funktionen

- **RPS** (Ranked Probability Score) — kanonisch für 1X2, bestraft falsche
  Reihenfolge der Wahrscheinlichkeiten.
- **Brier** — quadratischer Score, gut für O/U / BTTS.
- **LogLoss** — proper score, hart auf Über-Konfidenz.

**Empfehlung:** RPS für Tuning, Brier für Cross-Validation.

## 5. Cross-Validation einbauen

Tuning auf der **vollen** Backtest-CSV kann overfitten. Sicher:

```bash
python scripts/tune_models_offline.py history.csv \
  --trials 300 \
  --cv 5                  # 5-fold time-series CV
  --metric rps
```

Optuna sucht dann den Param-Satz, der auf **Held-out Folds** das beste
RPS liefert. Realistisch ist Δ-RPS = -2 bis -4 % vs Default.

## 6. Im Produktiv-Flow nutzen

Nach dem Fit liegt `config/runtime_blend.json`. Sie wird automatisch gelesen,
wenn vorhanden:
```python
# models_ml/poisson_goals.py:build_all_goal_models()
# liest runtime_blend.json falls vorhanden, sonst DEFAULT_BLEND_WEIGHTS
```

Oder explizit per Env:
```bash
BLEND_WEIGHTS=config/runtime_blend.json \
  python -m wm2026.cli predict --match config/matches/...
```

## 7. Verify

```bash
pytest tests/test_model_tuning.py -q                    # sicherheit der Tune-Pipeline
pytest tests/test_backtesting_rps.py -q                 # RPS-Math
```

## 8. Was du dem User reportest

```
Tuning auf 152 Spielen (WC22+EURO24+Copa24), 500 Trials, 5-fold-CV:
- RPS:    0.2017 (default)  →  0.1922 (-4.7 %)
- Brier:  0.1789 (default)  →  0.1734 (-3.1 %)

Best blend: DC 0.34 · NegBin 0.18 · GLM 0.28 · BiPoisson 0.20
Best ρ:     0.087

→ Aktiviert via config/runtime_blend.json. Nächste Prediction nutzt die
  optimierten Gewichte automatisch.
```

## 9. Wann es sich NICHT lohnt

- Backtest < 100 Spiele → Overfit
- RPS-Verbesserung < 1 % → Default-Blend OK lassen
- Optuna nicht installiert → Skill `calibrate-offline` reicht meist
