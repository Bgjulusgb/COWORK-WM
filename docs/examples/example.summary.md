# 🎯 Czech Republic vs South Africa · group_stage · 2026-06-18T18:00:00Z
mode `mock` · conf 🟡 0.59 · factors 11/20

## λ + CI (blended bootstrap)
- **Czech Republic**: λ = 1.69 [p5 1.27 / p95 2.11]
- **South Africa**: λ = 1.00 [p5 0.76 / p95 1.25]
- **1X2 (calibrated)**: H  49.0% · D  24.6% · A  26.4%
  - CI home_win:  53.8% [p5  42.6% / p95  64.1%]
  - CI draw:      22.2% [p5  18.9% / p95  26.0%]
  - CI away_win:  24.0% [p5  16.3% / p95  32.8%]
- **O/U 2.5**: Over  50.0% · Under  50.0%  ·  CI O2.5:  49.7% [p5  37.6% / p95  60.2%]
- **BTTS**:    Yes   49.6% · No     50.4%  ·  CI BTTS:  49.0% [p5  39.6% / p95  57.4%]

## Edges (top 3 by p5)
```
market        sel          odd    edge%      p5%    ½K   p5K  action
1X2           Home        2.10+13.50%-10.49% +6.13 +0.00  sanity-check
1X2           Draw        3.40-24.79%-35.87% +0.00 +0.00  no-bet
1X2           Away        3.20-23.73%-47.79% +0.00 +0.00  no-bet
```

## Recommendation
- ❌ **Pass** — no edge survives the bootstrap lower bound (p5).
- ⚠️ Sanity-check candidate (raw edge): 1X2 · Home @ 2.1 — edge **13.5%**, p5 **-10.49%** (sanity-check)

## ⚠️ Warnings
- all data sources are mock — predictions are illustrative, not live
- model-market divergence on 1X2 home: model 54.0% vs vig-free 44.0% (Δ 10.1% > 10pp) — verify the λ-driving inputs (xG/Elo) before trusting any edge here

> Forschung/Bildung — **keine Wett-Empfehlung**. Mock = illustrativ. ½-Kelly auf p5, niemals > 2 % Bankroll.