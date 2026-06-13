# 🏆 WM 2026 — Czech Republic vs South Africa

*group_stage · 2026-06-18T18:00:00Z · BMO Field, Toronto*
*Mode: `mock` · model `wm2026-workflow-1.1` · predicted 2026-06-12T08:31:56.161573+00:00*

## Executive Summary
- **Most likely 1X2:** Czech Republic ( 54.0%)  ·  Czech Republic  54.0% / Draw  22.1% / South Africa  23.8%
- **Expected goals (λ):** Czech Republic 1.69 — South Africa 1.00  ·  O2.5  49.9% · BTTS  49.6%
- **Top-3 driving factors:** goal_efficiency → home; elo_strength → home; form → home
- **Value pick:** 1X2 — Home @ 2.1 → edge **13.5%**, half-Kelly 6.13% (sanity-check)
- **Conservative pick (p5-survivor):** none — no edge survives the bootstrap lower bound (honest call: pass)
- **Confidence:** 🟡 (mock data — illustrative) (ensemble 0.59, 11/20 factors live)
- **Calibration (market-anchor):** Czech Republic  49.0% / Draw  24.6% / South Africa  26.4%

## ⚠️ Validation warnings
- all data sources are mock — predictions are illustrative, not live

## Factor Tornado (home ◀ favour ▶ away)
```
goal_efficiency    ██████████████████|                    +0.145
elo_strength                █████████|                    +0.066
form                            █████|                    +0.032
ml_blend                             |█                   -0.009
market_odds                         █|                    +0.004
injury_news                          |                    -0.002
lineup_strength                      |                    -0.001
fifa_ranking                         |                    +0.000
squad_value                          |                    +0.000
head_to_head                         |                    +0.000
tournament_context ··················|··················  (n/a)
sentiment          ··················|··················  (n/a)
squad_availability ··················|··················  (n/a)
rest_travel                          |                    +0.000
venue_altitude     ··················|··················  (n/a)
weather            ··················|··················  (n/a)
momentum_drift     ··················|··················  (n/a)
ml_blend_lgbm      ··················|··················  (n/a)
llm_sentiment      ··················|··················  (n/a)
network_strength   ··················|··················  (n/a)
                                home <┘└> away             
```

## Score Probability Matrix (rows = Czech Republic, cols = South Africa)
```
          0    1    2    3    4    5    6   (South Africa →)
   0  ░ 6.2▒ 8.0░ 3.5  1.2  0.3  0.1  0.0
   1  █12.5▓10.2░ 5.6  1.9  0.5  0.1  0.0
   2  ▓ 9.6▒ 9.3░ 4.6  1.6  0.4  0.1  0.0
   3  ░ 5.4░ 5.3  2.6  0.9  0.2  0.1  0.0
   4    2.4  2.3  1.2  0.4  0.1  0.0  0.0
   5    0.9  0.8  0.4  0.1  0.0  0.0  0.0
   6    0.3  0.3  0.1  0.0  0.0  0.0  0.0
(Czech Republic ↓)
```

**Top-5 correct scores:** 1-0 ( 12.6%)  1-1 ( 10.3%)  2-1 (  9.7%)  2-0 (  9.7%)  0-1 (  7.9%)

## Edge Table (Phase 6)
*`(p5)` columns are the conservative edge / half-Kelly on the bootstrap 5th-percentile — value that survives the model's own uncertainty.*
| Market | Selection | Model P | Fair P | Odd | Edge % | Edge% (p5) | ½-Kelly % | ½K (p5) | Action |
|---|---|---|---|---|---|---|---|---|---|
| 1X2 | Home |  54.0% |  44.0% | 2.1 | 13.5 | -12.26 | 6.13 | 0.0 | sanity-check |
| 1X2 | Draw |  22.1% |  27.2% | 3.4 | -24.79 | -36.51 | 0.0 | 0.0 | no-bet |
| 1X2 | Away |  23.8% |  28.9% | 3.2 | -23.73 | -47.51 | 0.0 | 0.0 | no-bet |
| O/U 2.5 | Over 2.5 |  50.0% |  51.3% | 1.85 | -7.6 | -31.45 | 0.0 | 0.0 | no-bet |
| O/U 2.5 | Under 2.5 |  50.0% |  48.7% | 1.95 | -2.4 | -22.55 | 0.0 | 0.0 | no-bet |
| BTTS | Yes |  49.6% |  52.6% | 1.8 | -10.78 | -29.2 | 0.0 | 0.0 | no-bet |
| BTTS | No |  50.4% |  47.4% | 2.0 | 0.86 | -13.95 | 0.43 | 0.0 | no-bet |

## Derived markets (Phase-1 math)
- **Double Chance:** 1X  76.2% · 12  77.9% · X2  46.0%   ·   **Draw-No-Bet:** Czech Republic  69.4% / South Africa  30.6%
- **Clean sheet:** Czech Republic  37.3% / South Africa  19.4%   ·   **Win-to-nil:** Czech Republic  31.1% / South Africa  13.2%   ·   **Goals:** odd  51.7% / even  48.3%

**Asian handicap** (home line · no-push probability):
| Line | Czech Republic | Push | South Africa |
|---|---|---|---|
| -1.5 |  29.1% |   0.0% |  70.9% |
| -1 |  38.8% |  24.9% |  61.2% |
| -0.5 |  54.0% |   0.0% |  46.0% |
| +0 |  69.4% |  22.1% |  30.6% |
| +0.5 |  76.2% |   0.0% |  23.8% |
| +1 |  90.1% |  15.5% |   9.9% |
| +1.5 |  91.7% |   0.0% |   8.3% |

**Alternative totals:**
| Line | Over | Under |
|---|---|---|
| 0.5 |  93.8% |   6.2% |
| 1.5 |  73.2% |  26.8% |
| 2.5 |  50.0% |  50.0% |
| 3.5 |  28.4% |  71.6% |
| 4.5 |  13.8% |  86.2% |

- **Winning margin:** Czech Republic +1  24.9% / +2  29.1% · Draw  22.1% · South Africa +1  15.5% / +2   8.3%
- **First goal:** Czech Republic  58.9% · South Africa  34.9% · none   6.2%
- **Total-goals bands:** 0-1  26.8% · 2-3  44.8% · 4-6  26.4% · 7+   2.1%

**HT/FT** (rows = halftime, cols = full-time · H/D/A):
| HT＼FT | Cze | Draw | Sou |
|---|---|---|---|
| Cze |  33.5% |   5.1% |   1.5% |
| Draw |  17.2% |  13.4% |   8.7% |
| Sou |   2.9% |   5.0% |  12.8% |

## Goal-model blend
| Model | Home | Draw | Away | O2.5 | BTTS |
|---|---|---|---|---|---|
| poisson |  54.6% |  21.9% |  23.5% |  50.4% |  50.5% |
| negbin |  53.3% |  21.6% |  25.1% |  48.8% |  46.8% |
| glm_poisson |  54.0% |  23.0% |  23.0% |  50.4% |  51.1% |

## Data sources (provenance)
- Mode counts: `mock`×18

---
*Generated by the WM 2026 workflow — not betting advice. Mock mode is illustrative; use `--mode live` with API keys for real data.*