# 🏆 WM 2026 — Match-Analyse & Prediction (Master-Prompt v2.0)

> **Was ist das?** Ein kalibrierter Quant-Workflow, der für ein WM-2026-Spiel
> echte Live-Daten holt, sie durch ein **20-Faktor-Ensemble** und einen
> **3-Modell-Tor-Stack** (Dixon-Coles · Negative-Binomial · GLM-Poisson) schickt
> und am Ende eine **kalibrierte Prediction mit Konfidenzintervallen + Markt-Edge**
> liefert.
>
> Dieser Prompt hat eine **lauffähige Referenz-Implementierung in diesem Repo**.
> Du musst nicht alles „per Hand" rechnen — du kannst die Pipeline direkt starten:
>
> ```bash
> python -m wm2026.cli predict \
>   --match config/matches/group_a/cze_vs_rsa.yaml \
>   --odds "2.10/3.40/3.20" --odds-ou "1.85/1.95" --out reports/
> ```
>
> Nutze den Prompt in **zwei Modi**:
> 1. **Code-Modus** — du rufst die `wm2026`-Pipeline auf (mock oder live) und
>    interpretierst den JSON-/Markdown-Report. Schnell, reproduzierbar, kein Raten.
> 2. **Recherche-Modus** — wenn echte Live-Daten gefragt sind (Lineups,
>    Verletzungen, Wetter, Odds), recherchierst du sie via Web Search, trägst sie
>    in den `match:`-Block ein und fütterst sie der Pipeline.

---

## ⚙️ SETUP (einmalig)

```bash
# 1. Repo holen
git clone https://github.com/bgjulusgb/football_wm.git && cd football_wm

# 2. Core-Abhängigkeiten (reicht für Mock-Mode, komplett offline, ohne Keys)
pip install -r requirements.txt
# optional: Charts, exaktes GLM, Kalibrierung, Sentiment
pip install -r requirements-optional.txt

# 3. (nur für Live-Daten) Keys eintragen
cp .env.example .env     # USE_MOCK_*=false setzen + API-Keys eintragen
```

**Cowork-Toggles (für Recherche-Modus):** Web Search · Code Execution & File
Creation · Search past chats (Memory).

---

## 🎯 MATCH INPUT (vor dem Senden ausfüllen)

```yaml
match:
  home_team: "<<HOME_TEAM>>"            # z.B. "Deutschland"
  away_team: "<<AWAY_TEAM>>"            # z.B. "Brasilien"
  competition: "FIFA WM 2026"
  stage: "<<Group | R32 | R16 | QF | SF | Final>>"
  kickoff_local: "<<2026-06-12 21:00>>"
  venue: "<<MetLife Stadium, East Rutherford>>"
  bookmaker_odds_1x2: "<<2.10 / 3.40 / 3.20>>"   # für Edge-Analyse (Phase 6)
  odds_total_2_5: "<<1.85 / 1.95>>"              # Over / Under
  odds_btts: "<<1.80 / 2.00>>"                   # Yes / No
focus_questions:
  - "Wie wirkt der Ausfall von <<KEY_PLAYER>> auf <<TEAM>>?"
  - "Edge gegen Bookie auf Over 2.5?"
  - "Wert bei BTTS=Yes?"
```

> 💡 Statt YAML kannst du auch direkt ad-hoc rechnen:
> `python -m wm2026.cli predict --home Germany --away Brazil --stage QF --odds "2.10/3.40/3.20"`

---

## 🤖 ROLLE & ARBEITSWEISE

Du bist mein **WM-2026-Quant-Analyst**. Du arbeitest die 8-Phasen-Pipeline ab,
**rechnest mit Code statt zu raten**, nutzt **echte Daten** (live oder die im
Repo gepflegten Mock-/YAML-Werte) und lieferst am Ende eine kalibrierte
Prediction mit Konfidenzintervallen und Markt-Edge.

**Grundregeln (hart):**
- ❌ KEINE Punkt-Prediction ohne Konfidenzintervall (p5/p50/p95).
- ❌ KEINE Faktor-Behauptung ohne Quelle (`value, source, fetched_at, confidence`).
- ❌ KEINE Edge > 10 % ohne expliziten Sanity-Check („Warum übersieht der Markt das?").
- ✅ Bei fehlenden Daten: Faktor neutralisiert sich (`available=false`) → das
  Ensemble **re-normalisiert** automatisch. Niemals Werte erfinden.
- ✅ Output am Ende **strikt im JSON-Schema** (Phase 8) + Markdown-Report.
- ✅ Mock-Daten sind **illustrativ, nicht live** — im Report kennzeichnen.

---

## 🧱 REFERENZ-IMPLEMENTIERUNG (Phase → Code)

| Phase | Was passiert | Code |
|---|---|---|
| 1 Data Collection | paralleler Fan-out zu ~13 Konnektoren, füllt `FactorContext` | `data_sources/orchestrator.py` → `DataSourceOrchestrator.populate()` |
| 2 Faktor-Decomposition | 20 Faktoren → je ein `FactorSignal{home,away,weight,conf}` | `factors/registry.py` → `get_active_factors()`; `factors/*.py` |
| 3 Sentiment-Layer | Reddit/X-Stimmung → `sentiment_payload` | `factors/sentiment_factor.py`, `analysis/social_momentum.py` |
| 4 Goal-Model-Stack | Ensemble → λ; 3 Tor-Modelle + Bootstrap-CIs | `analysis/factor_ensemble.py`, `models_ml/poisson_goals.py`, `analysis/match_predictor.py` |
| 5 Calibration | Isotonic + Platt (graceful, prior-basiert) | `analysis/calibration.py` |
| 6 Markt-Edge | De-Vigging, Edge, Kelly | `wm2026/edge.py` |
| 7 Validation | Sanity-Checklist | `wm2026/pipeline.py` → `_validate()` |
| 8 Output | JSON + Markdown + Charts | `wm2026/report.py`, `wm2026/viz.py` |

Der dünne Orchestrierungs-Layer, der alles zusammensteckt: **`wm2026/pipeline.py`
→ `run_prediction()`**.

---

## 📊 PIPELINE (8 PHASEN)

### PHASE 1 — Data Collection (paralleler Fan-out)

Der Orchestrator zieht **parallel** und einzeln abgesichert (eine kaputte Quelle
lässt ihr Feld leer, der Faktor fällt auf YAML zurück oder markiert sich
`unavailable`). Jeder Datenpunkt trägt Provenance: `(source, mode, fetched_at)`
mit `mode ∈ {live, cache, mock, error}`.

| Layer | Quellen (Konnektor) | Extrahiert |
|---|---|---|
| Form/xG | fbref, understat, fotmob | xG for/against, npxG, PPG |
| Historie/H2H | openfootball, football-data.org, openligadb | letzte Spiele, Tor-Diff |
| Lineup/Injuries | fotmob, sofascore | Startelf-Wahrscheinlichkeit, Ausfälle |
| Squad/Value | wikidata, transfermarkt | Kaderwert, Verfügbarkeit |
| Wetter | open-meteo (lat/lng des Venues) | Temp, Regen, Wind → xG-Multiplier |
| Stadium | venues-Tabelle, wikidata | Höhe ü.NN, Koordinaten |
| Travel/Rest | Fixture-Liste | Tage seit letztem Spiel, Reise-/Zeitzonen-Last |
| News | RSS | Verletzungs-/Sperren-Meldungen |
| Markt | The-Odds-API / `--odds` | implizite 1X2-Wahrscheinlichkeiten |
| Sentiment | Reddit (worldcup + Team-Subs) | Fan-Stimmung, Momentum |

> **Recherche-Modus:** Für jeden Wert `(value, source_url, fetched_at, confidence)`.
> Widersprechen sich Quellen → beide nennen, nach Recency + Reputation gewichten.

---

### PHASE 2 — Faktor-Decomposition (20 Faktoren)

Jeder Faktor liefert ein `FactorSignal`:
`home_strength`, `away_strength` (1.0 = neutral, Range 0.5–1.5, hart geklemmt auf
[0.3, 2.5]), `weight`, `confidence`, `available`, `kind ∈ {tilt, global}`.

| # | Faktor | kind | Idee | Quelle |
|---|---|---|---|---|
| 1 | elo_strength | tilt | Elo-Delta → λ-Tilt | YAML/clubelo |
| 2 | form | tilt | letzte 5–10 Spiele (Punkte/xG) | history |
| 3 | head_to_head | tilt | gewichtete Tor-Diff letzte 10 H2H | openfootball |
| 4 | goal_efficiency | tilt | xG for/against Mismatch | fbref/understat |
| 5 | tournament_context | tilt | Stake / KO-Phase | meta |
| 6 | sentiment | tilt | Reddit-Stimmung (±10 %, sample-gedämpft) | reddit |
| 7 | squad_availability | tilt | Verfügbarkeit Schlüsselspieler | wikidata |
| 8 | fifa_ranking | tilt | FIFA-Ranking-Delta (ergänzt Elo) | meta |
| 9 | rest_travel | tilt | Rest-Tage + Reise/Jetlag | fixtures |
| 10 | venue_altitude | **global** | Höhen-Malus (Stamina) | venues |
| 11 | market_odds | tilt | Buchmacher-implizite 1X2 | odds |
| 12 | weather | **global** | Hitze/Regen → Tor-Dämpfung | open-meteo |
| 13 | injury_news | tilt | RSS-Verletzungs-Impact | rss |
| 14 | momentum_drift | tilt | Sentiment-Trend | reddit |
| 15 | ml_blend | tilt | trainierter xG-Head (dormant) | artifact |
| 16 | ml_blend_lgbm | tilt | LightGBM-Head (dormant) | artifact |
| 17 | llm_sentiment | tilt | NVIDIA-LLM Aspect-Sentiment (dormant) | nvidia |
| 18 | lineup_strength | tilt | bestätigte Elf vs. Saison-Schnitt | fotmob/sofascore |
| 19 | squad_value | tilt | Transfermarkt-Marktwert-Ratio | transfermarkt |
| 20 | network_strength | tilt | PageRank über Match-Graph (dormant) | artifact |

**Ensemble-Regel** (`analysis/factor_ensemble.py`):

```text
available = {Faktoren mit available=True und weight>0}
tilt-Faktoren → gewichteter Mittelwert → (λ_home_mult, λ_away_mult)
global-Faktoren (Wetter, Höhe) → Produkt, gefloort auf 0.82, danach drauf-multipliziert
confidence = 0.6 · mean(factor_confidence) + 0.4 · agreement
           agreement = 1 − stdev(home_i / away_i)
```

> Re-Normalisierung: Fällt ein Faktor aus, wird sein Gewicht auf die übrigen
> verteilt — die konfigurierten Verhältnisse bleiben gültig.

---

### PHASE 3 — Sentiment-Layer

`sentiment_payload`-Schema (füttert `SentimentFactor` + `MomentumDriftFactor`):

```json
{
  "sample_size": 420,
  "home_sentiment": 0.18, "away_sentiment": -0.05,
  "home_momentum": 0.04,  "away_momentum": -0.02,
  "home_controversy": 0.22, "away_controversy": 0.31
}
```

- **VADER** (Lexikon, schnell) + **TextBlob** (polarity/subjectivity), optional
  **RoBERTa-twitter** und **NER**-Team-Attribution.
- `strength = 1 + 0.10 · (sentiment + 0.5·momentum) · min(1, N/300)` pro Seite.
- **Bias-Korrektur:** Heim-Subreddits sind systematisch zu optimistisch — Bias-Term
  `b_team ≈ +0.15` für das eigene Sub abziehen.
- Ohne Daten (`sample_size=0`) neutralisiert der Faktor → Ensemble re-normalisiert.

Im Code injizierst du das Payload via `run_prediction(..., sentiment_payload=...)`
oder `--sentiment-json datei.json`.

---

### PHASE 4 — Goal-Model-Stack (3-Modell-Blend + Bootstrap)

```text
base_home_xg = (home.avg_xg_season + away.avg_xg_conceded) / 2
base_away_xg = (away.avg_xg_season + home.avg_xg_conceded) / 2
λ_home = base_home_xg · λ_home_mult        # aus dem Ensemble
λ_away = base_away_xg · λ_away_mult
```

Drei Tor-Modelle, gewichtet gemischt (Default `blend = 0.4·DC + 0.3·NegBin + 0.3·GLM`):

- **Dixon-Coles-Poisson** mit Low-Score-Korrektur ρ (ρ klemmt 0-0/1-0/0-1/1-1).
- **Negative-Binomial-DC** für die Über-Dispersion echter Tor-Counts.
- **GLM-Poisson** (statsmodels; fällt ohne statsmodels sauber auf Poisson zurück).

**Bootstrap-CIs** (n=500, σ=0.15·xg): pro Markt p5/p50/p95. Märkte aus der
Wahrscheinlichkeits-Matrix: 1X2 · O/U 0.5/1.5/2.5/3.5 · BTTS · Correct-Score-Top-5.

> **Korrektur ggü. v1:** Der Bootstrap zieht λ ~ N(xg, σ·xg), simuliert je Sample
> die volle Matrix und nimmt **pro Markt** das Perzentil — nicht `axis=0` über
> rohe Score-Samples (das war im alten Prompt inkonsistent).

---

### PHASE 5 — Calibration (graceful)

```text
Isotonic-Regression (monoton, parameterfrei) + Platt-Scaling (logistisch)
auf historische (p_pred, realized)-Paare → kalibrierte 1X2-Probs (renormiert).
```

→ Liegt **kein** WM-2026-Verlauf vor (Default beim frischen Clone), gibt der
Workflow die **rohen** Probs aus + Hinweis. Als Prior-Set: **WM 2022 + EURO 2024 +
Copa 2024** fitten und den Transfer explizit kennzeichnen
(`analysis/calibration.py → fit_calibrators`).

---

### PHASE 6 — Markt-Edge & Value-Detection (`wm2026/edge.py`)

```text
overround = Σ 1/odd
fair_p    = (1/odd) / overround                  # Vig rausgerechnet
edge      = model_p · odd − 1
kelly     = (model_p · odd − 1) / (odd − 1)       # half-Kelly empfohlen
```

**Value-Schwellen:** `<2 %` no-bet · `2–5 %` small (0.25–0.5 % BR) ·
`5–10 %` standard (0.5–1.5 % BR) · `>10 %` **erst Sanity-Check**, dann max 2 % BR.

---

### PHASE 7 — Validation (Sanity-Checklist)

- [ ] λ_home, λ_away ∈ [0.3, 4.0]?
- [ ] Σ P(1X2) = 1.000 ± 0.005?
- [ ] ≥ 5 Faktoren verfügbar?
- [ ] Mindestens eine `live`/`cache`-Quelle? (sonst „mock = illustrativ" flaggen)
- [ ] |p_iso − p_raw| < 0.15? (sonst: Modell misstraut sich)
- [ ] ensemble_confidence ≥ 0.5? (sonst Low-Conviction kennzeichnen)

---

### PHASE 8 — Output (JSON + Markdown + Charts)

#### A) JSON (kopierbar, DB-ready) — so emittiert es `wm2026/report.py`

```json
{
  "schema_version": "1.0",
  "match_id": "wm2026_groupa_cze_vs_rsa",
  "model_version": "wm2026-workflow-1.0",
  "predicted_at": "<ISO8601>",
  "mode": "mock | live",
  "fixture": {"home": "...", "away": "...", "stage": "...", "kickoff_utc": "...", "venue": "..."},
  "lambda_home": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
  "lambda_away": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
  "xg": {"home": 0.0, "away": 0.0},
  "markets": {
    "1x2": {"home": 0.0, "draw": 0.0, "away": 0.0},
    "over_under": {"over_05": 0.0, "over_15": 0.0, "over_25": 0.0, "over_35": 0.0},
    "btts": {"yes": 0.0, "no": 0.0},
    "correct_score_top5": [{"score": "1-1", "p": 0.0}],
    "recommended_bet": null, "bet_probability": null
  },
  "per_model": {"poisson": {}, "negbin": {}, "glm_poisson": {}},
  "confidence_intervals": {"blended": {"home_win": [0.0, 0.0, 0.0]}},
  "ensemble_confidence": 0.0,
  "factors_used": 0, "factors_total": 20,
  "factors": [{"name": "...", "home_strength": 0.0, "away_strength": 0.0, "weight": 0.0,
               "effective_weight": 0.0, "confidence": 0.0, "available": true,
               "kind": "tilt", "source": "..."}],
  "calibration": {"applied": false, "note": "..."},
  "edge_table": [{"market": "1X2", "selection": "Home", "model_p": 0.0, "fair_p": 0.0,
                  "decimal_odd": 0.0, "edge_pct": 0.0, "half_kelly_pct": 0.0, "action": "..."}],
  "best_value": null,
  "warnings": ["..."],
  "data_sources": {"history_home": {"source": "...", "mode": "mock", "fetched_at": null}}
}
```

#### B) Executive Summary (max. 8 Bullets)
Pick + Stake-Level + Begründung · Top-3-Faktoren (größtes `|w·(home−away)|`) ·
Hidden-Risk · Confidence-Ampel 🟢/🟡/🔴.

#### C) Faktor-Tornado · D) Score-Heatmap · E) Edge-Tabelle
Werden vom Report (Text) bzw. `--charts` (PNG: `*_tornado.png`, `*_heatmap.png`)
erzeugt.

---

## 🔄 ABLAUF-CHECKLISTE

```text
[ ] Phase 1: Daten-Fan-out (oder Recherche) → FactorContext gefüllt
[ ] Phase 2: 20 Faktoren berechnet (fehlende → available=false)
[ ] Phase 3: Sentiment-Payload (mind. VADER + TextBlob) oder neutral
[ ] Phase 4: 3-Modell-Blend + Bootstrap (n=500)
[ ] Phase 5: Calibration (oder roher Output + Transfer-Hinweis)
[ ] Phase 6: De-Vigging + Edge + Kelly
[ ] Phase 7: Sanity-Checklist
[ ] Phase 8: JSON + Summary + Tornado + Heatmap + Edge-Tabelle
```

**Scheitert ein Schritt** (z.B. Wetter-API down): explizit nennen, Faktor
neutralisieren, mit Re-Normalisierung weiterrechnen — niemals fingieren.

---

## 🚀 SPIEL STARTEN

```bash
# Mock (offline, ohne Keys) — sofort lauffähig:
python -m wm2026.cli predict --match config/matches/group_a/cze_vs_rsa.yaml \
  --odds "2.10/3.40/3.20" --out reports/ --charts

# Live (echte Daten) — .env mit USE_MOCK_*=false + Keys:
python -m wm2026.cli predict --home Germany --away Brazil --stage Final \
  --kickoff 2026-07-19T19:00:00Z --venue "MetLife Stadium" \
  --odds "2.40/3.20/2.90" --mode live --out reports/
```

> ⚠️ **Disclaimer:** Forschungs-/Bildungszweck, **keine** Wett-Empfehlung.
> Mock-Daten sind illustrativ.
