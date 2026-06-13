# 📈 Verbesserungsplan — Mathematik & Cowork

Priorisierte Roadmap für die Vorhersage-Qualität. **Phase 1 ist umgesetzt**
(siehe „erledigt"); Phase 2/3 sind auf Wiederverwendung der bestehenden
Module ausgelegt — jede Position nennt Formel, Begründung und Ziel-Datei.

> Leitprinzip: jede Verbesserung bleibt **pure & getestet**, degradiert sauber
> ohne optionale Deps und ändert das JSON-Schema nur additiv.

---

## ✅ Phase 1 — erledigt (diese Iteration)

| # | Verbesserung | Mathematik | Datei |
|---|---|---|---|
| 1 | **Abgeleitete Märkte** aus der Score-Matrix | lineare Funktionale von `M[i][j]` | `wm2026/markets.py` |
| 2 | **Asian Handicap** inkl. Viertellinien | Half-Win/Half-Push-Settlement (`adj = (i−j)+line`) | `wm2026/markets.py` |
| 3 | **Double Chance · DNB · Team-Totals · Clean Sheet · Win-to-Nil · Odd/Even** | Teilsummen der Matrix | `wm2026/markets.py` |
| 4 | **Blended Score-Matrix** (alle 3 Modelle) für Heatmap + Derived Markets | `M̄ = Σ wₘ·Mₘ` — Märkte bleiben exakt blend-konsistent | `models_ml/poisson_goals.py` |
| 5 | **RPS** (Ranked Probability Score) im Backtesting | `RPS = 1/(r−1)·Σ(CPₖ−CYₖ)²`, geordnet Home>Draw>Away | `analysis/backtesting.py` |
| 6 | **Conservative Kelly** (p5-Edge) | Edge/Kelly auf der Bootstrap-Untergrenze statt p50 | `wm2026/edge.py` |

**Wirkung:** Das Markt-Board deckt jetzt die wichtigsten Wett-Märkte ab; die
`(p5)`-Spalten zeigen ehrlich, welche „Value"-Picks die eigene Modellunsicherheit
**nicht** überleben (z.B. eine 1X2-Home-Edge von +13.5 % fällt auf −12.8 % @ p5).

---

## 🔜 Phase 2 — Modell-Tiefe (größter erwarteter Brier/RPS-Gewinn)

### 2.1 Dixon-Coles-MLE-λ-Schätzer mit Zeitdecay ξ — ✅ erledigt
**Status:** `analysis/xg_estimator.py` schätzt Attack/Defence + Heimvorteil per
gewichteter **MLE** (`scipy.optimize`) aus der Historie, Gewicht
`w = exp(−ξ·Δt_days)·tier` (ξ=0.0065 ≈ ~2-Jahre-Halbwertszeit, Dixon-Coles 1997).
Identifizierbar via Sum-to-Zero (Σattack=Σdefence=0), ρ fix. Ersetzt das naive
`_base_xg` **nur** wenn genug identifizierbare Historie da ist, sonst harter
YAML-Fallback. Gated: `settings.use_mle_xg=False` (Default → Output unverändert).
Getestet (`tests/test_xg_estimator.py`): Recovery aus Synthetik, Sum-to-Zero,
Decay-Monotonie, Fallback, Default-Stabilität.

### 2.2 Echtes bivariates Poisson als 4. Modell — ✅ erledigt (opt-in im Blend)
**Status:** `BivariatePoisson` (Karlis-Ntzoufras, `λ₃`-Kovarianz) ist in
`models_ml/poisson_goals.py` umgesetzt + getestet (`tests/test_bivariate_poisson.py`):
`X=Y₁+Y₃, Y=Y₂+Y₃`, `Yᵢ~Poisson(λᵢ)`, `Cov=λ₃`. `λ₃` wird aus dem xG
herausgelöst (`λ₁=home−λ₃`, `λ₂=away−λ₃`) → **Marginal-Means bleiben exakt**,
nur die Korrelation (mehr Unentschieden) steigt; `λ₃→0` ⇒ unabhängiges Poisson.
Über `build_goal_model("bivariate")` wählbar.
**Phase-4-Update:** jetzt auch als **4. Blend-Modell** aktivierbar via
`settings.include_bivariate=True` (Env `INCLUDE_BIVARIATE=true`) —
`resolve_blend_weights()` schaltet dann auf `BLEND_WEIGHTS_WITH_BIVARIATE`
(0.34/0.25/0.25/0.16, renormalisiert) um. Default bleibt **off** → die
Standard-Outputs sind unverändert (Default-Stabilitäts-Contract).

### 2.3 Kalibrierung aktiviert — ✅ erledigt (zwei Wege, beide ohne sklearn)
**Problem war:** `analysis/calibration.py` fittete nur aus der DB *und* brauchte
sklearn; ein frischer Clone hatte nie ein Artefakt → Phase 5 lief leer.
**Jetzt:**
1. **Pure-Python-Fit (kern-deps):** PAV-Isotonic (`_pav`/`_isotonic_pav_curve`)
   + Newton/IRLS-Platt (`_platt_newton`) als sklearn-freie Fallbacks →
   `fit_calibrators` liefert echte Kurven mit nur numpy/scipy.
2. **Offline-Fit-Script:** `scripts/fit_calibration_offline.py` liest eine CSV
   `(home_win_prob,draw_prob,away_win_prob,home_score,away_score)` aus einem
   *berühmten Prior-Set* (WC2022 + EURO2024 + Copa2024), fittet, schreibt die
   Artefakte und zeigt den Brier-vorher/nachher. Danach greift `--calibrate auto`.
3. **Markt-Anker (pro Spiel, ohne Historie):** `calibration.market_anchor` zieht
   die 1X2 zur vig-freien Markt-Konsens-Quote — dem kanonisch *gut kalibrierten*
   Fußball-Forecaster (Constantinou & Fenton 2013). `--calibrate market`. Das ist
   der „pro Spiel im Claude-Workflow"-Pfad: Claude recherchiert die Quoten, das
   Modell kalibriert dagegen. Kompoundet mit dem Markt-Faktor (Doku-Hinweis).

**Famous-Referenzwerte (verankert):** Markt = bester kalibrierter Forecaster
(Constantinou & Fenton 2013); Dixon-Coles-Zeitdecay `ξ=0.0065` (Original, in
Halb-Wochen) ≈ 107-Tage-Halbwertszeit für 2.1.

### 2.4 Blend-Gewichte + ρ via Optuna tunen
**Problem:** `DEFAULT_BLEND_WEIGHTS` (0.4/0.3/0.3) und ρ=0.1 sind gesetzt, nicht
optimiert (Optuna tunt heute nur Faktor-Gewichte). **Lösung:** Zielmetrik
**RPS** (2.+Phase-1-#5) über das Backtest-Set; `analysis/weight_optimizer.py`
um einen `tune_goal_model_params()`-Pfad erweitern → `runtime_*`-Artefakt.

### 2.5 Geometrische λ-Aggregation — ✅ erledigt (Phase 4, opt-in)
**Status:** `FactorEnsemble(aggregation="geom")` mittelt die Tilt-Faktoren
log-linear: `λ_mult = exp(Σ wᵢ·ln sᵢ)` (Strengths geclamped auf ≥ 0.05 für den
Log). Geometrisch ist für multiplikative Tilts konsistent und **symmetrisch in
home/away** (Kehrwert-Invarianz: vertauschte Strengths ⇒ exakt invertierte
Multiplikatoren — als Test verankert). Default bleibt `"arith"`
(`settings.lambda_aggregation`, Env `LAMBDA_AGGREGATION=geom`) → Default-Output
unverändert. Getestet in `tests/test_factor_aggregation.py`.

---

## 🌍 Phase 3 — Reichweite & Cowork

- **Turnier-Monte-Carlo** — ✅ erledigt: `wm2026/tournament.py` + `wm2026 tournament`
  sampelt Gruppenphase → KO über die blend-konsistente Score-Matrix (gebackene
  CDFs + vektorisierte pmf) → **10k Sims des 48-Team-Felds in ~1,5 s**, Titel-/
  Finale-/Achtelfinal-% je Team. WC-2026-Format (12 Gruppen → 8 beste Dritte → 32).
- **HT/FT & First-Goal & Winning-Margin & Exact-Totals** — ✅ erledigt in
  `wm2026/markets.py` (Phase-1-Commit), inkl. Halbzeit-λ-Split `ht_lambda_share=0.45`.
- **Cowork-Overrides** — ✅ erledigt: `--overrides-json` + `wm2026 research`
  (`wm2026/context.py:apply_overrides`).
- **HTML-Report** — ✅ erledigt: `wm2026/report_html.py` + `--format html`.
- **Per-Quelle-Live-Toggle — ✅ erledigt (Phase 4):** `--live-sources weather,clubelo`
  (nur diese live, Rest mock; impliziert `--mode live`) und
  `--mock-sources reddit` (nur diese mock) in `wm2026/cli.py`.
- **Offen — Karten/Ecken:** eigenes (Negbin-)Zählmodell mit Referee-/Derby-Faktor
  (braucht sofascore-Stats).

---

## 🛠️ Phase 4 — Backend-Härtung & ehrliches Staking (autonome Session 2026-06-11)

Selbst-Audit des ersten Live-Durchlaufs (KOR vs CZE) ergab konkrete Lücken.
Jede Position: Befund → Fix → Test. Schema-Bump **1.2 → 1.3** (additiv).

### 4.1 🐞 Double-Chance-Edges ohne p5-Guard (BUGFIX, kritisch)
**Befund:** `compute_edges` lieferte für DC-Zeilen `edge_pct_cons=None`, aber
`action="standard"` rein aus der p50-Edge (KOR vs CZE: „DC 12 +9.94 % →
standard" ohne Konservativ-Check) — genau das Loch, das die p5-Regel schließen
soll. **Fix:** `bootstrap_markets` akkumuliert jetzt **pro Sample**
`dc_1x = P(H)+P(D)`, `dc_12`, `dc_x2` (Summen *innerhalb* des Samples ⇒
Korrelation korrekt erfasst, nicht Quantile addiert). Die Keys fließen durch
die bestehende Blended-CI-Aggregation; `compute_edges` verdrahtet sie via
`_lower(ci, "dc_*")`. **Invariante als Test:** `dc_12 ≡ 1 − draw` pro Sample ⇒
`p5(dc_12) = 1 − p95(draw)` exakt.

### 4.2 🐞 Asian-Handicap-Edges ohne Konservativ-Spalte (BUGFIX)
**Befund:** `evaluate_asian_handicap` hatte keine `model_p_lower`-Quelle — AH-
Empfehlungen (die meist-unterschätzten Märkte) liefen ohne p5-Disziplin.
**Fix:** neues `bootstrap_blend_metrics(models, λh, λa, fns)` in
`models_ml/poisson_goals.py` — sampelt λ-Paare **einmal**, baut pro Sample die
geblendete Matrix und wertet beliebige Metrik-Callables aus (mathematisch
sauberer als Quantil-Mittelung: bootstrappt den Blend direkt). Pipeline reicht
`home/away_prob_nopush`-p5 an `evaluate_asian_handicap(..., home_p_lower=,
away_p_lower=)` weiter → cons-Spalten wie bei 1X2.

### 4.3 `best_value_cons` — der ehrliche Pick (additiv)
**Befund:** `best_value` maximiert die rohe p50-Edge ⇒ zeigt systematisch den
Sanity-Check-Kandidaten statt des tradebaren Picks. **Fix:**
`best_value_cons_pick()` = höchste **p5-Edge > 0**; Report trägt beide Felder
(`best_value` für Transparenz, `best_value_cons` für die Empfehlung).

### 4.4 Bankroll-bewusstes Staking (`--bankroll`)
`--bankroll 1000` annotiert jede Edge-Zeile mit `stake_half_kelly` (p50) und
`stake_cons` (p5, **0 wenn die konservative Edge ≤ 0**) in Währungseinheiten;
Report-JSON trägt `bankroll`. Die ½-Kelly-auf-p5-Disziplin aus den Skills wird
damit direkt im Backend ausgewiesen statt nur im Briefing nachgerechnet.

### 4.5 Geometrische λ-Aggregation (= 2.5) + Bivariate im Blend (= 2.2)
Siehe oben — beides opt-in, Default-Stabilität getestet.

### 4.6 Per-Quelle-Live-Toggles (= Phase-3-Offen-Punkt)
`--live-sources` / `--mock-sources` — präzise Cowork-Runs („nur Wetter+Elo
live, Rest mock") ohne `.env`-Editing.

### Verifikation Phase 4
```bash
pytest tests/test_edge_conservative.py tests/test_factor_aggregation.py \
       tests/test_wm2026_pipeline.py tests/test_bivariate_poisson.py -q
python -m wm2026.cli predict --mode mock --match config/matches/group_a/kor_vs_cze.yaml \
  --odds "2.60/3.05/3.00" --odds-dc "1.30/1.45/1.55" --odds-ah=-0.25:1.92/1.98 \
  --bankroll 1000 --calibrate market
```

---

## Verifikation jeder Position
```bash
pytest tests/test_markets.py tests/test_edge_conservative.py \
       tests/test_backtesting_rps.py tests/test_wm2026_pipeline.py -q
python -m wm2026.cli predict --match config/matches/group_a/cze_vs_rsa.yaml \
  --odds "2.10/3.40/3.20" --odds-dc "1.28/1.30/1.55" --odds-ah=-0.5:1.95/1.95
```
Neue Mathematik kommt immer **mit Test** (Invariante oder Referenzwert) und hält
das bestehende Schema additiv.
