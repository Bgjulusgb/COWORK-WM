# CLAUDE.md — Agent & Developer Guide

Guidance for Claude Code (and humans) working in this repo. Keep changes
**pure, tested, and additive**; the offline mock path must always stay green.

## What this is
A calibrated WM-2026 football prediction workflow. The thin orchestration layer
`wm2026/` stitches the existing `data_sources/` · `factors/` · `analysis/` ·
`models_ml/` modules into one reproducible command. A fresh clone runs a full
prediction **offline** (mock data, no API keys) with only `requirements.txt`.

## Run & test
```bash
pip install -r requirements.txt
# Default is --mode live (real internet data). Use --mode mock for offline/repro:
python -m wm2026.cli predict --mode mock \
  --match config/matches/group_a/cze_vs_rsa.yaml --odds "2.10/3.40/3.20" --odds-ou "1.85/1.95"
python debug.py            # exercise every function on mock data (✅/❌ + summary)
pytest tests/test_wm2026_pipeline.py tests/test_markets.py tests/test_edge_conservative.py \
       tests/test_backtesting_rps.py tests/test_bivariate_poisson.py tests/test_calibration_offline.py -q
```
> **Live is the default; tests/CI/`debug.py` pin `--mode mock`.** In live mode the
> connectors fan out concurrently and degrade per-source to mock on failure; the
> pipeline then emits `claude_tasks` (the **Cowork-Auftrag** — the live-data gaps
> Claude must research via web search and feed back via the match YAML / `--odds*`
> / `--sentiment-json`). See `_claude_tasks` in `wm2026/pipeline.py`.
> **Tests run on _bare_ pytest by design.** `test_wm2026_pipeline.py` uses
> `asyncio.run()` directly. Do **not** add `pytest-asyncio` to satisfy the older
> `@pytest.mark.asyncio` factor tests — it breaks the bare-pytest suites that CI
> relies on. CI installs only `requirements.txt` + `pytest`.

## Architecture (data flow)
```
config YAML ─▶ wm2026.context.build_context ─▶ FactorContext
   Phase 1  data_sources/orchestrator.py   (parallel fan-out, mock|live|cache|error)
   Phase 2  factors/registry.py            20 factors → FactorSignal{home,away,weight,conf}
   Phase 3  factors/sentiment_factor.py    optional sentiment_payload
   Phase 4  analysis/factor_ensemble.py    signals → λ-multipliers
            analysis/match_predictor.py    base xG · λ-mult → λ_home/λ_away
            models_ml/poisson_goals.py      3 goal models → blended markets + bootstrap CIs
   Phase 5  analysis/calibration.py        isotonic/Platt (graceful — raw if no artifact)
   Phase 6  wm2026/edge.py                 de-vig, edge, Kelly, conservative p5-Kelly
            wm2026/markets.py              derived markets from the blended score matrix
   Phase 7  wm2026/pipeline.py _validate   sanity checklist → warnings
   Phase 8  wm2026/report.py               JSON + Markdown (+ wm2026/viz.py charts)
```
Entry point: `wm2026/pipeline.py → run_prediction()`. Output schema is documented
in `prompts/WM2026_MASTER_PROMPT.md` (Phase 8).

## The math layer (`wm2026/markets.py`, `models_ml/poisson_goals.py`)
- `M[i][j] = P(home i, away j)` is the **score matrix**. Every derived market
  (1X2, totals, Asian handicap, clean sheet …) is a **linear functional** of `M`.
- Derived markets + the heatmap use the **blended** matrix
  `blend_score_matrix(models, λ_home, λ_away)` = `Σ wₘ·Mₘ`, so they stay exactly
  consistent with the blended headline numbers. **Reuse this — don't re-derive
  markets from a single model's matrix.** Weights come from
  `resolve_blend_weights(model_names)` so opt-in models (e.g. bivariate)
  renormalise without touching the call sites.
- Asian-handicap / quarter lines: half-win/half-push settlement, see the
  `wm2026.markets` module docstring. The "quarter = average of its two
  neighbours" identity is the canonical correctness test.
- Conservative staking (`wm2026/edge.py`): edge & half-Kelly recomputed on the
  bootstrap **p5** (lower bound). Complement selections (Under/No) use `1 − p95`.
  Phase 4: **Double-Chance** (`dc_1x`/`dc_12`/`dc_x2` are accumulated per
  sample inside `bootstrap_markets` so the dependency is captured exactly:
  `dc_12 ≡ 1 − draw`) and **Asian-Handicap** (via `bootstrap_blend_metrics`,
  one resampling pass over arbitrary matrix functionals on the blended matrix)
  now carry the same p5 columns. `best_value_cons_pick()` exposes the
  honest pick (max p5 edge > 0); `--bankroll` annotates stake amounts that
  zero out when the conservative edge fails.
- Calibration (`analysis/calibration.py`, Phase 5) fits **without sklearn** —
  pure-Python PAV isotonic + Newton Platt fallbacks. Three modes via
  `--calibrate`: `auto` (fitted artifact if present, else raw), `market`
  (`market_anchor` → shrink 1X2 to the vig-free consensus, the per-match path),
  `none`. Fit an artifact offline with `scripts/fit_calibration_offline.py`.
- Phase 4 toggles (all opt-in; defaults keep the historical output identical):
  - `settings.include_bivariate` (`INCLUDE_BIVARIATE`) — adds Karlis-Ntzoufras
    BivariatePoisson as a 4th blend model.
  - `settings.lambda_aggregation` (`LAMBDA_AGGREGATION=geom`) — log-linear
    pooling of factor tilts (symmetric in home/away under inversion).
  - CLI: `--live-sources weather,clubelo` / `--mock-sources rss` for per-source
    live/mock control without `.env` editing.

## Conventions
- **No new runtime deps** for core features — `numpy`/`scipy` are the ceiling.
  Optional libs (sklearn, statsmodels, matplotlib) must degrade gracefully.
- A failed data source **never raises into the factor layer** — it returns
  `error`, the factor self-disables (`available=False`), the ensemble renormalises.
- New markets/metrics ship **with a test** (an invariant or a reference value)
  and extend the JSON schema **additively** (bump `schema_version`).
- German + English comments coexist; match the surrounding file's style.

## Where things live
| Need | File |
|---|---|
| Add a derived market | `wm2026/markets.py` (+ `tests/test_markets.py`) |
| Edge / Kelly / staking | `wm2026/edge.py` (+ `tests/test_edge_conservative.py`) |
| Goal models / blend / bootstrap | `models_ml/poisson_goals.py` |
| MLE attack/defence λ-estimator | `analysis/xg_estimator.py` (gated by `settings.use_mle_xg`) |
| Tournament Monte-Carlo | `wm2026/tournament.py` (+ `wm2026 tournament` CLI) |
| Accuracy metrics (Brier/LogLoss/RPS) | `analysis/backtesting.py` |
| Calibration (isotonic/Platt/market) | `analysis/calibration.py` (+ `scripts/fit_calibration_offline.py`) |
| A new factor | `factors/<name>_factor.py` + `factors/registry.py` + a weight in `config/settings.py` |
| CLI flags / `research` subcommand | `wm2026/cli.py` |
| Cowork overrides injection | `wm2026/context.py` (`apply_overrides`, `overrides_template`) |
| Report JSON/Markdown / HTML | `wm2026/report.py` · `wm2026/report_html.py` |
| Roadmap / next math | `verbesserungsplan.md` |

## Don't
- Don't commit a real `.env` (only `.env.example`); rotate any leaked key.
- Don't break the mock path — it's the contract that the repo runs out of the box.
- Don't present a prediction without its confidence interval, or an edge > 10 %
  without a sanity-check note.

## Cowork-Layer (`.claude/`)

The repo ships a complete Claude Code on the Web Cowork integration —
**setup-free**, idempotent, additive:

```
.claude/
  settings.json                # registers SessionStart hook + permissions allowlist
  hooks/
    session-start.sh           # idempotent bootstrap: pip install core+[viz,sentiment,stats], verify, smoke-test
  skills/                      # 11 specialized SKILL.md files
    predict-match/             #   ⭐ run the full 8-phase pipeline, interpret report
    research-fixture/          #   Cowork-Auftrag loop: Web Search → overrides.json
    read-report/               #   parse JSON, build UI-ready briefing
    analyze-edge/              #   3-stage filter (sanity → p5 → confidence), Kelly stake
    inspect-data/              #   token-budget tools (--compact, summary, --charts-external)
    compare-runs/              #   A/B diff two reports (overrides / calibration / bivariate / bankroll)
    tournament-sim/            #   10k tournament Monte-Carlo
    calibrate-offline/         #   fit isotonic/Platt artifacts from history CSV
    tune-models/               #   RPS-based Optuna blend-weight tuning
    list-fixtures/             #   browse the 104 match YAMLs
    cowork-setup/              #   first-time onboarding, debugging
  agents/
    wm-quant-analyst.md        # opus agent that orchestrates end-to-end (proactive)
```

The hook sets `.claude/.bootstrapped` on success → next session skips reinstall.
Force re-run with `WM2026_FORCE_BOOTSTRAP=1`. Setup walkthrough lives in
`SETUP_COWORK.md`.
