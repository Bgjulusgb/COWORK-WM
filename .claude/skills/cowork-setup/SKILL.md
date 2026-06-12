---
name: cowork-setup
description: First-time setup of the WM-2026 Cowork workspace — verify dependencies, configure the SessionStart hook, set up optional API keys, check the skill registry, run the bootstrap smoke test. Use when the user says "setup the project", "install everything", "is everything ready?", or starts a fresh container/session.
---

# Cowork-Setup — Repo von Null auf "predict ready"

Ein frischer Container/Session braucht ein paar Sekunden Bootstrap. Diese
Skill orchestriert das **deterministisch und idempotent**.

## 1. Bootstrap (was der SessionStart-Hook automatisch macht)

Wenn die Sitzung startet, führt `.claude/hooks/session-start.sh` das hier aus:
1. `python3 -m pip install -r requirements.txt` (core)
2. `python3 -m pip install '.[viz,sentiment,stats,test]'` (extras: matplotlib,
   vaderSentiment, statsmodels, sklearn, pytest)
3. Verify imports (core + extras)
4. **Smoke test:** `wm2026.cli predict --mode mock` → muss "match_id" liefern
5. Marker setzen: `.claude/.bootstrapped`

→ Wenn `.bootstrapped` da ist, läuft der Hook beim nächsten Start nicht erneut.

## 2. Manueller Verify

```bash
# Hook nochmal erzwingen (für Debugging)
WM2026_FORCE_BOOTSTRAP=1 bash .claude/hooks/session-start.sh

# Oder einfach prüfen ob alles geht
python debug.py                    # 63 Mock-Funktions-Checks, alles ✅
pytest tests/test_wm2026_pipeline.py tests/test_markets.py -q
python -m wm2026.cli list          # listet 104 Match-Configs
```

## 3. Skills-Inventar

Die Sitzung hat diese Skills geladen:

| Skill | Wofür |
|---|---|
| `predict-match` | Pipeline starten, Cowork-Loop, Briefing |
| `research-fixture` | Live-Daten-Lücken via Web Search recherchieren |
| `read-report` | JSON-Report parsen + UI-Briefing |
| `analyze-edge` | Edge / p5 / Kelly Tiefenanalyse |
| `tournament-sim` | 10 000 Sims, Title%, Advance% |
| `calibrate-offline` | History-Fit für `--calibrate auto` |
| `tune-models` | RPS-Tuning der Blend-Gewichte |
| `list-fixtures` | 104 Match-Configs browsen |
| `cowork-setup` | das hier — first-time onboarding |

## 4. Optionale API-Keys (Live-Modus)

`.env.example` kopieren und Keys eintragen — **alle optional**:

```bash
cp .env.example .env
# Editiere .env und setze die Keys, die du hast.
```

| Key | Wofür | Wo holen |
|---|---|---|
| `ODDS_API_KEY` | Live-Quoten (1X2, O/U, BTTS) | the-odds-api.com (free tier) |
| `FOOTBALL_DATA_API_KEY` | Spielplan-Cross-Check | football-data.org (free) |
| `REDDIT_CLIENT_ID`+`SECRET` | Reddit-Stimmung | reddit.com/prefs/apps |
| `NVIDIA_API_KEY` | LLM-Aspekt-Sentiment | build.nvidia.com |
| `OPENWEATHER_API_KEY` | Wetter (sonst RSS-Fallback) | openweathermap.org |

**Ohne Keys** läuft alles trotzdem — degradiert pro Quelle automatisch auf Mock,
und der Cowork-Auftrag fordert dich auf, die Lücken per Web Search zu füllen.

## 5. Claude-Seite (Web-/Desktop-Toggles)

In claude.ai / Claude Code:

| Feature | Empfehlung |
|---|---|
| **Web Search** | ☑️ Aktivieren — Pflicht für Live-Quoten / Lineups |
| **Code Execution** | ☑️ Aktivieren — für `wm2026 predict` |
| **File Creation** | ☑️ Aktivieren — `overrides.json` schreiben |
| **Memory / Past Chats** | optional — verbessert Match-Kontext |
| **MCP — Filesystem** | ☑️ wenn HTML-Report im Browser geöffnet werden soll |

## 6. Permissions (in `.claude/settings.json`)

Vorgenehmigt sind:
- Alle `wm2026 …` und `python -m wm2026.cli …` Calls
- `python debug.py`, `pytest`, `pip install -r requirements.txt`
- Web-Fetches auf bekannte Quoten-/Stats-Domains
- Read/Glob/Grep über das Repo

**Gesperrt** sind:
- `Write(.env)` — keine echte API-Key-Datei schreiben
- `rm -rf`, `git push --force`, `git reset --hard` — destruktiv

## 7. Quick-Sanity (für den User)

```
✅ Hook lief durch?       → .claude/.bootstrapped existiert
✅ Pipeline ready?         → python -m wm2026.cli predict --mode mock --home A --away B
✅ Skills geladen?         → 9 Skills sichtbar in der Sitzung
✅ Optional .env?          → ls .env  (oder ignorierbar)
✅ Reports-Ordner?         → mkdir -p reports/
```

## 8. Häufige Probleme

| Problem | Lösung |
|---|---|
| `ModuleNotFoundError: numpy` | Hook lief nicht; manuell `pip install -r requirements.txt` |
| `No module named matplotlib` (bei `--charts`) | `pip install '.[viz]'` |
| `--charts` schweigt | matplotlib-Backend prüfen; `MPLBACKEND=Agg` setzen für Headless |
| Pipeline hängt > 60 s | wahrscheinlich Live-Mode + langsame Quelle; `--mode mock` zum Testen |
| Live-Mode liefert nur Mock | keine Keys in `.env` → erwartet; Cowork-Auftrag abarbeiten |
| Schema-Mismatch im Report | `--bootstrap 0` für schnellen Schema-Check |

## 9. Repo-Map (wo finde ich was?)

```
wm2026/                  # Orchestrierungs-Layer (Pipeline, CLI, Edge, Markets)
  pipeline.py            #   ← die 8-Phasen-Engine
  cli.py                 #   ← predict, tournament, research, list
  markets.py             #   ← 15+ derived markets
  edge.py                #   ← edge, Kelly, p5-conservative
factors/                 # 20 factor heads (elo, form, h2h, sentiment, …)
models_ml/poisson_goals.py # 4 goal models (DC, NegBin, GLM, BiPoisson) + blend
data_sources/            # 13 connectors (mock|live|cache|error)
analysis/                # match_predictor, calibration, backtesting, xg_estimator
config/                  # settings + 104 match YAMLs + group structure
scripts/                 # fit_calibration_offline, tune_models_offline, ...
.claude/                 # hooks + skills + settings (dieser Cowork-Layer)
docs/examples/           # Referenz-Reports (Markdown + HTML + JSON + Charts)
```

Mehr Lesen: `CLAUDE.md` · `prompt.md` · `prompts/WM2026_MASTER_PROMPT.md` ·
`verbesserungsplan.md` · `SETUP_COWORK.md`.
