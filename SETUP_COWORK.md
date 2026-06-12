# 🤝 Cowork-Setup — WM-2026 Quant-Analyst-Workflow

**TL;DR** — Container starten → SessionStart-Hook installiert alles → Skills sind
live → Match-Frage stellen → kalibrierte Prognose mit p5-Edge zurückbekommen.
**Kein manuelles `pip install`, keine ad-hoc Bash-Sessions, keine vergessenen
Optional-Extras.**

---

## 0. Was ist neu vs. der "1. Durchlauf"

| Vorher (1. Run) | Jetzt |
|---|---|
| ❌ `pip install` manuell, oft vergessen | ✅ SessionStart-Hook installiert core + viz + sentiment + stats automatisch |
| ❌ Nur 1 Skill (`predict-match`) | ✅ 9 spezialisierte Skills (siehe unten) |
| ❌ Cowork-Auftrag ad-hoc in 5 Tasks | ✅ Dedizierter `research-fixture`-Skill mit overrides.json-Template |
| ❌ Edge-Interpretation flüchtig | ✅ `analyze-edge`-Skill mit 3-Stufen-Filter (Sanity → p5 → Konfidenz) |
| ❌ Math-Schichten halb genutzt | ✅ `predict-match` aktiviert MLE-xG, Markt-Anker, Bootstrap-CIs konsequent |
| ❌ Kein Smoke-Test → spät bemerkt, dass was kaputt war | ✅ Hook smokt einmal beim Start (`predict --mode mock`) |
| ❌ Charts nicht eingebettet (matplotlib fehlte) | ✅ Hook installiert `[viz]` → `--charts` produziert PNG + HTML mit eingebetteten Bildern |
| ❌ Bankroll/Kelly-Empfehlung im Briefing fehlte | ✅ `analyze-edge` gibt ½-Kelly auf p5 + Stake-Level (Pass/Token/Standard) |

---

## 1. Was passiert beim Sitzungsstart

Der **SessionStart-Hook** (`.claude/hooks/session-start.sh`) läuft einmal pro
Container und macht:

```
[wm2026-hook] python 3.12
[wm2026-hook] installiere core deps (numpy/scipy/httpx/pydantic/PyYAML/structlog)…
[wm2026-hook] installiere optional extras: viz, sentiment, stats, pytest…
[wm2026-hook] verify imports …
  core ok: 6/6  · fehlt: –
  extras:  6/6  · fehlt: – (alle optional)
  wm2026.pipeline.run_prediction: ok
[wm2026-hook] smoke test (mock predict)…
[wm2026-hook] smoke ok — Pipeline ist betriebsbereit.
[wm2026-hook] Bootstrap abgeschlossen. Skills bereit: predict-match · research-fixture · …
```

Wenn die Sitzung neu startet und `.claude/.bootstrapped` existiert, springt
der Hook sofort weg ("bereits installiert").

**Forcen** (Debug / nach Repo-Update):
```bash
WM2026_FORCE_BOOTSTRAP=1 bash .claude/hooks/session-start.sh
```

---

## 2. Die 11 Skills — was wann

| Skill | Wann triggern? |
|---|---|
| **`cowork-setup`** | First-time-Onboarding, "ist alles installiert?", "was kann das hier?" |
| **`list-fixtures`** | "Welche Spiele gibt es?", "wo liegt das YAML für DEU vs BRA?" |
| **`predict-match`** ⭐ | "Prognostiziere Spiel X", "Edge auf 1X2", "wer gewinnt KOR vs CZE?" |
| **`research-fixture`** | "Cowork-Auftrag abarbeiten", "Live-Quoten holen", "overrides.json bauen" |
| **`read-report`** | "Lies den Report", "fasse das JSON zusammen" |
| **`analyze-edge`** | "Wo ist der echte Wert?", "überlebt der Edge p5?", "was soll ich tippen?" |
| **`inspect-data`** | "Der JSON ist zu groß", "wie spare ich Tokens?", `--compact` / `summary` / `--charts-external` |
| **`compare-runs`** | "Wie viel hat das Override bewegt?", "Bivariate vs default", "vor/nach Kalibrierung" |
| **`tournament-sim`** | "Wer wird Weltmeister?", "Achtelfinal-Chance Gruppe X?" |
| **`calibrate-offline`** | "Ich habe ein historisches CSV", "kalibriere besser" |
| **`tune-models`** | "Optimiere Blend-Gewichte", "RPS-Tuning" |

Skills sind in **`.claude/skills/<name>/SKILL.md`** und werden von Claude Code
**automatisch erkannt**, sobald die Beschreibung zum User-Intent passt.

---

## 3. Sub-Agent: `wm-quant-analyst`

In `.claude/agents/wm-quant-analyst.md` wartet ein dedizierter **Opus**-Agent,
der das **End-to-End** macht: Setup-Check → Match-Resolve → Recherche → Pipeline
→ Cowork-Loop (max 2 Iterationen) → Report-Lesen → Edge-Analyse → Briefing.

Aktivieren (in Claude Code):
```
Use the wm-quant-analyst agent to predict <HOME> vs <AWAY>
```
Oder einfach: `@wm-quant-analyst` + Match-Beschreibung — die Skill-Beschreibung
sagt dem Agenten, dass er ihn proaktiv für WM-2026-Anfragen nutzen soll.

---

## 4. Plug-and-play Permissions (`.claude/settings.json`)

Vorgenehmigt (kein Permission-Prompt mehr):
- `wm2026 …` / `python -m wm2026.cli …` — alle Subcommands
- `python debug.py`, `pytest …`
- `pip install -r requirements.txt`, `pip install .[viz,sentiment,stats,test]`
- `WebSearch` allgemein
- `WebFetch` für ~20 Quoten-, Stats-, Wetter-Domains
- `Read`, `Glob`, `Grep` für das ganze Repo

Gesperrt:
- `Write(.env)` — keine echten Keys committen
- `rm -rf *`, `git push --force`, `git reset --hard` — destruktiv

---

## 5. Der ideale User-Prompt

```
Predict Südkorea vs Tschechien (Group A, 2026-06-12 04:00 MEZ).
Falls Quoten gerade verfügbar, frisch holen. Disclaimer am Ende.
```

Der Quant-Analyst-Agent:
1. Sucht das YAML (`config/matches/group_a/kor_vs_cze.yaml`)
2. Holt Live-Quoten + Lineups + Wetter via Web Search
3. Schreibt `reports/<match_id>.overrides.json`
4. Fährt `wm2026 predict --calibrate market --format html --out reports/ --charts`
5. Liest den JSON-Report, sortiert die Edge-Tabelle nach `edge_pct_cons`
6. Wendet den 3-Stufen-Filter an (Sanity > 10 % / p5-Test / Konfidenz-Ampel)
7. Liefert das Briefing inkl. Stake-Empfehlung + Disclaimer

---

## 6. Optionale Live-Modus-Keys (zero-config möglich)

Ohne API-Keys läuft alles trotzdem — Cowork-Auftrag fordert Web-Search-Werte
ein. Mit Keys ist es bequemer:

```bash
cp .env.example .env
# Editiere .env:
#   ODDS_API_KEY=…
#   FOOTBALL_DATA_API_KEY=…
#   REDDIT_CLIENT_ID=…
#   REDDIT_CLIENT_SECRET=…
#   OPENWEATHER_API_KEY=…
```

Free-Tier-Endpunkte reichen für ein Spiel pro Sitzung.

---

## 7. Math-Tiefe — was JETZT konsequent genutzt wird

Die Skills setzen explizit jede Schicht ein:

| Schicht | Aktivierung | Skill |
|---|---|---|
| **3-Modell-Blend** (Poisson · NegBin · GLM, blend-konsistent) | automatisch | `predict-match` |
| **Bivariates Poisson als 4. Blend-Modell** (Karlis-Ntzoufras λ₃) | `INCLUDE_BIVARIATE=true` | `predict-match`, `compare-runs` |
| **MLE-λ-Schätzer mit Zeitdecay ξ=0.0065** | `settings.use_mle_xg=True` | `predict-match` |
| **Geometrische λ-Aggregation** (log-linear, home/away-symmetrisch) | `LAMBDA_AGGREGATION=geom` | `predict-match` |
| **Bootstrap-CIs** (500 Sims, p5/p50/p95 — inkl. **DC** & **AH**) | `--bootstrap 500` (default) | alle |
| **Markt-Anker-Kalibrierung** (Constantinou & Fenton 2013) | `--calibrate market` | `predict-match` |
| **Isotonic + Platt Kalibrierung** (Pure-Python, ohne sklearn) | `--calibrate auto` | `calibrate-offline` |
| **Konservativer p5-Kelly** + `best_value_cons` (ehrlicher Pick) | automatisch in Phase 6 | `analyze-edge` |
| **Bankroll-Annotation** (`stake_half_kelly` / `stake_cons`) | `--bankroll 1000` | `analyze-edge` |
| **Per-Quelle Live/Mock-Toggle** | `--live-sources weather,clubelo` | `predict-match` |
| **Score-Heatmap + Faktor-Tornado** (PNG, embedded HTML) | `--charts` | `predict-match` |
| **RPS-Tuning der Blend-Gewichte** | manuell (Optuna) | `tune-models` |
| **10k Tournament-MC in 1.5 s** | `wm2026 tournament` | `tournament-sim` |

---

## 8. Debugging-Cheatsheet

```bash
# Hook neu erzwingen
WM2026_FORCE_BOOTSTRAP=1 bash .claude/hooks/session-start.sh

# Alle Funktionen einmal testen (mock, ~3 s)
python debug.py

# End-to-end Smoke
python -m wm2026.cli predict --mode mock --home A --away B --stage Group --format json | tail -1

# Tests
pytest tests/test_wm2026_pipeline.py tests/test_markets.py tests/test_edge_conservative.py -q

# Wenn `--charts` schweigt:
python -c "import matplotlib; print(matplotlib.__version__)"
MPLBACKEND=Agg python -m wm2026.cli predict --mode mock --home A --away B --charts --out reports/

# Welche Quellen sind degraded?
python -m wm2026.cli predict --mode live --home … --away … --format json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print('claude_tasks:', d.get('claude_tasks', []))"
```

---

## 9. Disclaimer

**Forschungs- und Bildungsprojekt — keine Wett-Empfehlung. Mock-Vorhersagen
sind illustrativ. Wette niemals mehr als 2 % deines Bankrolls auf eine
Selection. Closing Lines schlagen Edge.**
