---
name: research-fixture
description: Research live data for an upcoming WM-2026 fixture via Web Search — odds, lineups, injuries, weather, last-5 form/xG — and write an overrides.json that feeds the prediction pipeline. Use when the report's "🤝 Cowork-Auftrag (live data gaps)" section is non-empty, or before a first live run when no API keys are configured.
---

# Research-Fixture — Cowork-Auftrag abarbeiten

Wenn die `wm2026 predict --mode live`-Pipeline Lücken meldet (kein API-Key,
Quelle down), erzeugt sie eine priorisierte **`claude_tasks`-Liste**. Hier ist
der disziplinierte Loop, der jede Lücke schließt.

## 1. Template generieren

```bash
python -m wm2026.cli research \
  --home "<HOME>" --away "<AWAY>" --stage <STAGE> \
  --kickoff "<2026-06-12T21:00:00Z>" --venue "<City>" \
  --out reports/
```

Output:
- `reports/<match_id>.overrides.json` — leeres Template mit den Feldern, die
  du füllen sollst
- Markdown-Liste mit den priorisierten Cowork-Aufgaben

## 2. Recherchieren — strukturiert

**Jede Zahl bekommt drei Felder:** `value`, `source_url`, `fetched_at` (ISO8601).

| Slice | Was suchen | Bevorzugte Quellen |
|---|---|---|
| **Buchmacher-Quoten** (1X2, O/U 2.5, BTTS, DC, AH) | Decimal-Odds vom liquidesten Markt (Pinnacle > Bet365 > Tippmix-Konsens) | the-odds-api.com, oddsportal.com, sportsgambler.com |
| **Aufstellungen + Verletzungen + Sperren** | Predicted XI, Out-/Doubt-Listen | sportsmole.co.uk, aljazeera.com, transfermarkt.com, official federation pages |
| **Wetter am Anstoßort** | Temp °C, Wind km/h, Niederschlag %, Stadium altitude | accuweather.com, openweathermap.org, wikipedia (stadium) |
| **Letzte 5 Spiele** je Team | Resultat, xG-for, xG-against, Gegnerstärke (Elo/FIFA-Rank) | fbref.com, understat.com (xG), footystats.org, fotmob.com |
| **Elo-Ratings + FIFA-Rank** | Aktueller Wert (Tag des Anstoßes) | clubelo.com, eloratings.net, fifa.com/rank |
| **Reise/Rest-Tage** | Letzte Reise-Distanz + Tage seit letztem Match | wikipedia (Wettkampfkalender), team news |
| **Sentiment** (optional) | Reddit/Twitter-Polarity letzte 24-48h | reddit.com/r/soccer, r/<team> |

## 3. overrides.json ausfüllen

Schema (minimal — alles optional, leere Felder werden ignoriert):

```json
{
  "teams": {
    "home": {
      "avg_xg_season": 1.42,
      "avg_xg_conceded": 1.18,
      "elo": 1745,
      "fifa_rank": 22,
      "last5_results": ["W", "W", "D", "L", "W"],
      "last5_xg_for": [1.8, 2.1, 1.2, 0.9, 1.6],
      "last5_xg_against": [0.9, 1.1, 1.2, 1.7, 0.8],
      "injuries": [{"name": "Player X", "role": "CM", "status": "out"}],
      "suspensions": []
    },
    "away": { "...": "..." }
  },
  "weather": {
    "temp_c": 22,
    "wind_kph": 8,
    "precip_pct": 30,
    "altitude_m": 1566
  },
  "context": {
    "rest_days_home": 4,
    "rest_days_away": 5,
    "travel_km_home": 9800,
    "travel_km_away": 11200
  },
  "sentiment": {
    "home_polarity": 0.18,
    "away_polarity": -0.05,
    "n_posts_home": 124,
    "n_posts_away": 87
  },
  "sources": [
    {"slice": "odds", "value": "2.60/3.05/3.00", "url": "https://…", "fetched_at": "2026-06-11T21:00:00Z"},
    {"slice": "xg_home", "value": 1.42, "url": "https://fbref.com/…", "fetched_at": "2026-06-11T21:00:00Z"}
  ]
}
```

## 4. Re-Run mit Overrides

```bash
python -m wm2026.cli predict --match config/matches/<group>/<slug>.yaml \
  --overrides-json reports/<match_id>.overrides.json \
  --odds "<H/D/A>" --odds-ou "<O/U>" --odds-btts "<Y/N>" \
  --calibrate market --format html --out reports/ --charts
```

## 5. Stimmungs-Layer separat (optional)

Falls du Reddit/Twitter-Stimmung separater geliefert hast (oder
Sentiment-Score schon berechnet):

```bash
# sentiment_payload.json
# {"home_polarity": 0.18, "away_polarity": -0.05, "subjectivity": 0.41,
#  "n_posts_home": 124, "n_posts_away": 87, "src": "reddit/r/soccer"}
python -m wm2026.cli predict ... --sentiment-json sentiment_payload.json
```

## 6. Validieren — Schließe den Loop

Nach dem Re-Run im Report-JSON prüfen:
```bash
python3 -c "
import json, sys
d = json.load(open('reports/<match_id>.json'))
print('warnings:', d.get('warnings'))
print('claude_tasks remaining:', len(d.get('claude_tasks', [])))
print('factors_used:', d['factors_used'], '/', d['factors_total'])
print('mode:', d['mode'])
"
```

Ziel: **0 (oder minimal) verbleibende `claude_tasks`**, **≥18/20 factors_used**.
Erst dann ist die Prognose **nicht mehr mock-degradiert**.

## 7. Quellen-Hygiene

- **Jede Zahl mit URL belegen** — die `sources`-Liste im overrides.json ist
  die Audit-Trail. Im Briefing am Ende verlinken.
- **Closing-Line beobachten** — wenn sich Quoten zwischen Recherche und
  Anstoß stark bewegen (>5%), neu kalibrieren.
- **Konflikte transparent machen** — wenn FBref 1.8 xG und Understat 1.4 xG
  sagt, im Briefing erwähnen + mittleren Wert nehmen.
