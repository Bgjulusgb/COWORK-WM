# Eigene Trainingsdaten

Lege hier `.json` oder `.csv` Dateien mit historischen Matchergebnissen ab.
Das Trainings-Script (`train_xg_predictor.py`) liest diesen Ordner automatisch.

---

## CSV-Format

```
date,home,away,home_score,away_score,competition_tier
2023-06-20,Argentina,Brazil,1,0,2
2023-07-09,Argentina,France,3,3,1
```

**Spalten:**
- `date` — YYYY-MM-DD
- `home` / `away` — Teamname (Deutsch oder Englisch, auch FIFA-Code wie ARG, BRA)
- `home_score` / `away_score` — Tore nach regulärer Spielzeit
- `competition_tier` (optional) — 1=WM/EM, 2=Qualifikation, 3=Nations League, 4=Freundschaftsspiel

---

## JSON-Format (einfaches Array)

```json
[
  {"date": "2023-06-20", "home": "Argentina", "away": "Brazil",  "home_score": 1, "away_score": 0},
  {"date": "2023-07-09", "home": "Argentina", "away": "France",  "home_score": 3, "away_score": 3, "competition_tier": 1}
]
```

---

## JSON-Format (openfootball)

Die Dateien von https://github.com/openfootball/ funktionieren direkt:

```json
{
  "name": "CONMEBOL WM-Qualifikation 2026",
  "rounds": [
    {
      "matches": [
        {"date": "2023-09-07", "team1": "Argentina", "team2": "Ecuador", "score": {"ft": [1, 0]}}
      ]
    }
  ]
}
```

---

## Tipps für mehr Trainingsdaten

- **openfootball Qualifikation:** https://github.com/openfootball/world-cup.json — enthält auch Quali-Spiele
- **CONMEBOL-Ergebnisse:** https://github.com/openfootball/south-america.json
- **UEFA Nations League:** https://github.com/openfootball/euro.json
- **Eigene Exports:** Aus FootyStats, Sofascore oder FBref exportierte CSVs direkt hier ablegen
