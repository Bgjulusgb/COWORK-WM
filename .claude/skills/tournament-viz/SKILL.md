---
name: tournament-viz
description: Render the WM-2026 Monte-Carlo tournament result as a ranked bracket pyramid (champion / finalists / R16 entrants) instead of a flat ranked table. Use when the user asks "show me the bracket", "wer kommt am wahrscheinlichsten ins Finale?", or wants a visual / shareable overview of the simulation.
---

# Tournament-Viz — Ranked Bracket-Pyramide

Das `wm2026 tournament` Subcommand kann den Monte-Carlo-Output als
**dreistufige Bracket-Pyramide** rendern: Champion → Finale → Achtelfinale.

## 1. Bracket-Output

```bash
python -m wm2026.cli tournament --sims 10000 --format bracket
```

Beispiel-Output:
```
🏆 WM 2026 — TURNIER-BRACKET
   10000 Simulationen · neutral (kein Heimvorteil)

  ╔══════════════════════════════════════════════════════════════╗
  ║  🏆 WELTMEISTER  (P(Titel))                                  ║
  ╚══════════════════════════════════════════════════════════════╝
   1. Argentinien    ████████████··········   18.5%
   2. Brasilien      ██████████············   15.2%
   3. Frankreich     █████████·············   12.8%

  ┌──────────────────────────────────────────────────────────────┐
  │  🥇 FINALE  (P(Finale erreicht))                             │
  └──────────────────────────────────────────────────────────────┘
   1. Argentinien    ████████████████······   32.1%
   2. Brasilien      ███████████████·······   28.4%
   ...
```

## 2. Wann nutzen

- User fragt *"wer wird Weltmeister?"* → bracket > markdown-Tabelle für visual.
- User will **shareable Übersicht** für Forum/Slack — Unicode bar charts.
- Nach `predict-match` für ein Halbfinale → bracket gibt Turnier-Kontext.

## 3. Warum ranked pyramid statt echte Bracket

Eine "echte" Bracket-Anzeige bräuchte stabile Slot-Zuordnungen. Die
Monte-Carlo-Simulation hat aber **per-Sim verschiedene KO-Paarungen**
(Group-Winner-vs-Runner-up wechselt). Daher ist eine ranked Pyramide pro
Round die ehrlichere Darstellung der Ergebnisse — kein Slot wird fest
behauptet, der nur in 30% der Sims stimmt.

Per-Round-Daten kommen aus dem `TournamentResult`:
- `advance_prob` → Achtelfinale-Tier
- `final_prob`   → Finale-Tier
- `title_prob`   → Champion-Tier

## 4. Alternativ: JSON für eigenen Renderer

```bash
python -m wm2026.cli tournament --sims 10000 --format json > sims.json
```

Dann z.B. mit Python:
```python
import json
data = json.load(open("sims.json"))
top10 = sorted(data["title_prob"].items(), key=lambda kv: -kv[1])[:10]
```

## 5. Verify

```bash
pytest tests/test_tournament_viz.py -q       # 3 Tests
python -m wm2026.cli tournament --sims 100 --format bracket   # End-to-end
```
