---
name: inspect-data
description: Read large wm2026 prediction reports without blowing the token budget — use --compact JSON (~35% smaller), the wm2026 summary briefing (~400 tokens), --charts-external HTML (~92% smaller), targeted JSON queries, or gzip on disk. Use whenever a Read on the raw JSON would error with "Output too large", or before piping a report into another skill / agent.
---

# Inspect-Data — Reports lesen, ohne das Token-Budget zu sprengen

Ein voller Mock-Report ist **~3.95 k Tokens JSON + ~95 KB HTML**. In einer
Cowork-Sitzung mit 200 k Kontext klingt das wenig — ist es nicht, sobald du
3–5 Spiele vergleichst oder den HTML-Output liest. Hier sind die fünf Werkzeuge
des Token-Budgets in Reihenfolge der Aggressivität.

## 1. `wm2026 summary <json>` — die Briefing-Variante (~400 Tokens)

Erste Wahl, wenn du dem User antworten sollst. Liest die JSON von Platte
(oder `.json.gz`) und liefert einen deterministischen Block mit:

- λ + Bootstrap-CI je Team
- 1X2 (kalibriert wenn Phase 5 lief, sonst raw)
- O/U 2.5 + BTTS + Blended-CIs
- Top-5 Edges sortiert nach **konservativer p5-Edge** (mit Stake wenn `--bankroll` gesetzt war)
- Empfehlung: **`best_value_cons`** als ehrlicher Pick, `best_value` als
  Sanity-Check-Transparenz, **Pass** wenn keiner überlebt
- Cowork-Status + Warnings + Disclaimer

```bash
python -m wm2026.cli summary reports/<match_id>.json
python -m wm2026.cli summary reports/<match_id>.json.gz --top 8
```

Wird automatisch als `<match_id>.summary.md` neben dem Report geschrieben,
sobald `predict --out` läuft. **Lies das zuerst, bevor du den JSON öffnest.**

## 2. `--compact` JSON (~57 % der Größe)

Wenn du programmatisch durch den Report navigieren musst, aber kein
Raw-Provenance-Debug brauchst:

```bash
python -m wm2026.cli predict ... --compact --format json --out reports/
```

Was rausfliegt (Default-Schema bleibt intakt, neues Feld `"compact": true`):

| Block | Voll | Compact |
|---|---|---|
| `factors` | 20 (inkl. raw_data, cached_at) | nur `available=True`, 8 Skalar-Felder |
| `data_sources` | dict pro Slice (mode + source + payload) | nur Mode-Token (`"mock"`/`"live"`/...) |
| `confidence_intervals` | `poisson` + `negbin` + `glm_poisson` + `blended` | nur `blended` |
| `derived_markets.asian_handicap` | 13 Linien (−2 … +2 inkl. Viertel) | 5 Hauptlinien (−1, −0.5, 0, +0.5, +1) |
| `edge_table` | alle Märkte | Top-12 nach \|edge\| |
| `markets.correct_score_top5` | dabei | weg |
| `per_model` | dabei | weg |

`compact()` ist eine pure Funktion, ohne Side-Effects — du kannst sie auch
nachträglich anwenden:

```bash
python3 -c "
import json, sys
from wm2026.report import compact
js = json.load(open(sys.argv[1]))
print(json.dumps(compact(js), ensure_ascii=False))
" reports/<match_id>.json
```

## 3. `--ah-lines=-0.5,0,0.5` — Asian-Handicap-Linien begrenzen

13 AH-Linien sind ~1 600 Zeichen allein. Mit `--ah-lines` produziert die
Pipeline nur die gewünschten Linien — ohne Compact-Mode, ohne Schema-Bruch:

```bash
python -m wm2026.cli predict ... --ah-lines=-1,-0.5,0,0.5,1
# Achtung: --ah-lines MIT `=` schreiben, sonst halt argparse den Leading-Dash
# für ein Flag.
```

## 4. `--charts-external` HTML (~92 % kleiner)

Der HTML-Report bettet die PNG-Charts standardmäßig als base64 ein
(~95 KB). Mit `--charts-external` referenziert das HTML stattdessen die
PNG-Geschwister auf Platte — HTML schrumpft auf ~8 KB, der Browser lädt die
Charts ganz normal:

```bash
python -m wm2026.cli predict ... --format html --charts --charts-external --out reports/
ls -la reports/<match_id>{.html,_tornado.png,_heatmap.png}
```

Gut für `Read` des HTML-Reports + für Webserver-Mounts.

## 5. `--gzip` zusätzlicher `.json.gz`

```bash
python -m wm2026.cli predict ... --gzip --out reports/
ls -la reports/<match_id>.json.gz   # ~3-5 KB
python -m wm2026.cli summary reports/<match_id>.json.gz
```

`wm2026 summary` liest das transparent. Hilfreich beim Speichern vieler
Reports oder als Anhang in CI-Artefakten.

## Welches Werkzeug wann?

| Du willst … | Befehl |
|---|---|
| dem User antworten | `wm2026 summary` |
| eine andere Skill füttern | `--compact --format json` → an die Skill |
| zwei Runs A/B vergleichen | beide `--compact` ⇒ Skill `compare-runs` |
| den HTML in den Browser ziehen | `--charts-external` (kleiner & lazy-loaded) |
| auf Platte speichern | `--gzip` + ggf. `--compact` |
| nur eine spezifische AH-Linie | `--ah-lines=-0.5` |
| die Provenance debuggen | **kein** `--compact` — die Modi sind dann erhalten |

## Direkter Programm-Zugriff aus einem Notebook / Helper-Skript

```python
import json
from wm2026.report import compact
from wm2026.summary import summarise

js = json.load(open("reports/wm2026_groupa_kor_vs_cze.json"))

# Briefing:
print(summarise(js, top_edges=5))

# Token-budget JSON:
slim = compact(js)                    # ~57% smaller
print(len(json.dumps(slim)), "chars")

# Nur eine AH-Linie behalten:
slim2 = compact(js, ah_lines={-0.5})
```

## Beim Lesen: nicht den ganzen JSON, sondern Schlüssel

Wenn du wirklich nur einen Block brauchst, parse selektiv statt mit `Read`:

```bash
python3 -c "
import json
d = json.load(open('reports/<match_id>.json'))
print(json.dumps(d['confidence_intervals']['blended'], indent=2))
"
```

## Hilfsdiagnose, falls's hakt

```bash
python -m wm2026.cli doctor              # Dependencies + Pipeline-Self-Test
python -m wm2026.cli doctor --json       # CI-tauglicher Status
```
