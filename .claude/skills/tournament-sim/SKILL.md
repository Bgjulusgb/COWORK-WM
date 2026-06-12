---
name: tournament-sim
description: Run the WM-2026 tournament Monte-Carlo simulator and interpret per-team title / final / advance probabilities. Use when the user asks "who wins the World Cup?", "what's Germany's chance?", "advance probability for Group A", or wants 10k+ tournament simulations.
---

# Tournament-Sim — Wer wird Weltmeister?

`wm2026/tournament.py` simuliert das komplette 48-Team-WM-Feld (WC-2026-Format:
12 Gruppen × 4 → 8 beste Dritte → 32-Team-KO). Pro Spiel rollt die vektorisierte,
**blend-konsistente** Score-Matrix die Tore aus. **10 000 Sims in ~1,5 s** auf
einer Standard-CPU.

## 1. Schnellstart

```bash
# Default: 10 000 Sims, alle Gruppen, neutrales Stadion (kein Heimvorteil)
python -m wm2026.cli tournament --sims 10000

# Reproduzierbar
python -m wm2026.cli tournament --sims 50000 --seed 42

# Speichern + JSON
python -m wm2026.cli tournament --sims 10000 --format json --out reports/
```

Output (Markdown-Auszug):
```
# 🏆 WM 2026 — Turnier-Monte-Carlo
*10000 Simulationen · 12 Gruppen · neutral (kein Heimvorteil)*

| # | Team        | 🏆 Titel | Finale | Achtelfinale+ |
|---|-------------|---------|--------|---------------|
| 1 | Brazil      | 15.2 %  | 27.4 % | 88.1 %        |
| 2 | France      | 12.7 %  | 24.0 % | 85.6 %        |
| 3 | Germany     |  9.8 %  | 19.6 % | 82.2 %        |
| 4 | Spain       |  8.5 %  | 17.8 % | 79.5 %        |
...
```

## 2. Wie es funktioniert (kurz)

- **Pro Spiel**: λ_home, λ_away aus dem YAML-Match-Setup (avg_xg_season +
  avg_xg_conceded blend). Neutrales Stadion → kein Heimvorteil.
- **Score**: Vektorisiertes Sampling aus der blend-konsistenten
  `score_matrix` (Dixon-Coles + NegBin + GLM + BiPoisson, gewichtet).
- **Gruppenphase**: Punkte + Tordifferenz + Tore, wie bei der FIFA.
- **Beste Dritte**: 8 von 12 Drittplatzierten ins K.-o.-32. (WC-2026-Format).
- **K.-o.**: Sieg-Tor-Rule, bei Unentschieden Penalty (50/50).
- **Aggregation**: pro Team `title_prob`, `final_prob`, `advance_prob`
  (= Achtelfinale+).

## 3. Konvergenz — wie viele Sims?

| Sims     | Standardfehler für 10%-Wahrscheinlichkeit | Empfehlung |
|----------|-------------------------------------------|------------|
| 1 000    | ±0.95 % (eher Indikation)                | Quickcheck |
| 10 000   | ±0.30 % (Default)                        | **Standard** |
| 50 000   | ±0.13 % (verlässlich für kleine %)       | Forschung  |
| 100 000+ | ±0.09 %                                  | Overkill für Live |

**Faustregel:** mehr Sims sind sinnvoll, wenn du Kandidaten mit < 2 % Title-Prob
diskutierst — sonst dominiert Sample-Noise das Signal.

## 4. Was du dem User gibst

### Block 1 — Top-10-Titel-Wahrscheinlichkeiten
- Tabelle mit Title%, Finale%, Advance%
- Sims-Anzahl + Standardfehler ausweisen

### Block 2 — Gruppen-Ausblick (per Gruppe)
Wenn der User nach einer **Gruppe** fragt:
```bash
python3 - <<'PY'
import json
d = json.load(open("reports/tournament.json"))
# Für Gruppe A: filter auf KOR, CZE, RSA, MEX (oder welcher Code immer)
group_a_codes = ["KOR", "CZE", "RSA", "MEX"]
for c in group_a_codes:
    ap = d["advance_prob"].get(c, 0) * 100
    tp = d["title_prob"].get(c, 0) * 100
    print(f"{c}: advance {ap:.1f} %  ·  title {tp:.2f} %")
PY
```

### Block 3 — Wenn der User nach einer Wett-Quote (Outright) fragt
- Vergleich Modell-prob vs Buchmacher-prob (`1 / decimal_odd`)
- Edge auf Outright-Title bei großen Quoten oft signifikant — aber
  **niedrige Varianz hat hohe Sample-Anforderung**: für eine Edge von
  +2 % auf einem 10 %-Outright brauchst du mindestens 100 k Sims für
  saubere Konfidenz.

## 5. Was es NICHT macht

- **Keine Live-Daten** — liest nur die YAML-Configs. Wenn du frische
  Form/Verletzungen einspeisen willst, vorher die einzelnen Match-YAMLs
  via `--overrides-json` updaten (siehe Skill `research-fixture`).
- **Keine echten Buchmacher-Quoten** — nur Modell-Probs. Outright-Edge
  musst du selbst per Quoten-Lookup berechnen.
- **Neutraler Boden** — kein Heimvorteil (WC ist über mehrere Länder
  verteilt). Wenn du Heimvorteil simulieren willst, λ_home um +0.15 anheben
  in den YAMLs.

## 6. Beispiel-Ausgaben

`docs/examples/tournament.md` enthält einen 10 k-Sim-Beispiellauf der
Default-Configs. Hilfreich für Sanity-Check, dass deine eigenen Sims im
gleichen Range bleiben.

## 7. Disclaimer

> Forschung/Bildung. Outright-Wetten haben hohe Varianz und niedrigen ROI —
> Outright-Edges sollten typisch +5 % vor Stake erreichen, damit sie ½-Kelly
> sinnvoll sind.
