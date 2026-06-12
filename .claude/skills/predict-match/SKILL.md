---
name: predict-match
description: Run the WM-2026 prediction pipeline for a match and interpret the report. Use when the user asks for a match prediction, an edge/value analysis, Asian-handicap / over-under / BTTS / Double-Chance / Draw-No-Bet probabilities, a tournament/group simulation, or wants to run the wm2026 workflow for a fixture.
---

# Predict a WM-2026 match — end-to-end Cowork-Loop

Run the calibrated **8-phase pipeline** of this repo and return a fully-grounded
quant briefing. Deep methodology: `prompts/WM2026_MASTER_PROMPT.md`; conventions:
`CLAUDE.md`; math roadmap: `verbesserungsplan.md`. Hook installs deps; if you see
"smoke ok" in the session log, you're ready.

## 0. Vorgehen (in dieser Reihenfolge)

1. **Verify** (optional, falls Pipeline noch nie lief in der Sitzung):
   ```bash
   python -m wm2026.cli predict --mode mock --home A --away B --stage Group --format json | tail -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("ok" if "match_id" in d else "FAIL")'
   ```
2. **Research** (Cowork-Auftrag): live-Daten per Web Search holen (siehe Skill
   `research-fixture`).
3. **Predict** (siehe unten — Schritt 1 + 2).
4. **Read & explain** (siehe Skill `read-report` + `analyze-edge`).

## 1. Pipeline ausführen

### Ad-hoc (ohne YAML)
```bash
python -m wm2026.cli predict \
  --home "<HOME>" --away "<AWAY>" --stage <Group|R32|R16|QF|SF|Final> \
  --odds "<H/D/A>" --odds-ou "<O/U>" --odds-btts "<Y/N>" \
  --odds-dc "<1X/12/X2>" --odds-ah=<-0.5:HOME/AWAY> \
  --calibrate market --format html --out reports/ --charts
```

### Aus YAML
```bash
python -m wm2026.cli list                                 # alle 104 Configs auflisten
python -m wm2026.cli predict --match config/matches/<group>/<slug>.yaml \
  --odds "..." --odds-ou "..." --odds-btts "..." \
  --calibrate market --format html --out reports/ --charts
```

### Modi
- **`--mode live`** ist Default — ~13 Konnektoren parallel; pro Quelle fällt
  bei Fehler/Key-fehlt automatisch auf Mock zurück. Erzeugt
  `claude_tasks` (Cowork-Auftrag) für jede degradierte Slice.
- **`--mode mock`** = vollständig offline, reproduzierbar, illustrativ.
- **`--live-sources weather,clubelo`** (Phase 4): *nur* diese Konnektoren
  live, alles andere mock — präzise Cowork-Runs ohne `.env`-Editing
  (impliziert `--mode live`).
- **`--mock-sources reddit,transfermarkt`**: erzwingt einzelne Quellen auf
  mock (z.B. Rate-Limit-vermeidend).

### Token-budget Modi (Phase 5) — *immer mitnehmen*
- **`--compact`**: ~35 % kleinerer JSON (factors-availables-only, blended-CI
  only, AH-Long-Tail weg, kein per_model). Schema bleibt erhalten,
  `"compact": true` markiert es. Details: Skill `inspect-data`.
- **`--charts-external`** + `--charts`: HTML referenziert PNGs statt sie zu
  embedden → ~10 KB statt ~95 KB.
- **`--ah-lines=-0.5,0,0.5`** (mit `=`!): nur diese AH-Linien rendern.
- **`--gzip`**: zusätzlich `<id>.json.gz`. `wm2026 summary` liest .gz direkt.
- **`--format summary`**: druckt sofort die Token-budget-Briefing-Form
  (~400 Tokens). Wird zusätzlich als `<id>.summary.md` neben dem Report
  geschrieben — **immer**, auch ohne `--format summary`.

### Staking-Empfehlungen direkt im Report (Phase 4)
- **`--bankroll 1000`** annotiert jede Edge-Zeile mit konkretem Einsatz
  (`stake_half_kelly` p50 / `stake_cons` p5). `stake_cons=0.00` sobald die
  konservative Edge ≤ 0 ist — die ½-Kelly-auf-p5-Disziplin steht damit im
  Backend, nicht nur im Briefing.
- Das neue Feld **`best_value_cons`** im JSON ist der *ehrliche* Pick:
  höchste Edge, die auf der Bootstrap-Untergrenze (p5) positiv bleibt.
  `best_value` (alt) maximiert dagegen die rohe p50-Edge — perfekt um die
  Sanity-Check-Kandidaten transparent zu machen, **nie** als Empfehlung
  benutzen.

### Asian-Handicap-Syntax
Negative Linien **MIT `=`** wegen argparse:
`--odds-ah=-0.5:1.95/1.95` ✅   `--odds-ah -0.5:1.95/1.95` ❌

### Kalibrierung (Phase 5) — Pflicht für realistische Wahrscheinlichkeiten
- `--calibrate market` (Empfohlen wenn du Live-Quoten hast):
  ankert 1X2 an die **vig-freie Konsens-Quote** (Constantinou & Fenton 2013 —
  der kanonisch gut kalibrierte Forecaster).
- `--calibrate auto`: nutzt ein fitted Artefakt (siehe Skill `calibrate-offline`)
  wenn vorhanden, sonst raw.
- `--calibrate none`: nur für Debugging.

## 2. Cowork-Loop (Pflicht in Live-Modus)

Der Report enthält eine Sektion **„🤝 Cowork-Auftrag (live data gaps)"** mit
priorisierter Liste. **Jeden Eintrag abarbeiten:**

1. Wert per Web Search recherchieren, mit `(value, source_url, fetched_at)` belegen
2. Wie unter „einspeisen via" angegeben einspeisen:
   - **xG / Elo / Form** → in eine `overrides.json` (Template:
     `python -m wm2026.cli research --home … --away … --out reports/`)
   - **Quoten** → `--odds`, `--odds-ou`, `--odds-btts`, `--odds-dc`, `--odds-ah`
   - **Stimmung** → `--sentiment-json <file>`
3. Pipeline **erneut** fahren bis die "X/Y slices degraded"-Warnung minimal ist
4. **Ohne diesen Schritt ist die Prognose mock-degradiert (illustrativ)**

→ Details siehe Skill **`research-fixture`**.

## 3. Mathematik-Schichten — alle aktivieren

| Schicht | Aktivieren via |
|---|---|
| **Dixon-Coles + NegBin + GLM-Poisson** (3-Modell-Blend, default) | automatisch in Phase 4 |
| **Bivariates Poisson als 4. Blend-Modell** (Karlis-Ntzoufras λ₃) | `INCLUDE_BIVARIATE=true` (Env) — opt-in, Default off (Stabilitäts-Contract) |
| **MLE-λ-Schätzer mit Zeitdecay** (Attack/Defence aus Historie) | `settings.use_mle_xg=True` (Env `USE_MLE_XG=true`) |
| **Geometrische λ-Aggregation** (log-linear, symmetrisch in home/away) | `LAMBDA_AGGREGATION=geom` (Env) — opt-in, Default `arith` |
| **Bootstrap-CIs** (p5/p50/p95 für jede Headline-Zahl + DC + AH) | `--bootstrap 500` (default) |
| **Markt-Anker-Kalibrierung** | `--calibrate market` |
| **Konservative p5-Edge / p5-Kelly** | automatisch in Phase 6 (1X2 · O/U · BTTS · **DC** · **AH** — alle mit p5) |
| **Bankroll-Annotation** (`stake_half_kelly`, `stake_cons`) | `--bankroll 1000` |
| **Score-Heatmap + Faktor-Tornado (PNG)** | `--charts` (braucht matplotlib — vom Hook installiert) |

## 4. Antworten — was du IMMER mitlieferst

- **Headline 1X2** + **λ_home / λ_away** + **Bootstrap-CI [p5/p50/p95]**
- **Edge-Tabelle** inkl. **`(p5)`-Spalten** — eine Edge zählt nur, wenn sie
  positiv bleibt auf der konservativen Bootstrap-Untergrenze
- **Derived-Markets-Board**: Double Chance, Draw-No-Bet, Asian Handicap (inkl.
  Viertellinien), alternative Totals, Clean Sheet, Win-to-Nil, Odd/Even,
  Winning Margin, Multi-Goal-Bands, Exact Totals, First Goal, HT/FT
- **Cowork-Status**: erledigte vs offene Live-Daten-Gaps
- **Konfidenz-Ampel** (`ensemble_confidence`) + Disclaimer:
  *Forschung/Bildung, keine Wett-Empfehlung.*

## 5. Guardrails

- **Niemals** eine Punkt-Vorhersage ohne Konfidenzintervall.
- **Mock-Daten sind illustrativ** — explizit kennzeichnen.
- **Edge > 10 %** ⇒ Sanity-Check-Note („warum würde der Markt das verpassen?")
- Wenn p5-Edge negativ ist trotz positivem p50: **Pass** empfehlen, nicht
  beschönigen.
