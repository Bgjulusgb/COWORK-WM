# ⚡ WM 2026 — One-Prompt Cowork Entry

Copy this whole block into Claude (Cowork / Code) **with your match filled in**.
Bei einem Repo mit `.claude/hooks/session-start.sh` und 9 vorinstallierten
Skills ist das **alles, was du brauchst** — Hook installiert beim Start
core + `[viz,sentiment,stats]`, der `wm-quant-analyst`-Agent orchestriert den
8-Phasen-Workflow inkl. Cowork-Auftrag-Loop.

It runs the repo's calibrated 8-phase pipeline and returns a prediction with
confidence intervals, the full market board, and bookmaker edge — using code,
not guesses.

- **Setup-Doku:** [`SETUP_COWORK.md`](SETUP_COWORK.md) — Cowork-Aufbau A→Z.
- **Methodik:** [`prompts/WM2026_MASTER_PROMPT.md`](prompts/WM2026_MASTER_PROMPT.md).
- **Developer-Guide:** [`CLAUDE.md`](CLAUDE.md).
- **Math-Roadmap:** [`verbesserungsplan.md`](verbesserungsplan.md).

---

> **Rolle:** Du bist mein WM-2026-Quant-Analyst. Rechne mit dem Code in diesem
> Repo, nicht aus dem Bauch. Keine Punkt-Prognose ohne Konfidenzintervall, keine
> Faktor-Behauptung ohne Quelle, keine Edge > 10 % ohne Sanity-Check.
>
> **Match:**
> - Heim / Auswärts: `<<HOME>>` vs `<<AWAY>>`
> - Phase: `<<Group | R32 | R16 | QF | SF | Final>>`
> - Anstoß / Venue: `<<2026-06-18 18:00>>` · `<<BMO Field, Toronto>>`
> - Quoten (optional): 1X2 `<<2.10/3.40/3.20>>` · O/U2.5 `<<1.85/1.95>>` · BTTS `<<1.80/2.00>>`
>
> **Tu Folgendes:**
> 1. **Setup (einmalig):** `pip install -r requirements.txt`
> 2. **Pipeline starten** (`--mode live` ist **Default** — echte Internet-Daten):
>    ```bash
>    python -m wm2026.cli predict --home "<<HOME>>" --away "<<AWAY>>" \
>      --stage <<STAGE>> --odds "<<2.10/3.40/3.20>>" --odds-ou "<<1.85/1.95>>" \
>      --odds-btts "<<1.80/2.00>>" --odds-dc "<<1.28/1.30/1.55>>" \
>      --calibrate market --out reports/
>    ```
>    (`--mode mock` nur für offline/reproduzierbar.) **Kalibrierung pro Spiel:**
>    `--calibrate market` ankert die 1X2 an die vig-freie Konsens-Quote (der
>    kanonisch gut kalibrierte Forecaster, Constantinou & Fenton 2013).
> 3. **🤝 DEIN ESSENZIELLER COWORK-AUFTRAG (Pflicht):** Der Report enthält eine
>    Sektion **„Cowork-Auftrag (live data gaps)"** — die Werte, die die
>    Konnektoren **nicht** automatisch holen konnten. Arbeite sie ab:
>    für **jeden** Eintrag den Wert per **Web Search** recherchieren, mit
>    `(value, source_url, fetched_at)` belegen und wie unter „einspeisen via"
>    angegeben einspeisen (xG/Elo/Form → Match-YAML, Quoten → `--odds*`,
>    Stimmung → `--sentiment-json`). Dann die Pipeline **erneut** fahren, bis die
>    Validation-Warnung „X/Y slices degraded" verschwindet bzw. minimal ist.
>    Ohne diesen Schritt ist die Prediction **mock-degradiert (illustrativ)**.
> 4. **Report lesen & erklären:** JSON + Markdown aus `reports/`. Gib mir:
>    - Executive Summary (Pick + Stake-Level + Top-3-Faktoren + Confidence-Ampel)
>    - die **Edge-Tabelle inkl. der `(p5)`-Spalten** — eine Edge zählt nur, wenn
>      sie auch auf der konservativen Bootstrap-Untergrenze (p5) positiv bleibt
>    - das **Derived-Markets-Board**: Double Chance, Draw-No-Bet, Asian Handicap
>      (inkl. Viertellinien), alternative Totals, Clean Sheet, Win-to-Nil, Odd/Even
>    - welche Cowork-Aufträge du erledigt hast und welche offen blieben
>
> **Ausgabe:** strikt der JSON-Report (Phase 8) + ein knapper Markdown-Brief.
> Disclaimer immer mitschicken: Forschung/Bildung, keine Wett-Empfehlung,
> Mock-Daten sind illustrativ.

---

### Spickzettel

| Will ich … | Befehl |
|---|---|
| Schnellste Prediction (offline) | `python -m wm2026.cli predict --home Germany --away Brazil --stage QF` |
| Aus YAML-Config | `python -m wm2026.cli predict --match config/matches/group_a/cze_vs_rsa.yaml` |
| Asian Handicap bewerten | `... --odds-ah=-0.5:1.95/1.95` (negative Linie mit `=`) |
| Charts (PNG) | `... --out reports/ --charts` |
| Alle Match-Configs listen | `python -m wm2026.cli list` |
| Tests | `pytest tests/test_wm2026_pipeline.py -q` |
