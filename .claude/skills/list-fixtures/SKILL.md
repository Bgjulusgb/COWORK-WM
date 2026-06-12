---
name: list-fixtures
description: Browse the 104 pre-configured WM-2026 match YAML configs and pick the right one for a given group / matchday / team. Use when the user mentions a fixture but isn't sure of the YAML path or wants to see what's available.
---

# List-Fixtures — die 104 WM-2026-Configs durchsuchen

Im `config/matches/`-Tree liegen vorgefertigte YAML-Configs für alle 104
WM-2026-Spiele (Gruppenphase + K.-o.-Stub).

## 1. Komplette Liste

```bash
python -m wm2026.cli list
```
Output: `config/matches/group_a/cze_vs_rsa.yaml` etc., 104 Pfade.

Oder direkt:
```bash
find config/matches -name '*.yaml' -not -name '_*' | sort | head -20
```

## 2. Nach Team filtern

```bash
# Alle Spiele mit Deutschland
grep -l 'germany\|deutschland\|GER' config/matches/**/*.yaml 2>/dev/null
```

Oder per Glob:
```bash
ls config/matches/group_a/  # alle Gruppe-A-Spiele
ls config/matches/r16/      # alle Achtelfinale
```

## 3. YAML-Struktur (was steht drin)

```yaml
match:
  id: wm2026_groupa_kor_vs_cze
  stage: Group
  kickoff: 2026-06-12T04:00:00Z
  venue: Toronto / BMO Field
teams:
  home:
    name: South Korea
    code: KOR
    avg_xg_season: 1.30
    avg_xg_conceded: 1.20
    elo: 1745
    fifa_rank: 22
  away:
    name: Czech Republic
    code: CZE
    avg_xg_season: 1.42
    avg_xg_conceded: 1.30
    elo: 1718
    fifa_rank: 38
context:
  altitude_m: 75
  rest_days_home: 4
  rest_days_away: 5
```

## 4. Eigenes Match auf-Setzen (Ad-hoc, ohne YAML)

```bash
python -m wm2026.cli predict \
  --home "Germany" --away "Brazil" --stage QF \
  --kickoff "2026-07-04T21:00:00Z" --venue "MetLife Stadium" \
  --home-xg 1.55 --away-xg 1.65 \
  --home-xga 1.10 --away-xga 1.20 \
  --home-elo 1995 --away-elo 2025 \
  --odds "2.40/3.10/2.95" --calibrate market --out reports/
```

Nützlich für **Freundschaftsspiele** und **Friendly-Cup-Szenarien**, die nicht
im YAML-Tree liegen.

## 5. Auto-Match-Lookup-Helper

Wenn der User in natürlicher Sprache fragt „Korea Tschechien", finde den Slug:
```bash
python3 - <<'PY'
import yaml, glob
needles = ["korea", "czech"]
for p in sorted(glob.glob("config/matches/**/*.yaml", recursive=True)):
    if p.endswith("/_meta.yaml"): continue
    try:
        d = yaml.safe_load(open(p, encoding="utf-8")) or {}
    except Exception:
        continue
    teams = d.get("teams", {})
    h = (teams.get("home", {}) or {}).get("name", "").lower()
    a = (teams.get("away", {}) or {}).get("name", "").lower()
    blob = f"{h} {a}"
    if all(n in blob for n in needles):
        print(p, "→", teams.get("home", {}).get("name"), "vs", teams.get("away", {}).get("name"))
PY
```

## 6. Tournament-Sim (alle Gruppen auf einmal)

```bash
python -m wm2026.cli tournament --sims 10000 --out reports/
```
Liest **automatisch** alle `group_*/*.yaml` — siehe Skill `tournament-sim`.
