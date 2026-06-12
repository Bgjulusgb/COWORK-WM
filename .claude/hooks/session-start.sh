#!/usr/bin/env bash
# SessionStart hook — WM-2026 Cowork
# ──────────────────────────────────────────────────────────────────────────────
# Wird beim Sitzungsstart einmal ausgeführt (Claude Code Web/Desktop/CLI).
# Idempotent: läuft nur einmal pro Container (Marker-Datei .claude/.bootstrapped).
# Installiert: core (requirements.txt) + .[viz,sentiment,stats] + pytest.
# Verifiziert: Imports + CLI + ein 5-Sekunden-Smoke-Test gegen den Mock-Pfad.
# Niemals fatal — der Hook darf die Sitzung nie blockieren (immer Exit 0).
#
# Bei Netz-Hängern: pip-install hat ein 90s-Hartlimit pro Aufruf; bei Fehlschlag
# probiert der Hook den User-Site-Pfad (--user) und meldet den Rest als
# `wm2026 doctor` — die Pipeline läuft notfalls auch ohne extras.
# ──────────────────────────────────────────────────────────────────────────────

set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 0
MARKER="$ROOT/.claude/.bootstrapped"
PIP_TIMEOUT="${WM2026_PIP_TIMEOUT:-90}"

log() { printf '\033[2m[wm2026-hook]\033[0m %s\n' "$*" >&2; }
ok()  { printf '\033[32m[wm2026-hook]\033[0m %s\n' "$*" >&2; }
warn(){ printf '\033[33m[wm2026-hook]\033[0m %s\n' "$*" >&2; }

if [ -f "$MARKER" ] && [ "${WM2026_FORCE_BOOTSTRAP:-0}" != "1" ]; then
  ok "bereits installiert ($(cat "$MARKER")). Überspringe Bootstrap."
  exit 0
fi

# 1) Python prüfen
if ! command -v python3 >/dev/null 2>&1; then
  warn "python3 fehlt — kann nicht installieren."
  exit 0
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "?")
log "python ${PYV}"

# 2) pip Bootstrap, falls fehlt
if ! python3 -m pip --version >/dev/null 2>&1; then
  log "pip fehlt — bootstrap via ensurepip"
  python3 -m ensurepip --default-pip >/dev/null 2>&1 || true
fi

# pip_install_safe — try the system path, fall back to --user, never raise.
pip_install_safe() {
  local label="$1"; shift
  log "installiere ${label}…"
  if timeout "${PIP_TIMEOUT}" python3 -m pip install \
       --quiet --disable-pip-version-check "$@" 2>&1 | tail -3 >&2; then
    return 0
  fi
  warn "system-pip ${label}: Fehler oder Timeout (${PIP_TIMEOUT}s) — versuche --user"
  if timeout "${PIP_TIMEOUT}" python3 -m pip install --user \
       --quiet --disable-pip-version-check "$@" 2>&1 | tail -3 >&2; then
    ok "--user ${label} ok"
    return 0
  fi
  warn "${label} blieb optional — wm2026 doctor zeigt was fehlt"
  return 1
}

# 3) Core-Deps (requirements.txt) — Pflicht für die Pipeline
pip_install_safe "core deps (numpy/scipy/httpx/pydantic/PyYAML/structlog)" \
  -r requirements.txt

# 4) Optional-Extras über pyproject.toml [viz,sentiment,stats] + pytest
#    viz       → matplotlib (PNG-Charts in HTML eingebettet)
#    sentiment → vaderSentiment + textblob (Reddit-Stimmung)
#    stats     → statsmodels + scikit-learn (exaktes GLM-Poisson + Kalibrierung)
pip_install_safe "optional extras: viz, sentiment, stats, pytest" \
  ".[viz,sentiment,stats,test]"

# 5) Schneller Verify
log "verify imports …"
python3 - <<'PY' 2>&1 | sed 's/^/  /' >&2
import importlib.util as _util
import sys
core = ["numpy", "scipy", "httpx", "pydantic", "yaml", "structlog"]
extras = ["matplotlib", "sklearn", "statsmodels", "vaderSentiment", "textblob", "pytest"]
miss_c = [m for m in core if _util.find_spec(m) is None]
miss_e = [m for m in extras if _util.find_spec(m) is None]
print(f"core ok: {len(core)-len(miss_c)}/{len(core)}  · fehlt: {miss_c or '–'}")
print(f"extras:  {len(extras)-len(miss_e)}/{len(extras)}  · fehlt: {miss_e or '–'} (alle optional)")
try:
    from wm2026.pipeline import run_prediction  # type: ignore
    print("wm2026.pipeline.run_prediction: ok")
except Exception as exc:
    print(f"wm2026.pipeline: IMPORT-FEHLER -> {exc}")
    sys.exit(0)
PY

# 6) End-to-End Smoke: `wm2026 doctor --json` (Dep-Check + Pipeline-Smoke).
log "smoke test (wm2026 doctor)…"
DOC_OUT=$(python3 -m wm2026.cli doctor --json 2>/dev/null || true)
if echo "$DOC_OUT" | grep -q '"smoke_ok": true'; then
  ok "smoke ok — Pipeline ist betriebsbereit."
else
  warn "doctor smoke fehlgeschlagen — bitte 'python -m wm2026.cli doctor' manuell prüfen."
  echo "$DOC_OUT" | tail -6 | sed 's/^/    /' >&2
fi

# 7) Marker setzen
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER" 2>/dev/null || true
ok "Bootstrap abgeschlossen. Skills bereit:  predict-match · research-fixture · read-report · analyze-edge · inspect-data · compare-runs · tournament-sim · calibrate-offline · tune-models · list-fixtures · cowork-setup"

exit 0
