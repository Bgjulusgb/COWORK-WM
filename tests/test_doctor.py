"""Doctor self-check — exit codes + the Phase-5 API-keys reporting."""
from __future__ import annotations

import json
import subprocess
import sys


def _run_doctor(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "wm2026.cli", "doctor", *extra_args],
        capture_output=True, text=True, timeout=60,
    )


def test_doctor_human_output_shows_api_key_block_and_exits_zero():
    r = _run_doctor()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "API keys" in r.stdout
    for env_var in ("NVIDIA_API_KEY", "ODDS_API_KEY", "FOOTBALL_DATA_API_KEY"):
        assert env_var in r.stdout
    # Without keys configured the block must show ⚠️ (informational), not ❌.
    keys_section = r.stdout.split("API keys")[-1].split("##")[0]
    assert "❌" not in keys_section


def test_doctor_json_reports_api_keys_present():
    r = _run_doctor("--json")
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert "api_keys_present" in payload
    keys = payload["api_keys_present"]
    # Every supported key gets a stable boolean entry, even when unset.
    assert set(keys) == {"NVIDIA_API_KEY", "ODDS_API_KEY", "FOOTBALL_DATA_API_KEY"}
    for v in keys.values():
        assert isinstance(v, bool)


def test_doctor_human_shows_connector_dry_run():
    """Phase-5 extension: a per-connector dry-run section so the user sees
    which sources would go live with the *current* settings (not just keys)."""
    r = _run_doctor()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Connectors" in r.stdout
    # At least the auth-free connectors show ✅ — they need no key.
    for name in ("openfootball", "weather", "wikidata", "reddit"):
        assert name in r.stdout


def test_doctor_json_reports_connector_counts():
    r = _run_doctor("--json")
    payload = json.loads(r.stdout)
    assert "connectors_live" in payload
    assert "connectors_total" in payload
    assert payload["connectors_total"] >= 14         # 14 use_mock_* + LLM + reddit
    # Without any keys set, the key-gated connectors stay mock — but every
    # auth-free source goes live by default.
    assert payload["connectors_live"] >= 10


def test_doctor_smoke_still_runs_and_pins_schema_1_3():
    """The mock smoke that ensures the pipeline + schema work end-to-end."""
    r = _run_doctor("--json")
    payload = json.loads(r.stdout)
    assert payload["smoke_ok"] is True
    assert payload["schema_version"] == "1.3"
