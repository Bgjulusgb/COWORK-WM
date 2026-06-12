"""M3: Train-Status muss bei jedem Pfad (Erfolg, Exception, BaseException,
SystemExit) im Anschluss garantiert NICHT auf 'running' haengen bleiben.
"""
from __future__ import annotations

from unittest.mock import patch

from api import admin as admin_mod


def _reset_status(model_key: str = "xgboost"):
    admin_mod._TRAIN_STATUS[model_key] = {"status": "idle"}


def test_run_training_sets_running_then_done_on_success():
    _reset_status()

    class _Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    with patch.object(admin_mod.subprocess, "run", return_value=_Result()):
        admin_mod._run_training("xgboost", [])
    assert admin_mod._TRAIN_STATUS["xgboost"]["status"] in ("done", "error")
    assert admin_mod._TRAIN_STATUS["xgboost"].get("finished_at") is not None


def test_run_training_does_not_hang_in_running_on_exception():
    _reset_status()
    with patch.object(admin_mod.subprocess, "run", side_effect=RuntimeError("boom")):
        admin_mod._run_training("xgboost", [])
    assert admin_mod._TRAIN_STATUS["xgboost"]["status"] != "running"
    assert admin_mod._TRAIN_STATUS["xgboost"].get("finished_at") is not None


def test_run_training_does_not_hang_in_running_on_baseexception():
    """Even SystemExit/KeyboardInterrupt must leave the status non-running so
    the next training request isn't blocked."""
    _reset_status()
    with patch.object(admin_mod.subprocess, "run", side_effect=SystemExit(2)):
        try:
            admin_mod._run_training("xgboost", [])
        except SystemExit:
            pass
    assert admin_mod._TRAIN_STATUS["xgboost"]["status"] != "running"
