"""v3.3 — Admin & datasource-status endpoints.

* ``GET  /api/admin/weights``   — current target weights + active flag per factor.
* ``PATCH /api/admin/weights``  — write the slider values to
  ``models_ml/artifacts/runtime_weights.yaml`` and hot-reload via
  ``Settings.reload_runtime_weights()``.
* ``GET  /api/datasources/status`` — per-connector freshness, mode mix,
  last error, hit count from :class:`DataSourceCache`. Powers the Sources
  dashboard in the frontend.
* ``POST /api/admin/feature_flags`` — flip the small set of boolean toggles
  the UI exposes (NVIDIA LLM, goal model, Prefect).

Auth: every write endpoint requires an ``X-Admin-Key`` header matching
``settings.admin_api_key``. The read endpoints are open so the dashboard can
render without forcing the user to log in.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, BackgroundTasks, Body, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from analysis.weight_optimizer import _PRIOR_RANGES
from config.settings import settings
from db.database import get_session
from db.models import DataSourceCache, MatchPrediction
from utils.io import atomic_write_yaml

router = APIRouter(prefix="/api/admin", tags=["admin"])
sources_router = APIRouter(prefix="/api/datasources", tags=["datasources"])


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Header guard. The admin key is empty by default → endpoints stay locked
    until the operator configures one in ``.env``."""
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="admin api disabled (set ADMIN_API_KEY)")
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="invalid admin key")


# ── weights ────────────────────────────────────────────────────────────────

_RUNTIME_WEIGHTS_PATH: Path = settings.base_dir / "models_ml" / "artifacts" / "runtime_weights.yaml"


class WeightsPatch(BaseModel):
    """Slim payload: just ``factor_weight_*`` keys mapped to the target [0, 1]."""
    weights: dict[str, float] = Field(default_factory=dict)
    goal_model: str | None = None
    use_nvidia_llm: bool | None = None


@router.get("/weights")
async def get_weights() -> dict[str, Any]:
    out = []
    for key, (lo, hi) in _PRIOR_RANGES.items():
        if not hasattr(settings, key):
            continue
        value = float(getattr(settings, key))
        out.append({
            "name": key,
            "value": value,
            "active": value > 0,
            "lo": lo,
            "hi": hi,
        })
    return {
        "weights": out,
        "goal_model": settings.goal_model,
        "use_nvidia_llm": settings.use_nvidia_llm,
        "use_prefect": settings.use_prefect,
        "use_factor_ensemble": settings.use_factor_ensemble,
    }


@router.patch("/weights", dependencies=[Depends(require_admin)])
async def patch_weights(payload: WeightsPatch) -> dict[str, Any]:
    runtime: dict[str, float] = {}
    for key, val in payload.weights.items():
        if not key.startswith("factor_weight_"):
            raise HTTPException(status_code=400, detail=f"unknown setting: {key}")
        if not hasattr(settings, key):
            raise HTTPException(status_code=400, detail=f"unknown setting: {key}")
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"non-numeric weight {key}: {val}") from exc
        runtime[key] = max(0.0, min(1.0, fval))

    if payload.goal_model is not None:
        if payload.goal_model not in {"poisson", "negbin", "glm_poisson"}:
            raise HTTPException(status_code=400, detail="bad goal_model")
        object.__setattr__(settings, "goal_model", payload.goal_model)

    if payload.use_nvidia_llm is not None:
        object.__setattr__(settings, "use_nvidia_llm", bool(payload.use_nvidia_llm))

    if runtime:
        # Merge with existing artifact so partial PATCHes don't wipe the file.
        existing: dict[str, Any] = {}
        if _RUNTIME_WEIGHTS_PATH.exists():
            try:
                existing = yaml.safe_load(_RUNTIME_WEIGHTS_PATH.read_text(encoding="utf-8")) or {}
            except Exception:
                existing = {}
        existing.update(runtime)
        # K3: atomar schreiben, damit parallele reload_runtime_weights() Aufrufe
        # nie ein halb-geschriebenes YAML sehen koennen.
        atomic_write_yaml(_RUNTIME_WEIGHTS_PATH, existing)
        # Re-read the file we just wrote — using the same path so tests that
        # redirect _RUNTIME_WEIGHTS_PATH stay in sync.
        settings.reload_runtime_weights(path=_RUNTIME_WEIGHTS_PATH)

    return await get_weights()


# ── data source status ─────────────────────────────────────────────────────


@sources_router.get("/status")
async def datasources_status(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    """Per-connector freshness summary.

    Reads the ``data_source_cache`` table — every connector touches it on every
    successful live fetch — and aggregates the last fetch + count per connector.
    """
    now = datetime.now(timezone.utc)
    stmt = (
        select(
            DataSourceCache.connector,
            func.count(DataSourceCache.cache_key).label("entries"),
            func.max(DataSourceCache.fetched_at).label("last_fetch"),
        )
        .group_by(DataSourceCache.connector)
    )
    rows = (await session.execute(stmt)).all()

    known_connectors = [
        "openfootball", "thesportsdb", "openligadb", "wikidata", "weather",
        "rss_news", "football_data_org",
        # v3.3
        "fbref", "understat", "fotmob", "sofascore", "transfermarkt", "nvidia_llm",
    ]
    bucket: dict[str, dict[str, Any]] = {
        name: {
            "connector": name,
            "entries": 0,
            "last_fetch": None,
            "age_minutes": None,
            "configured_mock": _mock_flag(name),
            "active": _is_active(name),
        }
        for name in known_connectors
    }
    for connector, entries, last_fetch in rows:
        slot = bucket.setdefault(connector, {
            "connector": connector,
            "entries": 0,
            "last_fetch": None,
            "age_minutes": None,
            "configured_mock": _mock_flag(connector),
            "active": _is_active(connector),
        })
        slot["entries"] = int(entries or 0)
        slot["last_fetch"] = last_fetch.isoformat() if last_fetch else None
        if last_fetch is not None:
            age = (now - last_fetch.replace(tzinfo=timezone.utc)).total_seconds() / 60.0
            slot["age_minutes"] = round(age, 2)

    return {
        "as_of": now.isoformat(),
        "ttl_hours": settings.datasource_cache_ttl_hours,
        "connectors": list(bucket.values()),
    }


_MOCK_FLAGS = {
    "openfootball": "use_mock_openfootball",
    "thesportsdb": "use_mock_thesportsdb",
    "openligadb": "use_mock_openligadb",
    "wikidata": "use_mock_wikidata",
    "weather": "use_mock_weather",
    "rss_news": "use_mock_rss",
    "football_data_org": "use_mock_football_data",
    "fbref": "use_mock_fbref",
    "understat": "use_mock_understat",
    "fotmob": "use_mock_fotmob",
    "sofascore": "use_mock_sofascore",
    "transfermarkt": "use_mock_transfermarkt",
    "reddit": "use_mock_crawler",
    "nvidia_llm": "use_nvidia_llm",  # inverse semantics — handled below
}

_RUNTIME_FLAGS_PATH: Path = settings.base_dir / "models_ml" / "artifacts" / "runtime_flags.yaml"


def _persist_runtime_flag(key: str, value: object) -> None:
    """Merge {key: value} into runtime_flags.yaml so settings survive a restart.

    K3: atomarer Write via utils.io.atomic_write_yaml — parallele Reads sehen
    nie einen halben Schreibvorgang.
    """
    existing: dict[str, Any] = {}
    if _RUNTIME_FLAGS_PATH.exists():
        try:
            existing = yaml.safe_load(_RUNTIME_FLAGS_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            existing = {}
    existing[key] = value
    atomic_write_yaml(_RUNTIME_FLAGS_PATH, existing)


def _mock_flag(connector: str) -> bool:
    attr = _MOCK_FLAGS.get(connector)
    if attr is None:
        return False
    if connector == "nvidia_llm":
        return not bool(getattr(settings, attr, False))
    return bool(getattr(settings, attr, False))


def _is_active(connector: str) -> bool:
    """A connector counts as 'active' when it is allowed to talk to its live
    endpoint (i.e. mocking is off). nvidia_llm needs the API key on top."""
    if connector == "nvidia_llm":
        return settings.use_nvidia_llm and bool(settings.nvidia_api_key)
    attr = _MOCK_FLAGS.get(connector)
    if attr is None:
        return True
    return not bool(getattr(settings, attr, False))


# ── v3.6 — Datasource-Toggle ──────────────────────────────────────────────────


class DataSourceToggle(BaseModel):
    mock: bool


@sources_router.post("/{connector}/toggle", dependencies=[Depends(require_admin)])
async def toggle_datasource(connector: str, payload: DataSourceToggle) -> dict[str, Any]:
    """Live/Mock pro Connector umschalten. nvidia_llm hat inverse Semantik
    (mock=true bedeutet use_nvidia_llm=false)."""
    attr = _MOCK_FLAGS.get(connector)
    if attr is None:
        raise HTTPException(status_code=404, detail=f"unknown connector: {connector}")

    if connector == "nvidia_llm":
        new_value = not payload.mock
    else:
        new_value = bool(payload.mock)
    object.__setattr__(settings, attr, new_value)
    _persist_runtime_flag(attr, new_value)
    return {
        "connector": connector,
        "setting": attr,
        "value": new_value,
        "active": _is_active(connector),
        "configured_mock": _mock_flag(connector),
    }


# ── v3.6 — Training (BackgroundTask) ──────────────────────────────────────────

_TRAIN_STATUS: dict[str, dict[str, Any]] = {
    "xgboost": {"status": "idle"},
    "lgbm": {"status": "idle"},
}
_TRAIN_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_training(model_key: str, args: list[str]) -> None:
    with _TRAIN_LOCK:
        _TRAIN_STATUS[model_key] = {
            "status": "running",
            "started_at": _now_iso(),
            "finished_at": None,
            "error": None,
        }
    try:
        backend_dir = settings.base_dir
        py = sys.executable
        proc = subprocess.run(
            [py, "-m", "scripts.train_xg_predictor", *args],
            cwd=str(backend_dir),
            capture_output=True,
            text=True,
            timeout=60 * 30,
        )
        ok = proc.returncode == 0
        # Artifact-Groesse als grober Erfolgs-Indikator
        if model_key == "xgboost":
            artifact = backend_dir / "models_ml" / "artifacts" / "xg_predictor.json"
        else:
            artifact = backend_dir / "models_ml" / "artifacts" / "xg_predictor_lgbm.txt"
        artifact_size = artifact.stat().st_size if artifact.exists() else 0
        with _TRAIN_LOCK:
            _TRAIN_STATUS[model_key] = {
                "status": "done" if ok else "error",
                "started_at": _TRAIN_STATUS[model_key].get("started_at"),
                "finished_at": _now_iso(),
                "error": None if ok else (proc.stderr or "non-zero exit").strip()[-2000:],
                "artifact_size_bytes": artifact_size,
                "stdout_tail": (proc.stdout or "")[-2000:],
            }
    except Exception as exc:
        with _TRAIN_LOCK:
            _TRAIN_STATUS[model_key] = {
                "status": "error",
                "started_at": _TRAIN_STATUS[model_key].get("started_at"),
                "finished_at": _now_iso(),
                "error": str(exc),
            }
    finally:
        # M3: bei BaseException (SystemExit / KeyboardInterrupt) wuerde der
        # except oben nicht greifen, und der Status bliebe auf "running" haengen
        # — der naechste /train-Request bekommt dann 409. Wir reseten hier
        # garantiert, propagieren die Exception aber.
        with _TRAIN_LOCK:
            if _TRAIN_STATUS[model_key].get("status") == "running":
                _TRAIN_STATUS[model_key] = {
                    "status": "error",
                    "started_at": _TRAIN_STATUS[model_key].get("started_at"),
                    "finished_at": _now_iso(),
                    "error": "interrupted",
                }


@router.post("/train/xgboost", dependencies=[Depends(require_admin)])
async def train_xgboost(background: BackgroundTasks) -> dict[str, Any]:
    with _TRAIN_LOCK:
        if _TRAIN_STATUS["xgboost"].get("status") == "running":
            raise HTTPException(status_code=409, detail="xgboost training already running")
    background.add_task(_run_training, "xgboost", [])
    return {"model": "xgboost", "status": "queued"}


@router.post("/train/lgbm", dependencies=[Depends(require_admin)])
async def train_lgbm(background: BackgroundTasks) -> dict[str, Any]:
    with _TRAIN_LOCK:
        if _TRAIN_STATUS["lgbm"].get("status") == "running":
            raise HTTPException(status_code=409, detail="lgbm training already running")
    background.add_task(_run_training, "lgbm", ["--model", "lgbm"])
    return {"model": "lgbm", "status": "queued"}


@router.get("/train/status")
async def train_status() -> dict[str, Any]:
    with _TRAIN_LOCK:
        return {k: dict(v) for k, v in _TRAIN_STATUS.items()}


# ── v3.6 — Calibration Trigger ────────────────────────────────────────────────


@router.post("/calibrate", dependencies=[Depends(require_admin)])
async def calibrate(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    """Fittet Isotonic + Platt-Calibrator auf der DB-History (alle finished
    Predictions). Liefert Brier-Vorher/Nachher als Erfolgs-Metrik."""
    from analysis.backtesting import compute as bt_compute
    from analysis.calibration import apply as apply_calibration
    from analysis.calibration import fit_calibrators, load_isotonic, load_platt

    q = select(MatchPrediction).where(MatchPrediction.actual_home_score.is_not(None))
    rows = (await session.execute(q)).scalars().all()
    if len(rows) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"need >=10 finished predictions, have {len(rows)}",
        )

    pre = bt_compute(rows)
    iso, platt = fit_calibrators(rows)

    # Brier nach Isotonic-Anwendung
    class _Calibrated:
        def __init__(self, row, transformed):
            self.actual_home_score = row.actual_home_score
            self.actual_away_score = row.actual_away_score
            self.home_win_prob = transformed["home"]
            self.draw_prob = transformed["draw"]
            self.away_win_prob = transformed["away"]

    calibrated_rows = []
    for r in rows:
        cal = apply_calibration(iso, float(r.home_win_prob or 0), float(r.draw_prob or 0), float(r.away_win_prob or 0))
        if cal is None:
            continue
        calibrated_rows.append(_Calibrated(r, cal))
    post_iso = bt_compute(calibrated_rows)

    platt_rows = []
    for r in rows:
        cal = apply_calibration(platt, float(r.home_win_prob or 0), float(r.draw_prob or 0), float(r.away_win_prob or 0))
        if cal is None:
            continue
        platt_rows.append(_Calibrated(r, cal))
    post_platt = bt_compute(platt_rows)

    return {
        "n_trained_on": iso.n_trained_on,
        "brier_raw": round(pre.brier, 4),
        "brier_isotonic": round(post_iso.brier, 4),
        "brier_platt": round(post_platt.brier, 4),
        "log_loss_raw": round(pre.log_loss, 4),
        "log_loss_isotonic": round(post_iso.log_loss, 4),
        "log_loss_platt": round(post_platt.log_loss, 4),
    }


# ── v3.6 — Per-Model-Summary (Side-by-Side fuer Admin-Tab) ────────────────────


@router.get("/per_model_summary")
async def per_model_summary(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Letzte N Predictions mit Pro-Modell-Aufschluesselung. Datenquelle fuer
    den Vergleichs-Tab im Admin-Panel."""
    limit = max(1, min(100, limit))
    q = (
        select(MatchPrediction)
        .order_by(MatchPrediction.generated_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(q)).scalars().all()
    items = []
    for r in rows:
        per = r.per_model_markets or {}
        cis = r.confidence_intervals or {}
        items.append({
            "match_id": r.match_id,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
            "raw": {
                "home_win": r.home_win_prob,
                "draw": r.draw_prob,
                "away_win": r.away_win_prob,
            },
            "calibrated_isotonic": {
                "home_win": r.calibrated_home_win_prob,
                "draw": r.calibrated_draw_prob,
                "away_win": r.calibrated_away_win_prob,
            } if r.calibrated_home_win_prob is not None else None,
            "calibrated_platt": {
                "home_win": r.platt_home_win_prob,
                "draw": r.platt_draw_prob,
                "away_win": r.platt_away_win_prob,
            } if r.platt_home_win_prob is not None else None,
            "per_model": per,
            "confidence_intervals": cis,
            "actual_home_score": r.actual_home_score,
            "actual_away_score": r.actual_away_score,
        })
    return {"count": len(items), "items": items}


__all__ = ["router", "sources_router"]
