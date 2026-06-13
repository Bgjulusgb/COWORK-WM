"""Build a :class:`FactorContext` from a match config and toggle the mock/live
data profile — the two glue pieces between a YAML file and the factor fan-out.

The heavy lifting (fetching, factor maths) lives in the existing modules; this
module only adapts their inputs so the workflow stays a thin orchestration layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config.settings import settings
from factors.base import FactorContext

# Every ``use_mock_*`` flag the connectors honour. Forced True in mock mode so a
# clone runs fully offline with no API keys and no network round-trips.
_MOCK_FLAGS = (
    "use_mock_crawler",
    "use_mock_openfootball",
    "use_mock_thesportsdb",
    "use_mock_openligadb",
    "use_mock_wikidata",
    "use_mock_weather",
    "use_mock_rss",
    "use_mock_clubelo",
    "use_mock_football_data",
    "use_mock_fbref",
    "use_mock_understat",
    "use_mock_fotmob",
    "use_mock_sofascore",
    "use_mock_transfermarkt",
    "use_mock_odds_api",
)


def apply_runtime_profile(mode: str) -> None:
    """Flip the global ``settings`` singleton into ``mock`` or ``live`` mode.

    ``mock`` forces every connector to its deterministic offline payload — the
    default for ``wm2026 predict`` so the repo is runnable out of the box.
    ``live`` leaves the ``.env`` toggles untouched (each connector still
    degrades to its mock on a network error, per the connector contract).
    """
    mode = (mode or "mock").lower()
    if mode == "mock":
        for flag in _MOCK_FLAGS:
            if hasattr(settings, flag):
                # BaseSettings guards plain assignment; mirror settings.py's own
                # object.__setattr__ escape hatch used by reload_runtime_flags().
                object.__setattr__(settings, flag, True)
        # The NVIDIA LLM scorer needs a paid key — never call it in mock mode.
        object.__setattr__(settings, "use_nvidia_llm", False)
    elif mode != "live":
        raise ValueError(f"unknown mode {mode!r} (expected 'mock' or 'live')")


def load_match_config(path: str | Path) -> dict[str, Any]:
    """Load + lightly validate a match YAML into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"match config not found: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "match" not in cfg or "teams" not in cfg:
        raise ValueError(
            f"{p} is missing the required 'match:' and/or 'teams:' blocks"
        )
    return cfg


def _parse_kickoff(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        ko = raw
    else:
        text = str(raw or "").strip().replace("Z", "+00:00")
        try:
            ko = datetime.fromisoformat(text)
        except ValueError:
            ko = datetime.now(timezone.utc)
    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=timezone.utc)
    return ko


def build_context(cfg: dict[str, Any]) -> FactorContext:
    """Map a parsed match config onto a fresh :class:`FactorContext`.

    Only the always-present YAML fields are wired here; everything external
    (history, xG, weather, squads …) is filled afterwards by the
    :class:`DataSourceOrchestrator` in :mod:`wm2026.pipeline`.
    """
    match = cfg.get("match", {})
    teams = cfg.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})

    ctx = FactorContext(
        match_id=match.get("id") or "wm2026_match",
        config=cfg,
        home_code=str(home.get("code") or home.get("fifa_code") or "HOM").upper(),
        away_code=str(away.get("code") or away.get("fifa_code") or "AWY").upper(),
        kickoff_utc=_parse_kickoff(match.get("kickoff_utc")),
        venue=match.get("venue"),
        sentiment_payload=None,   # filled by the optional sentiment step
    )

    # Bookmaker-implied 1X2 (Phase 6 input). When the config carries odds we
    # pre-fill ctx.market_implied so the MarketOddsFactor tilts λ — exactly the
    # path match_service uses. The vig-free edge table is computed separately.
    odds = (match.get("bookmaker_odds_1x2") or cfg.get("odds_1x2"))
    implied = _implied_from_odds(odds)
    if implied is not None:
        ctx.market_implied = implied

    return ctx


def _implied_from_odds(odds: Any) -> tuple[float, float, float] | None:
    """Vig-free (home, draw, away) implied probabilities, or None."""
    from wm2026.edge import devig, parse_odds

    values = parse_odds(odds) if isinstance(odds, str) else odds
    if not values or len(values) < 3:
        return None
    fair, _ = devig(list(values)[:3])
    if len(fair) < 3:
        return None
    return (fair[0], fair[1], fair[2])


def apply_overrides(ctx: FactorContext, overrides: dict[str, Any] | None) -> list[str]:
    """Inject Claude-researched values into the context (Cowork v2).

    Applied **after** Phase-1 data collection so researched data overrides the
    connectors' mock/error fallbacks. Writes the λ-driving fields into the
    config (so ``_base_xg`` and the MLE estimator see them) and the typed slices
    the factors read, then stamps provenance ``{"mode": "research"}`` so the
    report shows a `research` badge and the "all sources mock" warning clears.

    Schema (each key optional)::

        {"xg": {"home": {"avg_xg_season": _, "avg_xg_conceded": _}, "away": {…}},
         "elo": {"home": _, "away": _},
         "weather": {"temp_c": _, "wind_kmh": _, "precipitation_mm": _, "humidity_pct": _},
         "sentiment": {"sample_size": _, "home_sentiment": _, …}}

    Returns the list of applied keys (for the report + logging).
    """
    if not overrides:
        return []
    applied: list[str] = []
    teams = ctx.config.setdefault("teams", {}) if isinstance(ctx.config, dict) else {}

    def _stamp(key: str) -> None:
        ctx.provenance[key] = {"source": "claude-research", "mode": "research", "fetched_at": None}

    xg = overrides.get("xg")
    if isinstance(xg, dict):
        for side in ("home", "away"):
            spec = xg.get(side)
            if isinstance(spec, dict):
                team = teams.setdefault(side, {})
                if "avg_xg_season" in spec:
                    team["avg_xg_season"] = float(spec["avg_xg_season"])
                if "avg_xg_conceded" in spec:
                    team["avg_xg_conceded"] = float(spec["avg_xg_conceded"])
                _stamp(f"xg_{side}")
                applied.append(f"xg.{side}")

    elo = overrides.get("elo")
    if isinstance(elo, dict):
        for side in ("home", "away"):
            if elo.get(side) is not None:
                teams.setdefault(side, {})["elo_rating"] = float(elo[side])
                applied.append(f"elo.{side}")

    weather = overrides.get("weather")
    if isinstance(weather, dict):
        try:
            from data_sources.schemas import WeatherInfo
            fields = {k: weather[k] for k in
                      ("temp_c", "humidity_pct", "wind_kmh", "precipitation_mm")
                      if k in weather and weather[k] is not None}
            ctx.weather = WeatherInfo(source="claude-research", **fields)
            _stamp("weather")
            applied.append("weather")
        except Exception:  # pragma: no cover - schema/optional-dep guard
            pass

    sentiment = overrides.get("sentiment")
    if isinstance(sentiment, dict):
        ctx.sentiment_payload = sentiment
        applied.append("sentiment")

    # Count markets (corners / cards) — λ values from a stats connector or hand-
    # researched. Stored on ctx.config so the pipeline picks them up without a
    # dedicated FactorContext field (no schema migration needed).
    corners = overrides.get("corners")
    if isinstance(corners, dict) and corners.get("lambda_home") is not None \
            and corners.get("lambda_away") is not None:
        ctx.config["corners"] = {
            "lambda_home": float(corners["lambda_home"]),
            "lambda_away": float(corners["lambda_away"]),
        }
        _stamp("corners")
        applied.append("corners")
    cards = overrides.get("cards")
    if isinstance(cards, dict) and cards.get("lambda_total") is not None:
        ctx.config["cards"] = {"lambda_total": float(cards["lambda_total"])}
        _stamp("cards")
        applied.append("cards")

    return applied


def overrides_template(cfg: dict[str, Any]) -> dict[str, Any]:
    """A blank ``--overrides-json`` scaffold for Claude to fill (research → re-run)."""
    teams = cfg.get("teams", {})
    home = (teams.get("home", {}) or {}).get("name", "home")
    away = (teams.get("away", {}) or {}).get("name", "away")
    return {
        "_instructions": (
            "Recherchiere die null-Werte per Web Search, trage value + _source ein, "
            "dann: wm2026 predict <match> --overrides-json DIESE_DATEI.json "
            "(Quoten zusätzlich via --odds/--odds-ou/--odds-btts)."
        ),
        "_fixture": f"{home} vs {away}",
        "xg": {
            "home": {"avg_xg_season": None, "avg_xg_conceded": None, "_source": None},
            "away": {"avg_xg_season": None, "avg_xg_conceded": None, "_source": None},
        },
        "elo": {"home": None, "away": None, "_source": None},
        "weather": {"temp_c": None, "wind_kmh": None, "precipitation_mm": None,
                    "humidity_pct": None, "_source": None},
        "sentiment": {"sample_size": None, "home_sentiment": None,
                      "away_sentiment": None, "_source": None},
        # Count markets — only emitted if you research / inject the λ values.
        "corners": {"lambda_home": None, "lambda_away": None, "_source": None},
        "cards":   {"lambda_total": None, "_source": None},
    }


def synth_config(
    *,
    home_team: str,
    away_team: str,
    home_code: str | None = None,
    away_code: str | None = None,
    stage: str = "Group",
    kickoff: str | None = None,
    venue: str | None = None,
    home_xg: float = 1.40,
    away_xg: float = 1.30,
    home_xga: float = 1.30,
    away_xga: float = 1.40,
    home_elo: int = 1700,
    away_elo: int = 1700,
    odds_1x2: str | None = None,
) -> dict[str, Any]:
    """Build a minimal in-memory match config from CLI flags.

    Lets ``wm2026 predict --home Germany --away Brazil`` work without writing a
    YAML file first. Sensible WC-neutral defaults fill anything not provided.
    """
    code_h = (home_code or home_team[:3]).upper()
    code_a = (away_code or away_team[:3]).upper()
    slug = f"{code_h.lower()}_vs_{code_a.lower()}"
    cfg: dict[str, Any] = {
        "match": {
            "id": f"wm2026_{slug}",
            "tournament": "FIFA World Cup 2026",
            "phase": stage,
            "kickoff_utc": kickoff or datetime.now(timezone.utc).isoformat(),
            "venue": venue,
        },
        "teams": {
            "home": {
                "name": home_team, "code": code_h, "fifa_code": code_h,
                "elo_rating": home_elo, "avg_xg_season": home_xg,
                "avg_xg_conceded": home_xga, "form_last5": [],
            },
            "away": {
                "name": away_team, "code": code_a, "fifa_code": code_a,
                "elo_rating": away_elo, "avg_xg_season": away_xg,
                "avg_xg_conceded": away_xga, "form_last5": [],
            },
        },
    }
    if odds_1x2:
        cfg["match"]["bookmaker_odds_1x2"] = odds_1x2
    return cfg


__all__ = [
    "apply_runtime_profile",
    "load_match_config",
    "build_context",
    "apply_overrides",
    "overrides_template",
    "synth_config",
]
