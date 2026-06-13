"""Factor-interface + FactorSignal schema for the multi-factor predictor.

A Factor is an independent signal source (Elo, form, head-to-head, sentiment …)
that produces a normalised home/away strength multiplier. The FactorEnsemble
combines all available signals into the lambda multipliers Dixon-Coles consumes.

Strength convention:
    1.0   = neutral, the factor has no opinion
    > 1.0 = advantage for that side
    < 1.0 = disadvantage for that side
The recommended live range is 0.5..1.5; values outside [0.3, 2.5] are clamped
so a single broken connector cannot dominate the ensemble.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FactorSignal(BaseModel):
    """One factor's contribution. Validated; used both internally and for API
    responses (via FactorSignalResponse mirror in db/schemas.py)."""
    model_config = ConfigDict(extra="forbid")

    name: str
    home_strength: float = Field(ge=0.0, le=3.0)
    away_strength: float = Field(ge=0.0, le=3.0)
    weight: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    available: bool
    source: str
    # "tilt"   = favours one side; averaged into the home/away λ split.
    # "global" = symmetric goal modifier (weather, altitude); multiplied onto
    #            both λ *after* the average so the averaging can't dilute it to
    #            near-1.0. home_strength (==away_strength) carries the factor.
    kind: str = "tilt"
    raw_data: dict[str, Any] | None = None
    cached_at: Optional[datetime] = None

    @field_validator("home_strength", "away_strength")
    @classmethod
    def _clamp_extremes(cls, v: float) -> float:
        # Guard against a single bad connector swinging the ensemble too hard.
        return max(0.3, min(2.5, v))


@dataclass
class FactorContext:
    """Everything a Factor needs to compute its signal.

    Populated by match_service.run_crawl_and_predict before fanning out to
    factors. Mutable so the orchestrator can fill external-data fields
    incrementally without rebuilding the dataclass each time.
    """
    match_id: str
    config: dict[str, Any]              # parsed YAML
    home_code: str
    away_code: str
    kickoff_utc: datetime
    venue: str | None = None

    # Sentiment pipeline output (filled by the existing Reddit path).
    sentiment_payload: dict[str, Any] | None = None

    # Filled by DataSourceOrchestrator. Default empty so factors that only
    # need the YAML (e.g. EloStrengthFactor) can run without any orchestrator.
    historical_matches_home: list[Any] = field(default_factory=list)
    historical_matches_away: list[Any] = field(default_factory=list)
    head_to_head: list[Any] = field(default_factory=list)
    fixture_meta: Any | None = None
    team_meta_home: Any | None = None
    team_meta_away: Any | None = None
    squad_meta_home: Any | None = None
    squad_meta_away: Any | None = None
    fixtures_for_context: list[Any] = field(default_factory=list)

    # v3 additional-factor inputs. Filled by match_service / orchestrator; each
    # factor that needs one of these falls back to a neutral signal when it's
    # missing, so populating them is purely additive.
    venue_info: Any | None = None                     # VenueInfo (coords/altitude)
    weather: Any | None = None                        # WeatherInfo at kickoff
    news_home: list[Any] = field(default_factory=list)   # InjuryNewsItem list
    news_away: list[Any] = field(default_factory=list)
    market_implied: tuple[float, float, float] | None = None  # (home, draw, away)
    rest_days_home: int | None = None
    rest_days_away: int | None = None
    # Travel since each side's previous fixture: {"km": float, "tz_shift": float}.
    travel_home: dict[str, float] | None = None
    travel_away: dict[str, float] | None = None

    # Per-data-source provenance, filled by DataSourceOrchestrator. Keyed by a
    # logical name ("history_home", "h2h", "squad_home", …) → {source, mode,
    # fetched_at}. Factors read this to set FactorSignal.source / cached_at so
    # the UI can show a live / cached / mock badge.
    provenance: dict[str, Any] = field(default_factory=dict)

    # v3.3 — new live data slices for FBref/Understat (xG truth), FotMob/
    # SofaScore (lineups + structured injuries) and Transfermarkt (squad value).
    # Each factor that uses one of these falls back to neutral when missing.
    xg_home: Any | None = None                    # XgInfo
    xg_away: Any | None = None                    # XgInfo
    lineup_home: Any | None = None                # LineupInfo
    lineup_away: Any | None = None                # LineupInfo
    structured_injuries_home: list[Any] = field(default_factory=list)
    structured_injuries_away: list[Any] = field(default_factory=list)
    squad_value_home: Any | None = None           # SquadValueInfo
    squad_value_away: Any | None = None           # SquadValueInfo
    # PageRank network strength — filled by NetworkStrengthFactor via an
    # offline-computed snapshot in ``models_ml/artifacts/network_strength.json``.
    network_strength_home: float | None = None
    network_strength_away: float | None = None

    # Live odds fetched by the_odds_api connector (free-OSS Phase 4). Filled by
    # the orchestrator when ODDS_API_KEY is set and the connector is live. The
    # pipeline falls back to these whenever the CLI didn't pass --odds-* flags
    # (so the edge table works without any manual entry). Shape is the same
    # parsed-list-of-decimals that ``parse_odds`` returns; keys are stable:
    #   "1x2"      → [home, draw, away]
    #   "ou_2_5"   → [over, under]
    #   "btts"     → [yes, no]
    live_odds: dict[str, list[float]] | None = None


class Factor(ABC):
    """Common shape every factor module implements.

    Subclasses declare `name` and `default_weight` as class attributes. The
    registry uses these to wire the factor into the ensemble.
    """
    name: str = "unnamed"
    default_weight: float = 0.0

    def __init__(self, weight: float | None = None) -> None:
        self.weight = self.default_weight if weight is None else weight

    @abstractmethod
    async def compute(self, ctx: FactorContext) -> FactorSignal: ...

    def _neutral(self, source: str = "neutral", reason: str = "not_available") -> FactorSignal:
        """Helper for the available=False path. Subclasses call this when their
        data source is missing so the ensemble can re-normalise the weights."""
        return FactorSignal(
            name=self.name,
            home_strength=1.0,
            away_strength=1.0,
            weight=self.weight,
            confidence=0.0,
            available=False,
            source=source,
            raw_data={"reason": reason},
        )


_PROV_RANK = {"live": 3, "cache": 2, "mock": 1, "error": 0}


def source_from_provenance(ctx: "FactorContext", *keys: str) -> tuple[str, Optional[datetime]]:
    """Collapse one-or-more ``ctx.provenance`` entries into a single
    ``(source, cached_at)`` pair for a FactorSignal.

    The orchestrator records ``{source, mode, fetched_at}`` per logical data
    slice ("history_home", "h2h", …). A factor often reads several slices, so
    we pick the *best* provenance (live > cache > mock > error) as the badge
    source and the newest ``fetched_at`` as ``cached_at``. Used so the UI can
    show a live / cached / mock badge that matches what actually fed the signal.
    """
    best_rank = -1
    source = "neutral"
    cached_at: Optional[datetime] = None
    for k in keys:
        p = ctx.provenance.get(k) or {}
        mode = p.get("mode")
        if not mode:
            continue
        rank = _PROV_RANK.get(mode, 0)
        if rank > best_rank:
            best_rank = rank
            source = p.get("source") or mode
        fetched = p.get("fetched_at")
        if isinstance(fetched, str):
            try:
                dt = datetime.fromisoformat(fetched)
                if cached_at is None or dt > cached_at:
                    cached_at = dt
            except ValueError:
                pass
        elif isinstance(fetched, datetime):
            if cached_at is None or fetched > cached_at:
                cached_at = fetched
    return source, cached_at
