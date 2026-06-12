"""Shared DTOs for every data-source connector.

Each connector maps its own response shape into these schemas so the factors
downstream can treat openfootball / TheSportsDB / OpenLigaDB output uniformly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MatchFixture(BaseModel):
    """A scheduled or finished match — schema union across all sources."""
    model_config = ConfigDict(extra="ignore")

    source: str                                # "openfootball" | "thesportsdb" | …
    source_match_id: Optional[str] = None
    tournament: Optional[str] = None
    # 1=WM/EM final tournament, 2=qualifier, 3=Nations League, 4=friendly.
    competition_tier: int = 4
    home_code: str                             # FIFA 3-letter code, upper-cased.
    away_code: str
    home_name: str
    away_name: str
    kickoff_utc: datetime
    venue: Optional[str] = None
    ground_country: Optional[str] = None       # for travel / time-zone math
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    is_finished: bool = False


class HistoricalMatch(MatchFixture):
    """A finished match — same shape, narrower semantics."""
    is_finished: bool = True


class TeamMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    code: str
    name: str
    logo_url: Optional[str] = None
    fifa_world_ranking: Optional[int] = None
    elo_rating: Optional[int] = None
    founded_year: Optional[int] = None


class SquadInfo(BaseModel):
    """Best-effort squad availability from Wikidata. Heavy nullable because
    Wikipedia infobox parsing is fragile."""
    model_config = ConfigDict(extra="ignore")

    source: str
    code: str
    squad_size: Optional[int] = None
    avg_age: Optional[float] = None
    star_players_available: Optional[int] = None
    notable_absences: list[str] = []
    # False = "we tried and got nothing useful". The squad factor will skip
    # itself and the ensemble will re-normalise.
    available: bool = True


class VenueInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    venue_id: Optional[str] = None
    name: str
    city: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    capacity: Optional[int] = None
    utc_offset_hours: Optional[float] = None    # for travel / jet-lag math


class WeatherInfo(BaseModel):
    """Forecast at a venue around kickoff (Open-Meteo)."""
    model_config = ConfigDict(extra="ignore")

    source: str
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_kmh: Optional[float] = None
    precipitation_mm: Optional[float] = None


class InjuryNewsItem(BaseModel):
    """One mined injury/availability headline attributed to a team."""
    model_config = ConfigDict(extra="ignore")

    source: str
    team_code: str
    headline: str
    # Severity-weighted impact 0..~1 (ruled-out > doubt > minor knock).
    impact: float = 0.0
    url: Optional[str] = None
    # Player name from spaCy NER (PERSON entity), when one was detected.
    player: Optional[str] = None


# v3.3 — DTOs for the new scraper connectors.


class XgInfo(BaseModel):
    """Per-team xG aggregate from FBref / Understat. Each connector populates
    a window over the last N matches (default 10) ⇒ rolling xG quality."""
    model_config = ConfigDict(extra="ignore")

    source: str
    code: str
    matches_considered: int = 0
    xg_for_avg: Optional[float] = None       # xG per match scored
    xg_against_avg: Optional[float] = None   # xG per match conceded
    shots_on_target_avg: Optional[float] = None
    goals_for_avg: Optional[float] = None
    goals_against_avg: Optional[float] = None


class StructuredInjury(BaseModel):
    """FotMob/SofaScore have authoritative injury lists (unlike RSS keyword
    mining). Severity maps to a goal-impact weight in InjuryNewsFactor."""
    model_config = ConfigDict(extra="ignore")

    source: str
    team_code: str
    player: str
    position: Optional[str] = None
    status: str = "doubt"     # out | doubt | suspended | returning
    severity: float = 0.4     # 0..1
    return_date: Optional[datetime] = None


class LineupInfo(BaseModel):
    """Confirmed/probable lineup (≈1h before kickoff). Stars/market-value are
    used by LineupStrengthFactor to compare against season average."""
    model_config = ConfigDict(extra="ignore")

    source: str
    code: str
    is_confirmed: bool = False
    starters: list[str] = []
    # Aggregate market value of the chosen XI (Transfermarkt cross-join, in €).
    starters_value_eur: Optional[float] = None
    # Reference: average XI market value used by the side this season.
    season_avg_value_eur: Optional[float] = None
    # Number of starters who are NOT in the season's top 11 by appearances.
    bench_promotions: int = 0


class SquadValueInfo(BaseModel):
    """Transfermarkt aggregate. log10(home/away) is the SquadValueFactor tilt."""
    model_config = ConfigDict(extra="ignore")

    source: str
    code: str
    total_value_eur: Optional[float] = None
    squad_size: Optional[int] = None
    avg_value_eur: Optional[float] = None
    top11_value_eur: Optional[float] = None
