"""Static geo data for the 16 FIFA World Cup 2026 host stadiums.

Used by the weather connector (lat/lon), the altitude factor (metres) and any
travel/jet-lag math (UTC offset in June, when all matches are played — DST is
already baked in; Mexico stays on CST year-round).

Coordinates are approximate stadium centroids; altitude in metres. Resolution is
by case-insensitive keyword match against `WM2026Match.venue`, which is stored
as "<Stadium>, <City>" in the YAML configs, so either token matches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    name: str
    city: str
    country: str
    latitude: float
    longitude: float
    altitude_m: float
    utc_offset_hours: float
    keywords: tuple[str, ...]


VENUES: tuple[Venue, ...] = (
    Venue("MetLife Stadium", "New York/New Jersey", "USA", 40.813, -74.074, 7, -4, ("metlife", "new york", "new jersey", "rutherford")),
    Venue("AT&T Stadium", "Dallas", "USA", 32.748, -97.093, 150, -5, ("at&t", "dallas", "arlington")),
    Venue("NRG Stadium", "Houston", "USA", 29.685, -95.411, 15, -5, ("nrg", "houston")),
    Venue("Arrowhead Stadium", "Kansas City", "USA", 39.049, -94.484, 230, -5, ("arrowhead", "kansas")),
    Venue("Mercedes-Benz Stadium", "Atlanta", "USA", 33.755, -84.401, 320, -4, ("mercedes", "atlanta")),
    Venue("Hard Rock Stadium", "Miami", "USA", 25.958, -80.239, 3, -4, ("hard rock", "miami")),
    Venue("Gillette Stadium", "Boston", "USA", 42.091, -71.264, 90, -4, ("gillette", "boston", "foxborough")),
    Venue("Lincoln Financial Field", "Philadelphia", "USA", 39.901, -75.168, 12, -4, ("lincoln financial", "philadelphia")),
    Venue("Levi's Stadium", "San Francisco Bay Area", "USA", 37.403, -121.970, 4, -7, ("levi", "san francisco", "santa clara", "bay area")),
    Venue("SoFi Stadium", "Los Angeles", "USA", 33.953, -118.339, 30, -7, ("sofi", "los angeles", "inglewood")),
    Venue("Lumen Field", "Seattle", "USA", 47.595, -122.331, 5, -7, ("lumen", "seattle")),
    Venue("BMO Field", "Toronto", "Canada", 43.633, -79.418, 80, -4, ("bmo", "toronto")),
    Venue("BC Place", "Vancouver", "Canada", 49.277, -123.112, 3, -7, ("bc place", "vancouver")),
    Venue("Estadio Azteca", "Mexico City", "Mexico", 19.303, -99.150, 2240, -6, ("azteca", "banorte", "mexico city")),
    Venue("Estadio Akron", "Guadalajara", "Mexico", 20.681, -103.463, 1566, -6, ("akron", "guadalajara", "zapopan")),
    Venue("Estadio BBVA", "Monterrey", "Mexico", 25.669, -100.244, 540, -6, ("bbva", "monterrey")),
)


def resolve(venue: str | None) -> Venue | None:
    """Best-effort lookup of a YAML venue string to a known host stadium."""
    if not venue:
        return None
    v = venue.lower()
    for entry in VENUES:
        if any(kw in v for kw in entry.keywords):
            return entry
    return None


def haversine_km(a: Venue, b: Venue) -> float:
    """Great-circle distance between two venues in km."""
    r = 6371.0
    p1, p2 = math.radians(a.latitude), math.radians(b.latitude)
    dphi = math.radians(b.latitude - a.latitude)
    dlmb = math.radians(b.longitude - a.longitude)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def travel_between(prev_venue: str | None, current_venue: str | None) -> dict[str, float] | None:
    """{'km', 'tz_shift'} for a team moving prev → current host stadium, or None
    when either venue can't be resolved (e.g. a team's tournament opener)."""
    a, b = resolve(prev_venue), resolve(current_venue)
    if a is None or b is None:
        return None
    return {"km": round(haversine_km(a, b), 1), "tz_shift": abs(a.utc_offset_hours - b.utc_offset_hours)}


__all__ = ["Venue", "VENUES", "resolve", "haversine_km", "travel_between"]
