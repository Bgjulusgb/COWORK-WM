"""Venue-altitude strength factor.

Endurance drops at altitude: VO2max and high-intensity running fall ~6-7% per
1000 m above ~1500 m, which suppresses total goals. For WC 2026 this hits the
Mexican venues (Mexico City 2240 m, Guadalajara 1566 m) hardest.

    excess        = max(0, altitude_m - 1500)
    damp          = 1 - 0.00003 * excess        (≈ -2.2% xG at 2240 m, both sides)
    home_strength = away_strength = damp
We apply it symmetrically (lower goal total) rather than as a home tilt — the
host-nation acclimatisation edge is already carried by TournamentContextFactor,
so this avoids double-counting. (McSharry 2007, BMJ — altitude & match outcome.)
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal

_THRESHOLD_M = 1500.0
_PER_M = 0.00003


def _altitude(ctx: FactorContext) -> float:
    # Prefer a resolved VenueInfo; fall back to the bundled altitude table.
    info = ctx.venue_info
    if info is not None and getattr(info, "altitude_m", None):
        return float(info.altitude_m)
    try:
        from scripts.team_real_data import get_venue_altitude

        return float(get_venue_altitude(ctx.venue) or 0.0)
    except Exception:
        return 0.0


class VenueAltitudeFactor(Factor):
    name = "venue_altitude"
    default_weight = 0.05

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        altitude = _altitude(ctx)
        if altitude <= _THRESHOLD_M:
            return self._neutral(source="venue", reason="lowland_venue")

        excess = altitude - _THRESHOLD_M
        damp = max(0.85, 1.0 - _PER_M * excess)

        return FactorSignal(
            name=self.name,
            home_strength=damp,
            away_strength=damp,
            weight=self.weight,
            confidence=0.60,
            available=True,
            source="venue",
            kind="global",  # symmetric goal damp, not a home/away tilt
            raw_data={"altitude_m": round(altitude, 0), "goal_damp": round(damp, 4)},
        )
