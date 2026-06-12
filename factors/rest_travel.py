"""Rest + travel strength factor.

Two recovery effects on the team that's more worn down going into kickoff:

1. **Rest days.** < 3 days' turnaround lowers high-intensity output and raises
   injury risk vs a better-rested opponent (Nédélec 2012; Dupont 2010).
       rest_term = 0.005 * clip(rest_home - rest_away, -3, 3)
2. **Travel / jet-lag.** Distance and time-zone shift since the previous fixture
   degrade performance (Fowler 2017 on long-haul travel in athletes).
       fatigue(side) = 0.000015·km + 0.010·|tz_shift|   (capped 0.10)
       travel_term   = fatigue(away) - fatigue(home)     (away more tired ⇒ home edge)

   home_strength = 1.0 + clip(rest_term + travel_term, -0.12, 0.12)

Travel data comes from the real 2026 schedule (orchestrator); group-stage
openers have no prior fixture, so it falls back to rest-only or neutral.
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal

_PER_DAY = 0.005
_REST_CLIP = 3
_KM_COEF = 0.000015
_TZ_COEF = 0.010
_FATIGUE_CAP = 0.10
_TILT_CLIP = 0.12


def _fatigue(travel: dict | None) -> float:
    if not travel:
        return 0.0
    km = float(travel.get("km", 0.0) or 0.0)
    tz = float(travel.get("tz_shift", 0.0) or 0.0)
    return min(_FATIGUE_CAP, _KM_COEF * km + _TZ_COEF * tz)


class RestTravelFactor(Factor):
    name = "rest_travel"
    default_weight = 0.06

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        rh, ra = ctx.rest_days_home, ctx.rest_days_away
        have_rest = rh is not None and ra is not None
        have_travel = ctx.travel_home is not None or ctx.travel_away is not None
        if not have_rest and not have_travel:
            return self._neutral(source="schedule", reason="schedule_unknown")

        rest_delta = max(-_REST_CLIP, min(_REST_CLIP, int(rh) - int(ra))) if have_rest else 0
        rest_term = _PER_DAY * rest_delta

        home_fatigue = _fatigue(ctx.travel_home)
        away_fatigue = _fatigue(ctx.travel_away)
        travel_term = away_fatigue - home_fatigue

        tilt = max(-_TILT_CLIP, min(_TILT_CLIP, rest_term + travel_term))
        return FactorSignal(
            name=self.name,
            home_strength=1.0 + tilt,
            away_strength=1.0 - tilt,
            weight=self.weight,
            confidence=0.55 if have_travel else 0.45,
            available=True,
            source="schedule",
            raw_data={
                "rest_home": rh, "rest_away": ra, "rest_delta": rest_delta,
                "home_travel_km": (ctx.travel_home or {}).get("km") if ctx.travel_home else None,
                "away_travel_km": (ctx.travel_away or {}).get("km") if ctx.travel_away else None,
                "home_fatigue": round(home_fatigue, 4),
                "away_fatigue": round(away_fatigue, 4),
            },
        )
