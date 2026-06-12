"""Injury-news strength factor.

Reads injury/availability items mined from football RSS feeds (BBC / Guardian /
ESPN) by `data_sources.rss_news`, attributed to each team via name + spaCy NER
and a keyword severity model. A side missing key players loses xG relative to a
fully-fit opponent.

    impact_side   = Σ item.impact            (per-player severity, 0..~1 each)
    delta         = clip(impact_away - impact_home, -3, 3)
    home_strength = 1.0 + 0.02 * delta       (fewer home injuries ⇒ home edge)
    away_strength = 1.0 - 0.02 * delta
Confidence is low (0.40): RSS headlines are a noisy, lagging signal.
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_K = 0.02
_CLIP = 3.0


def _impact(items: list) -> float:
    total = 0.0
    for it in items or []:
        total += float(getattr(it, "impact", 0.0) or 0.0)
    return total


class InjuryNewsFactor(Factor):
    name = "injury_news"
    default_weight = 0.06

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home_items = ctx.news_home or []
        away_items = ctx.news_away or []
        if not home_items and not away_items:
            return self._neutral(source="rss", reason="no_news")

        home_impact = _impact(home_items)
        away_impact = _impact(away_items)
        delta = max(-_CLIP, min(_CLIP, away_impact - home_impact))
        home_strength = 1.0 + _K * delta
        away_strength = 1.0 - _K * delta

        source, cached_at = source_from_provenance(ctx, "news")
        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=0.40,
            available=True,
            source=source if source != "neutral" else "rss",
            cached_at=cached_at,
            raw_data={
                "home_impact": round(home_impact, 2),
                "away_impact": round(away_impact, 2),
                "home_items": len(home_items),
                "away_items": len(away_items),
            },
        )
