from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from analysis.factor_ensemble import EnsembleResult, FactorEnsemble
from factors.base import FactorSignal
from models_ml.poisson_goals import (
    DEFAULT_BLEND_WEIGHTS,
    MODEL_NAMES,
    blend_markets,
    bootstrap_markets,
    build_all_goal_models,
    build_goal_model,
    resolve_blend_weights,
)


@dataclass
class PredictionInput:
    home_elo: int
    away_elo: int
    home_avg_xg: float
    away_avg_xg: float
    home_avg_xg_conceded: float
    away_avg_xg_conceded: float
    home_form_pts: int  # 0..15
    away_form_pts: int
    home_sentiment: float  # [-1, 1]
    away_sentiment: float
    # Optional advanced signals (default 0.0 = neutral)
    home_momentum: float = 0.0
    away_momentum: float = 0.0
    home_controversy: float = 0.0      # 0..1, high = disagreement → lower confidence
    away_controversy: float = 0.0
    sentiment_sample_size: int = 0     # total scored posts; small N → dampen sentiment nudge
    # Venue / rest signals (defaults stay neutral so callers can ignore them).
    home_advantage: float = 0.0        # nominal home boost, [0, 0.18]. WC matches are mostly neutral.
    venue_altitude_m: float = 0.0      # ≥ 1500m starts costing stamina
    home_rest_days: int = 4
    away_rest_days: int = 4
    # Market-derived implied probabilities (priors). All-zero = no market signal.
    market_home_prob: float = 0.0
    market_draw_prob: float = 0.0
    market_away_prob: float = 0.0
    # Head-to-head record over last N meetings.
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0
    h2h_avg_goals: float = 0.0


@dataclass
class PredictionOutput:
    home_xg: float
    away_xg: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_15: float
    over_25: float
    over_35: float
    btts: float
    top_scores: List[Dict]
    confidence: float
    recommended_bet: str | None
    bet_probability: float | None
    features: Dict


class MatchPredictor:
    """Heuristic xG -> Dixon-Coles.

    Blends per side:
      - team's own attacking xG average,
      - opponent's defensive concession average,
      - Elo-delta nudge (BUG-03 fixed: half-strength so the gap doesn't double),
      - sentiment nudge (sample-size-damped),
      - form nudge,
      - home venue advantage,
      - altitude / rest-day penalty.
    The blended xG goes into Dixon-Coles. A market-implied prior (if present)
    is blended into the 1X2 output as a Bayesian-style posterior.
    """

    def __init__(self, rho: float = 0.1, max_goals: int = 6,
                 goal_model: str = "poisson", negbin_size: float = 8.0,
                 *, combine: str = "blend",
                 bootstrap_n: int = 0, bootstrap_xg_sigma: float = 0.15,
                 include_bivariate: bool = False,
                 bivariate_lambda3: float = 0.12) -> None:
        self.rho = rho
        self.max_goals = max_goals
        self.negbin_size = negbin_size
        self.goal_model_name = (goal_model or "poisson").lower()
        self.combine = (combine or "blend").lower()
        self.bootstrap_n = max(0, int(bootstrap_n))
        self.bootstrap_xg_sigma = float(bootstrap_xg_sigma)

        # v3.6 — alle drei Tor-Modelle parallel; `self.poisson` zeigt auf das
        # per `goal_model` gewaehlte Modell (Back-Compat mit MatchPredictor.predict).
        # Phase 4: optionales 4. Modell (BivariatePoisson) — Default off, damit
        # der Standard-Output unveraendert bleibt.
        self.models = build_all_goal_models(
            rho=rho, max_goals=max_goals, negbin_size=negbin_size,
            include_bivariate=include_bivariate,
            bivariate_lambda3=bivariate_lambda3,
        )
        self.blend_weights = resolve_blend_weights(self.models.keys())
        primary_key = self.goal_model_name if self.goal_model_name in self.models else "poisson"
        self.poisson = self.models[primary_key]

    def predict(self, x: PredictionInput) -> PredictionOutput:
        elo_delta = x.home_elo - x.away_elo
        # BUG-03 fix: 0.0008 per side compounded to ~35% over a 200-elo gap.
        # Halved to 0.0004 so the per-side nudge stays inside ±~8% and the
        # combined home/away effect lands near the intended ±16%.
        elo_nudge_home = 1.0 + 0.0004 * elo_delta
        elo_nudge_away = 1.0 - 0.0004 * elo_delta

        # Dampen sentiment influence when the sample is tiny.
        # 0 posts → 0% weight, 300+ posts → full weight, linear in between.
        # 300 is the scientifically defensible floor for WC crowd sentiment;
        # below this the mean is too noisy to move the line.
        sample_weight = min(1.0, x.sentiment_sample_size / 300.0)

        eff_home_sent = x.home_sentiment + 0.5 * x.home_momentum
        eff_away_sent = x.away_sentiment + 0.5 * x.away_momentum

        sent_nudge_home = 1.0 + 0.10 * eff_home_sent * sample_weight
        sent_nudge_away = 1.0 + 0.10 * eff_away_sent * sample_weight

        # Form: 5 wins (15pts) vs 5 losses (0) is a ±2.5% nudge — small but
        # meaningful when paired with sentiment and elo.
        form_nudge_home = 1.0 + 0.0033 * (x.home_form_pts - 7)
        form_nudge_away = 1.0 + 0.0033 * (x.away_form_pts - 7)

        # Venue advantage — applied only to the side designated as 'home' in
        # the fixture. WC 2026 fixtures are mostly neutral, callers can set 0.
        home_adv = 1.0 + max(0.0, min(0.18, x.home_advantage))

        # Altitude penalty: ~3% xG malus per 1000m above 1500m, both sides
        # (the lowland team usually suffers more but we don't model that yet).
        altitude_excess = max(0.0, x.venue_altitude_m - 1500.0)
        altitude_pen = 1.0 - 0.00003 * altitude_excess

        # Rest-day delta: each extra day vs opponent ≈ +0.5% xG, capped.
        rest_delta_home = max(-3, min(3, x.home_rest_days - x.away_rest_days))
        rest_nudge_home = 1.0 + 0.005 * rest_delta_home
        rest_nudge_away = 1.0 - 0.005 * rest_delta_home

        base_home = (x.home_avg_xg + x.away_avg_xg_conceded) / 2.0
        base_away = (x.away_avg_xg + x.home_avg_xg_conceded) / 2.0

        # H2H tilt — last-10 record nudges xG by up to ±5%.
        h2h_total = x.h2h_home_wins + x.h2h_draws + x.h2h_away_wins
        h2h_nudge_home = h2h_nudge_away = 1.0
        if h2h_total > 0:
            h2h_score = (x.h2h_home_wins - x.h2h_away_wins) / h2h_total
            h2h_nudge_home = 1.0 + 0.05 * h2h_score
            h2h_nudge_away = 1.0 - 0.05 * h2h_score

        home_xg = max(
            0.15,
            base_home * elo_nudge_home * sent_nudge_home * form_nudge_home
            * home_adv * altitude_pen * rest_nudge_home * h2h_nudge_home,
        )
        away_xg = max(
            0.15,
            base_away * elo_nudge_away * sent_nudge_away * form_nudge_away
            * altitude_pen * rest_nudge_away * h2h_nudge_away,
        )

        matrix = self.poisson.predict_matrix(home_xg, away_xg)
        markets = self.poisson.markets(matrix)

        # Optional Bayesian-style blend with bookmaker odds. Equal weight to
        # the market prior and the model when both are present.
        market_total = x.market_home_prob + x.market_draw_prob + x.market_away_prob
        if 0.95 <= market_total <= 1.05:
            mh = x.market_home_prob / market_total
            md = x.market_draw_prob / market_total
            ma = x.market_away_prob / market_total
            markets["home_win"] = 0.5 * markets["home_win"] + 0.5 * mh
            markets["draw"] = 0.5 * markets["draw"] + 0.5 * md
            markets["away_win"] = 0.5 * markets["away_win"] + 0.5 * ma

        top_prob = max(markets["home_win"], markets["draw"], markets["away_win"])
        spread = top_prob - min(markets["home_win"], markets["draw"], markets["away_win"])
        base_conf = 0.5 + 0.5 * spread

        controversy = max(x.home_controversy, x.away_controversy)
        controversy_penalty = 0.15 * controversy
        sample_penalty = 0.10 * (1.0 - sample_weight)
        confidence = max(0.0, min(1.0, base_conf - controversy_penalty - sample_penalty))

        bet, bet_p = self._recommend_bet(markets)

        return PredictionOutput(
            home_xg=home_xg,
            away_xg=away_xg,
            home_win_prob=markets["home_win"],
            draw_prob=markets["draw"],
            away_win_prob=markets["away_win"],
            over_15=markets["over_15"],
            over_25=markets["over_25"],
            over_35=markets["over_35"],
            btts=markets["btts"],
            top_scores=markets["top_scores"],
            confidence=confidence,
            recommended_bet=bet,
            bet_probability=bet_p,
            features={
                "elo_delta": elo_delta,
                "home_sentiment": x.home_sentiment,
                "away_sentiment": x.away_sentiment,
                "home_momentum": x.home_momentum,
                "away_momentum": x.away_momentum,
                "home_form_pts": x.home_form_pts,
                "away_form_pts": x.away_form_pts,
                "home_controversy": x.home_controversy,
                "away_controversy": x.away_controversy,
                "sample_size": x.sentiment_sample_size,
                "sample_weight": round(sample_weight, 3),
                "home_advantage": x.home_advantage,
                "venue_altitude_m": x.venue_altitude_m,
                "rest_delta": rest_delta_home,
                "h2h_home_wins": x.h2h_home_wins,
                "h2h_draws": x.h2h_draws,
                "h2h_away_wins": x.h2h_away_wins,
                "market_used": 0.95 <= market_total <= 1.05,
            },
        )

    def predict_from_signals(
        self,
        signals: List[FactorSignal],
        base_home_xg: float,
        base_away_xg: float,
        *,
        market_prior: tuple[float, float, float] | None = None,
    ) -> tuple[PredictionOutput, EnsembleResult]:
        """MULTIFACTOR-08 path. Replaces the per-feature nudges from predict()
        with a single re-normalised ensemble of FactorSignals. base_*_xg are
        the same blended attack/defence averages predict() uses; everything
        beyond that is delegated to the factors.

        v3.6: Rechnet alle 3 Tor-Modelle parallel (poisson, negbin, glm_poisson).
        Primaere Ausgabe ist der gewichtete Blend (combine='blend') oder das
        per goal_model gewaehlte Modell (combine='primary'). Pro-Modell-Markets
        und Bootstrap-CIs liegen in PredictionOutput.features unter
        "per_model" und "confidence_intervals".

        Returns the PredictionOutput plus the EnsembleResult so the caller can
        persist the per-factor breakdown for explainability.
        """
        ensemble = FactorEnsemble().combine(signals)
        home_xg = max(0.15, base_home_xg * ensemble.lambda_home_multiplier)
        away_xg = max(0.15, base_away_xg * ensemble.lambda_away_multiplier)

        # Alle 3 Modelle parallel — gleiche Lambdas, unterschiedliche
        # Marginal-/Korrektur-Konfiguration -> distinkte Vorhersagen.
        per_model_markets: Dict[str, Dict[str, Any]] = {}
        per_model_ci: Dict[str, Dict[str, Any]] = {}
        for name, model in self.models.items():
            mk = model.markets(model.predict_matrix(home_xg, away_xg))
            per_model_markets[name] = mk
            if self.bootstrap_n > 0:
                ci = bootstrap_markets(
                    model, home_xg, away_xg,
                    n=self.bootstrap_n, xg_sigma=self.bootstrap_xg_sigma,
                )
                per_model_ci[name] = {k: list(v) for k, v in ci.items()}

        # Phase 4: Gewichte folgen der tatsaechlichen Modell-Menge (inkl.
        # optionalem bivariate) — fuer das Standard-Trio identisch zu
        # DEFAULT_BLEND_WEIGHTS, d.h. der Default-Output bleibt unveraendert.
        blend_w = self.blend_weights
        blended = blend_markets(per_model_markets, blend_w)
        if self.bootstrap_n > 0:
            blended_ci_arrays: Dict[str, list[float]] = {}
            for name, ci in per_model_ci.items():
                w = blend_w.get(name, 0.0)
                for key, (p5, p50, p95) in ci.items():
                    arr = blended_ci_arrays.setdefault(key, [0.0, 0.0, 0.0])
                    arr[0] += w * p5
                    arr[1] += w * p50
                    arr[2] += w * p95
            total_w = sum(blend_w.values()) or 1.0
            per_model_ci["blended"] = {
                k: [v[0] / total_w, v[1] / total_w, v[2] / total_w]
                for k, v in blended_ci_arrays.items()
            }

        if self.combine == "primary":
            markets = dict(per_model_markets.get(self.goal_model_name) or blended)
        else:
            markets = dict(blended)

        # Optional market-prior blend — selbe Logik wie zuvor; nur auf das
        # primaere Modell angewandt (per Modell unkalibriert speichern wir
        # in per_model_markets weiter, damit Vergleichs-UI ehrlich bleibt).
        if market_prior is not None:
            mh, md, ma = market_prior
            total = mh + md + ma
            if 0.95 <= total <= 1.05:
                mh, md, ma = mh / total, md / total, ma / total
                markets["home_win"] = 0.5 * markets["home_win"] + 0.5 * mh
                markets["draw"] = 0.5 * markets["draw"] + 0.5 * md
                markets["away_win"] = 0.5 * markets["away_win"] + 0.5 * ma

        bet, bet_p = self._recommend_bet(markets)

        feature_payload = {
            "lambda_home_multiplier": ensemble.lambda_home_multiplier,
            "lambda_away_multiplier": ensemble.lambda_away_multiplier,
            "ensemble_confidence": ensemble.confidence,
            "base_home_xg": base_home_xg,
            "base_away_xg": base_away_xg,
            "active_factors": [s.name for s in signals if s.available],
            "skipped_factors": [s.name for s in signals if not s.available],
            "market_used": market_prior is not None,
            "goal_model_combine": self.combine,
            "goal_model_primary": self.goal_model_name,
            "blend_weights": dict(blend_w),
            "per_model": per_model_markets,
            "confidence_intervals": per_model_ci,
        }

        return (
            PredictionOutput(
                home_xg=home_xg,
                away_xg=away_xg,
                home_win_prob=markets["home_win"],
                draw_prob=markets["draw"],
                away_win_prob=markets["away_win"],
                over_15=markets["over_15"],
                over_25=markets["over_25"],
                over_35=markets["over_35"],
                btts=markets["btts"],
                top_scores=markets["top_scores"],
                # Confidence comes from the ensemble (factor agreement +
                # average factor confidence). The legacy spread-based confidence
                # is intentionally dropped here.
                confidence=ensemble.confidence,
                recommended_bet=bet,
                bet_probability=bet_p,
                features=feature_payload,
            ),
            ensemble,
        )

    @staticmethod
    def _recommend_bet(markets: Dict) -> tuple[str | None, float | None]:
        candidates = [
            ("home_win",  markets["home_win"]),
            ("draw",      markets["draw"]),
            ("away_win",  markets["away_win"]),
            ("over_25",   markets["over_25"]),
            ("under_25",  1 - markets["over_25"]),
            ("btts_yes",  markets["btts"]),
            ("btts_no",   1 - markets["btts"]),
        ]
        candidates.sort(key=lambda c: -c[1])
        best = candidates[0]
        if best[1] >= 0.55:
            return best[0], float(best[1])
        return None, None
