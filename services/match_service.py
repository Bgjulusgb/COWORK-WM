from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger("services.match_service")

from analysis.advanced_metrics import compute_advanced
from analysis.ensemble_scorer import SentimentEnsemble
from analysis.match_predictor import MatchPredictor, PredictionInput
from analysis.social_momentum import momentum, post_velocity, weighted_sentiment
from analysis.trend_analyzer import compute_trend
from config.settings import settings
from crawler.mock_reddit import MockRedditCrawler
from db.models import (
    FactorSnapshot,
    MatchPrediction,
    PostFlag,
    RedditPost,
    SentimentScore,
    SentimentSnapshot,
    WM2026Match,
)
from factors.base import FactorContext
from factors.registry import get_active_factors
from preprocessing.bot_filter import is_bot_author, is_low_quality
from preprocessing.language_detector import detect_language, translate_to_english_async
from preprocessing.pipeline import PreprocessingPipeline
from utils.config_loader import load_match_config


_FORM_PTS = {"W": 3, "D": 1, "L": 0}


def _form_to_points(form: List[str]) -> int:
    return sum(_FORM_PTS.get(c, 0) for c in form)


async def upsert_match_from_config(session: AsyncSession, config_path: Path) -> WM2026Match:
    cfg = load_match_config(config_path)
    m = cfg["match"]
    h, a = cfg["teams"]["home"], cfg["teams"]["away"]

    existing = await session.get(WM2026Match, m["id"])
    if existing is None:
        match = WM2026Match(
            id=m["id"],
            group=m["group"],
            phase=m["phase"],
            home_team=h["code"],
            away_team=a["code"],
            home_name=h["name"],
            away_name=a["name"],
            home_flag=h.get("flag_emoji"),
            away_flag=a.get("flag_emoji"),
            kickoff_utc=_parse_iso(m["kickoff_utc"]),
            venue=m.get("venue"),
            status="scheduled",
            config_path=str(config_path),
        )
        session.add(match)
    else:
        existing.group = m["group"]
        existing.phase = m["phase"]
        existing.home_team = h["code"]
        existing.away_team = a["code"]
        existing.home_name = h["name"]
        existing.away_name = a["name"]
        existing.home_flag = h.get("flag_emoji")
        existing.away_flag = a.get("flag_emoji")
        existing.kickoff_utc = _parse_iso(m["kickoff_utc"])
        existing.venue = m.get("venue")
        existing.config_path = str(config_path)
        match = existing
    await session.flush()
    return match


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


async def run_crawl_and_predict(
    session: AsyncSession,
    match: WM2026Match,
    *,
    crawl_seed: int | None = None,
) -> Tuple[int, int, MatchPrediction]:
    cfg = load_match_config(Path(match.config_path))
    slang = cfg["preprocessing"]["sport_slang_expansion"].get("custom_dict", {})
    ta = cfg["sentiment_config"]["team_attribution"]
    pipeline = PreprocessingPipeline(
        slang_dict=slang,
        home_keywords=ta.get("home_keywords", []),
        away_keywords=ta.get("away_keywords", []),
    )

    vader_lex = cfg["sentiment_config"].get("vader_config", {}).get("custom_lexicon", {})
    weights = cfg["sentiment_config"].get("ensemble_weights", None)
    scorer = SentimentEnsemble(weights=weights, custom_vader_lexicon=vader_lex)

    log.info("crawl_begin", match_id=match.id)
    if settings.use_mock_crawler:
        crawler = MockRedditCrawler(seed=crawl_seed)
        fetched = await crawler.crawl(cfg)
    elif settings.use_arctic_shift:
        from crawler.parallel_reddit import ParallelRedditCrawler
        crawler = ParallelRedditCrawler()
        try:
            fetched = await crawler.crawl(cfg)
        finally:
            await crawler.aclose()
    else:
        from crawler.http_reddit import HttpRedditCrawler
        crawler = HttpRedditCrawler()
        try:
            fetched = await crawler.crawl(cfg)
        finally:
            await crawler.aclose()

    log.info("crawl_fetched", match_id=match.id, posts_fetched=len(fetched))
    posts_added = 0
    scored = 0
    bot_filtered = 0
    low_quality_filtered = 0
    translated_count = 0
    scored_records: List[Tuple[RedditPost, float, float]] = []

    # BUG-01 fix: batch-load existing post IDs for this match (one query) instead
    # of per-post round-trips. Reduces 200+ DB queries → 1 for large crawls.
    existing_ids = {
        row[0]
        for row in (
            await session.execute(
                select(RedditPost.id).where(RedditPost.match_id == match.id)
            )
        ).all()
    }

    new_posts = []
    for fp in fetched:
        post_pk = f"{match.id}:{fp.subreddit}:{fp.post_id}"
        if post_pk in existing_ids:
            continue
        existing_ids.add(post_pk)

        if is_bot_author(fp.author):
            bot_filtered += 1
            session.add(PostFlag(post_id=post_pk, match_id=match.id, flag="bot_author"))
            continue
        if is_low_quality(fp.body):
            low_quality_filtered += 1
            session.add(PostFlag(post_id=post_pk, match_id=match.id, flag="low_quality"))
            continue

        new_posts.append((post_pk, fp))

    to_translate: list[tuple[int, str, str]] = []
    detected_langs: dict[int, str] = {}
    for i, (post_pk, fp) in enumerate(new_posts):
        lang = detect_language(fp.body)
        detected_langs[i] = lang
        if lang != "en":
            to_translate.append((i, fp.body, lang))

    translated_texts: dict[int, str] = {}
    if to_translate:
        tasks = [translate_to_english_async(text, lang) for _, text, lang in to_translate]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (idx, text, lang), result in zip(to_translate, results):
            if isinstance(result, str):
                translated_texts[idx] = result
                translated_count += 1
            else:
                log.warning("translation_error", index=idx, error=str(result))
                translated_texts[idx] = text

    for i, (post_pk, fp) in enumerate(new_posts):
        lang = detected_langs[i]
        text_for_analysis = translated_texts.get(i, fp.body)
        was_translated = i in translated_texts

        processed = pipeline.process(
            text_for_analysis,
            score=fp.score,
            num_comments=fp.num_comments,
            upvote_ratio=fp.upvote_ratio,
        )
        sentiment = scorer.score(processed.cleaned, source_language=lang)

        post = RedditPost(
            id=post_pk,
            match_id=match.id,
            subreddit=fp.subreddit,
            tier=fp.tier,
            post_id=fp.post_id,
            title=fp.title,
            body=fp.body,
            score=fp.score,
            upvote_ratio=fp.upvote_ratio,
            num_comments=fp.num_comments,
            created_utc=fp.created_utc.replace(tzinfo=None),
            author=fp.author,
            flair=fp.flair,
            is_comment=fp.is_comment,
            source=fp.source,
            raw_text=fp.body,
            processed_text=processed.cleaned,
            detected_language=lang,
            translation_used=was_translated,
            team_attribution=processed.team_attribution,
        )
        session.add(post)
        scored_records.append((post, sentiment.ensemble_score, processed.engagement_weight))

        team_label = {
            "home": match.home_team,
            "away": match.away_team,
            "neutral": "NEUTRAL",
        }[processed.team_attribution]

        session.add(
            SentimentScore(
                post_id=post_pk,
                match_id=match.id,
                team=team_label,
                vader_score=sentiment.vader_score,
                textblob_polarity=sentiment.textblob_polarity,
                textblob_subjectivity=sentiment.textblob_subjectivity,
                roberta_positive=sentiment.roberta_positive,
                roberta_neutral=sentiment.roberta_neutral,
                roberta_negative=sentiment.roberta_negative,
                roberta_emotion=sentiment.roberta_emotion,
                ensemble_score=sentiment.ensemble_score,
                engagement_weight=processed.engagement_weight,
                source_language=lang,
            )
        )
        posts_added += 1
        scored += 1

    await session.flush()
    # IMPROVE-06: record completion time so the next crawl can skip already-
    # fetched windows when the crawler supports the `after` parameter.
    match.last_crawled_at = datetime.now(timezone.utc).replace(tzinfo=None)
    log.info(
        "posts_stored",
        match_id=match.id,
        new=posts_added,
        scored=scored,
        bot_filtered=bot_filtered,
        low_quality_filtered=low_quality_filtered,
        translated=translated_count,
    )

    # Pull all scored posts for this match (incl. ones from earlier crawls)
    all_posts = await _fetch_all_scored(session, match.id)

    kickoff = match.kickoff_utc.replace(tzinfo=timezone.utc) if match.kickoff_utc else None
    home_sentiment, _ = weighted_sentiment(all_posts, team="home", kickoff_utc=kickoff)
    away_sentiment, _ = weighted_sentiment(all_posts, team="away", kickoff_utc=kickoff)
    home_momentum = momentum(all_posts, team="home", window_hours=6, kickoff_utc=kickoff)
    away_momentum = momentum(all_posts, team="away", window_hours=6, kickoff_utc=kickoff)
    home_velocity = post_velocity(all_posts, team="home", hours=6)
    away_velocity = post_velocity(all_posts, team="away", hours=6)

    # Advanced analytics
    adv = compute_advanced(all_posts)
    home_trend = compute_trend(all_posts, team="home")
    away_trend = compute_trend(all_posts, team="away")

    advanced_payload = {
        "tier_breakdown": adv.tier_breakdown,
        "subreddit_influence": adv.subreddit_influence,
        "home_anomalies": [
            {
                "bucket_start": a.bucket_start.isoformat(),
                "delta": a.delta,
                "z_score": a.z_score,
                "direction": a.direction,
            }
            for a in home_trend.anomalies
        ],
        "away_anomalies": [
            {
                "bucket_start": a.bucket_start.isoformat(),
                "delta": a.delta,
                "z_score": a.z_score,
                "direction": a.direction,
            }
            for a in away_trend.anomalies
        ],
        "home_trend_direction": home_trend.direction,
        "away_trend_direction": away_trend.direction,
        "home_r_squared": home_trend.r_squared,
        "away_r_squared": away_trend.r_squared,
        "top_decile_engagement": adv.top_decile_engagement,
        "neutral_share": adv.neutral_share,
        "filter_stats": {
            "bot_filtered": bot_filtered,
            "low_quality_filtered": low_quality_filtered,
        },
    }

    snapshot = SentimentSnapshot(
        match_id=match.id,
        hours_to_kickoff=_hours_to_kickoff(match.kickoff_utc),
        home_sentiment=home_sentiment,
        away_sentiment=away_sentiment,
        home_momentum=home_momentum,
        away_momentum=away_momentum,
        home_post_velocity=home_velocity,
        away_post_velocity=away_velocity,
        total_posts_crawled=len(all_posts),
        home_controversy=adv.home_controversy,
        away_controversy=adv.away_controversy,
        home_hype_ratio=adv.home_hype_ratio,
        away_hype_ratio=adv.away_hype_ratio,
        home_cope_ratio=adv.home_cope_ratio,
        away_cope_ratio=adv.away_cope_ratio,
        polarization=adv.polarization,
        fan_balance=adv.fan_balance,
        engagement_density=adv.engagement_density,
        home_emotion=adv.home_emotion,
        away_emotion=adv.away_emotion,
        home_trend_slope=home_trend.slope_per_hour,
        away_trend_slope=away_trend.slope_per_hour,
        home_volatility=home_trend.volatility,
        away_volatility=away_trend.volatility,
        unique_authors=adv.unique_authors,
        advanced_payload=advanced_payload,
    )
    session.add(snapshot)

    h_team = cfg["teams"]["home"]
    a_team = cfg["teams"]["away"]
    predictor = MatchPredictor(
        rho=cfg["prediction_config"]["goal_prediction"].get("dixon_coles_rho", 0.1),
        max_goals=cfg["prediction_config"]["goal_prediction"].get("max_goals_display", 6),
        goal_model=settings.goal_model,
        negbin_size=settings.negbin_size,
        combine=settings.goal_model_combine,
        bootstrap_n=settings.bootstrap_n,
        bootstrap_xg_sigma=settings.bootstrap_xg_sigma,
    )

    # Optional features: market odds + H2H + venue altitude/host advantage.
    market_home = market_draw = market_away = 0.0
    if settings.enable_odds_integration:
        try:
            from crawler.odds_api import fetch_odds, implied_probabilities
            fixtures = await fetch_odds()
            implied = implied_probabilities(h_team["name"], a_team["name"], fixtures)
            if implied:
                market_home = implied["home"]
                market_draw = implied["draw"]
                market_away = implied["away"]
        except Exception as exc:
            log.debug("odds_lookup_failed", error=str(exc))

    h2h_home_wins = h2h_draws = h2h_away_wins = 0
    h2h_avg_goals = 0.0
    if settings.enable_h2h:
        try:
            from crawler.h2h_data import lookup as h2h_lookup
            rec = h2h_lookup(h_team["code"], a_team["code"])
            if rec:
                h2h_home_wins = rec["home_wins"]
                h2h_draws = rec["draws"]
                h2h_away_wins = rec["away_wins"]
                h2h_avg_goals = rec["avg_goals"]
        except Exception as exc:
            log.debug("h2h_lookup_failed", error=str(exc))

    venue = match.venue
    try:
        from scripts.team_real_data import get_home_advantage, get_venue_altitude
        venue_altitude = get_venue_altitude(venue)
        home_advantage = get_home_advantage(h_team["code"], venue)
    except Exception:
        venue_altitude = 0.0
        home_advantage = 0.0

    breakdown_payload: dict | None = None
    factor_signals_to_persist: list = []

    if settings.use_factor_ensemble:
        # MULTIFACTOR-09: new path. Build a FactorContext with everything the
        # already-running pipelines produced, fan out to the active factors in
        # parallel, then feed the signals into the new predictor entrypoint.
        ctx = FactorContext(
            match_id=match.id,
            config=cfg,
            home_code=h_team["code"],
            away_code=a_team["code"],
            kickoff_utc=match.kickoff_utc.replace(tzinfo=timezone.utc)
            if match.kickoff_utc and match.kickoff_utc.tzinfo is None
            else match.kickoff_utc,
            venue=match.venue,
            sentiment_payload={
                "home_sentiment": home_sentiment,
                "away_sentiment": away_sentiment,
                "home_momentum": home_momentum,
                "away_momentum": away_momentum,
                "home_controversy": adv.home_controversy,
                "away_controversy": adv.away_controversy,
                "home_trend_slope": home_trend.slope_per_hour,
                "away_trend_slope": away_trend.slope_per_hour,
                "sample_size": len(all_posts),
            },
        )

        # Market-implied 1X2 also feeds the MarketOddsFactor as a small λ-tilt
        # (the full prior is still blended into the outcome below).
        if 0.95 <= (market_home + market_draw + market_away) <= 1.05:
            ctx.market_implied = (market_home, market_draw, market_away)

        # MULTIFACTOR-09 fix: actually fetch the external data the factors need.
        # Without this the orchestrator was dead code and every factor except
        # Elo/Sentiment fell back to neutral. Guarded so a source outage can
        # never break a crawl — the ensemble just re-normalises what it gets.
        from data_sources.orchestrator import DataSourceOrchestrator

        orchestrator = DataSourceOrchestrator()
        try:
            await orchestrator.populate(ctx)
        except Exception as exc:
            log.warning("datasource_populate_failed", match_id=match.id, error=str(exc))

        # v3.3 — NVIDIA-LLM aspect sentiment, optional. Hooked here so it sees
        # the same pre-cleaned post list the classical scorers used. Failures
        # are logged but don't disturb the ensemble (LlmSentimentFactor
        # self-disables when the payload is missing).
        if settings.use_nvidia_llm and settings.factor_weight_llm_sentiment > 0:
            try:
                from analysis.nvidia_llm_scorer import score_match as llm_score_match

                llm_payload = await llm_score_match(
                    ctx.home_code, ctx.away_code, all_posts,
                )
                if llm_payload and ctx.sentiment_payload is not None:
                    ctx.sentiment_payload["llm"] = llm_payload
            except Exception as exc:
                log.warning("llm_scoring_failed", match_id=match.id, error=str(exc))

        factors = get_active_factors(settings)
        if factors:
            results = await asyncio.gather(
                *(f.compute(ctx) for f in factors),
                return_exceptions=True,
            )
            signals = []
            for f, result in zip(factors, results):
                if isinstance(result, Exception):
                    log.warning("factor_compute_failed", name=f.name, error=str(result))
                    signals.append(
                        f._neutral(source="error", reason=type(result).__name__)
                    )
                else:
                    signals.append(result)
        else:
            log.warning("no_active_factors")
            signals = []

        base_home_xg = (h_team["avg_xg_season"] + a_team["avg_xg_conceded"]) / 2.0
        base_away_xg = (a_team["avg_xg_season"] + h_team["avg_xg_conceded"]) / 2.0
        market_prior = None
        # C1: avoid double-counting the market. When MarketOddsFactor is active
        # it already tilts λ via ctx.market_implied, so we drop the separate 1X2
        # prior. Only blend the prior when the factor is switched off.
        if (
            settings.factor_weight_market <= 0
            and 0.95 <= (market_home + market_draw + market_away) <= 1.05
        ):
            market_prior = (market_home, market_draw, market_away)

        pred, ensemble = predictor.predict_from_signals(
            signals,
            base_home_xg=base_home_xg,
            base_away_xg=base_away_xg,
            market_prior=market_prior,
        )
        breakdown_payload = ensemble.breakdown_payload
        factor_signals_to_persist = list(zip(signals, breakdown_payload["signals"]))
    else:
        # Legacy heuristic-xG path. Untouched so a USE_FACTOR_ENSEMBLE=false
        # rollback returns bit-for-bit identical predictions.
        pred = predictor.predict(
            PredictionInput(
                home_elo=h_team["elo_rating"],
                away_elo=a_team["elo_rating"],
                home_avg_xg=h_team["avg_xg_season"],
                away_avg_xg=a_team["avg_xg_season"],
                home_avg_xg_conceded=h_team["avg_xg_conceded"],
                away_avg_xg_conceded=a_team["avg_xg_conceded"],
                home_form_pts=_form_to_points(h_team["form_last5"]),
                away_form_pts=_form_to_points(a_team["form_last5"]),
                home_sentiment=home_sentiment,
                away_sentiment=away_sentiment,
                home_momentum=home_momentum,
                away_momentum=away_momentum,
                home_controversy=adv.home_controversy,
                away_controversy=adv.away_controversy,
                sentiment_sample_size=len(all_posts),
                home_advantage=home_advantage,
                venue_altitude_m=venue_altitude,
                market_home_prob=market_home,
                market_draw_prob=market_draw,
                market_away_prob=market_away,
                h2h_home_wins=h2h_home_wins,
                h2h_draws=h2h_draws,
                h2h_away_wins=h2h_away_wins,
                h2h_avg_goals=h2h_avg_goals,
            )
        )

    # v3.6 — Kalibrierungs-Layer: Isotonic + Platt-Scaling auf die rohen 1X2.
    # v3.7 (K1) — Bootstrap-CIs werden mit DERSELBEN Kurve transformiert, sonst
    # liegen die Baender nicht um den kalibrierten Punktwert. Wir persistieren
    # raw/isotonic/platt nebeneinander, damit das UI explizit waehlen kann.
    cal_home = cal_draw = cal_away = None
    platt_home = platt_draw = platt_away = None
    iso = platt = None
    try:
        from analysis.calibration import apply as apply_calibration
        from analysis.calibration import load_isotonic, load_platt, transform_intervals
        iso = load_isotonic()
        platt = load_platt()
        iso_out = apply_calibration(iso, pred.home_win_prob, pred.draw_prob, pred.away_win_prob)
        if iso_out is not None:
            cal_home, cal_draw, cal_away = iso_out["home"], iso_out["draw"], iso_out["away"]
        platt_out = apply_calibration(platt, pred.home_win_prob, pred.draw_prob, pred.away_win_prob)
        if platt_out is not None:
            platt_home, platt_draw, platt_away = platt_out["home"], platt_out["draw"], platt_out["away"]
    except Exception as exc:
        log.debug("calibration_apply_failed", error=str(exc))

    per_model_markets = (pred.features or {}).get("per_model")
    raw_intervals = (pred.features or {}).get("confidence_intervals")
    if raw_intervals is not None:
        try:
            from analysis.calibration import transform_intervals
            iso_intervals = transform_intervals(raw_intervals, iso) if iso else None
            platt_intervals = transform_intervals(raw_intervals, platt) if platt else None
        except Exception as exc:
            log.debug("calibration_intervals_failed", error=str(exc))
            iso_intervals = platt_intervals = None
        confidence_intervals = {
            "raw": raw_intervals,
            "isotonic": iso_intervals,
            "platt": platt_intervals,
        }
    else:
        confidence_intervals = None

    # M2: alte Predictions fuer diesen match auf is_latest=False demoten,
    # bevor wir die frische Row einfuegen. Damit hat jede match_id genau eine
    # `is_latest=True`-Zeile, und Dashboards/Backtests koennen ohne
    # ORDER BY generated_at DESC LIMIT 1 die aktuelle Vorhersage holen.
    await session.execute(
        update(MatchPrediction)
        .where(MatchPrediction.match_id == match.id, MatchPrediction.is_latest.is_(True))
        .values(is_latest=False)
    )

    record = MatchPrediction(
        match_id=match.id,
        home_win_prob=pred.home_win_prob,
        draw_prob=pred.draw_prob,
        away_win_prob=pred.away_win_prob,
        confidence=pred.confidence,
        home_xg=pred.home_xg,
        away_xg=pred.away_xg,
        # Mirror xG into the legacy goals_expected columns. With the current
        # heuristic predictor the Poisson λ equals the nudged xG; a future ML
        # head can diverge by setting these independently (BUG-13).
        home_goals_expected=pred.home_xg,
        away_goals_expected=pred.away_xg,
        over_25_prob=pred.over_25,
        over_15_prob=pred.over_15,
        over_35_prob=pred.over_35,
        btts_prob=pred.btts,
        top_scores=pred.top_scores,
        recommended_bet=pred.recommended_bet,
        bet_probability=pred.bet_probability,
        features_snapshot=pred.features,
        factor_breakdown=breakdown_payload,
        calibrated_home_win_prob=cal_home,
        calibrated_draw_prob=cal_draw,
        calibrated_away_win_prob=cal_away,
        platt_home_win_prob=platt_home,
        platt_draw_prob=platt_draw,
        platt_away_win_prob=platt_away,
        per_model_markets=per_model_markets,
        confidence_intervals=confidence_intervals,
        is_latest=True,
    )
    session.add(record)
    await session.flush()

    # MULTIFACTOR-10: persist each FactorSignal as its own row now that we
    # have prediction_id. Keeps the breakdown queryable without parsing JSON.
    for signal, payload in factor_signals_to_persist:
        session.add(
            FactorSnapshot(
                match_id=match.id,
                prediction_id=record.id,
                factor_name=signal.name,
                home_strength=signal.home_strength,
                away_strength=signal.away_strength,
                weight=signal.weight,
                effective_weight=payload.get("effective_weight", 0.0),
                confidence=signal.confidence,
                available=signal.available,
                source=signal.source,
                raw_data=signal.raw_data,
                cached_at=signal.cached_at.replace(tzinfo=None)
                if signal.cached_at and signal.cached_at.tzinfo is not None
                else signal.cached_at,
            )
        )

    await session.commit()
    await session.refresh(record)

    # IMPROVE-15: invalidate cached views for this match.
    try:
        from utils.cache import cache
        await cache.invalidate(f"match:{match.id}:")
        await cache.invalidate("matches:list:")
    except Exception:
        pass

    # EXTEND-08: push live update to WebSocket / SSE subscribers.
    try:
        from api.live import publish_prediction
        await publish_prediction(session, match.id)
    except Exception as exc:
        log.debug("live_publish_failed", error=str(exc))

    log.info(
        "prediction_stored",
        match_id=match.id,
        home_win=round(record.home_win_prob, 3),
        draw=round(record.draw_prob, 3),
        away_win=round(record.away_win_prob, 3),
        bet=record.recommended_bet,
    )
    return posts_added, scored, record


def _hours_to_kickoff(kickoff: datetime) -> float:
    if kickoff is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    return (kickoff - now).total_seconds() / 3600.0


async def _fetch_all_scored(session: AsyncSession, match_id: str):
    """Return objects exposing team_attribution + ensemble_score + engagement_weight + created_utc."""
    q = (
        select(RedditPost, SentimentScore)
        .join(SentimentScore, SentimentScore.post_id == RedditPost.id)
        .where(RedditPost.match_id == match_id)
    )
    result = await session.execute(q)

    class _Joined:
        __slots__ = (
            "team_attribution", "ensemble_score", "engagement_weight", "created_utc",
            "tier", "subreddit", "author",
        )

        def __init__(self, p: RedditPost, s: SentimentScore):
            self.team_attribution = p.team_attribution
            self.ensemble_score = s.ensemble_score
            self.engagement_weight = s.engagement_weight
            self.created_utc = p.created_utc.replace(tzinfo=timezone.utc) if p.created_utc else datetime.now(timezone.utc)
            self.tier = p.tier
            self.subreddit = p.subreddit
            self.author = p.author

    return [_Joined(p, s) for p, s in result.all()]
