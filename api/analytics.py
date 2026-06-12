"""Advanced analytics endpoints — controversy, trend, anomalies, exports."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import (
    MatchPrediction,
    SentimentSnapshot,
    WM2026Match,
)
from db.schemas import AdvancedMetricsResponse, AnomalyPoint, SubredditInfluence
from utils.security import require_admin_key


class MatchResultIn(BaseModel):
    """IMPROVE-16: validate final-score input.

    Scores are clamped to a realistic range so a typo (e.g. 100-0) cannot
    silently corrupt the accuracy stats."""

    home_score: int = Field(..., ge=0, le=20)
    away_score: int = Field(..., ge=0, le=20)

router = APIRouter(prefix="/api/matches", tags=["analytics"])
stats_router = APIRouter(prefix="/api/stats", tags=["analytics"])


def _latest_snapshot_q(match_id: str):
    return (
        select(SentimentSnapshot)
        .where(SentimentSnapshot.match_id == match_id)
        .order_by(desc(SentimentSnapshot.snapshot_time))
        .limit(1)
    )


@router.get("/{match_id}/analytics", response_model=AdvancedMetricsResponse)
async def advanced_metrics(match_id: str, session: AsyncSession = Depends(get_session)):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    snap = (await session.execute(_latest_snapshot_q(match_id))).scalar_one_or_none()
    if snap is None:
        raise HTTPException(404, "No analytics yet — call /crawl first")

    payload = snap.advanced_payload or {}
    tier_breakdown = payload.get("tier_breakdown", {})
    sub_influence = [
        SubredditInfluence(**item) for item in payload.get("subreddit_influence", [])
    ]
    anomalies_raw = payload.get("home_anomalies", []) + payload.get("away_anomalies", [])
    # Sort anomalies by absolute z-score descending so the most striking ones lead
    anomalies_raw.sort(key=lambda a: abs(a.get("z_score", 0)), reverse=True)
    anomalies = [AnomalyPoint(**a) for a in anomalies_raw[:10]]

    return AdvancedMetricsResponse(
        match_id=match_id,
        snapshot_time=snap.snapshot_time,
        polarization=snap.polarization or 0.0,
        fan_balance=snap.fan_balance or 0.0,
        home_controversy=snap.home_controversy or 0.0,
        away_controversy=snap.away_controversy or 0.0,
        home_hype_ratio=snap.home_hype_ratio or 0.0,
        away_hype_ratio=snap.away_hype_ratio or 0.0,
        home_cope_ratio=snap.home_cope_ratio or 0.0,
        away_cope_ratio=snap.away_cope_ratio or 0.0,
        engagement_density=snap.engagement_density or 0.0,
        home_emotion=snap.home_emotion or "neutral",
        away_emotion=snap.away_emotion or "neutral",
        home_trend_slope=snap.home_trend_slope or 0.0,
        away_trend_slope=snap.away_trend_slope or 0.0,
        home_volatility=snap.home_volatility or 0.0,
        away_volatility=snap.away_volatility or 0.0,
        unique_authors=snap.unique_authors or 0,
        tier_breakdown=tier_breakdown,
        subreddit_influence=sub_influence,
        anomalies=anomalies,
    )


@router.get("/{match_id}/prediction/export")
async def export_prediction(match_id: str, session: AsyncSession = Depends(get_session)):
    """CSV export combining prediction + latest sentiment snapshot.

    v3.6: enthaelt zusaetzlich Pro-Modell-Markets (poisson/negbin/glm_poisson) inkl.
    Bootstrap-CI-Baender (p5/p50/p95), kalibrierte Wahrscheinlichkeiten (isotonic +
    platt) und einen Pro-Faktor-Block (alle 20 FactorSignals)."""
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    pred = (await session.execute(
        select(MatchPrediction)
        .where(MatchPrediction.match_id == match_id)
        .order_by(desc(MatchPrediction.generated_at))
        .limit(1)
    )).scalar_one_or_none()
    if pred is None:
        raise HTTPException(404, "No prediction yet — call /crawl first")
    snap = (await session.execute(_latest_snapshot_q(match_id))).scalar_one_or_none()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["field", "value"])
    writer.writerow(["match_id", match.id])
    writer.writerow(["matchup", f"{match.home_name} vs {match.away_name}"])
    writer.writerow(["kickoff_utc", match.kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if match.kickoff_utc else ""])
    writer.writerow(["home_win_prob", f"{pred.home_win_prob:.4f}"])
    writer.writerow(["draw_prob", f"{pred.draw_prob:.4f}"])
    writer.writerow(["away_win_prob", f"{pred.away_win_prob:.4f}"])
    writer.writerow(["home_xg", f"{pred.home_xg:.3f}"])
    writer.writerow(["away_xg", f"{pred.away_xg:.3f}"])
    writer.writerow(["over_25_prob", f"{pred.over_25_prob:.4f}"])
    writer.writerow(["btts_prob", f"{pred.btts_prob:.4f}"])
    writer.writerow(["confidence", f"{pred.confidence:.4f}"])
    writer.writerow(["recommended_bet", pred.recommended_bet or ""])
    if snap:
        writer.writerow(["home_sentiment", f"{snap.home_sentiment:.3f}"])
        writer.writerow(["away_sentiment", f"{snap.away_sentiment:.3f}"])
        writer.writerow(["polarization", f"{(snap.polarization or 0):.3f}"])
        writer.writerow(["home_emotion", snap.home_emotion or ""])
        writer.writerow(["away_emotion", snap.away_emotion or ""])
        writer.writerow(["home_controversy", f"{(snap.home_controversy or 0):.3f}"])
        writer.writerow(["away_controversy", f"{(snap.away_controversy or 0):.3f}"])
        writer.writerow(["total_posts_crawled", snap.total_posts_crawled])
        n = snap.total_posts_crawled or 0
        dq = "high" if n >= 300 else ("medium" if n >= 100 else "low")
        writer.writerow(["data_quality", dq])

    # v3.6 — kalibrierte Wahrscheinlichkeiten
    if pred.calibrated_home_win_prob is not None:
        writer.writerow(["calibrated_home_win_prob", f"{pred.calibrated_home_win_prob:.4f}"])
        writer.writerow(["calibrated_draw_prob", f"{pred.calibrated_draw_prob:.4f}"])
        writer.writerow(["calibrated_away_win_prob", f"{pred.calibrated_away_win_prob:.4f}"])
    if pred.platt_home_win_prob is not None:
        writer.writerow(["platt_home_win_prob", f"{pred.platt_home_win_prob:.4f}"])
        writer.writerow(["platt_draw_prob", f"{pred.platt_draw_prob:.4f}"])
        writer.writerow(["platt_away_win_prob", f"{pred.platt_away_win_prob:.4f}"])

    # v3.6 — Pro-Modell-Markets (poisson, negbin, glm_poisson)
    per_model = pred.per_model_markets or {}
    if isinstance(per_model, dict):
        for model_name in ("poisson", "negbin", "glm_poisson"):
            mk = per_model.get(model_name)
            if not isinstance(mk, dict):
                continue
            for key in ("home_win", "draw", "away_win", "over_25", "btts"):
                v = mk.get(key)
                if isinstance(v, (int, float)):
                    writer.writerow([f"{model_name}_{key}", f"{float(v):.4f}"])

    # v3.6 — Bootstrap-Konfidenzintervalle pro Modell
    cis = pred.confidence_intervals or {}
    if isinstance(cis, dict):
        for model_name, ci in cis.items():
            if not isinstance(ci, dict):
                continue
            for key in ("home_win", "draw", "away_win", "over_25", "btts"):
                tri = ci.get(key)
                if isinstance(tri, (list, tuple)) and len(tri) == 3:
                    writer.writerow([f"{model_name}_{key}_p5", f"{float(tri[0]):.4f}"])
                    writer.writerow([f"{model_name}_{key}_p50", f"{float(tri[1]):.4f}"])
                    writer.writerow([f"{model_name}_{key}_p95", f"{float(tri[2]):.4f}"])

    # v3.6 — Pro-Faktor-Detail aus factor_breakdown
    fb = pred.factor_breakdown or {}
    signals = fb.get("signals") if isinstance(fb, dict) else None
    if isinstance(signals, list):
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            name = sig.get("name") or "unknown"
            writer.writerow([f"factor_{name}_home_strength", f"{float(sig.get('home_strength', 1.0)):.4f}"])
            writer.writerow([f"factor_{name}_away_strength", f"{float(sig.get('away_strength', 1.0)):.4f}"])
            writer.writerow([f"factor_{name}_weight", f"{float(sig.get('weight', 0.0)):.4f}"])
            writer.writerow([f"factor_{name}_effective_weight", f"{float(sig.get('effective_weight', 0.0)):.4f}"])
            writer.writerow([f"factor_{name}_confidence", f"{float(sig.get('confidence', 0.0)):.4f}"])
            writer.writerow([f"factor_{name}_available", "true" if sig.get("available") else "false"])
            writer.writerow([f"factor_{name}_source", str(sig.get("source") or "")])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{match.id}_prediction.csv"'},
    )


@router.get("/{match_id}/prediction/full")
async def full_prediction(match_id: str, session: AsyncSession = Depends(get_session)):
    """v3.6 — vollstaendige JSON-Sicht auf eine Prediction (alle 3 Modelle,
    Kalibrierung, Bootstrap-CIs, Pro-Faktor-Breakdown). Datenquelle fuer das
    Admin-Panel Tab 'Pro-Modell' und Tab 'Datenquellen'."""
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    pred = (await session.execute(
        select(MatchPrediction)
        .where(MatchPrediction.match_id == match_id)
        .order_by(desc(MatchPrediction.generated_at))
        .limit(1)
    )).scalar_one_or_none()
    if pred is None:
        raise HTTPException(404, "No prediction yet — call /crawl first")
    return {
        "match_id": match.id,
        "matchup": f"{match.home_name} vs {match.away_name}",
        "kickoff_utc": match.kickoff_utc.isoformat() if match.kickoff_utc else None,
        "generated_at": pred.generated_at.isoformat() if pred.generated_at else None,
        "raw": {
            "home_win_prob": pred.home_win_prob,
            "draw_prob": pred.draw_prob,
            "away_win_prob": pred.away_win_prob,
            "home_xg": pred.home_xg,
            "away_xg": pred.away_xg,
            "over_25_prob": pred.over_25_prob,
            "btts_prob": pred.btts_prob,
            "confidence": pred.confidence,
        },
        "calibrated": {
            "isotonic": {
                "home_win": pred.calibrated_home_win_prob,
                "draw": pred.calibrated_draw_prob,
                "away_win": pred.calibrated_away_win_prob,
            } if pred.calibrated_home_win_prob is not None else None,
            "platt": {
                "home_win": pred.platt_home_win_prob,
                "draw": pred.platt_draw_prob,
                "away_win": pred.platt_away_win_prob,
            } if pred.platt_home_win_prob is not None else None,
        },
        "per_model": pred.per_model_markets,
        "confidence_intervals": pred.confidence_intervals,
        "factor_breakdown": pred.factor_breakdown,
        "recommended_bet": pred.recommended_bet,
        "bet_probability": pred.bet_probability,
    }


@router.post("/{match_id}/result", dependencies=[Depends(require_admin_key)])
async def record_result(
    match_id: str,
    payload: MatchResultIn = Body(...),
    session: AsyncSession = Depends(get_session),
):
    """Manually record the final score → enables accuracy tracking."""
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    home_score = payload.home_score
    away_score = payload.away_score

    match.home_score = home_score
    match.away_score = away_score
    match.status = "finished"

    # Update all predictions for this match with prediction_correct
    preds = (await session.execute(
        select(MatchPrediction).where(MatchPrediction.match_id == match_id)
    )).scalars().all()
    if home_score > away_score:
        actual = "home"
    elif away_score > home_score:
        actual = "away"
    else:
        actual = "draw"
    for p in preds:
        top = max(
            ("home", p.home_win_prob),
            ("draw", p.draw_prob),
            ("away", p.away_win_prob),
            key=lambda x: x[1],
        )[0]
        p.prediction_correct = (top == actual)
        p.actual_home_score = home_score
        p.actual_away_score = away_score

    # EXTEND-11: settle outstanding tips so the leaderboard reflects this match.
    tips_scored = 0
    try:
        from api.tipping import score_pending_tips
        tips_scored = await score_pending_tips(session, match_id, home_score, away_score)
    except Exception:
        tips_scored = 0

    await session.commit()
    return {
        "match_id": match_id,
        "home_score": home_score,
        "away_score": away_score,
        "actual_outcome": actual,
        "predictions_updated": len(preds),
        "tips_scored": tips_scored,
    }


@stats_router.get("/backtesting")
async def backtesting(session: AsyncSession = Depends(get_session)):
    """EXTEND-10: Brier / log-loss / calibration over all evaluated predictions."""
    from analysis.backtesting import compute
    q = select(MatchPrediction).where(MatchPrediction.actual_home_score.is_not(None))
    rows = (await session.execute(q)).scalars().all()
    report = compute(rows)
    return {
        "n_evaluated": report.n_evaluated,
        "accuracy": round(report.accuracy, 4),
        "brier": round(report.brier, 4),
        "log_loss": round(report.log_loss, 4),
        "rps": round(report.rps, 4),
        "calibration": [
            {
                "bucket_lo": round(b.bucket_lo, 2),
                "bucket_hi": round(b.bucket_hi, 2),
                "n": b.n,
                "mean_predicted": round(b.mean_predicted, 3),
                "mean_actual": round(b.mean_actual, 3),
            }
            for b in report.calibration
        ],
    }


@stats_router.get("/accuracy")
async def accuracy_stats(session: AsyncSession = Depends(get_session)):
    """Dashboard widget: how many predictions matched the real result?"""
    q = (
        select(MatchPrediction)
        .where(MatchPrediction.prediction_correct.is_not(None))
    )
    rows = (await session.execute(q)).scalars().all()
    total = len(rows)
    correct = sum(1 for r in rows if r.prediction_correct)
    accuracy = (correct / total) if total else 0.0
    return {
        "predictions_evaluated": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
    }
