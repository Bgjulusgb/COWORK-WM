"""The 8-phase prediction pipeline (master prompt → runnable code).

``run_prediction`` ties the existing modules together into one async call:

    Phase 1  Data Collection      data_sources.DataSourceOrchestrator.populate
    Phase 2  Factor Decomposition factors.registry.get_active_factors → compute
    Phase 3  Sentiment Layer       optional sentiment_payload (best-effort)
    Phase 4  Goal-Model Stack      analysis.MatchPredictor.predict_from_signals
             + Bootstrap CIs        models_ml.poisson_goals.bootstrap_markets
    Phase 5  Calibration          analysis.calibration (graceful, prior-based)
    Phase 6  Market Edge          wm2026.edge.compute_edges
    Phase 7  Validation           sanity checks → list of warnings
    Phase 8  Output               handled by wm2026.report.build_report

The function returns a single ``result`` dict carrying every raw piece the
report needs. Nothing here imports FastAPI / SQLAlchemy / Prefect, so a fresh
clone runs the whole thing offline with only the core requirements.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Sequence

import structlog

from analysis.factor_ensemble import EnsembleResult
from analysis.match_predictor import MatchPredictor
from config.settings import settings
from data_sources.orchestrator import DataSourceOrchestrator
from factors.base import FactorContext
from factors.registry import get_active_factors
from wm2026 import edge as edge_mod
from wm2026 import markets as markets_mod
from wm2026.context import apply_overrides, apply_runtime_profile, build_context

log = structlog.get_logger("wm2026.pipeline")


def _team(cfg: dict[str, Any], side: str) -> dict[str, Any]:
    return (cfg.get("teams", {}) or {}).get(side, {}) or {}


def _base_xg(cfg: dict[str, Any]) -> tuple[float, float]:
    """Base attack/defence xG blend — identical to services.match_service.

        base_home = (home.avg_xg_season + away.avg_xg_conceded) / 2
        base_away = (away.avg_xg_season + home.avg_xg_conceded) / 2
    """
    h, a = _team(cfg, "home"), _team(cfg, "away")
    h_xg = float(h.get("avg_xg_season", 1.40) or 1.40)
    a_xg = float(a.get("avg_xg_season", 1.30) or 1.30)
    h_xga = float(h.get("avg_xg_conceded", 1.30) or 1.30)
    a_xga = float(a.get("avg_xg_conceded", 1.40) or 1.40)
    return (h_xg + a_xga) / 2.0, (a_xg + h_xga) / 2.0


async def _populate(ctx: FactorContext) -> dict[str, Any]:
    """Phase 1 — fan-out external data. Never raises into the caller."""
    orch = DataSourceOrchestrator()
    try:
        await orch.populate(ctx)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("orchestrator_failed", error=str(exc), exc_info=True)
    finally:
        await orch.aclose()
    return dict(ctx.provenance)


async def _signals(ctx: FactorContext) -> list:
    """Phase 2/3 — run every active factor concurrently → FactorSignals."""
    factors = get_active_factors(settings)
    signals = await asyncio.gather(
        *(f.compute(ctx) for f in factors), return_exceptions=True
    )
    out = []
    for f, s in zip(factors, signals):
        if isinstance(s, Exception):
            log.warning("factor_failed", name=getattr(f, "name", "?"), error=str(s))
            continue
        out.append(s)
    return out


def _lambda_ci(xg: float, sigma: float) -> dict[str, float]:
    """Normal-approx 90 % interval for λ from the bootstrap σ (relative).

    The bootstrap samples λ ~ N(xg, sigma·xg); its 5/50/95 percentiles are
    ``xg·(1 ± 1.645·sigma)``. Cheaper than a second bootstrap and identical in
    distribution. Floored at 0.05 so a CI never goes non-physical.
    """
    z = 1.645
    return {
        "p5": round(max(0.05, xg * (1.0 - z * sigma)), 3),
        "p50": round(xg, 3),
        "p95": round(max(0.05, xg * (1.0 + z * sigma)), 3),
    }


def _maybe_calibrate(
    out: Any,
    *,
    mode: str = "auto",
    market: tuple[float, float, float] | None = None,
    market_weight: float = 0.5,
) -> dict[str, Any]:
    """Phase 5 — calibrate the 1X2 line. Three modes:

    * ``auto`` (default) — apply a fitted isotonic/Platt artifact if one exists,
      else report raw probabilities (a fresh clone has no WC-2026 history).
    * ``market`` — anchor toward the vig-free market consensus (the canonical
      well-calibrated football forecaster; no historical data needed). This is
      the per-match calibration Claude can drive in Cowork by researching odds.
    * ``none`` — never calibrate.

    Returns ``{"applied": bool, "method": str|None, "note": str,
    "calibrated": {home_win,draw,away_win}|None}``.
    """
    from analysis import calibration as calib

    raw = (out.home_win_prob, out.draw_prob, out.away_win_prob)

    if mode == "none":
        return {"applied": False, "method": None,
                "note": "Calibration disabled (--calibrate none).", "calibrated": None}

    if mode == "market":
        cal = calib.market_anchor(*raw, market, weight=market_weight)
        if cal is not None:
            return {
                "applied": True,
                "method": "market-anchor",
                "note": (
                    f"Anchored toward the vig-free market consensus "
                    f"(weight {market_weight:.2f}; Constantinou & Fenton 2013 — "
                    f"closing odds are well calibrated). Compounds with the "
                    f"market factor; lower factor_weight_market to avoid."
                ),
                "calibrated": {
                    "home_win": round(cal["home"], 4),
                    "draw": round(cal["draw"], 4),
                    "away_win": round(cal["away"], 4),
                },
            }
        return {"applied": False, "method": None,
                "note": "Market calibration requested but no usable odds were supplied.",
                "calibrated": None}

    # mode == "auto" (or unknown) → fitted artifact if present, else raw.
    try:
        iso = calib.load_isotonic()
        platt = calib.load_platt()
        artifact = iso or platt
        calibrated = calib.apply(artifact, raw[0], raw[1], raw[2])
        if calibrated is None:
            raise FileNotFoundError("no usable calibration artifact")
        return {
            "applied": True,
            "method": "isotonic" if iso is not None else "platt",
            "note": "Calibrated against the fitted historical artifact.",
            "calibrated": {
                "home_win": round(calibrated["home"], 4),
                "draw": round(calibrated["draw"], 4),
                "away_win": round(calibrated["away"], 4),
            },
        }
    except Exception:
        return {
            "applied": False,
            "method": None,
            "note": (
                "No fitted calibration artifact (no WC-2026 history yet). Raw "
                "model probabilities are reported. Either fit one offline "
                "(scripts/fit_calibration_offline.py on WC 2022 + EURO 2024 + "
                "Copa 2024) or pass --calibrate market to anchor to the odds."
            ),
            "calibrated": None,
        }


def _provenance_summary(provenance: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-slice provenance modes into one summary dict.

    Looks at every ctx.provenance entry the orchestrator filled in (history,
    weather, lineup, odds_api, …) and returns counts plus the live-coverage
    ratio. The report surfaces this so the user can see at a glance whether a
    ``mode: live`` run is *substantively* live or mostly fail-soft-mocked.
    """
    modes: dict[str, int] = {}
    for entry in (provenance or {}).values():
        mode = entry.get("mode") if isinstance(entry, dict) else entry
        if mode is None:
            mode = "unknown"
        modes[mode] = modes.get(mode, 0) + 1
    total = sum(modes.values())
    live_like = modes.get("live", 0) + modes.get("cache", 0) + modes.get("research", 0)
    coverage = round(live_like / total, 3) if total else 0.0
    return {
        "modes": modes,
        "total_slices": total,
        "live_slices": live_like,
        "mock_slices": modes.get("mock", 0),
        "error_slices": modes.get("error", 0),
        "live_coverage_pct": round(100.0 * coverage, 1),
    }


def _validate(
    out: Any, ensemble: EnsembleResult, signals: list, provenance: dict[str, Any]
) -> list[str]:
    """Phase 7 — sanity self-check. Returns human-readable warnings (never
    raises). Mirrors the master prompt's ``/data:validate-data`` checklist."""
    warnings: list[str] = []
    lam_h, lam_a = out.home_xg, out.away_xg
    if not (0.3 <= lam_h <= 4.0) or not (0.3 <= lam_a <= 4.0):
        warnings.append(
            f"λ out of sane range [0.3, 4.0]: home={lam_h:.2f} away={lam_a:.2f}"
        )
    p_sum = out.home_win_prob + out.draw_prob + out.away_win_prob
    if abs(p_sum - 1.0) > 0.005:
        warnings.append(f"1X2 probabilities sum to {p_sum:.4f} (expected 1.000 ± 0.005)")
    n_avail = sum(1 for s in signals if s.available)
    if n_avail < 5:
        warnings.append(f"only {n_avail} factors available (low coverage)")
    modes = {p.get("mode") for p in provenance.values() if isinstance(p, dict)}
    non_mock = {"live", "cache", "research"} & modes
    if not non_mock:
        warnings.append("all data sources are mock — predictions are illustrative, not live")
    else:
        # Partial-live nuance: a "live" run with <40% real slices is honest but
        # thin — surface that so the user doesn't read "mode: live" as "all real".
        summary = _provenance_summary(provenance)
        if summary["total_slices"] >= 5 and summary["live_coverage_pct"] < 40.0:
            warnings.append(
                f"partial-live: only {summary['live_slices']}/{summary['total_slices']} "
                f"data slices ({summary['live_coverage_pct']}%) came back live — "
                "the rest fell back to mock"
            )
    if ensemble.confidence < 0.5:
        warnings.append(
            f"ensemble confidence {ensemble.confidence:.2f} < 0.50 — treat the pick as low-conviction"
        )
    return warnings


# Map provenance slices → ONE Claude research task per data category. (category,
# priority, slices, what-to-find, how-to-feed-it-back).
_RESEARCH_CATEGORIES: tuple[tuple[str, str, tuple[str, ...], str, str], ...] = (
    ("xg", "high", ("xg_home", "xg_away"),
     "xG erzielt/zugelassen (Understat/FBref, letzte ~10 Spiele)",
     "Match-YAML: teams.home/away.avg_xg_season + avg_xg_conceded"),
    ("lineup", "high", ("lineup_home", "lineup_away"),
     "voraussichtliche Startelf + bestätigte Ausfälle",
     "qualitativ → avg_xg_season in der YAML anpassen (Schlüsselausfall senkt sie)"),
    ("injuries", "high", ("injuries_structured_home", "injuries_structured_away"),
     "Verletzungen/Sperren der Schlüsselspieler",
     "wie lineup — in die xG/Elo-Schätzung der YAML einarbeiten"),
    ("weather", "medium", ("weather",),
     "Wetter (Temp/Regen/Wind) am Venue zur Anstoßzeit",
     "qualitativ → Tor-Erwartung in der YAML (Hitze/Regen dämpft)"),
    ("news", "medium", ("news",),
     "aktuelle Team-News (Form, Trainerwechsel, Motivation)",
     "--sentiment-json oder xG-Anpassung in der YAML"),
    ("history", "medium", ("history_home", "history_away", "h2h"),
     "letzte Ergebnisse + Head-to-Head",
     "YAML: avg_xg_season / form_last5"),
    ("squad_value", "low", ("squad_value_home", "squad_value_away"),
     "Kaderwert (Transfermarkt)", "informativ"),
)

_PRIO_ORDER = {"high": 0, "medium": 1, "low": 2}


def _claude_tasks(
    provenance: dict[str, Any], cfg: dict[str, Any], *, mode: str, has_odds: bool
) -> list[dict[str, Any]]:
    """Phase 1.5 — Claude's essential Cowork assignment.

    Lists the LIVE data the automated connectors could **not** fetch (each slice
    came back ``mock`` or ``error``), so Claude researches it via web search and
    feeds it back. Empty in mock mode (no live gaps by design); odds are always
    requested when absent since they drive the edge table + market calibration.
    """
    teams = cfg.get("teams", {})
    home = (teams.get("home", {}) or {}).get("name", "Home")
    away = (teams.get("away", {}) or {}).get("name", "Away")
    venue = (cfg.get("match", {}) or {}).get("venue") or "venue"
    tasks: list[dict[str, Any]] = []

    if mode == "live":
        for cat, prio, slices, what, fill in _RESEARCH_CATEGORIES:
            degraded = [s for s in slices
                        if (provenance.get(s, {}) or {}).get("mode") in ("mock", "error")]
            if degraded:
                tasks.append({
                    "priority": prio, "category": cat,
                    "task": f"Recherchiere {what} für {home} vs {away}",
                    "fill_via": fill, "missing_slices": degraded,
                })

    if not has_odds:
        tasks.append({
            "priority": "high", "category": "odds",
            "task": (f"Recherchiere die Buchmacher-Konsens-Quoten "
                     f"(1X2, O/U 2.5, BTTS) für {home} vs {away} ({venue})"),
            "fill_via": "--odds \"H/D/A\" --odds-ou \"O/U\" --odds-btts \"Y/N\" (+ --calibrate market)",
            "missing_slices": ["odds"],
        })

    tasks.sort(key=lambda t: _PRIO_ORDER.get(t["priority"], 3))
    return tasks


async def run_prediction(
    cfg: dict[str, Any],
    *,
    mode: str = "mock",
    bootstrap_n: int | None = None,
    sentiment_payload: dict[str, Any] | None = None,
    odds_1x2: list[float] | None = None,
    odds_ou25: list[float] | None = None,
    odds_btts: list[float] | None = None,
    odds_dc: list[float] | None = None,
    odds_ah: tuple[float, float | None, float | None] | None = None,
    calibrate: str = "auto",
    overrides: dict[str, Any] | None = None,
    bankroll: float | None = None,
    ah_lines: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run all phases for one match config and return the raw ``result`` dict.

    Parameters
    ----------
    cfg
        A parsed match config (see ``wm2026.context.load_match_config`` /
        ``synth_config``).
    mode
        ``"mock"`` (offline, deterministic — the Python-API default for tests
        and reproducibility) or ``"live"`` (uses ``.env`` toggles for real
        connectors). The ``wm2026 predict`` CLI defaults to ``"live"`` instead
        — pass ``--mode mock`` to pin offline from the shell.
    bootstrap_n
        Override the bootstrap sample count (default: ``settings.bootstrap_n``).
    sentiment_payload
        Optional pre-computed sentiment dict (keys: ``sample_size``,
        ``home_sentiment``, ``away_sentiment``, ``home_momentum`` …). When None
        the sentiment + momentum factors neutralise and the ensemble
        re-normalises.
    odds_1x2 / odds_ou25 / odds_btts
        Decimal odds for the Phase-6 edge table. ``odds_1x2`` also seeds the
        market factor when the config didn't already carry odds.
    """
    started = datetime.now(timezone.utc)
    apply_runtime_profile(mode)

    ctx = build_context(cfg)
    if sentiment_payload:
        ctx.sentiment_payload = sentiment_payload
    if odds_1x2 and ctx.market_implied is None:
        fair, _ = edge_mod.devig(odds_1x2[:3])
        if len(fair) >= 3:
            ctx.market_implied = (fair[0], fair[1], fair[2])

    # Phase 1 — data collection.
    provenance = await _populate(ctx)

    # Phase 1.5 — inject Claude-researched overrides over the connectors' fallbacks
    # (Cowork v2). Re-read provenance so the `research` stamps reach the report.
    overrides_applied = apply_overrides(ctx, overrides)
    if overrides_applied:
        provenance = dict(ctx.provenance)

    # Phase 1.6 — fall back to live odds from the_odds_api connector whenever
    # the CLI didn't pass --odds* flags. The orchestrator already fetched them
    # in Phase 1; this lets the edge table populate without any manual entry.
    if ctx.live_odds:
        if odds_1x2 is None and ctx.live_odds.get("1x2"):
            odds_1x2 = list(ctx.live_odds["1x2"])
            if ctx.market_implied is None:
                fair, _ = edge_mod.devig(odds_1x2[:3])
                if len(fair) >= 3:
                    ctx.market_implied = (fair[0], fair[1], fair[2])
        if odds_ou25 is None and ctx.live_odds.get("ou_2_5"):
            odds_ou25 = list(ctx.live_odds["ou_2_5"])
        if odds_btts is None and ctx.live_odds.get("btts"):
            odds_btts = list(ctx.live_odds["btts"])

    # Phase 2/3 — factor decomposition (+ injected sentiment).
    signals = await _signals(ctx)

    # Phase 4 — goal-model stack + bootstrap CIs.
    boot = settings.bootstrap_n if bootstrap_n is None else int(bootstrap_n)
    base_home_xg, base_away_xg = _base_xg(cfg)
    # Optional Dixon-Coles MLE base xG (gated off by default → output unchanged).
    base_xg_source = "yaml"
    if getattr(settings, "use_mle_xg", False):
        try:
            from analysis.xg_estimator import estimate_base_xg
            mle, diag = estimate_base_xg(ctx, ctx.home_code, ctx.away_code, settings=settings)
            base_xg_source = diag.get("source", "yaml")
            if mle is not None:
                base_home_xg, base_away_xg = mle
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("mle_xg_failed", error=str(exc), exc_info=True)
    predictor = MatchPredictor(
        rho=getattr(settings, "dixon_coles_rho", 0.1),
        goal_model=settings.goal_model,
        negbin_size=settings.negbin_size,
        combine=settings.goal_model_combine,
        bootstrap_n=boot,
        bootstrap_xg_sigma=settings.bootstrap_xg_sigma,
        # Phase 4 (2.2): optionales 4. Blend-Modell — Default off.
        include_bivariate=getattr(settings, "include_bivariate", False),
        bivariate_lambda3=getattr(settings, "bivariate_lambda3", 0.12),
    )
    # market_prior stays None: when odds are present the MarketOddsFactor
    # already tilts λ, so a second 1X2 blend would double-count (match_service
    # drops it for exactly this reason).
    out, ensemble = predictor.predict_from_signals(
        signals, base_home_xg=base_home_xg, base_away_xg=base_away_xg,
        market_prior=None,
    )

    # Blended score-probability matrix (all 3 models at the final λ) — used for
    # BOTH the heatmap and every derived market (Phase-1 math upgrade). Because
    # the markets are linear functionals of the matrix, deriving them here keeps
    # Double-Chance / Asian-Handicap / totals exactly consistent with the
    # blended headline 1X2 + O/U numbers.
    from models_ml.poisson_goals import blend_score_matrix

    derived_markets: dict[str, Any] = {}
    score_matrix: list[list[float]] = []
    try:
        matrix = blend_score_matrix(predictor.models, out.home_xg, out.away_xg)
        score_matrix = [[float(matrix[i][j]) for j in range(matrix.shape[1])]
                        for i in range(matrix.shape[0])]
        corners_cfg = (cfg.get("corners") or {}) if isinstance(cfg, dict) else {}
        cards_cfg = (cfg.get("cards") or {}) if isinstance(cfg, dict) else {}
        derived_markets = markets_mod.derive_all(
            matrix,
            p1x2=(out.home_win_prob, out.draw_prob, out.away_win_prob),
            ah_lines=ah_lines,
            lam_home=out.home_xg, lam_away=out.away_xg,
            models=predictor.models, ht_share=settings.ht_lambda_share,
            corners_lam_home=corners_cfg.get("lambda_home"),
            corners_lam_away=corners_cfg.get("lambda_away"),
            cards_lam_total=cards_cfg.get("lambda_total"),
        )
    except Exception:  # pragma: no cover - defensive
        score_matrix = []

    # Phase 5 — calibration. mode=auto uses a fitted artifact if present, else
    # raw; mode=market anchors to the vig-free consensus (ctx.market_implied was
    # de-vigged from the config odds or --odds in build_context/run_prediction).
    calibration = _maybe_calibrate(
        out,
        mode=(calibrate or "auto").lower(),
        market=ctx.market_implied,
        market_weight=getattr(settings, "calibration_market_weight", 0.5),
    )

    # Phase 6 — market edge / value detection.
    blended = {
        "home_win": out.home_win_prob,
        "draw": out.draw_prob,
        "away_win": out.away_win_prob,
        "over_25": out.over_25,
        "btts": out.btts,
    }
    # Blended bootstrap CI → conservative (p5) edge / half-Kelly columns.
    ci_blended = (out.features.get("confidence_intervals") or {}).get("blended")
    edge_rows = edge_mod.compute_edges(
        blended, odds_1x2=odds_1x2, odds_ou25=odds_ou25, odds_btts=odds_btts,
        odds_dc=odds_dc, ci=ci_blended,
    )
    # Optional Asian-handicap value line (model probs from the score matrix).
    if odds_ah is not None:
        ah_line, ah_home_odd, ah_away_odd = odds_ah
        try:
            ah_model = markets_mod.asian_handicap(matrix, float(ah_line))
            # Phase 4 (4.2): konservative p5-Spalten auch fuer AH — bootstrappt
            # die No-Push-Wahrscheinlichkeiten direkt auf der Blend-Matrix
            # (ein Sampling-Pass fuer beide Seiten).
            ah_home_lo = ah_away_lo = None
            if boot > 0:
                from models_ml.poisson_goals import bootstrap_blend_metrics
                _line = float(ah_line)
                ah_ci = bootstrap_blend_metrics(
                    predictor.models, out.home_xg, out.away_xg,
                    {
                        "home_np": lambda M: markets_mod.asian_handicap(M, _line)["home_prob_nopush"],
                        "away_np": lambda M: markets_mod.asian_handicap(M, _line)["away_prob_nopush"],
                    },
                    n=min(boot, 300), xg_sigma=settings.bootstrap_xg_sigma,
                )
                if "home_np" in ah_ci:
                    ah_home_lo = ah_ci["home_np"][0]
                if "away_np" in ah_ci:
                    ah_away_lo = ah_ci["away_np"][0]
            edge_rows.extend(edge_mod.evaluate_asian_handicap(
                ah_model, home_odd=ah_home_odd, away_odd=ah_away_odd,
                home_p_lower=ah_home_lo, away_p_lower=ah_away_lo,
            ))
        except Exception:  # pragma: no cover - defensive
            log.warning("asian_handicap_edge_failed", line=ah_line)

    # Phase 4 (4.4): Bankroll-bewusstes Staking — annotiert jede Edge-Zeile mit
    # konkreten Einsatz-Betraegen. stake_cons folgt der ½-Kelly-auf-p5-Disziplin:
    # 0.00 sobald die konservative Edge nicht positiv ist.
    if bankroll is not None and bankroll > 0:
        for r in edge_rows:
            hk = r.get("half_kelly_pct")
            r["stake_half_kelly"] = round(bankroll * hk / 100.0, 2) if hk else 0.0
            hkc = r.get("half_kelly_cons")
            cons_edge = r.get("edge_pct_cons")
            r["stake_cons"] = (
                round(bankroll * hkc / 100.0, 2)
                if hkc and cons_edge is not None and cons_edge > 0 else 0.0
            )

    best_value = edge_mod.best_value_pick(edge_rows)
    best_value_cons = edge_mod.best_value_cons_pick(edge_rows)

    # Phase 7 — validation + Claude's Cowork research assignment.
    warnings = _validate(out, ensemble, signals, provenance)
    claude_tasks = _claude_tasks(provenance, cfg, mode=mode, has_odds=bool(odds_1x2))
    if mode == "live":
        slices = [p for p in provenance.values() if isinstance(p, dict)]
        degraded = sum(1 for p in slices if p.get("mode") in ("mock", "error"))
        if degraded:
            warnings.append(
                f"live mode: {degraded}/{len(slices)} data slices degraded to mock — "
                f"Claude must research the gaps (see claude_tasks / the Cowork-Auftrag section)"
            )

    return {
        "config": cfg,
        "started_at": started,
        "mode": mode,
        "base_home_xg": base_home_xg,
        "base_away_xg": base_away_xg,
        "base_xg_source": base_xg_source,
        "lambda_home_ci": _lambda_ci(out.home_xg, settings.bootstrap_xg_sigma),
        "lambda_away_ci": _lambda_ci(out.away_xg, settings.bootstrap_xg_sigma),
        "prediction": out,
        "ensemble": ensemble,
        "signals": signals,
        "score_matrix": score_matrix,
        "derived_markets": derived_markets,
        "calibration": calibration,
        "provenance": provenance,
        "provenance_summary": _provenance_summary(provenance),
        "edges": edge_rows,
        "best_value": best_value,
        "best_value_cons": best_value_cons,
        "bankroll": bankroll,
        "warnings": warnings,
        "claude_tasks": claude_tasks,
        "overrides_applied": overrides_applied,
        "bootstrap_n": boot,
    }


__all__ = ["run_prediction"]
