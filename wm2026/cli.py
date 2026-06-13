"""``wm2026`` command-line entry point.

    wm2026 predict --match config/matches/group_a/cze_vs_rsa.yaml
    wm2026 predict --home Germany --away Brazil --stage QF --odds "2.10/3.40/3.20"
    wm2026 list

``predict`` runs the full 8-phase pipeline and prints the Markdown report; with
``--out DIR`` it also writes ``<match_id>.json`` + ``<match_id>.md`` (and PNG
charts when ``--charts`` and matplotlib are available).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def _quiet_logs(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    try:
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(level)
        )
    except Exception:
        pass


def _add_predict_args(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group("match input (use --match OR --home/--away)")
    src.add_argument("--match", "-m", help="path to a match config YAML")
    src.add_argument("--home", help="home team name (ad-hoc match)")
    src.add_argument("--away", help="away team name (ad-hoc match)")
    src.add_argument("--home-code", help="home FIFA code (default: first 3 letters)")
    src.add_argument("--away-code", help="away FIFA code")
    src.add_argument("--stage", default="Group", help="Group | R32 | R16 | QF | SF | Final")
    src.add_argument("--kickoff", help="ISO8601 kickoff, e.g. 2026-06-12T21:00:00Z")
    src.add_argument("--venue", help="stadium / city")
    src.add_argument("--home-xg", type=float, default=1.40)
    src.add_argument("--away-xg", type=float, default=1.30)
    src.add_argument("--home-xga", type=float, default=1.30)
    src.add_argument("--away-xga", type=float, default=1.40)
    src.add_argument("--home-elo", type=int, default=1700)
    src.add_argument("--away-elo", type=int, default=1700)

    odds = p.add_argument_group("odds (Phase 6 edge table)")
    odds.add_argument("--odds", help="1X2 decimal odds, e.g. \"2.10/3.40/3.20\"")
    odds.add_argument("--odds-ou", help="Over/Under 2.5 odds, e.g. \"1.85/1.95\"")
    odds.add_argument("--odds-btts", help="BTTS Yes/No odds, e.g. \"1.80/2.00\"")
    odds.add_argument("--odds-dc", help="Double Chance odds 1X/12/X2, e.g. \"1.25/1.30/1.55\"")
    odds.add_argument("--odds-ah", help="Asian-handicap line + home/away odds, "
                                        "\"LINE:HOME/AWAY\". Use = for negative lines, "
                                        "e.g. --odds-ah=-0.5:1.95/1.95")

    run = p.add_argument_group("run options")
    run.add_argument("--mode", choices=["mock", "live"], default="live",
                     help="live = fetch real internet data (DEFAULT); "
                          "mock = offline, no keys/network (reproducible, for tests/CI)")
    run.add_argument("--live-sources",
                     help="comma list: ONLY these connectors go live, the rest mock "
                          "(implies --mode live), e.g. --live-sources weather,clubelo")
    run.add_argument("--mock-sources",
                     help="comma list: force ONLY these connectors to mock, "
                          "e.g. --mock-sources transfermarkt,rss")
    run.add_argument("--bankroll", type=float, default=None,
                     help="bankroll in your currency — annotates the edge table with "
                          "stake amounts (½-Kelly p50 + conservative p5)")
    run.add_argument("--bootstrap", type=int, default=None,
                     help="bootstrap samples for CIs (default: settings.bootstrap_n)")
    run.add_argument("--calibrate", choices=["auto", "market", "none"], default="auto",
                     help="Phase 5: auto = fitted artifact if present else raw; "
                          "market = anchor 1X2 to the vig-free odds; none = off")
    run.add_argument("--sentiment-json", help="path to a sentiment_payload JSON to inject")
    run.add_argument("--overrides-json", help="path to a Claude-researched overrides JSON "
                                              "(xg/elo/weather/sentiment) — see `wm2026 research`")
    run.add_argument("--out", "-o", help="output directory for JSON/MD/PNG")
    run.add_argument("--format", choices=["markdown", "json", "html", "summary"],
                     default="markdown",
                     help="output format (default markdown); html = self-contained report; "
                          "summary = token-budget briefing (~400 tokens)")
    run.add_argument("--json-only", action="store_true", help="alias for --format json")
    run.add_argument("--charts", action="store_true", help="also render PNG charts (needs matplotlib)")
    run.add_argument("--compact", action="store_true",
                     help="token-budget JSON: drop raw provenance, per-model markets, "
                          "per-model CIs, raw_data on factors, AH long tail (~57% smaller)")
    run.add_argument("--charts-external", action="store_true",
                     help="HTML references on-disk PNGs instead of inlining base64 "
                          "(~95 KB → ~10 KB HTML); pairs with --charts + --out")
    run.add_argument("--ah-lines", default=None,
                     help="comma list of Asian-handicap lines, e.g. \"-0.5,0,0.5\" "
                          "(default: full spread −2..+2 incl. quarters)")
    run.add_argument("--gzip", action="store_true",
                     help="also write reports/<id>.json.gz (smaller on disk; "
                          "downstream tools can stream it)")
    run.add_argument("--verbose", "-v", action="store_true", help="show debug logs")


def _parse_ah(spec: str | None) -> tuple[float, float | None, float | None] | None:
    """Parse ``"LINE:HOME/AWAY"`` (e.g. ``"-0.5:1.95/1.95"``) into
    ``(line, home_odd, away_odd)``. Returns ``None`` for an empty/invalid spec."""
    if not spec:
        return None
    from wm2026.edge import parse_odds

    try:
        line_part, odds_part = spec.split(":", 1)
        line = float(line_part)
    except ValueError:
        return None
    odds = parse_odds(odds_part) or []
    home_odd = odds[0] if len(odds) > 0 else None
    away_odd = odds[1] if len(odds) > 1 else None
    return (line, home_odd, away_odd)


def _parse_ah_lines(spec: str | None) -> list[float] | None:
    """Parse a comma list of AH lines, e.g. ``"-0.5,0,0.5"``. None ⇒ default set."""
    if not spec:
        return None
    out: list[float] = []
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            raise SystemExit(f"error: --ah-lines: cannot parse {tok!r} as float")
    return out or None


def _build_cfg(args: argparse.Namespace):
    from wm2026.context import load_match_config, synth_config

    if args.match:
        return load_match_config(args.match)
    if args.home and args.away:
        return synth_config(
            home_team=args.home, away_team=args.away,
            home_code=args.home_code, away_code=args.away_code,
            stage=args.stage, kickoff=args.kickoff, venue=args.venue,
            home_xg=args.home_xg, away_xg=args.away_xg,
            home_xga=args.home_xga, away_xga=args.away_xga,
            home_elo=args.home_elo, away_elo=args.away_elo,
            odds_1x2=args.odds,
        )
    raise SystemExit("error: provide --match <yaml> or both --home and --away")


# Per-source live/mock toggles (Phase 4 / Verbesserungsplan Phase-3-Offen-Punkt).
# Lowercase source name ⇄ USE_MOCK_* env key, mirroring config.settings.
# Discovered dynamically from ``Settings.model_fields`` so a new connector in
# config/settings.py auto-syncs the CLI flags without a manual list update here.
def _list_source_names() -> tuple[str, ...]:
    """Connector names derived from ``Settings``'s ``use_mock_*`` fields.

    Lazy import: ``config.settings`` is light (pydantic-settings only — no
    numpy/scipy), so calling this from the CLI startup path is cheap, but the
    import still sits inside the function so ``wm2026 --help`` or
    ``wm2026 list`` never pulls it in.
    """
    from config.settings import Settings
    return tuple(
        name[len("use_mock_"):]
        for name in Settings.model_fields
        if name.startswith("use_mock_")
    )


def _parse_source_list(spec: str | None, flag: str) -> list[str]:
    """Validate a comma list of connector names → lowercase list (or exit)."""
    if not spec:
        return []
    sources = _list_source_names()
    names = [t.strip().lower() for t in spec.split(",") if t.strip()]
    unknown = sorted(set(names) - set(sources))
    if unknown:
        raise SystemExit(
            f"error: {flag}: unknown source(s) {', '.join(unknown)} — "
            f"valid: {', '.join(sorted(sources))}")
    return names


def _seed_source_toggles(args: argparse.Namespace) -> None:
    """Translate --mode / --live-sources / --mock-sources into USE_MOCK_* env.

    Must run before the heavy imports so the import-time ``settings`` singleton
    already sees the toggles (same contract as the original mock pre-seed).
    Explicit CLI flags overwrite the environment (they beat any ``.env``).
    """
    live_only = _parse_source_list(args.live_sources, "--live-sources")
    mock_only = _parse_source_list(args.mock_sources, "--mock-sources")

    if live_only and args.mode == "mock":
        # Selective-live makes no sense in full-mock mode — flip to live.
        print("[--live-sources given → switching --mode to live]", file=sys.stderr)
        args.mode = "live"

    sources = _list_source_names()
    if args.mode == "mock":
        for name in sources:
            os.environ.setdefault(f"USE_MOCK_{name.upper()}", "true")
    elif live_only:
        # ONLY the listed connectors go live; everything else is mock.
        for name in sources:
            os.environ[f"USE_MOCK_{name.upper()}"] = (
                "false" if name in live_only else "true")
    for name in mock_only:
        os.environ[f"USE_MOCK_{name.upper()}"] = "true"


def _cmd_predict(args: argparse.Namespace) -> int:
    _quiet_logs(args.verbose)
    # Pre-seed mock/live toggles before the heavy imports so even import-time
    # settings reflect the chosen profile (apply_runtime_profile re-asserts the
    # full-mock case anyway).
    _seed_source_toggles(args)

    from wm2026.edge import parse_odds
    from wm2026.pipeline import run_prediction
    from wm2026.report import build_report

    cfg = _build_cfg(args)
    sentiment = None
    if args.sentiment_json:
        sentiment = json.loads(Path(args.sentiment_json).read_text(encoding="utf-8"))
    overrides = None
    if args.overrides_json:
        overrides = json.loads(Path(args.overrides_json).read_text(encoding="utf-8"))

    result = asyncio.run(run_prediction(
        cfg,
        mode=args.mode,
        bootstrap_n=args.bootstrap,
        sentiment_payload=sentiment,
        odds_1x2=parse_odds(args.odds),
        odds_ou25=parse_odds(args.odds_ou),
        odds_btts=parse_odds(args.odds_btts),
        odds_dc=parse_odds(args.odds_dc),
        odds_ah=_parse_ah(args.odds_ah),
        calibrate=args.calibrate,
        overrides=overrides,
        bankroll=args.bankroll,
        ah_lines=_parse_ah_lines(args.ah_lines),
    ))
    report = build_report(result)
    # Optional Token-Sparmodus: ein dünnerer JSON-Schnappschuss (faktisch
    # ohne data_sources-raw, per_model-CIs, factor raw_data). Schema bleibt
    # gleich; das Feld ``compact: true`` signalisiert es konsumierende Skills.
    if args.compact:
        from wm2026.report import compact as _compact
        report["json"] = _compact(report["json"])
    fmt = "json" if args.json_only else args.format

    html_str: str | None = None
    if fmt == "html":
        from wm2026.report_html import build_html
        ext_prefix = report["json"]["match_id"] if args.charts_external else None
        html_str = build_html(result, report["json"],
                              external_charts_prefix=ext_prefix)
        print(html_str)
    elif fmt == "json":
        print(json.dumps(report["json"], indent=2, ensure_ascii=False))
    elif fmt == "summary":
        from wm2026.summary import summarise
        print(summarise(report["json"]))
    else:
        print(report["markdown"])

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        mid = report["json"]["match_id"]
        json_text = json.dumps(report["json"], indent=2, ensure_ascii=False)
        (out_dir / f"{mid}.json").write_text(json_text, encoding="utf-8")
        (out_dir / f"{mid}.md").write_text(report["markdown"], encoding="utf-8")
        written = [f"{mid}.json", f"{mid}.md"]
        if args.gzip:
            import gzip
            with gzip.open(out_dir / f"{mid}.json.gz", "wt", encoding="utf-8") as fh:
                fh.write(json_text)
            written.append(f"{mid}.json.gz")
        if fmt == "html":
            if html_str is None:
                from wm2026.report_html import build_html
                ext_prefix = mid if args.charts_external else None
                html_str = build_html(result, report["json"],
                                      external_charts_prefix=ext_prefix)
            (out_dir / f"{mid}.html").write_text(html_str, encoding="utf-8")
            written.append(f"{mid}.html")
        if args.charts:
            try:
                from wm2026.viz import render_charts
                written += [p.name for p in render_charts(result, out_dir, mid)]
            except Exception as exc:
                print(f"[charts skipped: {exc}]", file=sys.stderr)
        # Always also emit the token-budget summary — it's tiny and the most-
        # consumed artefact in the Cowork loop.
        try:
            from wm2026.summary import summarise
            (out_dir / f"{mid}.summary.md").write_text(
                summarise(report["json"]), encoding="utf-8")
            written.append(f"{mid}.summary.md")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[summary skipped: {exc}]", file=sys.stderr)
        print(f"\n→ wrote {', '.join(written)} to {out_dir}/", file=sys.stderr)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Run the dependency + pipeline self-check (see wm2026.doctor)."""
    from wm2026.doctor import main as doctor_main
    argv = []
    if args.verbose:
        argv.append("-v")
    if args.json:
        argv.append("--json")
    return doctor_main(argv)


def _cmd_summary(args: argparse.Namespace) -> int:
    """Print a token-budget summary of an existing JSON report."""
    from wm2026.summary import summarise

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            js = json.load(fh)
    else:
        js = json.loads(path.read_text(encoding="utf-8"))
    print(summarise(js, top_edges=args.top))
    return 0


def _cmd_tournament(args: argparse.Namespace) -> int:
    """Monte-Carlo the whole tournament (group → knockout) → per-team title /
    final / knockout probabilities, from the YAML group configs."""
    _quiet_logs(args.verbose)
    import yaml
    from models_ml.poisson_goals import build_all_goal_models
    from wm2026.tournament import simulate_tournament

    root = Path(args.groups or "config/matches")
    groups: dict[str, list[str]] = {}
    team_data: dict[str, tuple[float, float, str]] = {}
    for gdir in sorted(root.glob("group_*")):
        gname = gdir.name.replace("group_", "").upper()
        codes: list[str] = []
        for yf in sorted(gdir.glob("*.yaml")):
            if yf.name.startswith("_"):
                continue
            cfg = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
            for side in ("home", "away"):
                t = (cfg.get("teams", {}) or {}).get(side, {}) or {}
                code = str(t.get("code") or t.get("fifa_code") or t.get("name", "")).upper()
                if not code:
                    continue
                team_data.setdefault(code, (
                    float(t.get("avg_xg_season", 1.3) or 1.3),
                    float(t.get("avg_xg_conceded", 1.3) or 1.3),
                    t.get("name", code)))
                if code not in codes:
                    codes.append(code)
        if len(codes) >= 2:
            groups[gname] = codes[:4]
    if not groups:
        print(f"no group_* configs under {root}/", file=sys.stderr)
        return 1

    def lam(a: str, b: str):
        aa, ad, _ = team_data.get(a, (1.3, 1.3, a))
        ba, bd, _ = team_data.get(b, (1.3, 1.3, b))
        return ((aa + bd) / 2.0, (ba + ad) / 2.0)     # neutral venue (no home edge)

    res = simulate_tournament(groups, lam_provider=lam, models=build_all_goal_models(),
                              n_sims=args.sims, seed=args.seed)
    names = {c: team_data.get(c, (0, 0, c))[2] for c in res.title_prob}

    if args.format == "json":
        out = {"n_sims": res.n_sims, "groups": len(groups),
               "title_prob": res.title_prob, "final_prob": res.final_prob,
               "advance_prob": res.advance_prob}
        text = json.dumps(out, indent=2, ensure_ascii=False)
    elif args.format == "bracket":
        from wm2026.tournament_viz import render_bracket
        text = render_bracket(res, names)
    else:
        L = [f"# 🏆 WM 2026 — Turnier-Monte-Carlo",
             f"*{res.n_sims} Simulationen · {len(groups)} Gruppen · neutral (kein Heimvorteil)*", "",
             "| # | Team | 🏆 Titel | Finale | Achtelfinale+ |", "|---|---|---|---|---|"]
        for i, (c, p) in enumerate(res.ranked("title_prob")[:24], 1):
            L.append(f"| {i} | {names.get(c, c)} | {100*p:.1f}% | "
                     f"{100*res.final_prob[c]:.1f}% | {100*res.advance_prob[c]:.1f}% |")
        text = "\n".join(L)
    print(text)
    if args.out:
        out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
        ext = "json" if args.format == "json" else "md"
        (out_dir / f"tournament.{ext}").write_text(text, encoding="utf-8")
        print(f"\n→ wrote tournament.{ext} to {out_dir}/", file=sys.stderr)
    return 0


def _cmd_research(args: argparse.Namespace) -> int:
    """Emit Claude's Cowork assignment (live-data gaps) + an overrides-JSON
    template to fill, then re-run ``predict --overrides-json``."""
    _quiet_logs(args.verbose)
    _seed_source_toggles(args)    # same per-source toggle contract as predict
    from wm2026.context import overrides_template
    from wm2026.edge import parse_odds
    from wm2026.pipeline import run_prediction

    cfg = _build_cfg(args)
    result = asyncio.run(run_prediction(
        cfg, mode=args.mode, bootstrap_n=0, odds_1x2=parse_odds(args.odds),
    ))
    teams = cfg.get("teams", {})
    home = (teams.get("home", {}) or {}).get("name", "Home")
    away = (teams.get("away", {}) or {}).get("name", "Away")
    tasks = result.get("claude_tasks", [])

    lines = [f"# 🤝 Cowork-Auftrag — {home} vs {away}  (mode: {args.mode})", ""]
    if tasks:
        for i, t in enumerate(tasks, 1):
            lines.append(f"{i}. **[{t['priority']}]** {t['task']}")
            lines.append(f"   → einspeisen via: `{t['fill_via']}`")
    else:
        lines.append("_Keine offenen Lücken — alle Quellen live._")
    print("\n".join(lines))

    template = overrides_template(cfg)
    payload = json.dumps(template, indent=2, ensure_ascii=False)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        mid = cfg.get("match", {}).get("id", "wm2026_match")
        path = out_dir / f"{mid}.overrides.json"
        path.write_text(payload, encoding="utf-8")
        print(f"\n→ Overrides-Template: {path}", file=sys.stderr)
        print(f"  Ausfüllen, dann: wm2026 predict ... --overrides-json {path}", file=sys.stderr)
    else:
        print("\n## Overrides-Template (ausfüllen → --overrides-json)\n```json")
        print(payload)
        print("```")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.dir or "config/matches")
    files = sorted(p for p in root.rglob("*.yaml") if not p.name.startswith("_"))
    if not files:
        print(f"no match configs under {root}/", file=sys.stderr)
        return 1
    for p in files:
        print(p)
    print(f"\n{len(files)} match configs", file=sys.stderr)
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    """Phase 2.4 — Optuna RPS-tune of blend weights + Dixon-Coles ρ.

    Thin wrapper around ``scripts/tune_models_offline.py`` so the same engine
    that powers the ``tune-models`` skill is a first-class subcommand. Needs
    the optional ``[tune]`` extra (Optuna) — clear error if missing.
    """
    _quiet_logs(args.verbose)
    from scripts.tune_models_offline import main as tune_main
    argv = [args.history]
    if args.trials is not None:
        argv += ["--trials", str(args.trials)]
    return tune_main(argv)


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Score the JSON reports under ``--reports`` against actual match outcomes.

    Joins ``<reports>/*.json`` with a ground-truth CSV (``match_id``,
    ``home_score``, ``away_score``) and reports Brier / RPS / LogLoss for 1X2
    plus, when available, the hit-rate of ``best_value_cons`` and a naïve
    half-Kelly ROI. Closes the loop the ``analyze-edge`` / p5 rule promises.
    """
    _quiet_logs(args.verbose)
    from wm2026.backtest import run_backtest

    report = run_backtest(
        reports_dir=Path(args.reports),
        truth_csv=Path(args.truth),
        bankroll=args.bankroll,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        from wm2026.backtest import format_briefing
        print(format_briefing(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wm2026",
        description="WM 2026 — Match-Analyse & Prediction workflow (8-phase pipeline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="run the full prediction pipeline")
    _add_predict_args(p_predict)
    p_predict.set_defaults(func=_cmd_predict)

    p_research = sub.add_parser(
        "research", help="emit the Cowork research assignment + an overrides-JSON template")
    _add_predict_args(p_research)
    p_research.set_defaults(func=_cmd_research)

    p_tour = sub.add_parser("tournament", help="Monte-Carlo the whole tournament (group → knockout)")
    p_tour.add_argument("--sims", type=int, default=10000, help="number of simulations (default 10000)")
    p_tour.add_argument("--seed", type=int, default=0, help="RNG seed (deterministic)")
    p_tour.add_argument("--groups", help="config root with group_* dirs (default config/matches)")
    p_tour.add_argument("--mode", choices=["mock", "live"], default="mock", help="(reads YAML; mock/live parity)")
    p_tour.add_argument("--format", choices=["markdown", "json", "bracket"], default="markdown",
                        help="markdown = ranked table · json = raw probabilities · "
                             "bracket = ranked pyramid (champion / final / R16)")
    p_tour.add_argument("--out", "-o", help="also write tournament.md/json here")
    p_tour.add_argument("--verbose", "-v", action="store_true")
    p_tour.set_defaults(func=_cmd_tournament)

    p_list = sub.add_parser("list", help="list available match configs")
    p_list.add_argument("--dir", help="match config root (default: config/matches)")
    p_list.set_defaults(func=_cmd_list)

    p_doc = sub.add_parser(
        "doctor",
        help="dependency + pipeline self-check (clear ✅/⚠️/❌ per group)")
    p_doc.add_argument("--verbose", "-v", action="store_true")
    p_doc.add_argument("--json", action="store_true",
                       help="emit a compact JSON status (CI-friendly)")
    p_doc.set_defaults(func=_cmd_doctor)

    p_sum = sub.add_parser(
        "summary",
        help="token-budget briefing for an existing JSON report (~400 tokens)")
    p_sum.add_argument("path", help="path to <match_id>.json (or .json.gz)")
    p_sum.add_argument("--top", type=int, default=5,
                       help="cap the edge table at the top-N rows by p5 edge (default 5)")
    p_sum.set_defaults(func=_cmd_summary)

    p_tune = sub.add_parser(
        "tune",
        help="Phase 2.4 — Optuna RPS-tune blend weights + Dixon-Coles ρ from a "
             "history CSV (writes models_ml/artifacts/tuned_model_params.yaml)")
    p_tune.add_argument("history",
                        help="history CSV with columns home_xg, away_xg, home_score, away_score")
    p_tune.add_argument("--trials", type=int, default=None,
                        help="number of Optuna trials (script default: 200)")
    p_tune.add_argument("--verbose", "-v", action="store_true")
    p_tune.set_defaults(func=_cmd_tune)

    p_bt = sub.add_parser(
        "backtest",
        help="score saved JSON reports against actual outcomes "
             "(Brier / RPS / LogLoss + p5 hit-rate + naïve Kelly ROI)")
    p_bt.add_argument("--reports", required=True,
                      help="directory containing predict-output JSON reports")
    p_bt.add_argument("--truth", required=True,
                      help="ground-truth CSV: match_id, home_score, away_score")
    p_bt.add_argument("--bankroll", type=float, default=1000.0,
                      help="bankroll for the naïve Kelly ROI calculation (default 1000)")
    p_bt.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_bt.add_argument("--verbose", "-v", action="store_true")
    p_bt.set_defaults(func=_cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
