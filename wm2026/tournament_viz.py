"""Tournament-Bracket-Viz — turn a :class:`TournamentResult` into a ranked
pyramid showing the most likely champions, finalists, and knockout entrants.

The Monte-Carlo simulation produces three per-team probabilities:
``P(reach R16)`` (``advance_prob``), ``P(reach final)`` (``final_prob``) and
``P(title)`` (``title_prob``). A "true" bracket changes every sim, so this
viz collapses them into three ranked tiers — the rows you'd expect to see
in a 16-team / 2-team / 1-team bracket if you sampled the modal field.

Pure stdlib, no new deps. The CLI exposes it via
``wm2026 tournament --format bracket``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wm2026.tournament import TournamentResult


_BAR_WIDTH = 22


def _bar(p: float) -> str:
    """Unicode bar of width ``_BAR_WIDTH`` proportional to ``p ∈ [0, 1]``."""
    filled = max(0, min(_BAR_WIDTH, int(round(p * _BAR_WIDTH))))
    return "█" * filled + "·" * (_BAR_WIDTH - filled)


def _row(rank: int, name: str, prob: float, name_width: int) -> str:
    return f"  {rank:>2}. {name:<{name_width}}  {_bar(prob)}  {100 * prob:5.1f}%"


def _ranked(prob_map: dict[str, float], names: dict[str, str],
            top_n: int) -> list[tuple[str, float]]:
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(names.get(c, c), p) for c, p in items if p > 0.0]


def render_bracket(res: "TournamentResult", names: dict[str, str]) -> str:
    """Build the three-tier ranked pyramid for terminal output.

    ``names`` maps the FIFA code → display name (the CLI already builds it).
    Teams with zero probability in a tier are dropped from that tier.
    """
    champion = _ranked(res.title_prob, names, top_n=3)
    finalists = _ranked(res.final_prob, names, top_n=8)
    r16 = _ranked(res.advance_prob, names, top_n=16)

    nw = max((len(n) for n, _ in champion + finalists + r16), default=4)

    L = [
        "🏆 WM 2026 — TURNIER-BRACKET",
        f"   {res.n_sims} Simulationen · neutral (kein Heimvorteil)",
        "",
        "  ╔══════════════════════════════════════════════════════════════╗",
        "  ║  🏆 WELTMEISTER  (P(Titel))                                  ║",
        "  ╚══════════════════════════════════════════════════════════════╝",
    ]
    for i, (n, p) in enumerate(champion, 1):
        L.append(_row(i, n, p, nw))
    if not champion:
        L.append("   (keine Wahrscheinlichkeit > 0)")

    L += [
        "",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  🥇 FINALE  (P(Finale erreicht))                             │",
        "  └──────────────────────────────────────────────────────────────┘",
    ]
    for i, (n, p) in enumerate(finalists, 1):
        L.append(_row(i, n, p, nw))

    L += [
        "",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  🏟  ACHTELFINALE  (P(K.o.-Runde erreicht))                  │",
        "  └──────────────────────────────────────────────────────────────┘",
    ]
    for i, (n, p) in enumerate(r16, 1):
        L.append(_row(i, n, p, nw))

    return "\n".join(L)


__all__ = ["render_bracket"]
