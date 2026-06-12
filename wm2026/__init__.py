"""WM 2026 — Match-Analyse & Prediction Workflow.

A thin, dependency-light orchestration layer on top of the existing
``data_sources`` / ``factors`` / ``analysis`` / ``models_ml`` modules. It turns
the 8-phase master prompt (``prompts/WM2026_MASTER_PROMPT.md``) into a single,
reproducible command:

    wm2026 predict --match config/matches/group_a/cze_vs_rsa.yaml --mode mock

The package is intentionally free of the FastAPI / database / Prefect stack so
that *anyone* can clone the repo and run a full prediction offline (mock mode)
with only the core requirements installed.

Public API
----------
``run_prediction``      Phase 1-7 — fan-out data, run the factor ensemble,
                        blend the goal models, bootstrap confidence intervals.
``build_report``        Phase 8 — assemble the structured JSON + Markdown report.
``compute_edges``       Phase 6 — de-vig bookmaker odds, edge & Kelly staking.

Imports are lazy (PEP 562) so ``import wm2026`` — and lightweight commands like
``wm2026 list`` / ``--help`` — never pull in numpy/scipy or the factor chain.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.1.0"
MODEL_VERSION = "wm2026-workflow-1.1"

__all__ = [
    "__version__",
    "MODEL_VERSION",
    "run_prediction",
    "build_report",
    "compute_edges",
    "devig",
]

if TYPE_CHECKING:  # pragma: no cover
    from wm2026.edge import compute_edges, devig
    from wm2026.pipeline import run_prediction
    from wm2026.report import build_report


def __getattr__(name: str) -> Any:
    if name in ("compute_edges", "devig"):
        from wm2026 import edge
        return getattr(edge, name)
    if name == "run_prediction":
        from wm2026.pipeline import run_prediction
        return run_prediction
    if name == "build_report":
        from wm2026.report import build_report
        return build_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
