"""Zentrale Warning-Suppression fuer bekannte, harmlose Warnings.

Wird ganz oben in jedem Entry-Point importiert BEVOR pytensor/arviz/pymc
importiert werden -- sonst greifen die Filter zu spaet.
"""
from __future__ import annotations

import logging
import os
import warnings


def silence_known_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"arviz.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"arviz.*")

    logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)
    logging.getLogger("pytensor.tensor.blas").setLevel(logging.ERROR)
    logging.getLogger("pytensor.link.c.cmodule").setLevel(logging.ERROR)

    os.environ.setdefault("PYTENSOR_FLAGS", "")
