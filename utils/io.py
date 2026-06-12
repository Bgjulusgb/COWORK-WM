"""I/O helpers — atomic file writes for hot-reload configs.

Used by api/admin.py for runtime_weights.yaml / runtime_flags.yaml so a parallel
reader (settings.reload_runtime_*()) never sees a half-written file. The
strategy is the POSIX-classic write-temp-then-rename pattern; ``os.replace``
is atomic on Windows too as of Python 3.3.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


_REPLACE_ATTEMPTS = 20
_REPLACE_BACKOFF_S = 0.002       # 2 ms initial backoff
_REPLACE_BACKOFF_CAP_S = 0.1     # cap exponential backoff at 100 ms


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as YAML to *path* atomically.

    Creates parents if needed, dumps to a sibling temp file, fsyncs, then
    ``os.replace`` swaps it in. Concurrent readers either see the previous
    content or the new content — never a truncated/empty file.

    Windows-Spezifika: ``os.replace`` schlaegt mit ``PermissionError`` fehl,
    wenn ein anderer Prozess das Ziel-File gerade zum Lesen geoeffnet hat.
    Wir retryn mit kurzem Backoff bis :data:`_REPLACE_ATTEMPTS`.
    """
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=True)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass

        last_exc: Exception | None = None
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_exc = exc
                backoff = min(
                    _REPLACE_BACKOFF_S * (2 ** attempt),
                    _REPLACE_BACKOFF_CAP_S,
                )
                time.sleep(backoff)
        raise last_exc  # type: ignore[misc]
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


__all__ = ["atomic_write_yaml"]
