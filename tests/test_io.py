"""Tests für utils.io.atomic_write_yaml — K3.

Stellt sicher, dass parallele Reads während eines Writes nie ein halb
geschriebenes oder leeres File sehen.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import yaml

from utils.io import atomic_write_yaml


def test_atomic_write_creates_file(tmp_path: Path):
    target = tmp_path / "x" / "config.yaml"
    atomic_write_yaml(target, {"key": "value", "n": 42})
    assert target.exists()
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data == {"key": "value", "n": 42}


def test_atomic_write_no_tmp_file_leftover(tmp_path: Path):
    target = tmp_path / "config.yaml"
    atomic_write_yaml(target, {"key": "value"})
    # tempfile must be cleaned up — only the target should exist.
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*"))
    assert leftovers == [], f"tmp leftovers: {leftovers}"


def test_atomic_write_overwrites_existing(tmp_path: Path):
    target = tmp_path / "config.yaml"
    target.write_text("stale: true\n", encoding="utf-8")
    atomic_write_yaml(target, {"fresh": True})
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data == {"fresh": True}


def test_atomic_write_handles_concurrent_readers(tmp_path: Path):
    """K3-Regression: bei realistischen Reload-Frequenzen darf weder ein Reader
    einen leeren/teilweise geschriebenen Read sehen, noch ein Writer mit
    PermissionError abbrechen."""
    target = tmp_path / "config.yaml"
    atomic_write_yaml(target, {"iteration": 0, "payload": "init"})

    stop = threading.Event()
    errors: list[str] = []
    reader_observations: list[dict] = []
    writer_exc: list[BaseException] = []

    def writer():
        try:
            for i in range(1, 51):  # 50 Iterations
                atomic_write_yaml(target, {"iteration": i, "payload": "x" * 1024})
                time.sleep(0.002)  # Writer 2 ms zwischen den Calls — realistischer
        except BaseException as exc:
            writer_exc.append(exc)
        finally:
            stop.set()

    def reader():
        while not stop.is_set():
            try:
                content = target.read_text(encoding="utf-8")
                if not content.strip():
                    errors.append("reader saw empty file")
                    continue
                data = yaml.safe_load(content)
                if not isinstance(data, dict) or "iteration" not in data:
                    errors.append(f"reader saw malformed yaml: {content[:80]!r}")
                    continue
                reader_observations.append(data)
            except yaml.YAMLError as exc:
                errors.append(f"yaml error: {exc}")
            except FileNotFoundError:
                errors.append("file vanished between rename and read")
            except PermissionError:
                pass  # Windows: kurzes Race; OK, weil Helper retryd.
            time.sleep(0.005)  # 5 ms Reader-Tick — realistisch fuer settings reload

    threads = [threading.Thread(target=reader) for _ in range(2)]
    writer_thread = threading.Thread(target=writer)
    for t in threads:
        t.start()
    writer_thread.start()
    writer_thread.join(timeout=15.0)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    assert writer_exc == [], f"writer raised: {writer_exc!r}"
    assert errors == [], f"concurrent reader errors: {errors[:5]}"
    assert len(reader_observations) >= 2
