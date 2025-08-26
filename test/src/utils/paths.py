"""Pfad-Helfer."""

from __future__ import annotations
from pathlib import Path
from config import settings


def ensure_dir(name: str) -> Path:
    p = settings.TEST_SRC_ROOT / "data" / name
    p.mkdir(parents=True, exist_ok=True)
    return p
