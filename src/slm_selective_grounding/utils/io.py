from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return the Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
