from __future__ import annotations

import uuid
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return the Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_dirs(paths: list[str | Path]) -> list[Path]:
    """Ensure directories exist and are writable."""
    ensured: list[Path] = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.exists() and not path_obj.is_dir():
            _raise_path_error(path_obj, "Path exists and is not a directory.")
        path_obj.mkdir(parents=True, exist_ok=True)
        _ensure_writable(path_obj)
        ensured.append(path_obj)
    return ensured


def _ensure_writable(path_obj: Path) -> None:
    test_file = path_obj / f".write_test_{uuid.uuid4().hex}"
    try:
        with test_file.open("w", encoding="utf-8") as handle:
            handle.write("ok")
        test_file.unlink()
    except OSError as exc:
        if test_file.exists():
            try:
                test_file.unlink()
            except OSError:
                pass
        _raise_path_error(path_obj, "Path is not writable.", exc)


def _raise_path_error(path_obj: Path, reason: str, exc: Exception | None = None) -> None:
    message = (
        f"{reason}\n"
        f"Path: {path_obj}\n"
        f"CWD: {Path.cwd()}\n"
        "Hint: On Windows/OneDrive, ensure the folder isn't read-only or blocked by OneDrive permissions."
    )
    raise RuntimeError(message) from exc
