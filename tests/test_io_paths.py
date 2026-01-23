from __future__ import annotations

from pathlib import Path

import pytest

from slm_selective_grounding.utils.io import ensure_dirs


def test_ensure_dirs_creates(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    ensure_dirs([target])
    assert target.is_dir()


def test_ensure_dirs_writable(tmp_path: Path) -> None:
    target = tmp_path / "writable"
    ensure_dirs([target])
    test_file = target / "probe.txt"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink()


def test_ensure_dirs_raises_on_file_path(tmp_path: Path) -> None:
    target = tmp_path / "not_a_dir"
    target.write_text("nope", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ensure_dirs([target])
