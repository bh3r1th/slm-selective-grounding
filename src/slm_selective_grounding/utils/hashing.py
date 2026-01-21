from __future__ import annotations

import hashlib


def stable_hash(text: str) -> str:
    """Return a stable hex hash for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
