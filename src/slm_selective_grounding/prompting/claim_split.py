from __future__ import annotations


def split_claims(text: str) -> list[str]:
    """Split a paragraph into claims (placeholder)."""
    return [segment.strip() for segment in text.split(".") if segment.strip()]
