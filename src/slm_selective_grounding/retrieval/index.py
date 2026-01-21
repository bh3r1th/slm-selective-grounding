from __future__ import annotations

from slm_selective_grounding.retrieval.bm25 import build_bm25_index
from slm_selective_grounding.retrieval.dense import build_dense_index


def build_indexes() -> None:
    """Build all retrieval indexes (placeholder)."""
    build_bm25_index()
    build_dense_index()
