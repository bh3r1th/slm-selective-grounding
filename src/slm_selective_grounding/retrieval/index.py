from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.retrieval.bm25 import build_bm25_index
from slm_selective_grounding.utils.io import ensure_dir
from slm_selective_grounding.utils.pipeline import write_json


def _latest_file(pattern: str) -> Path:
    matches = sorted(Path(".").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[-1]


def build_indexes(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
    corpus: str | None = None,
    corpus_path_override: Path | None = None,
    index_dir_override: Path | None = None,
) -> Path:
    """Build retrieval indexes (BM25 only for now)."""
    retriever_cfg = dict(config.get("retriever", {}))
    rtype = retriever_cfg.get("type")
    if not rtype:
        raise ValueError("retriever.type is required")

    if rtype != "bm25":
        raise ValueError(f"Only retriever.type=bm25 is supported right now (got {rtype})")

    bm25_cfg = dict(config.get("bm25", {}))
    if corpus == "wiki_leads":
        index_dir = Path("artifacts/indexes/bm25/wiki_leads")
        corpus_path = Path("artifacts/corpus/wiki_leads.jsonl")
    else:
        index_dir = Path(bm25_cfg.get("index_dir", "artifacts/indexes/bm25"))
        # Find corpus. Prefer explicit config, else pick latest data/corpus_*.jsonl
        corpus_override = bm25_cfg.get("corpus_path")
        corpus_path = Path(corpus_override) if corpus_override else _latest_file("data/corpus_*.jsonl")

    if corpus_path_override is not None:
        corpus_path = corpus_path_override
    if index_dir_override is not None:
        index_dir = index_dir_override

    ensure_dir(index_dir)

    # Build BM25 (implement in retrieval/bm25.py)
    build_bm25_index(
        corpus_path=corpus_path,
        index_dir=index_dir,
        k1=float(bm25_cfg.get("k1", 0.9)),
        b=float(bm25_cfg.get("b", 0.4)),
    )

    payload = {
        "run_id": run_id,
        "status": "ok",
        "retriever": {"type": "bm25"},
        "corpus_path": str(corpus_path),
        "bm25": {"index_dir": str(index_dir), "k1": bm25_cfg.get("k1", 0.9), "b": bm25_cfg.get("b", 0.4)},
    }
    write_json(output_path, payload)
    return output_path
