from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.retrieval.bm25 import build_bm25_index
from slm_selective_grounding.retrieval.index import build_indexes
from slm_selective_grounding.utils.io import ensure_dirs
from slm_selective_grounding.utils.pipeline import (
    build_output_path,
    compute_run_id,
    load_config,
    log_run_start,
    write_json,
)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _safe_corpus_name(path: Path) -> str:
    stem = path.stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retriever indexes")
    parser.add_argument("--config", required=True, help="Path to Hydra config")
    parser.add_argument(
        "--corpus",
        choices=["default", "wiki_leads"],
        default="default",
        help="Which corpus to index",
    )
    parser.add_argument("--corpus_path", help="Path to JSONL corpus to index")
    args = parser.parse_args()

    ensure_dirs(["artifacts/indexes"])
    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    run_id = compute_run_id(config, inputs=[args.config])
    log_run_start(logger, run_id)

    output_path = build_output_path("artifacts", "retriever_index", run_id)
    corpus_choice = None if args.corpus == "default" else args.corpus

    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
        corpus_name = _safe_corpus_name(corpus_path)
        index_dir = Path("artifacts/indexes/bm25") / corpus_name
        ensure_dirs([index_dir])

        doc_count = 0
        for row in _iter_jsonl(corpus_path):
            if isinstance(row.get("doc_id"), str) and isinstance(row.get("text"), str):
                doc_count += 1

        bm25_cfg = dict(config.get("bm25", {}))
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
        logger.info(
            "indexed corpus=%s docs=%s index_dir=%s summary=%s",
            corpus_path,
            doc_count,
            index_dir,
            output_path,
        )
        return

    build_indexes(config, output_path, run_id, corpus=corpus_choice)
    logger.info("index_summary=%s", output_path)


if __name__ == "__main__":
    main()
