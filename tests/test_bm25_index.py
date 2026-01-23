from __future__ import annotations

import json
from pathlib import Path

from slm_selective_grounding.retrieval.bm25 import build_bm25_index, load_bm25_index, search


def _write_corpus(path: Path) -> None:
    rows = [
        {"doc_id": "d1", "title": "Cats", "text": "Cats are small domesticated animals."},
        {"doc_id": "d2", "title": "Dogs", "text": "Dogs are loyal animals and often friendly."},
        {"doc_id": "d3", "title": "Birds", "text": "Birds can fly and have feathers."},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def test_bm25_search_top_doc(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    _write_corpus(corpus_path)

    build_bm25_index(corpus_path, index_dir)
    bm25 = load_bm25_index(index_dir)
    results = search(bm25, "cats domesticated", topn=2)

    assert results
    assert results[0]["doc_id"] == "d1"
