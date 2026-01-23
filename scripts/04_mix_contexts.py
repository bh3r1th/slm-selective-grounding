from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from slm_selective_grounding.contexts.mixer import mix_contexts
from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.utils.io import ensure_dirs


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
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return cleaned or "corpus"


def _load_corpus_lookup(path: Path) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for row in _iter_jsonl(path):
        doc_id = row.get("doc_id") or row.get("passage_id") or row.get("id")
        if doc_id is None:
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        title = row.get("title")
        title_str = str(title).strip() if isinstance(title, str) else ""
        lookup[str(doc_id)] = (title_str, text.strip())
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix retrieval contexts with labels")
    parser.add_argument("--dataset", required=False, help="Dataset name (for paths)")
    parser.add_argument(
        "--dataset-path",
        "--in",
        dest="dataset_path",
        default=None,
        help="Path to dataset JSONL (defaults to artifacts/corpus/<dataset>.jsonl)",
    )
    parser.add_argument(
        "--corpus",
        default="artifacts/corpus/wiki_leads.jsonl",
        help="Corpus name (artifacts/corpus/<name>.jsonl) or JSONL path",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="BM25 index directory",
    )
    parser.add_argument(
        "--corpus-db",
        dest="corpus_db",
        default=None,
        help="SQLite corpus path for passage lookup",
    )
    parser.add_argument("--topn", type=int, default=50, help="BM25 candidate size")
    parser.add_argument("--k", type=int, default=12, help="Contexts per claim")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default=None, help="Output JSONL path override")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after N input rows",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Progress logging frequency",
    )
    parser.add_argument(
        "--profile",
        choices=["clean", "mixed", "toxic"],
        default=None,
        help="Force a single mix profile",
    )
    parser.add_argument(
        "--profile-schedule",
        default="clean:0.34,mixed:0.33,toxic:0.33",
        help="Sampling schedule for profiles",
    )
    args = parser.parse_args()

    configure_logging()
    dataset_name = args.dataset or "custom"
    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path
        else Path("artifacts") / "corpus" / f"{dataset_name}.jsonl"
    )
    output_path = (
        Path(args.out)
        if args.out
        else Path("artifacts") / "contexts" / f"{dataset_name}_k{args.k}.jsonl"
    )
    ensure_dirs([output_path.parent])

    corpus_arg = Path(args.corpus)
    corpus_db_path = Path(args.corpus_db) if args.corpus_db else None
    if corpus_arg.exists():
        corpus_path = corpus_arg
        corpus_name = _safe_corpus_name(corpus_arg)
        corpus_lookup = None if corpus_db_path else _load_corpus_lookup(corpus_path)
    else:
        corpus_name = args.corpus
        corpus_path = Path("artifacts") / "corpus" / f"{corpus_name}.jsonl"
        corpus_lookup = None

    index_dir = Path(args.index_dir) if args.index_dir else Path("artifacts") / "indexes" / "bm25" / corpus_name

    written = mix_contexts(
        dataset_path=dataset_path,
        output_path=output_path,
        corpus_path=corpus_path,
        index_dir=index_dir,
        topn=args.topn,
        k=args.k,
        seed=args.seed,
        profile=args.profile,
        profile_schedule=args.profile_schedule,
        corpus_lookup=corpus_lookup,
        corpus_db_path=corpus_db_path,
        max_rows=args.max_rows,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
