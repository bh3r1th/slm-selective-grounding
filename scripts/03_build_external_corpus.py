from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_selective_grounding.datasets.external_corpus_schema import (
    validate_external_passage,
)
from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.utils.io import ensure_dirs


_TEXT_KEYS = [
    "text",
    "content",
    "article",
    "body",
    "wikipedia_text",
    "wiki_text",
    "raw_text",
]
_TITLE_KEYS = ["title", "page_title", "article_title", "name"]
_ID_KEYS = ["id", "page_id", "pageid", "article_id", "doc_id", "document_id"]
_NESTED_KEYS = ["document", "doc", "page", "article"]


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(parts).strip()
    return str(value).strip()


def _extract_first_str(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key not in row:
            continue
        value = _coerce_text(row.get(key))
        if value:
            return value
    return ""


def _extract_text(row: Mapping[str, Any]) -> str:
    text = _extract_first_str(row, _TEXT_KEYS)
    if text:
        return text
    for key in _NESTED_KEYS:
        nested = row.get(key)
        if isinstance(nested, Mapping):
            nested_text = _extract_text(nested)
            if nested_text:
                return nested_text
    return ""


def _extract_title(row: Mapping[str, Any]) -> str:
    return _extract_first_str(row, _TITLE_KEYS)


def _extract_article_id(row: Mapping[str, Any]) -> str:
    return _extract_first_str(row, _ID_KEYS)


def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _chunk_tokens(tokens: list[str], max_tokens: int, overlap: int) -> Iterable[list[str]]:
    if max_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    step = max_tokens if overlap == 0 else max_tokens - overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_tokens")
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        yield tokens[start:end]
        if end == len(tokens):
            break
        start += step


def _chunk_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    return [" ".join(chunk) for chunk in _chunk_tokens(tokens, max_tokens, overlap)]


def _load_dataset(
    dataset: str,
    subset: str,
    split: str,
    logger: logging.Logger,
) -> tuple[Iterable[Mapping[str, Any]], bool]:
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset, subset, split=split, streaming=True)
        logger.info("loaded dataset=%s subset=%s split=%s streaming=true", dataset, subset, split)
        return ds, True
    except Exception as exc:
        logger.warning(
            "streaming load failed dataset=%s subset=%s split=%s err=%r; falling back to non-streaming",
            dataset,
            subset,
            split,
            exc,
        )
        ds = load_dataset(dataset, subset, split=split)
        logger.info("loaded dataset=%s subset=%s split=%s streaming=false", dataset, subset, split)
        return ds, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external wiki retrieval corpus")
    parser.add_argument("--dataset", default="wikimedia/wikipedia", help="HF dataset id")
    parser.add_argument("--subset", default="20231101.en", help="HF dataset subset/config")
    parser.add_argument("--split", default="train", help="HF split name")
    parser.add_argument("--max_articles", type=int, default=200000, help="Max articles to process")
    parser.add_argument("--chunk_tokens", type=int, default=180, help="Chunk size in tokens")
    parser.add_argument("--chunk_overlap", type=int, default=30, help="Chunk overlap in tokens")
    parser.add_argument("--min_chars", type=int, default=300, help="Minimum article length")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (unused)")
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    output_path = Path(args.out)
    ensure_dirs([output_path.parent])

    dataset_iter, streaming = _load_dataset(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        logger=logger,
    )

    articles_seen = 0
    passages_written = 0
    skipped = 0
    flush_every = 1000

    with output_path.open("w", encoding="utf-8") as handle:
        for row_idx, row in enumerate(dataset_iter):
            if args.max_articles is not None and articles_seen >= args.max_articles:
                break
            articles_seen += 1

            if not isinstance(row, Mapping):
                skipped += 1
                continue

            text = _extract_text(row)
            if not text or len(text) < args.min_chars:
                skipped += 1
                continue

            title = _extract_title(row)
            article_id = _extract_article_id(row)
            if article_id:
                article_key = article_id
            elif title:
                article_key = f"title:{_stable_hash(title)}"
            else:
                article_key = f"row:{_stable_hash(text[:512])}"

            chunks = _chunk_text(text, args.chunk_tokens, args.chunk_overlap)
            if not chunks:
                skipped += 1
                continue

            for chunk_idx, chunk in enumerate(chunks):
                payload = {
                    "doc_id": f"{article_key}:{chunk_idx}",
                    "source": args.dataset,
                    "subset": args.subset,
                    "title": title,
                    "text": chunk,
                    "metadata": {"article_id": article_id or article_key, "chunk_index": chunk_idx},
                }
                try:
                    validate_external_passage(payload)
                except ValueError:
                    skipped += 1
                    continue
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
                passages_written += 1

            if articles_seen % flush_every == 0:
                handle.flush()

            if articles_seen % 5000 == 0:
                logger.info(
                    "progress articles=%s passages=%s skipped=%s streaming=%s",
                    articles_seen,
                    passages_written,
                    skipped,
                    streaming,
                )

    logger.info(
        "completed articles=%s passages=%s skipped=%s output=%s",
        articles_seen,
        passages_written,
        skipped,
        output_path,
    )


if __name__ == "__main__":
    main()
