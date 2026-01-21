from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_selective_grounding.utils.io import ensure_dir
from slm_selective_grounding.utils.pipeline import config_to_dict


def _chunk_tokens(tokens: list[str], max_tokens: int, stride: int) -> Iterable[list[str]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if stride < 0:
        raise ValueError("stride must be non-negative")
    step = max_tokens if stride == 0 else max_tokens - stride
    if step <= 0:
        raise ValueError("stride must be smaller than max_tokens")
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        yield tokens[start:end]
        if end == len(tokens):
            break
        start += step


def _chunk_text(text: str, max_tokens: int, stride: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    return [" ".join(chunk) for chunk in _chunk_tokens(tokens, max_tokens, stride)]


def build_corpus(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Build a JSONL corpus of passage chunks from normalized datasets."""
    if config.__class__.__name__ == "DictConfig":
        config_payload = config_to_dict(config)  # type: ignore[arg-type]
    else:
        config_payload = dict(config)

    datasets_path = Path(
        config_payload.get("datasets_path", f"data/datasets_{run_id}.json")
    )
    passage_tokens = int(config_payload.get("passage_tokens", 200))
    passage_stride = int(config_payload.get("passage_stride", 0))
    dry_run = bool(config_payload.get("dry_run", False))

    raw_payload = json.loads(datasets_path.read_text())
    datasets = raw_payload.get("datasets", {})
    ensure_dir(output_path.parent)

    with output_path.open("w", encoding="utf-8") as handle:
        for dataset_name, examples in datasets.items():
            if not isinstance(examples, list):
                continue
            trimmed_examples = examples[:10] if dry_run else examples
            for ex_idx, example in enumerate(trimmed_examples):
                docs = example.get("docs", [])
                query = example.get("query")
                for doc in docs:
                    doc_id = str(doc.get("doc_id", ""))
                    title = str(doc.get("title", ""))
                    text = str(doc.get("text", ""))
                    chunks = _chunk_text(text, passage_tokens, passage_stride)
                    for chunk_idx, chunk in enumerate(chunks):
                        passage = {
                            "passage_id": f"{doc_id}::chunk{chunk_idx}",
                            "doc_id": doc_id,
                            "title": title,
                            "text": chunk,
                            "metadata": {
                                "dataset": dataset_name,
                                "example_index": ex_idx,
                                "query": query,
                                "run_id": run_id,
                            },
                        }
                        handle.write(json.dumps(passage, ensure_ascii=False))
                        handle.write("\n")
    return output_path
