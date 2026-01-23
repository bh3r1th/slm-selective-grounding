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
    """
    Build a JSONL corpus of passage chunks from normalized datasets.

    Expected upstream artifact:
      scripts/00_download_datasets.py writes a JSON file at:
        data/datasets_{run_id}.json
      with payload:
        {"datasets": { "<dataset_id>": [ { "query": ..., "docs": [...] }, ... ] } }
    """
    # Normalize Hydra DictConfig -> dict
    if config.__class__.__name__ == "DictConfig":
        config_payload = config_to_dict(config)  # type: ignore[arg-type]
    else:
        config_payload = dict(config)

    # Back-compat: accept old keys, but prefer new config structure
    datasets_path = Path(config_payload.get("datasets_path", f"data/datasets_{run_id}.json"))

    # Prefer chunking config block if present
    chunking_cfg = config_payload.get("chunking", {}) or {}
    passage_tokens = int(chunking_cfg.get("chunk_size", config_payload.get("passage_tokens", 200)))
    passage_stride = int(chunking_cfg.get("overlap", config_payload.get("passage_stride", 0)))

    dry_run = bool(config_payload.get("dry_run", False))

    if not datasets_path.exists():
        raise FileNotFoundError(
            f"Missing normalized datasets file: {datasets_path}. "
            "Run scripts/00_download_datasets.py first."
        )

    raw_payload = json.loads(datasets_path.read_text(encoding="utf-8"))
    datasets_obj = raw_payload.get("datasets")

    # Accept both formats:
    # A) {"datasets": {"dataset_id": [examples...]}}
    # B) {"datasets": [{"id": "...", "examples": [...]}, ...]}
    if isinstance(datasets_obj, dict):
        datasets_map = datasets_obj
    elif isinstance(datasets_obj, list):
        datasets_map = {}
        for entry in datasets_obj:
            if not isinstance(entry, dict):
                continue
            name = entry.get("id") or entry.get("dataset_id") or entry.get("name")
            examples = entry.get("examples") or entry.get("rows") or entry.get("data") or []
            if isinstance(name, str) and isinstance(examples, list):
                datasets_map[name] = examples
    else:
        raise ValueError(
            f"Invalid datasets payload in {datasets_path}: expected dict or list under key 'datasets'. "
            f"Got {type(datasets_obj)}"
        )

    ensure_dir(output_path.parent)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for dataset_name, examples in datasets_map.items():
            if not isinstance(examples, list):
                continue

            trimmed_examples = examples[:10] if dry_run else examples

            for ex_idx, example in enumerate(trimmed_examples):
                if not isinstance(example, dict):
                    continue

                docs = example.get("docs") or []
                query = example.get("query")

                if not docs:
                    # Fallback: build a pseudo-document from any available textual fields
                    candidates = [
                        "gold_answer",
                        "long_answer",
                        "answer",
                        "output",
                        "response",
                        "final_answer",
                        "context",
                        "prompt",
                        "question",
                        "query",
                        "claim",
                        "instruction",
                        "input",
                        "text",
                    ]
                    text = ""
                    for key in candidates:
                        val = example.get(key)
                        if isinstance(val, str) and val.strip():
                            text = val.strip()
                            break

                    # Last-resort: serialize the example so we still index something in dry-run
                    if not text:
                        safe = {k: v for k, v in example.items() if k not in {"docs", "metadata"}}
                        text = json.dumps(safe, ensure_ascii=False)

                    docs = [{"doc_id": f"{dataset_name}::ex{ex_idx}", "title": "", "text": text}]



                if not isinstance(docs, list):
                    continue

                for doc in docs:
                    if not isinstance(doc, dict):
                        continue

                    doc_id = str(doc.get("doc_id", "")).strip()
                    title = str(doc.get("title", "")).strip()
                    text = str(doc.get("text", "")).strip()

                    if not doc_id or not text:
                        continue

                    chunks = _chunk_text(text, passage_tokens, passage_stride)
                    for chunk_idx, chunk in enumerate(chunks):
                        passage = {
                            "passage_id": f"{dataset_name}::{doc_id}::chunk{chunk_idx}",
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
                        written += 1

    if written == 0:
        raise RuntimeError(
            "Corpus build produced 0 passages. Most common causes:\n"
            "- Normalized dataset examples have docs=[] (adapter not extracting docs)\n"
            "- doc_id/text missing in docs\n"
            "- passage_tokens too small / text empty\n"
        )

    return output_path
