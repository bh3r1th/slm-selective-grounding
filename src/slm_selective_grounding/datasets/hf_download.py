from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_selective_grounding.datasets.schema import Document, Example
from slm_selective_grounding.utils.pipeline import config_to_dict, write_json


ALCE_DATASET = "tatsu-lab/alce"
FEVER_DATASET = "fever"


def _get_first(example: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in example:
            return example[key]
    return None


def _normalize_docs(raw_docs: Any) -> list[Document]:
    docs: list[Document] = []
    if not raw_docs:
        return docs
    if isinstance(raw_docs, Mapping):
        raw_docs = [raw_docs]
    for idx, raw_doc in enumerate(raw_docs):
        if not isinstance(raw_doc, Mapping):
            continue
        doc_id = str(raw_doc.get("doc_id") or raw_doc.get("id") or idx)
        title = str(raw_doc.get("title") or raw_doc.get("source") or "")
        text = str(raw_doc.get("text") or raw_doc.get("content") or "")
        docs.append(Document(doc_id=doc_id, title=title, text=text))
    return docs


def _normalize_alce(example: Mapping[str, Any], dataset_name: str) -> Example:
    query = _get_first(example, ["question", "query", "prompt", "instruction"]) or ""
    gold_answer = _get_first(example, ["answer", "output", "reference_answer"])
    docs = _normalize_docs(_get_first(example, ["docs", "documents", "contexts", "ctxs"]))
    metadata = {
        "dataset": dataset_name,
        "id": _get_first(example, ["id", "example_id"]),
    }
    return Example(
        query=str(query),
        gold_answer=str(gold_answer) if gold_answer is not None else None,
        gold_claims=None,
        docs=docs,
        metadata=metadata,
    )


def _normalize_fever(example: Mapping[str, Any]) -> Example:
    claim = _get_first(example, ["claim", "query", "text"]) or ""
    evidence = _get_first(example, ["evidence", "evidence_sentences", "evidence_sets"])
    metadata = {
        "dataset": "fever",
        "id": _get_first(example, ["id", "example_id"]),
        "label": _get_first(example, ["label", "verdict"]),
        "evidence": evidence,
    }
    return Example(
        query=str(claim),
        gold_answer=None,
        gold_claims=None,
        docs=[],
        metadata=metadata,
    )


def _load_dataset(name: str, split: str, subset: str | None, limit: int | None) -> list[Mapping[str, Any]]:
    from datasets import load_dataset

    dataset_kwargs = {"split": split}
    if subset is not None:
        dataset = load_dataset(name, subset, **dataset_kwargs)
    else:
        dataset = load_dataset(name, **dataset_kwargs)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def download_datasets(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Download datasets from Hugging Face and normalize to a unified schema."""
    if config.__class__.__name__ == "DictConfig":
        config_payload = config_to_dict(config)  # type: ignore[arg-type]
    else:
        config_payload = dict(config)

    datasets_cfg = config_payload.get(
        "datasets",
        [
            {"name": "alce_asqa"},
            {"name": "alce_qampari"},
            {"name": "fever"},
        ],
    )
    if not isinstance(datasets_cfg, list):
        raise ValueError("datasets must be a list")
    dry_run = bool(config_payload.get("dry_run", False))
    limit = 10 if dry_run else None

    normalized: dict[str, list[dict[str, Any]]] = {}
    for entry in datasets_cfg:
        if not isinstance(entry, Mapping):
            raise ValueError("datasets entries must be mappings")
        name = str(entry.get("name"))
        if name == "alce_asqa":
            raw_rows = _load_dataset(ALCE_DATASET, split="train", subset="asqa", limit=limit)
            normalized[name] = [
                _normalize_alce(row, dataset_name=name).to_dict() for row in raw_rows
            ]
        elif name == "alce_qampari":
            raw_rows = _load_dataset(
                ALCE_DATASET, split="train", subset="qampari", limit=limit
            )
            normalized[name] = [
                _normalize_alce(row, dataset_name=name).to_dict() for row in raw_rows
            ]
        elif name == "fever":
            raw_rows = _load_dataset(FEVER_DATASET, split="train", subset=None, limit=limit)
            normalized[name] = [_normalize_fever(row).to_dict() for row in raw_rows]
        else:
            raise ValueError(f"Unsupported dataset: {name}")

    payload = {
        "run_id": run_id,
        "status": "ok",
        "dry_run": dry_run,
        "datasets": normalized,
    }
    write_json(output_path, payload)
    return output_path

