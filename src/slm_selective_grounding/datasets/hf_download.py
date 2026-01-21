from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.datasets.schema import normalize_example
from slm_selective_grounding.utils.io import ensure_dir
from slm_selective_grounding.utils.pipeline import config_to_dict, write_json


def _sanitize_dataset_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _load_dataset(
    dataset_id: str,
    config_name: str | None,
    split: str,
    limit: int | None,
) -> list[Mapping[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config_name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def _parse_dataset_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id = entry.get("id")
    if not dataset_id:
        raise ValueError("Each dataset entry must include an id")
    config_name = entry.get("config_name")
    split = entry.get("split") or "train"
    n_examples = entry.get("n_examples")
    text_fields = entry.get("text_fields")
    return {
        "dataset_id": str(dataset_id),
        "config_name": str(config_name) if config_name is not None else None,
        "split": str(split),
        "n_examples": int(n_examples) if n_examples is not None else None,
        "text_fields": dict(text_fields) if isinstance(text_fields, Mapping) else None,
    }


def download_datasets(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
    output_root: Path | None = None,
) -> Path:
    """Download datasets from Hugging Face and normalize to a unified schema."""
    logger = logging.getLogger(__name__)
    if config.__class__.__name__ == "DictConfig":
        config_payload = config_to_dict(config)  # type: ignore[arg-type]
    else:
        config_payload = dict(config)

    datasets_cfg = config_payload.get("datasets", [])
    if not isinstance(datasets_cfg, list):
        raise ValueError("datasets must be a list")
    dry_run = bool(config_payload.get("dry_run", False))
    output_root = output_root or Path("data") / "raw"

    manifest_entries: list[dict[str, Any]] = []
    for entry in datasets_cfg:
        if not isinstance(entry, Mapping):
            raise ValueError("datasets entries must be mappings")
        parsed = _parse_dataset_entry(entry)
        dataset_id = parsed["dataset_id"]
        config_name = parsed["config_name"]
        split = parsed["split"]
        n_examples = parsed["n_examples"]
        limit = n_examples if dry_run else None
        if dry_run and limit is None:
            limit = 10

        raw_rows = _load_dataset(dataset_id, config_name, split=split, limit=limit)
        normalized_rows: list[Mapping[str, Any]] = []
        for row in raw_rows:
            example = normalize_example(row, dataset_id, config_name, split)
            if parsed["text_fields"] is not None:
                example = example.with_metadata(text_fields=parsed["text_fields"])
            normalized_rows.append(example.to_dict())

        dataset_dir = output_root / _sanitize_dataset_id(dataset_id)
        dataset_path = dataset_dir / f"{split}.jsonl"
        _write_jsonl(dataset_path, normalized_rows)

        logger.info(
            "dataset_id=%s config_name=%s split=%s row_count=%s",
            dataset_id,
            config_name,
            split,
            len(normalized_rows),
        )
        manifest_entries.append(
            {
                "dataset_id": dataset_id,
                "config_name": config_name,
                "split": split,
                "row_count": len(normalized_rows),
                "path": str(dataset_path),
            }
        )

    payload = {
        "run_id": run_id,
        "status": "ok",
        "dry_run": dry_run,
        "datasets": manifest_entries,
    }
    write_json(output_path, payload)
    return output_path
