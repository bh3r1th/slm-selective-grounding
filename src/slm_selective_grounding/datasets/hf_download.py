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
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_dataset(
    dataset_id: str,
    config_name: str | None,
    split: str,
    limit: int | None,
) -> list[Mapping[str, Any]]:
    """
    Loads a HF dataset split and returns it as a list[dict].
    NOTE: Some HF datasets use "script" loaders that may fail depending on datasets version.
    """
    from datasets import load_dataset  # local import to keep module import light

    ds = load_dataset(dataset_id, config_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def _parse_dataset_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id = entry.get("id")
    if not dataset_id:
        raise ValueError("Each dataset entry must include an 'id' field.")
    return {
        "dataset_id": str(dataset_id),
        "config_name": entry.get("config_name"),
        "split": str(entry.get("split") or "train"),
        "n_examples": int(entry["n_examples"]) if entry.get("n_examples") is not None else None,
        "text_fields": dict(entry["text_fields"]) if isinstance(entry.get("text_fields"), Mapping) else None,
    }


def download_datasets(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
    output_root: Path | None = None,
) -> Path:
    """
    Download datasets from Hugging Face, normalize to unified schema, and write:
      1) data/raw/<dataset_id_sanitized>/<split>.jsonl  (normalized rows)
      2) data/datasets_<run_id>.json                   (aggregate payload used by build_corpus)

    The aggregate JSON MUST include actual normalized examples under payload["datasets"].
    """
    logger = logging.getLogger(__name__)

    if config.__class__.__name__ == "DictConfig":
        config_payload = config_to_dict(config)  # type: ignore[arg-type]
    else:
        config_payload = dict(config)

    datasets_cfg = config_payload.get("datasets", [])
    if not isinstance(datasets_cfg, list):
        raise ValueError("Config field 'datasets' must be a list of entries.")

    dry_run = bool(config_payload.get("dry_run", False))
    output_root = output_root or (Path("data") / "raw")

    # This is the critical structure build_corpus needs:
    # datasets_map[dataset_id] = [normalized_example_dict, ...]
    datasets_map: dict[str, list[Mapping[str, Any]]] = {}

    manifest_entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for entry in datasets_cfg:
        if not isinstance(entry, Mapping):
            raise ValueError("Each item in config 'datasets' must be a mapping/dict.")

        parsed = _parse_dataset_entry(entry)
        dataset_id: str = parsed["dataset_id"]
        config_name = parsed["config_name"]
        split: str = parsed["split"]
        n_examples = parsed["n_examples"]

        # Dry run limit
        limit = None
        if dry_run:
            limit = n_examples if n_examples is not None else 10

        try:
            raw_rows = _load_dataset(dataset_id, config_name, split=split, limit=limit)
        except Exception as e:  # keep running even if one dataset fails
            logger.error("FAILED dataset_id=%s config_name=%s split=%s err=%r", dataset_id, config_name, split, e)
            failures.append(
                {"dataset_id": dataset_id, "config_name": config_name, "split": split, "error": repr(e)}
            )
            continue

        normalized_rows: list[Mapping[str, Any]] = []
        raw_rows_with_meta: list[Mapping[str, Any]] = []
        for row in raw_rows:
            raw_meta: dict[str, Any] = {}
            existing_meta = row.get("metadata")
            if isinstance(existing_meta, Mapping):
                raw_meta.update(existing_meta)
            for key, value in {
                "dataset_id": dataset_id,
                "config_name": config_name,
                "split": split,
            }.items():
                if key not in raw_meta:
                    raw_meta[key] = value
            raw_row = dict(row)
            raw_row["metadata"] = raw_meta
            raw_rows_with_meta.append(raw_row)

            ex = normalize_example(row, dataset_id, config_name, split)
            if parsed["text_fields"] is not None:
                ex = ex.with_metadata(text_fields=parsed["text_fields"])
            normalized_rows.append(ex.to_dict())

        # Store normalized examples in the aggregate JSON
        datasets_map[dataset_id] = normalized_rows

        # Also write per-dataset JSONL under data/raw/...
        dataset_dir = output_root / _sanitize_dataset_id(dataset_id)
        dataset_path = dataset_dir / f"{split}.jsonl"
        _write_jsonl(dataset_path, raw_rows_with_meta)

        logger.info(
            "dataset_id=%s config_name=%s split=%s row_count=%s",
            dataset_id,
            config_name,
            split,
            len(normalized_rows),
        )

        manifest_entries.append(
            {
                "id": dataset_id,
                "config_name": config_name,
                "split": split,
                "n_examples": n_examples,
                "row_count": len(normalized_rows),
                "path": str(dataset_path),
            }
        )


    payload = {
        "run_id": run_id,
        "status": "ok" if not failures else "partial",
        "dry_run": dry_run,
        # tests expect payload["datasets"] to be a LIST with ["path"]
        "datasets": manifest_entries,
        # keep the big in-memory normalized examples under a different key (optional)
        "datasets_map": datasets_map,
        "failures": failures,
    }

    write_json(output_path, payload)
    return output_path
