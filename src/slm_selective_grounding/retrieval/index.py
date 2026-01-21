from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.utils.pipeline import write_json


def build_indexes(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Build all retrieval indexes (placeholder)."""
    retriever_cfg = dict(config.get("retriever", {}))
    if "type" not in retriever_cfg:
        raise ValueError("retriever.type is required")
    payload = {
        "run_id": run_id,
        "status": "placeholder",
        "retriever": retriever_cfg,
    }
    write_json(output_path, payload)
    return output_path
