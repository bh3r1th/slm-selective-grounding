from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.utils.pipeline import write_json


def make_preference_pairs(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Create preference pairs (placeholder)."""
    payload = {
        "run_id": run_id,
        "status": "placeholder",
        "config": dict(config.get("preference_pairs", {})),
    }
    write_json(output_path, payload)
    return output_path
