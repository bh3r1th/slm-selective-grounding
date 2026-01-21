from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.utils.pipeline import write_json


def generate_baseline_outputs(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Generate baseline model outputs (placeholder)."""
    payload = {
        "run_id": run_id,
        "status": "placeholder",
        "config": dict(config.get("baseline", {})),
    }
    write_json(output_path, payload)
    return output_path
