from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.eval.metrics import compute_metrics
from slm_selective_grounding.utils.pipeline import write_json


def run_eval(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Run evaluation and report metrics (placeholder)."""
    metrics = compute_metrics([])
    payload = {
        "run_id": run_id,
        "status": "placeholder",
        "metrics": metrics,
        "config": dict(config.get("eval", {})),
    }
    write_json(output_path, payload)
    return output_path
