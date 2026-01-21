from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.mixtures.controls import apply_controls
from slm_selective_grounding.utils.pipeline import write_json


def build_mixtures(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Build data mixtures (placeholder)."""
    mixtures_cfg = dict(config.get("mixtures", {}))
    size = int(mixtures_cfg.get("size", 0))
    ratios = mixtures_cfg.get(
        "ratios",
        {"support": 1.0, "decoy": 0.0, "conflict": 0.0},
    )
    counts = apply_controls(size, ratios)
    payload = {
        "run_id": run_id,
        "status": "placeholder",
        "size": size,
        "counts": {
            "support": counts.support,
            "decoy": counts.decoy,
            "conflict": counts.conflict,
        },
    }
    write_json(output_path, payload)
    return output_path
