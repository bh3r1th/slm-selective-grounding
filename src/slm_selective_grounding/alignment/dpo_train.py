from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from slm_selective_grounding.utils.pipeline import write_json


def train_dpo(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Train a DPO model (placeholder)."""
    dpo_cfg = dict(config.get("dpo", {}))
    if "beta" not in dpo_cfg:
        raise ValueError("dpo.beta is required")
    payload = {
        "run_id": run_id,
        "status": "placeholder",
        "dpo": dpo_cfg,
    }
    write_json(output_path, payload)
    return output_path
