from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from slm_selective_grounding.utils.pipeline import config_to_dict, write_json


def download_datasets(
    config: Mapping[str, Any],
    output_path: Path,
    run_id: str,
) -> Path:
    """Download datasets from Hugging Face (placeholder)."""
    if config.__class__.__name__ == "DictConfig":
        config_payload = config_to_dict(config)  # type: ignore[arg-type]
    else:
        config_payload = dict(config)
    payload = {
        "run_id": run_id,
        "status": "placeholder",
        "config": config_payload,
    }
    write_json(output_path, payload)
    return output_path


if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import DictConfig
