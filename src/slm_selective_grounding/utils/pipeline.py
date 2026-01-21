from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from slm_selective_grounding.utils.hashing import stable_hash
from slm_selective_grounding.utils.io import ensure_dir


def load_config(config_path: str) -> "DictConfig":
    """Load a Hydra config from an explicit path."""
    from hydra import compose, initialize_config_dir

    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = Path(config_path).stem
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name)


def config_to_dict(config: "DictConfig") -> dict[str, Any]:
    """Convert a DictConfig to a plain dict with resolved values."""
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected mapping config")
    return payload


def compute_run_id(config: Mapping[str, Any], inputs: Sequence[str]) -> str:
    """Compute a deterministic run id from config and inputs."""
    if config.__class__.__name__ == "DictConfig":
        config_dict = config_to_dict(config)  # type: ignore[arg-type]
    else:
        config_dict = dict(config)
    payload = {"config": config_dict, "inputs": list(inputs)}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return stable_hash(serialized)


if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import DictConfig


def build_output_path(base_dir: str | Path, stem: str, run_id: str, ext: str = ".json") -> Path:
    """Create a deterministic output path under a base directory."""
    base_path = ensure_dir(base_dir)
    return Path(base_path) / f"{stem}_{run_id}{ext}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload to disk."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def log_run_start(logger: logging.Logger, run_id: str) -> None:
    """Log the run id for a pipeline step."""
    logger.info("run_id=%s", run_id)
