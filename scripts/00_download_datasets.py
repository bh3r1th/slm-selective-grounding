from __future__ import annotations

import argparse
import logging
from pathlib import Path

from slm_selective_grounding.datasets.hf_download import download_datasets
from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.utils.pipeline import (
    build_output_path,
    compute_run_id,
    load_config,
    log_run_start,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--config", required=True, help="Path to Hydra config")
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    run_id = compute_run_id(config, inputs=[args.config])
    log_run_start(logger, run_id)

    output_path = build_output_path("data", "datasets", run_id)
    output_root = Path("data") / "raw"
    download_datasets(config, output_path, run_id, output_root=output_root)


if __name__ == "__main__":
    main()
