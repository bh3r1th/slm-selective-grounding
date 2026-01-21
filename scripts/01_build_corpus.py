from __future__ import annotations

import argparse
import logging

from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.utils.pipeline import (
    build_output_path,
    compute_run_id,
    config_to_dict,
    load_config,
    log_run_start,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus")
    parser.add_argument("--config", required=True, help="Path to Hydra config")
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    run_id = compute_run_id(config, inputs=[args.config])
    log_run_start(logger, run_id)

    output_path = build_output_path("data", "corpus", run_id)
    payload = {"run_id": run_id, "status": "placeholder", "config": config_to_dict(config)}
    write_json(output_path, payload)


if __name__ == "__main__":
    main()
