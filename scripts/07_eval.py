from __future__ import annotations

import argparse
import logging

from slm_selective_grounding.eval.report import run_eval
from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.utils.pipeline import (
    build_output_path,
    compute_run_id,
    load_config,
    log_run_start,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--config", required=True, help="Path to Hydra config")
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    run_id = compute_run_id(config, inputs=[args.config])
    log_run_start(logger, run_id)

    output_path = build_output_path("artifacts", "eval_metrics", run_id)
    run_eval(config, output_path, run_id)


if __name__ == "__main__":
    main()
