from __future__ import annotations

import argparse
import logging

from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.retrieval.index import build_indexes
from slm_selective_grounding.utils.pipeline import (
    build_output_path,
    compute_run_id,
    load_config,
    log_run_start,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retriever indexes")
    parser.add_argument("--config", required=True, help="Path to Hydra config")
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    run_id = compute_run_id(config, inputs=[args.config])
    log_run_start(logger, run_id)

    output_path = build_output_path("artifacts", "retriever_index", run_id)
    build_indexes(config, output_path, run_id)


if __name__ == "__main__":
    main()
