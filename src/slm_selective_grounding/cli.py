from __future__ import annotations

import argparse

from slm_selective_grounding.alignment.dpo_train import train_dpo
from slm_selective_grounding.alignment.formatting import make_preference_pairs
from slm_selective_grounding.datasets.hf_download import download_datasets
from slm_selective_grounding.eval.report import run_eval
from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.mixtures.builder import build_mixtures
from slm_selective_grounding.prompting.templates import generate_baseline_outputs
from slm_selective_grounding.retrieval.index import build_indexes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slm-selective-grounding",
        description="Selective grounding research CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download datasets")
    subparsers.add_parser("index", help="Build retrieval indexes")
    subparsers.add_parser("mix", help="Build mixtures")
    subparsers.add_parser("generate", help="Generate baseline outputs")
    subparsers.add_parser("pairs", help="Create preference pairs")
    subparsers.add_parser("train", help="Train DPO")
    subparsers.add_parser("eval", help="Run evaluation")

    return parser


def main() -> None:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    command_map = {
        "download": download_datasets,
        "index": build_indexes,
        "mix": build_mixtures,
        "generate": generate_baseline_outputs,
        "pairs": make_preference_pairs,
        "train": train_dpo,
        "eval": run_eval,
    }

    command = command_map[args.command]
    command()


if __name__ == "__main__":
    main()
