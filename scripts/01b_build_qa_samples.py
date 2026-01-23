from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from slm_selective_grounding.datasets.qa_samples import (
    extract_alce_sample,
    extract_asqa_sample,
    load_jsonl,
)
from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.utils.io import ensure_dirs


def _resolve_raw_path(dataset: str, split: str) -> Path:
    if dataset == "asqa":
        return Path("data") / "raw" / "din0s__asqa" / f"{split}.jsonl"
    if dataset == "alce":
        return Path("data") / "raw" / "princeton-nlp__ALCE-data" / f"{split}.jsonl"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA samples from raw datasets")
    parser.add_argument("--dataset", choices=["asqa", "alce"], required=True)
    parser.add_argument("--split", required=True, help="Dataset split (train/dev/test)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to write")
    parser.add_argument("--out-name", default=None, help="Override output filename")
    parser.add_argument("--seed", type=int, default=42, help="Unused, for compatibility")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    raw_path = _resolve_raw_path(args.dataset, args.split)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw dataset file: {raw_path}")

    out_name = args.out_name or f"{args.dataset}_{args.split}_qa.jsonl"
    output_path = Path("artifacts") / "corpus" / out_name
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output_path}")

    ensure_dirs([output_path.parent])

    read_count = 0
    written_count = 0
    skipped_count = 0
    samples: list[dict[str, object]] = []

    for row in load_jsonl(raw_path):
        read_count += 1
        if args.dataset == "asqa":
            sample = extract_asqa_sample(row, args.dataset, args.split, read_count - 1)
        else:
            sample = extract_alce_sample(row, args.dataset, args.split, read_count - 1)

        if sample is None:
            skipped_count += 1
            continue

        samples.append(sample)
        written_count += 1
        if args.limit is not None and written_count >= args.limit:
            break

    _write_jsonl(output_path, samples)

    logger.info(
        "read=%s written=%s skipped=%s output=%s",
        read_count,
        written_count,
        skipped_count,
        output_path,
    )


if __name__ == "__main__":
    main()
