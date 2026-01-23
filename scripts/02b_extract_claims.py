from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from slm_selective_grounding.logging import configure_logging
from slm_selective_grounding.utils.io import ensure_dirs


def _split_claims(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(".")]
    return [part for part in parts if len(part) >= 10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract claims from QA answers")
    parser.add_argument("--dataset", required=True, help="Dataset basename without .jsonl")
    parser.add_argument(
        "--out-suffix",
        default="_with_claims",
        help="Suffix to append to output filename",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    input_path = Path("artifacts") / "corpus" / f"{args.dataset}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_path = Path("artifacts") / "corpus" / f"{args.dataset}{args.out_suffix}.jsonl"
    ensure_dirs([output_path.parent])

    read_count = 0
    written_count = 0
    skipped_count = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            read_count += 1
            row = json.loads(line)
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers:
                skipped_count += 1
                continue

            claims: list[dict[str, str]] = []
            for answer in answers:
                if not isinstance(answer, str):
                    continue
                for claim_text in _split_claims(answer):
                    claims.append(
                        {
                            "claim_id": f"c{len(claims)}",
                            "text": claim_text,
                        }
                    )

            if not claims:
                skipped_count += 1
                continue

            row["claims"] = claims
            dst.write(json.dumps(row, ensure_ascii=False))
            dst.write("\n")
            written_count += 1

    logger.info(
        "read=%s written=%s skipped=%s output=%s",
        read_count,
        written_count,
        skipped_count,
        output_path,
    )


if __name__ == "__main__":
    main()
