from __future__ import annotations

import json
from pathlib import Path

from slm_selective_grounding.datasets.qa_samples import (
    extract_alce_sample,
    extract_asqa_sample,
    load_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def test_asqa_parsing_smoke(tmp_path: Path) -> None:
    rows = [
        {"question": "Who wrote Hamlet?", "answers": [{"text": "William Shakespeare"}]},
        {"query": "Capital of France?", "answers": "Paris"},
    ]
    path = tmp_path / "asqa.jsonl"
    _write_jsonl(path, rows)

    samples = []
    for idx, row in enumerate(load_jsonl(path)):
        sample = extract_asqa_sample(row, "asqa", "train", idx)
        assert sample is not None
        samples.append(sample)

    assert samples
    for sample in samples:
        assert sample["qid"]
        assert sample["question"]
        assert sample["answers"]


def test_alce_parsing_smoke(tmp_path: Path) -> None:
    rows = [{"instruction": "Explain gravity.", "output": "Gravity attracts masses."}]
    path = tmp_path / "alce.jsonl"
    _write_jsonl(path, rows)

    samples = []
    for idx, row in enumerate(load_jsonl(path)):
        sample = extract_alce_sample(row, "alce", "train", idx)
        assert sample is not None
        samples.append(sample)

    assert samples[0]["question"] == "Explain gravity."
    assert samples[0]["answers"] == ["Gravity attracts masses."]
