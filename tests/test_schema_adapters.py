from __future__ import annotations

import pytest

from slm_selective_grounding.datasets.schema import (
    normalize_alce_data,
    normalize_asqa,
    normalize_fever,
    normalize_qampari,
)


@pytest.mark.parametrize(
    ("adapter", "row", "dataset_id", "config_name"),
    [
        (
            normalize_alce_data,
            {"question": "q", "answer": "a", "docs": [{"id": "1", "text": "t"}]},
            "princeton-nlp/ALCE-data",
            "asqa",
        ),
        (
            normalize_asqa,
            {"question": "q", "long_answer": "a"},
            "din0s/asqa",
            None,
        ),
        (
            normalize_qampari,
            {"question": "q", "answers": ["a1", "a2"]},
            "iohadrubin/qampari",
            None,
        ),
        (
            normalize_fever,
            {"claim": "c", "label": "SUPPORTS", "evidence": [["doc", 1]]},
            "fever/fever",
            None,
        ),
    ],
)
def test_schema_adapters(adapter, row, dataset_id, config_name) -> None:
    example = adapter(row, dataset_id, config_name, "train")
    payload = example.to_dict()

    assert "query" in payload
    assert "gold_answer" in payload
    assert "gold_claims" in payload
    assert "docs" in payload
    assert "metadata" in payload
    assert isinstance(payload["docs"], list)
    assert isinstance(payload["metadata"], dict)
