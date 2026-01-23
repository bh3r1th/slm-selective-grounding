from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_selective_grounding.contexts.mixer import mix_contexts


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _make_doc(doc_id: str, title: str, text: str) -> dict[str, str]:
    return {"doc_id": doc_id, "title": title, "text": text, "source": "wiki"}


def test_mix_contexts_exact_k(tmp_path: Path) -> None:
    corpus_path = tmp_path / "wiki_leads.jsonl"
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "contexts.jsonl"
    index_dir = tmp_path / "index"

    claim = "Marie Curie discovered radium in 1898."

    docs = []
    for i in range(7):
        docs.append(
            _make_doc(
                f"s{i}",
                "Marie Curie",
                f"Marie Curie discovered radium in 1898. ({i})",
            )
        )
    for i in range(4):
        docs.append(
            _make_doc(
                f"d{i}",
                "Radium",
                "The element radium was discovered in a laboratory setting.",
            )
        )
    docs.append(
        _make_doc(
            "c0",
            "Curie",
            "Marie Curie did not discover radium in 1898.",
        )
    )
    for i in range(4):
        docs.append(
            _make_doc(
                f"r{i}",
                "Banana",
                "Bananas are yellow fruit and grow in tropical climates.",
            )
        )

    _write_jsonl(corpus_path, docs)
    _write_jsonl(
        dataset_path,
        [{"qid": "q1", "question": "Who discovered radium?", "claims": [claim]}],
    )

    written = mix_contexts(
        dataset_path=dataset_path,
        output_path=output_path,
        corpus_path=corpus_path,
        index_dir=index_dir,
        topn=20,
        k=12,
        seed=7,
        profile="clean",
        profile_schedule="clean:1.0",
    )
    assert written == 1

    row: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    contexts = row["contexts"]
    assert len(contexts) == 12


def test_mix_contexts_label_distribution_with_shortfall(tmp_path: Path) -> None:
    corpus_path = tmp_path / "wiki_leads.jsonl"
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "contexts.jsonl"
    index_dir = tmp_path / "index"

    claim = "Marie Curie discovered radium in 1898."

    docs = []
    for i in range(7):
        docs.append(
            _make_doc(
                f"s{i}",
                "Marie Curie",
                f"Marie Curie discovered radium in 1898. ({i})",
            )
        )
    for i in range(4):
        docs.append(
            _make_doc(
                f"d{i}",
                "Radium",
                "The element radium was discovered in a laboratory setting.",
            )
        )
    docs.append(
        _make_doc(
            "c0",
            "Curie",
            "Marie Curie did not discover radium in 1898.",
        )
    )
    for i in range(4):
        docs.append(
            _make_doc(
                f"r{i}",
                "Banana",
                "Bananas are yellow fruit and grow in tropical climates.",
            )
        )

    _write_jsonl(corpus_path, docs)
    _write_jsonl(
        dataset_path,
        [{"qid": "q1", "question": "Who discovered radium?", "claims": [claim]}],
    )

    mix_contexts(
        dataset_path=dataset_path,
        output_path=output_path,
        corpus_path=corpus_path,
        index_dir=index_dir,
        topn=20,
        k=12,
        seed=7,
        profile="clean",
        profile_schedule="clean:1.0",
    )

    row: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    contexts = row["contexts"]
    counts = {"support": 0, "decoy": 0, "conflict": 0, "irrelevant": 0}
    for ctx in contexts:
        label = str(ctx["label"])
        counts[label] += 1

    recipe = {"support": 8, "decoy": 3, "conflict": 0, "irrelevant": 1}
    for label, expected in recipe.items():
        assert expected - 1 <= counts[label] <= expected + 1
