from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Example:
    query: str
    gold_answer: str | None
    gold_claims: list[str] | None
    docs: list[Document]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "gold_answer": self.gold_answer,
            "gold_claims": self.gold_claims,
            "docs": [doc.__dict__ for doc in self.docs],
            "metadata": dict(self.metadata),
        }
