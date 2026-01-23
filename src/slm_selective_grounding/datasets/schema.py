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

    def with_metadata(self, **extra: Any) -> "Example":
        merged = dict(self.metadata)
        merged.update(extra)
        return Example(
            query=self.query,
            gold_answer=self.gold_answer,
            gold_claims=self.gold_claims,
            docs=self.docs,
            metadata=merged,
        )


def _get_first(example: Mapping[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in example:
            return example[key]
    return None


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _extract_text(raw_doc: Mapping[str, Any]) -> str:
    """
    Many HF datasets store passage text under different keys.
    This function tries common variants.
    """
    for k in [
        "text",
        "content",
        "paragraph_text",
        "passage",
        "sentence",
        "sent",
        "ctx",
        "context",
        "wikipedia_text",
        "wiki_text",
        "body",
    ]:
        v = raw_doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Sometimes it's nested, e.g., {"document": {"text": "..."}}
    for k in ["document", "doc", "passage", "ctx"]:
        v = raw_doc.get(k)
        if isinstance(v, Mapping):
            t = _extract_text(v)
            if t:
                return t

    return ""


def _extract_title(raw_doc: Mapping[str, Any]) -> str:
    for k in ["title", "source", "article_title", "wiki_title", "doc_title"]:
        v = raw_doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_doc_id(raw_doc: Mapping[str, Any], fallback: str) -> str:
    for k in ["doc_id", "id", "docid", "document_id", "passage_id", "pid", "idx"]:
        v = raw_doc.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return fallback


def _normalize_docs(raw_docs: Any) -> list[Document]:
    docs: list[Document] = []
    if not raw_docs:
        return docs

    # Allow string doc directly
    if isinstance(raw_docs, str):
        raw_docs = [{"text": raw_docs}]

    # Allow single mapping
    if isinstance(raw_docs, Mapping):
        raw_docs = [raw_docs]

    # Allow nested structures like {"ctxs": [...]}
    if isinstance(raw_docs, Mapping) and "ctxs" in raw_docs:
        raw_docs = raw_docs["ctxs"]

    # Now iterate
    for idx, raw_doc in enumerate(_as_list(raw_docs)):
        if not isinstance(raw_doc, Mapping):
            continue

        doc_id = _extract_doc_id(raw_doc, fallback=str(idx))
        title = _extract_title(raw_doc)
        text = _extract_text(raw_doc)

        # Handle common ctxs format: {"title": "...", "text": "..."} already handled,
        # but some use {"title": "...", "sent": [...]} or list fields.
        if not text:
            for k in ["sentences", "sents", "sent", "sentence"]:
                v = raw_doc.get(k)
                if isinstance(v, list) and v:
                    joined = " ".join(str(x).strip() for x in v if str(x).strip())
                    if joined.strip():
                        text = joined.strip()
                        break

        if not text:
            continue  # do not keep empty docs

        docs.append(Document(doc_id=str(doc_id), title=str(title), text=str(text)))
    return docs


def normalize_alce_data(
    example: Mapping[str, Any],
    dataset_id: str,
    config_name: str | None,
    split: str,
) -> Example:
    query = _get_first(example, ["question", "query", "prompt", "instruction"]) or ""
    gold_answer = _get_first(example, ["answer", "output", "reference_answer", "gold_answer"])

    # ALCE variants: docs/ctxs/contexts/passages sometimes nested
    raw_docs = _get_first(example, ["docs", "documents", "passages", "contexts", "ctxs", "retrieved_docs"])
    docs = _normalize_docs(raw_docs)

    citations = _get_first(example, ["citations", "citation", "source_citations"])
    metadata = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "id": _get_first(example, ["id", "example_id"]),
        "citations": citations,
    }
    return Example(
        query=str(query),
        gold_answer=str(gold_answer) if gold_answer is not None else None,
        gold_claims=None,
        docs=docs,
        metadata=metadata,
    )


def normalize_asqa(
    example: Mapping[str, Any],
    dataset_id: str,
    config_name: str | None,
    split: str,
) -> Example:
    question = _get_first(example, ["question", "query", "prompt"]) or ""
    long_answer = _get_first(example, ["long_answer", "answer", "response"])

    # ASQA mirrors often store contexts in ctxs / docs / references
    raw_docs = _get_first(example, ["docs", "documents", "contexts", "ctxs", "references", "passages"])
    docs = _normalize_docs(raw_docs)

    metadata = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "id": _get_first(example, ["id", "example_id"]),
    }
    return Example(
        query=str(question),
        gold_answer=str(long_answer) if long_answer is not None else None,
        gold_claims=None,
        docs=docs,
        metadata=metadata,
    )


def normalize_qampari(
    example: Mapping[str, Any],
    dataset_id: str,
    config_name: str | None,
    split: str,
) -> Example:
    question = _get_first(example, ["question", "query", "prompt"]) or ""
    answers = _get_first(example, ["answers", "targets", "gold_answers", "answer"])
    if isinstance(answers, str):
        answers_list = [answers]
    elif isinstance(answers, list):
        answers_list = [str(answer) for answer in answers]
    else:
        answers_list = []
    metadata = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "id": _get_first(example, ["id", "example_id"]),
    }
    return Example(
        query=str(question),
        gold_answer=None,
        gold_claims=answers_list or None,
        docs=[],
        metadata=metadata,
    )


def normalize_fever(
    example: Mapping[str, Any],
    dataset_id: str,
    config_name: str | None,
    split: str,
) -> Example:
    claim = _get_first(example, ["claim", "query", "text"]) or ""
    label = _get_first(example, ["label", "verdict"])
    evidence = _get_first(example, ["evidence", "evidence_sentences", "evidence_sets"])
    metadata = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "id": _get_first(example, ["id", "example_id"]),
        "label": label,
        "evidence": evidence,
    }
    return Example(
        query=str(claim),
        gold_answer=None,
        gold_claims=None,
        docs=[],
        metadata=metadata,
    )


def normalize_example(
    example: Mapping[str, Any],
    dataset_id: str,
    config_name: str | None,
    split: str,
) -> Example:
    if dataset_id == "princeton-nlp/ALCE-data":
        return normalize_alce_data(example, dataset_id, config_name, split)
    if dataset_id == "din0s/asqa":
        return normalize_asqa(example, dataset_id, config_name, split)
    if dataset_id == "iohadrubin/qampari":
        return normalize_qampari(example, dataset_id, config_name, split)
    if dataset_id == "fever/fever":
        return normalize_fever(example, dataset_id, config_name, split)
    raise ValueError(f"Unsupported dataset_id: {dataset_id}")
