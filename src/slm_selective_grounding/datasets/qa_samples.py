from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> Iterable[dict[str, object]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def _as_non_empty_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # reject placeholder / junk “questions”
    junk = {"?", "??", "???", "n/a", "na", "none", "null", "<unk>", "unknown"}
    if text.lower() in junk:
        return None

    return text



def _unique_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_question(row: dict[str, object], keys: list[str]) -> str | None:
    for key in keys:
        question = _as_non_empty_str(row.get(key))
        if question:
            return question
    return None

def _extract_question_from_metadata(row: dict[str, object]) -> str | None:
    md = row.get("metadata")
    if not isinstance(md, dict):
        return None

    # try common direct keys
    for key in ["question", "original_question", "query", "prompt", "input", "instruction", "title"]:
        q = _as_non_empty_str(md.get(key))
        if q:
            return q

    # sometimes nested
    for nested_key in ["question", "query", "prompt", "input"]:
        nested = md.get(nested_key)
        if isinstance(nested, dict):
            for key in ["text", "value", "question", "query", "prompt", "input"]:
                q = _as_non_empty_str(nested.get(key))
                if q:
                    return q

    return None



def _extract_answers(row: dict[str, object]) -> list[str]:
    answers: list[str] = []
    raw = row.get("gold_answer") if "gold_answer" in row else row.get("gold_answers")
    raw = raw if raw is not None else row.get("answers")

    if isinstance(raw, str):
        answers.append(raw.strip())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                if item.strip():
                    answers.append(item.strip())
            elif isinstance(item, dict):
                text = _as_non_empty_str(item.get("text"))
                if text:
                    answers.append(text)

    if not answers:
        for key in ["gold_answer", "answer", "output", "reference_answer"]:
            fallback = _as_non_empty_str(row.get(key))
            if fallback:
                answers.append(fallback)
                break

    return _unique_ordered([ans for ans in answers if ans.strip()])


def extract_asqa_sample(
    row: dict[str, object],
    dataset: str,
    split: str,
    index: int,
) -> dict[str, object] | None:
    question = _as_non_empty_str(row.get("ambiguous_question")) or _extract_question(row, ["question", "query", "q"])
    if not question:
        question = _extract_question_from_metadata(row)

    if not question:
        logging.warning("Skipping ASQA row %s: missing question", index)
        return None

    # ASQA: prefer long answers from annotations
    answers: list[str] = []
    anns = row.get("annotations")
    if isinstance(anns, list):
        for a in anns:
            if isinstance(a, dict):
                la = _as_non_empty_str(a.get("long_answer"))
                if la:
                    answers.append(la)

    # Fallback: short answers from qa_pairs
    if not answers:
        qa_pairs = row.get("qa_pairs")
        if isinstance(qa_pairs, list):
            for qa in qa_pairs:
                if isinstance(qa, dict):
                    sas = qa.get("short_answers")
                    if isinstance(sas, list):
                        for s in sas:
                            t = _as_non_empty_str(s)
                            if t:
                                answers.append(t)

    # de-dupe answers (preserve order)
    seen = set()
    answers = [a for a in answers if not (a in seen or seen.add(a))]

    if not answers:
        logging.warning("Skipping ASQA row %s: missing answers", index)
        return None

    # ASQA in this schema does not provide gold_claims directly; leave empty for now
    claims: list[dict[str, str]] = []


    qid = f"{dataset}_{split}_{index:06d}"
    return {
        "qid": qid,
        "dataset": dataset,
        "split": split,
        "question": question,
        "answers": answers,
        "claims": claims,
        "meta": {
            "sample_id": row.get("sample_id"),
            "n_annotations": len(row.get("annotations") or []) if isinstance(row.get("annotations"), list) else None,
            "n_qa_pairs": len(row.get("qa_pairs") or []) if isinstance(row.get("qa_pairs"), list) else None,
            "n_wikipages": len(row.get("wikipages") or []) if isinstance(row.get("wikipages"), list) else None,
            }
    }

def _extract_asqa_long_answers(row: dict[str, object]) -> list[str]:
    anns = row.get("annotations")
    out: list[str] = []
    if isinstance(anns, list):
        for a in anns:
            if isinstance(a, dict):
                la = _as_non_empty_str(a.get("long_answer"))
                if la:
                    out.append(la)
    # de-dupe preserve order
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


def extract_alce_sample(
    row: dict[str, object],
    dataset: str,
    split: str,
    index: int,
) -> dict[str, object] | None:
    question = _extract_question(row, ["question", "query", "instruction", "input"])
    if not question:
        logging.warning("Skipping ALCE row %s: missing question", index)
        return None

    answer = _as_non_empty_str(
        row.get("answer")
        or row.get("output")
        or row.get("response")
        or row.get("gold")
        or row.get("reference")
    )
    if not answer:
        logging.warning("Skipping ALCE row %s: missing answer", index)
        return None

    meta: dict[str, object] = {}
    for key in ["id", "example_id", "task", "dataset_name", "source", "citation"]:
        if key in row:
            meta[key] = row[key]

    qid = f"{dataset}_{split}_{index:06d}"
    return {
        "qid": qid,
        "dataset": dataset,
        "split": split,
        "question": question,
        "answers": [answer],
        "meta": meta,
    }
