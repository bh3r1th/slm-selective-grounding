from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from slm_selective_grounding.retrieval.bm25 import (
    build_bm25_index,
    load_bm25_index,
    search as bm25_search,
)
from slm_selective_grounding.utils.io import ensure_dirs


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+\b")
_NUMBER_RE = re.compile(r"\b\d+\b")
_NEGATION_CUES = ("not", "never", "no evidence", "false", "untrue", "incorrect")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}

_PROFILE_RECIPES = {
    "clean": {"support": 8, "decoy": 3, "conflict": 0, "irrelevant": 1},
    "mixed": {"support": 5, "decoy": 4, "conflict": 2, "irrelevant": 1},
    "toxic": {"support": 3, "decoy": 4, "conflict": 4, "irrelevant": 1},
}


@dataclass(frozen=True)
class ClaimInfo:
    text: str
    keywords: set[str]
    entities: set[str]
    numbers: set[str]


@dataclass(frozen=True)
class CandidateDoc:
    doc_id: str
    title: str
    text: str
    rank: int


@dataclass(frozen=True)
class LabeledContext:
    doc_id: str
    title: str
    text: str
    label: str
    rank: int


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _extract_keywords(text: str) -> set[str]:
    tokens = _tokenize(text)
    return {tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 1}


def _extract_entities(text: str) -> set[str]:
    return {match.group(0).lower() for match in _ENTITY_RE.finditer(text)}


def _extract_numbers(text: str) -> set[str]:
    return {match.group(0) for match in _NUMBER_RE.finditer(text)}


def build_claim_info(text: str) -> ClaimInfo:
    return ClaimInfo(
        text=text,
        keywords=_extract_keywords(text),
        entities=_extract_entities(text),
        numbers=_extract_numbers(text),
    )


def _keyword_overlap(claim_keywords: set[str], doc_tokens: set[str]) -> int:
    return len(claim_keywords.intersection(doc_tokens))


def label_candidate(candidate: CandidateDoc, claim: ClaimInfo) -> str:
    doc_tokens = set(_tokenize(candidate.text))
    overlap = _keyword_overlap(claim.keywords, doc_tokens)
    doc_entities = _extract_entities(candidate.text)
    doc_numbers = _extract_numbers(candidate.text)
    entity_overlap = bool(claim.entities.intersection(doc_entities))
    number_overlap = bool(claim.numbers.intersection(doc_numbers))
    doc_lower = candidate.text.lower()
    has_negation = any(cue in doc_lower for cue in _NEGATION_CUES)
    mismatched_number = bool(claim.numbers and doc_numbers and not number_overlap)

    if overlap >= 2 and (entity_overlap or number_overlap):
        if has_negation or mismatched_number:
            return "conflict"
        return "support"

    if overlap >= 2:
        return "decoy"

    return "irrelevant"


def _scaled_recipe(profile: str, k: int) -> dict[str, int]:
    base = _PROFILE_RECIPES[profile]
    if k == 12:
        return dict(base)
    scaled: dict[str, int] = {}
    for label, count in base.items():
        scaled[label] = max(0, int(round(count * k / 12)))
    total = sum(scaled.values())
    if total == 0:
        scaled["irrelevant"] = k
        return scaled
    while total < k:
        scaled["support"] += 1
        total += 1
    while total > k:
        for label in ("irrelevant", "decoy", "conflict", "support"):
            if scaled[label] > 0 and total > k:
                scaled[label] -= 1
                total -= 1
    return scaled


def _sample_pool(
    pool: list[CandidateDoc],
    n: int,
    rng: random.Random,
) -> list[CandidateDoc]:
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        return list(pool)
    pool_copy = list(pool)
    rng.shuffle(pool_copy)
    return pool_copy[:n]


def select_contexts_for_claim(
    claim: ClaimInfo,
    candidates: list[CandidateDoc],
    k: int,
    profile: str,
    rng: random.Random,
) -> list[LabeledContext]:
    recipe = _scaled_recipe(profile, k)
    labeled: dict[str, list[CandidateDoc]] = {
        "support": [],
        "decoy": [],
        "conflict": [],
        "irrelevant": [],
    }

    for candidate in candidates:
        label = label_candidate(candidate, claim)
        labeled[label].append(candidate)

    tail_start = int(len(candidates) * 0.6)
    tail_irrelevant = [
        cand
        for cand in candidates[tail_start:]
        if label_candidate(cand, claim) == "irrelevant"
    ]
    if tail_irrelevant:
        labeled["irrelevant"] = tail_irrelevant

    selected: list[tuple[CandidateDoc, str]] = []
    used_keys: set[tuple[str, int]] = set()

    for label in ("support", "decoy", "conflict", "irrelevant"):
        picks = _sample_pool(labeled[label], recipe.get(label, 0), rng)
        for cand in picks:
            key = (cand.doc_id, cand.rank)
            if key in used_keys:
                continue
            selected.append((cand, label))
            used_keys.add(key)

    if len(selected) < k:
        remaining = [
            cand
            for cand in candidates
            if (cand.doc_id, cand.rank) not in used_keys
        ]
        rng.shuffle(remaining)
        for cand in remaining:
            if len(selected) >= k:
                break
            label = label_candidate(cand, claim)
            selected.append((cand, label))
            used_keys.add((cand.doc_id, cand.rank))

    if len(selected) < k and candidates:
        while len(selected) < k:
            cand = rng.choice(candidates)
            label = label_candidate(cand, claim)
            selected.append((cand, label))

    selected = sorted(selected, key=lambda item: item[0].rank)[:k]
    return [
        LabeledContext(
            doc_id=cand.doc_id,
            title=cand.title,
            text=cand.text,
            label=label,
            rank=cand.rank,
        )
        for cand, label in selected
    ]


def parse_profile_schedule(schedule: str) -> list[tuple[str, float]]:
    items = []
    for entry in schedule.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, weight = entry.split(":")
        name = name.strip()
        weight_val = float(weight.strip())
        if name not in _PROFILE_RECIPES:
            raise ValueError(f"Unknown profile in schedule: {name}")
        if weight_val <= 0:
            continue
        items.append((name, weight_val))
    if not items:
        raise ValueError("Profile schedule is empty")
    return items


def choose_profile(schedule: list[tuple[str, float]], rng: random.Random) -> str:
    total = sum(weight for _, weight in schedule)
    roll = rng.random() * total
    acc = 0.0
    for name, weight in schedule:
        acc += weight
        if roll <= acc:
            return name
    return schedule[-1][0]


def ensure_bm25_index(corpus_path: Path, index_dir: Path, k1: float, b: float) -> None:
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        docs_path = Path(meta.get("docs_path", index_dir / "docs.jsonl"))
        df_path = Path(meta.get("df_path", index_dir / "df.json"))
        if (
            meta.get("corpus_path") == str(corpus_path)
            and docs_path.exists()
            and df_path.exists()
        ):
            return
    build_bm25_index(corpus_path=corpus_path, index_dir=index_dir, k1=k1, b=b)


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _reservoir_sample_corpus(
    corpus_path: Path,
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if sample_size <= 0:
        return []
    reservoir: list[dict[str, str]] = []
    seen = 0
    for payload in _iter_jsonl(corpus_path):
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        doc_id = payload.get("doc_id") or payload.get("passage_id") or payload.get("id")
        if doc_id is None:
            continue
        title = str(payload.get("title", "")).strip()
        record = {"doc_id": str(doc_id), "title": title, "text": text}
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(record)
        else:
            pick = rng.randint(0, seen - 1)
            if pick < sample_size:
                reservoir[pick] = record
    return reservoir


def _sample_sqlite_backfill(
    conn: sqlite3.Connection,
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if sample_size <= 0:
        return []
    cursor = conn.cursor()
    row = cursor.execute("SELECT MAX(rowid) FROM passages").fetchone()
    if not row or row[0] is None:
        return []
    max_rowid = int(row[0])
    pool: list[dict[str, str]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = sample_size * 5
    while len(pool) < sample_size and attempts < max_attempts:
        rowid = rng.randint(1, max_rowid)
        fetched = cursor.execute(
            "SELECT doc_id, title, text FROM passages WHERE rowid >= ? LIMIT 1",
            (rowid,),
        ).fetchone()
        attempts += 1
        if not fetched:
            continue
        doc_id, title, text = fetched
        if doc_id is None or text is None:
            continue
        doc_id_str = str(doc_id)
        if not doc_id_str or doc_id_str in seen:
            continue
        text_str = str(text).strip()
        if not text_str:
            continue
        title_str = title.strip() if isinstance(title, str) else ""
        pool.append({"doc_id": doc_id_str, "title": title_str, "text": text_str})
        seen.add(doc_id_str)
    return pool


def _backfill_contexts(
    contexts: list[dict[str, object]],
    backfill_pool: list[dict[str, str]],
    k: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    if len(contexts) >= k or not backfill_pool:
        return contexts
    needed = k - len(contexts)
    existing_ids = {
        ctx.get("doc_id")
        for ctx in contexts
        if isinstance(ctx.get("doc_id"), str)
    }
    indices = list(range(len(backfill_pool)))
    rng.shuffle(indices)
    for idx in indices:
        if needed <= 0:
            break
        entry = backfill_pool[idx]
        if entry["doc_id"] in existing_ids:
            continue
        contexts.append(
            {
                "doc_id": entry["doc_id"],
                "title": entry.get("title", ""),
                "text": entry["text"],
                "label": "irrelevant",
                "rank": len(contexts) + 1,
                "metadata": {
                    "backfill": True,
                    "backfill_reason": "insufficient_retrieval",
                },
            }
        )
        existing_ids.add(entry["doc_id"])
        needed -= 1
    while needed > 0 and backfill_pool:
        entry = rng.choice(backfill_pool)
        contexts.append(
            {
                "doc_id": entry["doc_id"],
                "title": entry.get("title", ""),
                "text": entry["text"],
                "label": "irrelevant",
                "rank": len(contexts) + 1,
                "metadata": {
                    "backfill": True,
                    "backfill_reason": "insufficient_retrieval",
                },
            }
        )
        needed -= 1
    return contexts


def _as_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _candidate_docs(results: list[dict[str, object]]) -> list[CandidateDoc]:
    return [
        CandidateDoc(
            doc_id=str(res["doc_id"]),
            title=str(res.get("title", "")),
            text=str(res.get("text", "")),
            rank=int(res.get("rank", 0)),
        )
        for res in results
    ]


def mix_contexts(
    dataset_path: Path,
    output_path: Path,
    corpus_path: Path,
    index_dir: Path,
    topn: int,
    k: int,
    seed: int,
    profile: str | None,
    profile_schedule: str,
    k1: float = 0.9,
    b: float = 0.4,
    corpus_lookup: dict[str, tuple[str, str]] | None = None,
    corpus_db_path: Path | None = None,
    max_rows: int | None = None,
    log_every: int = 200,
) -> int:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing wiki leads corpus: {corpus_path}")
    if corpus_db_path is not None and not corpus_db_path.exists():
        raise FileNotFoundError(f"Missing corpus db: {corpus_db_path}")

    ensure_dirs([output_path.parent, index_dir])

    ensure_bm25_index(corpus_path=corpus_path, index_dir=index_dir, k1=k1, b=b)
    bm25 = load_bm25_index(index_dir)

    rng = random.Random(seed)
    backfill_rng = random.Random(seed + 17)
    schedule = parse_profile_schedule(profile_schedule) if profile is None else []

    backfill_pool_size = max(k * 5, 1000)
    db_conn: sqlite3.Connection | None = None
    if corpus_db_path is not None:
        db_conn = sqlite3.connect(str(corpus_db_path))
        backfill_pool = _sample_sqlite_backfill(
            conn=db_conn,
            sample_size=backfill_pool_size,
            rng=backfill_rng,
        )
    else:
        backfill_pool = _reservoir_sample_corpus(
            corpus_path=corpus_path,
            sample_size=backfill_pool_size,
            rng=backfill_rng,
        )

    written = 0
    processed_rows = 0
    start_time = time.perf_counter()
    placeholder_json = '{"gold_answer": null, "gold_claims": null, "query": ""}'
    warned_placeholder = False
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for row_idx, row in enumerate(_iter_jsonl(dataset_path)):
                if max_rows is not None and processed_rows >= max_rows:
                    break
                qid = str(
                    row.get("qid") or row.get("id") or row.get("example_id") or row_idx
                )
                question = str(
                    row.get("question") or row.get("query") or row.get("prompt") or ""
                )
                answers = _as_list(
                    row.get("answers") or row.get("answer") or row.get("gold_answer")
                )
                claims = _as_list(row.get("claims") or row.get("gold_claims"))

                if not claims:
                    logging.warning(
                        "TODO: claims missing for qid=%s; using answers as proxy.", qid
                    )
                    claims = answers

                if not claims:
                    claims = [question] if question else []

                if not claims:
                    processed_rows += 1
                    if log_every and processed_rows % log_every == 0:
                        elapsed = time.perf_counter() - start_time
                        rows_per_sec = processed_rows / elapsed if elapsed else 0.0
                        print(
                            f"processed_rows={processed_rows} "
                            f"rows_per_sec={rows_per_sec:.2f} "
                            f"elapsed_sec={elapsed:.1f}"
                        )
                    continue

                mix_profile = profile or choose_profile(schedule, rng)

                for claim_idx, claim_text in enumerate(claims):
                    claim_info = build_claim_info(claim_text)
                    results = bm25_search(bm25, claim_text, topn=topn)
                    candidates = _candidate_docs(results)
                    contexts = select_contexts_for_claim(
                        claim=claim_info,
                        candidates=candidates,
                        k=k,
                        profile=mix_profile,
                        rng=rng,
                    )

                    context_rows: list[dict[str, object]] = []
                    for ctx in contexts:
                        title = ctx.title
                        text = ctx.text
                        metadata = None
                        if db_conn is not None:
                            has_doc_id = bool(ctx.doc_id) and ctx.doc_id != "None"
                            if has_doc_id:
                                fetched = db_conn.execute(
                                    "SELECT title, text FROM passages WHERE doc_id = ?",
                                    (ctx.doc_id,),
                                ).fetchone()
                                if fetched is None:
                                    metadata = {"missing_doc": True}
                                else:
                                    title = fetched[0] or ""
                                    text = fetched[1] or ""
                                    metadata = {}
                            else:
                                metadata = {"missing_doc": True}
                        elif corpus_lookup is not None:
                            mapped = corpus_lookup.get(ctx.doc_id)
                            if mapped is not None:
                                title, text = mapped
                        if (
                            __debug__
                            and db_conn is not None
                            and not warned_placeholder
                            and text == placeholder_json
                        ):
                            logging.warning(
                                "SQLite corpus provided but placeholder text used"
                            )
                            warned_placeholder = True
                        context_payload = {
                            "doc_id": ctx.doc_id,
                            "title": title,
                            "text": text,
                            "label": ctx.label,
                            "rank": ctx.rank,
                        }
                        if metadata is not None:
                            context_payload["metadata"] = metadata
                        context_rows.append(context_payload)
                    if len(context_rows) < k:
                        context_rows = _backfill_contexts(
                            contexts=context_rows,
                            backfill_pool=backfill_pool,
                            k=k,
                            rng=backfill_rng,
                        )

                    payload = {
                        "qid": qid,
                        "question": question,
                        "claim_id": f"{qid}::c{claim_idx}",
                        "claim_text": claim_text,
                        "mix_profile": mix_profile,
                        "contexts": context_rows,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False))
                    handle.write("\n")
                    written += 1

                processed_rows += 1
                if log_every and processed_rows % log_every == 0:
                    elapsed = time.perf_counter() - start_time
                    rows_per_sec = processed_rows / elapsed if elapsed else 0.0
                    print(
                        f"processed_rows={processed_rows} "
                        f"rows_per_sec={rows_per_sec:.2f} "
                        f"elapsed_sec={elapsed:.1f}"
                    )
    finally:
        if db_conn is not None:
            db_conn.close()

    print(f"output_path={output_path} rows_written={written}")

    return written
