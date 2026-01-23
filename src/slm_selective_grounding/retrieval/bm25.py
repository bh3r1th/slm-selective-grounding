from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from slm_selective_grounding.datasets.schema import Document


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Result:
    doc_id: str
    title: str
    text: str
    score: float
    rank: int


class BM25Index:
    def __init__(
        self,
        docs: list[Document],
        df: dict[str, int],
        avgdl: float,
        k1: float,
        b: float,
    ) -> None:
        self._docs = docs
        self._df = df
        self._avgdl = avgdl if avgdl > 0 else 1.0
        self._k1 = k1
        self._b = b
        self._doc_tfs: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        for doc in docs:
            tokens = _tokenize(doc.text)
            self._doc_tfs.append(dict(Counter(tokens)))
            self._doc_lens.append(len(tokens))

    @classmethod
    def from_index_dir(cls, index_dir: Path) -> "BM25Index":
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        df = json.loads((index_dir / "df.json").read_text(encoding="utf-8"))
        docs_path = Path(meta.get("docs_path", index_dir / "docs.jsonl"))
        docs = list(_load_docs(docs_path))
        return cls(
            docs=docs,
            df={str(k): int(v) for k, v in df.items()},
            avgdl=float(meta.get("avgdl", 0.0)),
            k1=float(meta.get("k1", 0.9)),
            b=float(meta.get("b", 0.4)),
        )

    def search(self, query: str, topn: int = 50) -> list[BM25Result]:
        tokens = _tokenize(query)
        if not tokens or not self._docs:
            return []
        query_terms = list(dict.fromkeys(tokens))
        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self._doc_tfs):
            score = 0.0
            dl = self._doc_lens[idx] or 1
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((len(self._docs) - df + 0.5) / (df + 0.5) + 1.0)
                denom = freq + self._k1 * (1.0 - self._b + self._b * (dl / self._avgdl))
                score += idf * ((freq * (self._k1 + 1.0)) / denom)
            if score > 0.0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        results: list[BM25Result] = []
        for rank, (idx, score) in enumerate(scores[:topn], start=1):
            doc = self._docs[idx]
            results.append(
                BM25Result(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=score,
                    rank=rank,
                )
            )
        return results


def load_bm25_index(index_dir: Path) -> BM25Index:
    return BM25Index.from_index_dir(index_dir)


def search(index: BM25Index, query: str, topn: int = 50) -> list[dict[str, object]]:
    return [
        {
            "doc_id": result.doc_id,
            "title": result.title,
            "text": result.text,
            "score": result.score,
            "rank": result.rank,
        }
        for result in index.search(query, topn=topn)
    ]


def _load_docs(corpus_path: Path) -> Iterable[Document]:
    for idx, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        doc_id = payload.get("doc_id") or payload.get("passage_id") or payload.get("id") or str(idx)
        title = str(payload.get("title", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        yield Document(doc_id=str(doc_id), title=title, text=text)


def build_bm25_index(corpus_path: Path, index_dir: Path, k1: float = 0.9, b: float = 0.4) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
    df_counts: Counter[str] = Counter()
    total_len = 0

    for doc in _load_docs(corpus_path):
        tokens = _tokenize(doc.text)
        if not tokens:
            continue
        df_counts.update(set(tokens))
        total_len += len(tokens)
        docs.append(doc)

    doc_count = len(docs)
    avgdl = total_len / doc_count if doc_count else 0.0

    docs_path = index_dir / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(
                json.dumps(
                    {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text},
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    df_path = index_dir / "df.json"
    df_path.write_text(json.dumps(df_counts, ensure_ascii=False), encoding="utf-8")

    meta = {
        "k1": k1,
        "b": b,
        "corpus_path": str(corpus_path),
        "docs_path": str(docs_path),
        "df_path": str(df_path),
        "doc_count": doc_count,
        "avgdl": avgdl,
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (index_dir / "corpus.jsonl").write_text(docs_path.read_text(encoding="utf-8"), encoding="utf-8")
