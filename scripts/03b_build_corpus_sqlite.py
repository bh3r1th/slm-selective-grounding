from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from slm_selective_grounding.utils.io import ensure_dirs


def _iter_jsonl(path: Path) -> tuple[int, int, list[tuple[str, str, str]]]:
    inserted = 0
    skipped = 0
    batch: list[tuple[str, str, str]] = []
    processed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                processed += 1
                if processed % 50000 == 0:
                    print(f"processed={processed} inserted={inserted} skipped={skipped}")
                continue
            if not isinstance(payload, dict):
                skipped += 1
                processed += 1
                if processed % 50000 == 0:
                    print(f"processed={processed} inserted={inserted} skipped={skipped}")
                continue
            doc_id = payload.get("doc_id")
            text = payload.get("text")
            if doc_id is None or not isinstance(text, str) or not text.strip():
                skipped += 1
                processed += 1
                if processed % 50000 == 0:
                    print(f"processed={processed} inserted={inserted} skipped={skipped}")
                continue
            title = payload.get("title")
            title_str = title.strip() if isinstance(title, str) else ""
            batch.append((str(doc_id), title_str, text.strip()))
            inserted += 1
            processed += 1
            if processed % 50000 == 0:
                print(f"processed={processed} inserted={inserted} skipped={skipped}")
            if len(batch) >= 50000:
                yield inserted, skipped, batch
                batch = []
    if batch:
        yield inserted, skipped, batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SQLite corpus for fast lookup")
    parser.add_argument("--corpus_jsonl", required=True, help="Path to JSONL passages")
    parser.add_argument("--out_db", required=True, help="Output SQLite path")
    args = parser.parse_args()

    corpus_path = Path(args.corpus_jsonl)
    out_db = Path(args.out_db)
    ensure_dirs([out_db.parent])

    conn = sqlite3.connect(str(out_db))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS passages (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                text TEXT
            )
            """
        )
        conn.commit()
        cursor = conn.cursor()
        for inserted, skipped, batch in _iter_jsonl(corpus_path):
            cursor.executemany(
                "INSERT OR REPLACE INTO passages (doc_id, title, text) VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
