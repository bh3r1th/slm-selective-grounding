from __future__ import annotations

import json
import re
from pathlib import Path

from slm_selective_grounding.utils.io import ensure_dirs


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def build_wiki_leads_corpus(
    input_path: Path,
    output_path: Path,
    min_chars: int = 200,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing wiki leads file: {input_path}")

    ensure_dirs([output_path.parent])

    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            payload = json.loads(line)
            doc_id = str(payload.get("doc_id", "")).strip()
            title = str(payload.get("title", "")).strip()
            text = normalize_text(str(payload.get("text", "")))
            source = str(payload.get("source", "wiki")).strip() or "wiki"

            if len(text) < min_chars:
                continue
            if not doc_id:
                doc_id = title or f"doc_{written}"

            doc = {"doc_id": doc_id, "title": title, "text": text, "source": source}
            dst.write(json.dumps(doc, ensure_ascii=False))
            dst.write("\n")
            written += 1

    if written == 0:
        raise RuntimeError("No wiki leads written: input file may be empty or too short.")

    return written
