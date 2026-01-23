from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _iter_jsonl(path: Path) -> Iterable[dict]:
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


def _write_head_jsonl(source: Path, dest: Path, limit: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8") as src, dest.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            out.write(line)
            written += 1
            if written >= limit:
                break
    return written


def _safe_corpus_name(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return cleaned or "corpus"


def _latest_file(pattern: str) -> Path | None:
    matches = sorted(Path(".").glob(pattern))
    return matches[-1] if matches else None


def _validate_contexts(path: Path, k: int) -> None:
    if not path.exists():
        raise SystemExit(f"Missing contexts output: {path}")
    for idx, row in enumerate(_iter_jsonl(path), start=1):
        contexts = row.get("contexts")
        if not isinstance(contexts, list):
            raise SystemExit(f"Row {idx} missing contexts list")
        if len(contexts) != k:
            raise SystemExit(f"Row {idx} contexts length {len(contexts)} != {k}")
        for ctx in contexts:
            text = ctx.get("text") if isinstance(ctx, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f"Row {idx} has empty context text")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"

    corpus_out = repo_root / "artifacts" / "corpus" / "wiki_smoke.jsonl"
    dataset_src = repo_root / "artifacts" / "corpus" / "asqa_train_qa_with_claims.jsonl"
    dataset_trim = repo_root / "artifacts" / "corpus" / "asqa_train_qa_with_claims_smoke50.jsonl"
    dataset_name = "asqa_train_qa_with_claims_smoke"
    k = 6
    contexts_out = repo_root / "artifacts" / "contexts" / f"{dataset_name}_k{k}.jsonl"
    config_path = repo_root / "configs" / "default.yaml"

    if not dataset_src.exists():
        raise SystemExit(f"Missing source dataset for smoke test: {dataset_src}")

    _run(
        [
            sys.executable,
            str(scripts_dir / "03_build_external_corpus.py"),
            "--out",
            str(corpus_out),
            "--max_articles",
            "2000",
        ]
    )

    _run(
        [
            sys.executable,
            str(scripts_dir / "02_build_retriever_indexes.py"),
            "--config",
            str(config_path),
            "--corpus_path",
            str(corpus_out),
        ]
    )

    rows_written = _write_head_jsonl(dataset_src, dataset_trim, limit=50)
    if rows_written == 0:
        raise SystemExit(f"No rows written to trimmed dataset: {dataset_trim}")

    index_dir = repo_root / "artifacts" / "indexes" / "bm25" / _safe_corpus_name(corpus_out)
    _run(
        [
            sys.executable,
            str(scripts_dir / "04_mix_contexts.py"),
            "--dataset",
            dataset_name,
            "--dataset-path",
            str(dataset_trim),
            "--corpus",
            str(corpus_out),
            "--index-dir",
            str(index_dir),
            "--k",
            str(k),
        ]
    )

    _validate_contexts(contexts_out, k)

    summary = _latest_file("artifacts/retriever_index_*.json")
    print(f"external_corpus={corpus_out}")
    print(f"bm25_index_dir={index_dir}")
    if summary is not None:
        print(f"retriever_summary={summary}")
    print(f"trimmed_dataset={dataset_trim}")
    print(f"contexts_output={contexts_out}")


if __name__ == "__main__":
    main()
