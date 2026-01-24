from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

from slm_selective_grounding.utils.io import ensure_dirs

_NLP = None
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _get_id(ex: dict, fallback: int | None = None) -> str | int | None:
    return ex.get("id") or ex.get("qid") or ex.get("question_id") or fallback


def _jsonl_reader(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _jsonl_writer(path: Path):
    ensure_dirs([path.parent])
    return path.open("w", encoding="utf-8")


def _flatten_context(ctx: Any) -> str:
    if isinstance(ctx, list):
        parts = []
        for item in ctx:
            if isinstance(item, dict):
                parts.append(item.get("text") or str(item))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(ctx)


def _load_spacy():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy  # type: ignore

        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = False
    return _NLP


def load_contexts_index(contexts_jsonl: Path) -> dict[str, Any]:
    ctx_by_id: dict[str, Any] = {}
    for ex in _jsonl_reader(contexts_jsonl):
        _id = _get_id(ex)
        if _id is None:
            continue
        ctx_by_id[str(_id)] = ex.get("contexts") or ex.get("context") or ex.get(
            "retrieval_context"
        )
    return ctx_by_id


def generate_answers(
    contexts_jsonl: Path,
    out_jsonl: Path,
    model_id: str,
    max_input_len: int,
    max_new_tokens: int,
    do_sample: bool,
    prompt_style: Literal["base", "refusal"],
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto",
    )

    def build_prompt(ex: dict) -> str:
        q = ex.get("question") or ex.get("query") or ex.get("prompt") or ""
        ctx = ex.get("contexts") or ex.get("context") or ex.get("retrieval_context") or ""
        if prompt_style == "base":
            return (
                "Answer using only the context. If unsupported, say you don't know.\n\n"
                f"Question: {q}\n\nContext:\n{_flatten_context(ctx)}\n\nAnswer:"
            )
        return (
            "Answer using the context. For each claim you are unsure is supported, "
            "explicitly say: \"I don't know.\" Do not guess.\n\n"
            f"Question: {q}\n\nContext:\n{_flatten_context(ctx)}\n\nAnswer:"
        )

    count = 0
    with _jsonl_writer(out_jsonl) as handle:
        for idx, ex in enumerate(_jsonl_reader(contexts_jsonl)):
            prompt = build_prompt(ex)
            inputs = tok(
                prompt, return_tensors="pt", truncation=True, max_length=max_input_len
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
                )
            text = tok.decode(gen[0], skip_special_tokens=True)
            answer = text.split("Answer:", 1)[-1].strip() if "Answer:" in text else text
            row = {
                "id": _get_id(ex, idx),
                "answer": answer.strip(),
                "model": model_id,
                "mode": "base" if prompt_style == "base" else "selective_refusal",
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    print(f"Wrote: {out_jsonl} rows: {count}")


def extract_claims(answers_jsonl: Path, out_jsonl: Path) -> None:
    nlp = _load_spacy()
    n_in = 0
    n_claims = 0

    def split_sentences(text: str) -> list[str]:
        if nlp:
            return [s.text.strip() for s in nlp(text).sents]
        return [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]

    with _jsonl_writer(out_jsonl) as handle:
        for ex in _jsonl_reader(answers_jsonl):
            ans = ex.get("answer", "") or ""
            sentences = split_sentences(ans)
            claims = [s for s in sentences if len(s) >= 15]
            for i, claim in enumerate(claims):
                row = {
                    "id": _get_id(ex, n_in),
                    "claim_id": i,
                    "claim": claim,
                    "answer": ans,
                }
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
                n_claims += 1
            n_in += 1
    print(f"Read: {answers_jsonl} rows: {n_in}")
    print(f"Wrote: {out_jsonl} claim rows: {n_claims}")


def join_claims_with_contexts(
    claims_jsonl: Path, contexts_jsonl: Path, out_jsonl: Path
) -> None:
    ctx_by_id = load_contexts_index(contexts_jsonl)
    n = 0
    with _jsonl_writer(out_jsonl) as handle:
        for ex in _jsonl_reader(claims_jsonl):
            _id = _get_id(ex)
            ctx = ctx_by_id.get(str(_id)) if _id is not None else None
            if _id is None or ctx is None:
                continue
            row = {
                "id": _id,
                "claim_id": ex["claim_id"],
                "claim": ex["claim"],
                "contexts": ctx,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            n += 1
    print(f"Wrote: {out_jsonl} rows: {n}")


def nli_score_pairs(
    pairs_jsonl: Path, out_jsonl: Path, nli_model_id: str, max_len: int
) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(nli_model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(nli_model_id).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    mdl.eval()

    def score_pair(premise: str, hypothesis: str) -> tuple[str, dict[str, float]]:
        inputs = tok(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=max_len
        ).to(mdl.device)
        with torch.no_grad():
            logits = mdl(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        c, n, e = probs
        if e >= c and e >= n:
            lab = "support"
        elif c >= e and c >= n:
            lab = "conflict"
        else:
            lab = "irrelevant"
        return lab, {"support": e, "irrelevant": n, "conflict": c}

    n = 0
    with _jsonl_writer(out_jsonl) as handle:
        for ex in _jsonl_reader(pairs_jsonl):
            claim = ex["claim"]
            ctx = _flatten_context(ex["contexts"])
            lab, scores = score_pair(ctx, claim)
            row = {
                "id": ex["id"],
                "claim_id": ex["claim_id"],
                "claim": claim,
                "label": lab,
                "scores": scores,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            n += 1
    print(f"Wrote: {out_jsonl} rows: {n}")


def compute_answer_metrics(scores_jsonl: Path, out_jsonl: Path) -> None:
    by_id = defaultdict(lambda: {"support": 0, "conflict": 0, "irrelevant": 0})
    for ex in _jsonl_reader(scores_jsonl):
        by_id[ex["id"]][ex["label"]] += 1

    with _jsonl_writer(out_jsonl) as handle:
        for _id, counts in by_id.items():
            total = sum(counts.values())
            support_frac = counts["support"] / total if total else 0.0
            conflict_frac = counts["conflict"] / total if total else 0.0
            irrelevant_frac = counts["irrelevant"] / total if total else 0.0
            row = {
                "id": _id,
                "total_claims": total,
                "support_frac": support_frac,
                "conflict_frac": conflict_frac,
                "irrelevant_frac": irrelevant_frac,
                "partial_hallucination": conflict_frac > 0,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    print(f"Wrote: {out_jsonl}")


def ground_answers_from_supported(
    scores_jsonl: Path, out_jsonl: Path, all_ids_jsonl: Path | None = None
) -> None:
    by_id: dict[str | int, list[str]] = defaultdict(list)
    for ex in _jsonl_reader(scores_jsonl):
        if ex["label"] == "support":
            by_id[ex["id"]].append(ex["claim"])

    ids: list[str | int] | None = None
    if all_ids_jsonl is not None:
        ids = []
        for idx, ex in enumerate(_jsonl_reader(all_ids_jsonl)):
            _id = _get_id(ex, idx)
            if _id is None:
                continue
            ids.append(_id)

    with _jsonl_writer(out_jsonl) as handle:
        rows_written = 0
        if ids is None:
            id_iter = by_id.items()
        else:
            id_iter = ((_id, by_id.get(_id, [])) for _id in ids)
        for _id, claims in id_iter:
            if claims:
                answer = " ".join(claims)
            else:
                answer = "I don't know."
            row = {
                "id": _id,
                "answer": answer,
                "mode": "claim_level_grounded",
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            rows_written += 1
    print(f"Wrote: {out_jsonl} rows: {rows_written}")
