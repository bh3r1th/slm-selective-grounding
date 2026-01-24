from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from slm_selective_grounding.pipeline import (
    compute_answer_metrics,
    extract_claims,
    generate_answers,
    ground_answers_from_supported,
    join_claims_with_contexts,
    nli_score_pairs,
)


def _resolve_path(arg_path: str | None, default_path: Path) -> Path:
    return Path(arg_path) if arg_path else default_path


def _jsonl_reader(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_phase1(args: argparse.Namespace) -> None:
    out_path = _resolve_path(
        args.phase1_jsonl, Path("artifacts") / f"phase1_gen_{args.tag}.jsonl"
    )
    generate_answers(
        contexts_jsonl=Path(args.contexts_jsonl),
        out_jsonl=out_path,
        model_id="google/gemma-3-1b-it",
        max_input_len=1536,
        max_new_tokens=128,
        do_sample=False,
        prompt_style="base",
    )


def run_phase2(args: argparse.Namespace) -> None:
    inp_path = _resolve_path(
        args.phase1_jsonl, Path("artifacts") / f"phase1_gen_{args.tag}.jsonl"
    )
    out_path = _resolve_path(
        args.phase2_jsonl, Path("artifacts") / f"phase2_claims_{args.tag}.jsonl"
    )
    extract_claims(inp_path, out_path)


def run_phase3(args: argparse.Namespace) -> None:
    claims_path = _resolve_path(
        args.phase2_jsonl, Path("artifacts") / f"phase2_claims_{args.tag}.jsonl"
    )
    pairs_path = _resolve_path(
        args.phase3_jsonl,
        Path("artifacts") / f"phase3_claim_context_pairs_{args.tag}.jsonl",
    )
    join_claims_with_contexts(claims_path, Path(args.contexts_jsonl), pairs_path)

    scores_path = _resolve_path(
        args.phase3_scores_jsonl,
        Path("artifacts") / f"phase3_claim_scores_{args.tag}.jsonl",
    )
    nli_score_pairs(
        pairs_jsonl=pairs_path,
        out_jsonl=scores_path,
        nli_model_id="cross-encoder/nli-deberta-v3-base",
        max_len=512,
    )

    metrics_path = _resolve_path(
        args.phase3_metrics_jsonl,
        Path("artifacts") / f"phase3_answer_metrics_{args.tag}.jsonl",
    )
    compute_answer_metrics(scores_path, metrics_path)


def run_phase4(args: argparse.Namespace) -> None:
    gen_path = _resolve_path(
        args.phase4_jsonl, Path("artifacts") / f"phase4_gen_refusal_{args.tag}.jsonl"
    )
    generate_answers(
        contexts_jsonl=Path(args.contexts_jsonl),
        out_jsonl=gen_path,
        model_id="google/gemma-3-1b-it",
        max_input_len=1536,
        max_new_tokens=128,
        do_sample=False,
        prompt_style="refusal",
    )

    claims_path = _resolve_path(
        args.phase4_claims_jsonl,
        Path("artifacts") / f"phase4_claims_refusal_{args.tag}.jsonl",
    )
    extract_claims(gen_path, claims_path)

    pairs_path = _resolve_path(
        args.phase4_pairs_jsonl,
        Path("artifacts") / f"phase4_claim_context_pairs_refusal_{args.tag}.jsonl",
    )
    join_claims_with_contexts(claims_path, Path(args.contexts_jsonl), pairs_path)

    scores_path = _resolve_path(
        args.phase4_scores_jsonl,
        Path("artifacts") / f"phase4_claim_scores_refusal_{args.tag}.jsonl",
    )
    nli_score_pairs(
        pairs_jsonl=pairs_path,
        out_jsonl=scores_path,
        nli_model_id="cross-encoder/nli-deberta-v3-base",
        max_len=512,
    )


def run_phase5(args: argparse.Namespace) -> None:
    base_scores_path = _resolve_path(
        args.phase3_scores_jsonl,
        Path("artifacts") / f"phase3_claim_scores_{args.tag}.jsonl",
    )
    grounded_path = _resolve_path(
        args.phase5_jsonl,
        Path("artifacts") / f"phase5_grounded_answers_{args.tag}.jsonl",
    )
    ground_answers_from_supported(
        base_scores_path,
        grounded_path,
        all_ids_jsonl=Path(args.contexts_jsonl),
    )
    grounded_rows = sum(1 for _ in _jsonl_reader(grounded_path))
    contexts_rows = sum(1 for _ in _jsonl_reader(Path(args.contexts_jsonl)))
    print(
        "Phase 5 grounded rows: "
        f"{grounded_rows} (contexts: {contexts_rows})"
    )

    claims_path = _resolve_path(
        args.phase5_claims_jsonl,
        Path("artifacts") / f"phase5_claims_grounded_{args.tag}.jsonl",
    )
    extract_claims(grounded_path, claims_path)

    pairs_path = _resolve_path(
        args.phase5_pairs_jsonl,
        Path("artifacts") / f"phase5_claim_context_pairs_{args.tag}.jsonl",
    )
    join_claims_with_contexts(claims_path, Path(args.contexts_jsonl), pairs_path)

    scores_path = _resolve_path(
        args.phase5_scores_jsonl,
        Path("artifacts") / f"phase5_claim_scores_{args.tag}.jsonl",
    )
    nli_score_pairs(
        pairs_jsonl=pairs_path,
        out_jsonl=scores_path,
        nli_model_id="cross-encoder/nli-deberta-v3-base",
        max_len=512,
    )


def run_report(args: argparse.Namespace) -> None:
    def dist(path: Path) -> Counter:
        counts = Counter()
        for ex in _jsonl_reader(path):
            counts[ex["label"]] += 1
        return counts

    def per_id_metrics(path: Path) -> tuple[dict[str, dict[str, float]], int]:
        by_id = {}
        totals = {}
        counts = {}
        for ex in _jsonl_reader(path):
            _id = str(ex["id"])
            totals[_id] = totals.get(_id, 0) + 1
            if _id not in counts:
                counts[_id] = {"support": 0, "conflict": 0, "irrelevant": 0}
            counts[_id][ex["label"]] += 1
        for _id, c in counts.items():
            tot = totals[_id]
            by_id[_id] = {
                "total": tot,
                "support_frac": c["support"] / tot if tot else 0.0,
                "conflict_frac": c["conflict"] / tot if tot else 0.0,
                "irrelevant_frac": c["irrelevant"] / tot if tot else 0.0,
                "conflict": c["conflict"],
            }
        return by_id, len(by_id)

    phase3_scores = _resolve_path(
        args.phase3_scores_jsonl,
        Path("artifacts") / f"phase3_claim_scores_{args.tag}.jsonl",
    )
    phase4_scores = _resolve_path(
        args.phase4_scores_jsonl,
        Path("artifacts") / f"phase4_claim_scores_refusal_{args.tag}.jsonl",
    )
    phase5_scores = _resolve_path(
        args.phase5_scores_jsonl,
        Path("artifacts") / f"phase5_claim_scores_{args.tag}.jsonl",
    )

    counts = {}
    for label, path in [
        ("phase3", phase3_scores),
        ("phase4", phase4_scores),
        ("phase5", phase5_scores),
    ]:
        c = dist(path)
        counts[label] = {
            "support": c["support"],
            "conflict": c["conflict"],
            "irrelevant": c["irrelevant"],
            "total": sum(c.values()),
        }

    print(
        "Phase 3 (base): "
        f"support={counts['phase3']['support']}, "
        f"conflict={counts['phase3']['conflict']}, "
        f"irrelevant={counts['phase3']['irrelevant']}, "
        f"total={counts['phase3']['total']}"
    )
    print(
        "Phase 4 (prompt refusal): "
        f"support={counts['phase4']['support']}, "
        f"conflict={counts['phase4']['conflict']}, "
        f"irrelevant={counts['phase4']['irrelevant']}, "
        f"total={counts['phase4']['total']}"
    )
    print(
        "Phase 5 (claim-level grounding): "
        f"support={counts['phase5']['support']}, "
        f"conflict={counts['phase5']['conflict']}, "
        f"irrelevant={counts['phase5']['irrelevant']}, "
        f"total={counts['phase5']['total']}"
    )

    p3, n3 = per_id_metrics(phase3_scores)
    p4, n4 = per_id_metrics(phase4_scores)
    p5, n5 = per_id_metrics(phase5_scores)
    shared_ids = sorted(set(p3) & set(p4) & set(p5))
    n_ids = len(shared_ids)

    def avg(metrics: dict[str, dict[str, float]], key: str) -> float:
        if not shared_ids:
            return 0.0
        return sum(metrics[i][key] for i in shared_ids) / n_ids

    def avg_all(metrics: dict[str, dict[str, float]], key: str) -> float:
        if not metrics:
            return 0.0
        return sum(m[key] for m in metrics.values()) / len(metrics)

    phase3_macro = {
        "support_frac": avg_all(p3, "support_frac"),
        "conflict_frac": avg_all(p3, "conflict_frac"),
        "irrelevant_frac": avg_all(p3, "irrelevant_frac"),
    }
    phase4_macro = {
        "support_frac": avg_all(p4, "support_frac"),
        "conflict_frac": avg_all(p4, "conflict_frac"),
        "irrelevant_frac": avg_all(p4, "irrelevant_frac"),
    }
    phase5_macro = {
        "support_frac": avg_all(p5, "support_frac"),
        "conflict_frac": avg_all(p5, "conflict_frac"),
        "irrelevant_frac": avg_all(p5, "irrelevant_frac"),
    }

    phase3_partial = sum(m["conflict"] > 0 for m in p3.values())
    phase4_partial = sum(m["conflict"] > 0 for m in p4.values())
    phase5_partial = sum(m["conflict"] > 0 for m in p5.values())

    phase3_shared_macro = {
        "support_frac": avg(p3, "support_frac"),
        "conflict_frac": avg(p3, "conflict_frac"),
        "irrelevant_frac": avg(p3, "irrelevant_frac"),
    }
    phase4_shared_macro = {
        "support_frac": avg(p4, "support_frac"),
        "conflict_frac": avg(p4, "conflict_frac"),
        "irrelevant_frac": avg(p4, "irrelevant_frac"),
    }
    phase5_shared_macro = {
        "support_frac": avg(p5, "support_frac"),
        "conflict_frac": avg(p5, "conflict_frac"),
        "irrelevant_frac": avg(p5, "irrelevant_frac"),
    }

    phase3_shared_partial = sum(p3[i]["conflict"] > 0 for i in shared_ids)
    phase4_shared_partial = sum(p4[i]["conflict"] > 0 for i in shared_ids)
    phase5_shared_partial = sum(p5[i]["conflict"] > 0 for i in shared_ids)

    print(
        f"Phase 3 macro avg (N={n3}): "
        f"support_frac={phase3_macro['support_frac']}, "
        f"conflict_frac={phase3_macro['conflict_frac']}, "
        f"irrelevant_frac={phase3_macro['irrelevant_frac']}, "
        f"partial_hallucinations={phase3_partial}"
    )
    print(
        f"Phase 4 macro avg (N={n4}): "
        f"support_frac={phase4_macro['support_frac']}, "
        f"conflict_frac={phase4_macro['conflict_frac']}, "
        f"irrelevant_frac={phase4_macro['irrelevant_frac']}, "
        f"partial_hallucinations={phase4_partial}"
    )
    print(
        f"Phase 5 macro avg (N={n5}): "
        f"support_frac={phase5_macro['support_frac']}, "
        f"conflict_frac={phase5_macro['conflict_frac']}, "
        f"irrelevant_frac={phase5_macro['irrelevant_frac']}, "
        f"partial_hallucinations={phase5_partial}"
    )
    print(f"Shared ids across all phases: N={n_ids}")
    print(
        "Shared-only macro avg (Phase 3): "
        f"support_frac={phase3_shared_macro['support_frac']}, "
        f"conflict_frac={phase3_shared_macro['conflict_frac']}, "
        f"irrelevant_frac={phase3_shared_macro['irrelevant_frac']}, "
        f"partial_hallucinations={phase3_shared_partial}"
    )
    print(
        "Shared-only macro avg (Phase 4): "
        f"support_frac={phase4_shared_macro['support_frac']}, "
        f"conflict_frac={phase4_shared_macro['conflict_frac']}, "
        f"irrelevant_frac={phase4_shared_macro['irrelevant_frac']}, "
        f"partial_hallucinations={phase4_shared_partial}"
    )
    print(
        "Shared-only macro avg (Phase 5): "
        f"support_frac={phase5_shared_macro['support_frac']}, "
        f"conflict_frac={phase5_shared_macro['conflict_frac']}, "
        f"irrelevant_frac={phase5_shared_macro['irrelevant_frac']}, "
        f"partial_hallucinations={phase5_shared_partial}"
    )

    report_path = Path("artifacts") / f"report_{args.tag}.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        row = {
            "tag": args.tag,
            "ids_compared": n_ids,
            "phase3": {
                "ids": n3,
                "counts": counts["phase3"],
                "macro_avg": phase3_macro,
                "partial_hallucinations": phase3_partial,
            },
            "phase4": {
                "ids": n4,
                "counts": counts["phase4"],
                "macro_avg": phase4_macro,
                "partial_hallucinations": phase4_partial,
            },
            "phase5": {
                "ids": n5,
                "counts": counts["phase5"],
                "macro_avg": phase5_macro,
                "partial_hallucinations": phase5_partial,
            },
            "shared_ids": n_ids,
        }
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase pipeline outputs")
    parser.add_argument("--tag", default="sample100", help="Output tag suffix")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--phase1_jsonl", help="Override phase1 JSONL")
        sub.add_argument("--phase2_jsonl", help="Override phase2 JSONL")
        sub.add_argument("--phase3_jsonl", help="Override phase3 pairs JSONL")
        sub.add_argument("--phase3_scores_jsonl", help="Override phase3 scores JSONL")
        sub.add_argument("--phase3_metrics_jsonl", help="Override phase3 metrics JSONL")
        sub.add_argument("--phase4_jsonl", help="Override phase4 gen JSONL")
        sub.add_argument("--phase4_claims_jsonl", help="Override phase4 claims JSONL")
        sub.add_argument("--phase4_pairs_jsonl", help="Override phase4 pairs JSONL")
        sub.add_argument("--phase4_scores_jsonl", help="Override phase4 scores JSONL")
        sub.add_argument("--phase5_jsonl", help="Override phase5 gen JSONL")
        sub.add_argument("--phase5_claims_jsonl", help="Override phase5 claims JSONL")
        sub.add_argument("--phase5_pairs_jsonl", help="Override phase5 pairs JSONL")
        sub.add_argument("--phase5_scores_jsonl", help="Override phase5 scores JSONL")

    phase1 = subparsers.add_parser("phase1", help="Run Phase 1 generation")
    phase1.add_argument("--contexts_jsonl", required=True, help="Contexts JSONL")
    add_common(phase1)

    phase2 = subparsers.add_parser("phase2", help="Run Phase 2 claim extraction")
    add_common(phase2)

    phase3 = subparsers.add_parser("phase3", help="Run Phase 3 join/scoring")
    phase3.add_argument("--contexts_jsonl", required=True, help="Contexts JSONL")
    add_common(phase3)

    phase4 = subparsers.add_parser("phase4", help="Run Phase 4 refusal pipeline")
    phase4.add_argument("--contexts_jsonl", required=True, help="Contexts JSONL")
    add_common(phase4)

    phase5 = subparsers.add_parser("phase5", help="Run Phase 5 grounded pipeline")
    phase5.add_argument("--contexts_jsonl", required=True, help="Contexts JSONL")
    add_common(phase5)

    report = subparsers.add_parser("report", help="Summarize phase scores")
    add_common(report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "phase1":
        run_phase1(args)
    elif args.command == "phase2":
        run_phase2(args)
    elif args.command == "phase3":
        run_phase3(args)
    elif args.command == "phase4":
        run_phase4(args)
    elif args.command == "phase5":
        run_phase5(args)
    elif args.command == "report":
        run_report(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
