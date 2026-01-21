from __future__ import annotations

from collections.abc import Mapping, Sequence

LABELS = ("support", "decoy", "conflict")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(records: Sequence[Mapping[str, str]] | None = None) -> dict[str, float]:
    """Compute CSP/UCR/FCR/SCC/FRR/CDR metrics for labeled records."""
    if not records:
        return {metric: 0.0 for metric in ("CSP", "UCR", "FCR", "SCC", "FRR", "CDR")}

    totals = {label: 0 for label in LABELS}
    correct = {label: 0 for label in LABELS}
    predictions = {label: 0 for label in LABELS}
    for record in records:
        label = record["label"]
        prediction = record["prediction"]
        if label not in totals or prediction not in predictions:
            raise ValueError("Unknown label or prediction")
        totals[label] += 1
        predictions[prediction] += 1
        if label == prediction:
            correct[label] += 1

    total = sum(totals.values())
    support_total = totals["support"]
    conflict_total = totals["conflict"]

    metrics = {
        "CSP": _safe_div(correct["support"], support_total),
        "UCR": _safe_div(correct["decoy"], totals["decoy"]),
        "FCR": _safe_div(correct["conflict"], conflict_total),
        "SCC": _safe_div(predictions["support"], total),
        "FRR": _safe_div(support_total - correct["support"], support_total),
        "CDR": _safe_div(predictions["conflict"], total),
    }
    return metrics
