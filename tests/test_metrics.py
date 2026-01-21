from slm_selective_grounding.eval.metrics import compute_metrics


def test_metrics_on_tiny_example() -> None:
    records = [
        {"label": "support", "prediction": "support"},
        {"label": "support", "prediction": "decoy"},
        {"label": "decoy", "prediction": "decoy"},
        {"label": "conflict", "prediction": "conflict"},
        {"label": "conflict", "prediction": "support"},
    ]
    metrics = compute_metrics(records)
    assert metrics["CSP"] == 0.5
    assert metrics["UCR"] == 1.0
    assert metrics["FCR"] == 0.5
    assert metrics["SCC"] == 2 / 5
    assert metrics["FRR"] == 0.5
    assert metrics["CDR"] == 1 / 5
