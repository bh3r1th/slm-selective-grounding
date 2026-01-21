from slm_selective_grounding.eval.metrics import compute_metrics


def test_compute_metrics_defaults() -> None:
    metrics = compute_metrics()
    assert "faithfulness" in metrics
    assert "coverage" in metrics
