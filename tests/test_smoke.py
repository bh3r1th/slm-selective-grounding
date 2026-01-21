from slm_selective_grounding.utils.pipeline import compute_run_id


def test_deterministic_run_id() -> None:
    config = {"seed": 42, "mixtures": {"size": 10}}
    run_id_first = compute_run_id(config, inputs=["a.txt", "b.txt"])
    run_id_second = compute_run_id(config, inputs=["a.txt", "b.txt"])
    assert run_id_first == run_id_second

    run_id_changed = compute_run_id(config, inputs=["a.txt", "c.txt"])
    assert run_id_first != run_id_changed
