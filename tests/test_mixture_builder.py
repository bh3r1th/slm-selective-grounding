from slm_selective_grounding.mixtures.controls import compute_mixture_counts


def test_mixture_ratio_controls() -> None:
    counts = compute_mixture_counts(
        total=10,
        ratios={"support": 0.5, "decoy": 0.3, "conflict": 0.2},
    )
    assert counts.support == 5
    assert counts.decoy == 3
    assert counts.conflict == 2
