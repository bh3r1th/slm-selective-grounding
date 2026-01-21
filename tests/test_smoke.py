from slm_selective_grounding.utils.hashing import stable_hash


def test_stable_hash() -> None:
    value = stable_hash("hello")
    assert len(value) == 64
