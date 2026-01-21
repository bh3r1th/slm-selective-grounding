from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MixtureCounts:
    support: int
    decoy: int
    conflict: int

    @property
    def total(self) -> int:
        return self.support + self.decoy + self.conflict


def compute_mixture_counts(total: int, ratios: Mapping[str, float]) -> MixtureCounts:
    """Compute deterministic mixture counts from ratios."""
    if total < 0:
        raise ValueError("total must be non-negative")
    required_keys = {"support", "decoy", "conflict"}
    missing = required_keys - set(ratios)
    if missing:
        raise ValueError(f"Missing ratio keys: {sorted(missing)}")
    ratio_sum = sum(float(ratios[key]) for key in required_keys)
    if ratio_sum <= 0:
        raise ValueError("ratio sum must be positive")
    normalized = {key: float(ratios[key]) / ratio_sum for key in required_keys}
    raw_counts = {key: total * normalized[key] for key in required_keys}
    floors = {key: int(raw_counts[key]) for key in required_keys}
    remainder = total - sum(floors.values())
    fractional = sorted(
        required_keys,
        key=lambda key: (raw_counts[key] - floors[key], key),
        reverse=True,
    )
    for key in fractional[:remainder]:
        floors[key] += 1
    return MixtureCounts(
        support=floors["support"],
        decoy=floors["decoy"],
        conflict=floors["conflict"],
    )


def apply_controls(total: int, ratios: Mapping[str, float]) -> MixtureCounts:
    """Apply mixture controls and return final counts."""
    return compute_mixture_counts(total, ratios)
