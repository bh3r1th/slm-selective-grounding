from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    """Seed python random generators."""
    random.seed(seed)
