from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

PerGame = Mapping[str, float]
SeedResults = Mapping[int, PerGame]


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    replicates: int
    confidence: float
    resampled_seeds: bool
    seed_count: int
    game_count: int


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def paired_hierarchical_bootstrap(
    treatment: SeedResults,
    control: SeedResults,
    *,
    replicates: int = 50_000,
    confidence: float = 0.95,
    rng_seed: int = 42,
    resample_seeds: bool = True,
) -> BootstrapResult:
    """Paired game bootstrap, optionally including training-seed uncertainty."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    seed_ids = sorted(treatment)
    if not seed_ids or seed_ids != sorted(control):
        raise ValueError("treatment and control must contain the same non-empty seed set")
    game_ids = sorted(treatment[seed_ids[0]])
    if not game_ids:
        raise ValueError("per-game results cannot be empty")
    expected_games = set(game_ids)
    for seed in seed_ids:
        if set(treatment[seed]) != expected_games or set(control[seed]) != expected_games:
            raise ValueError(f"paired game keys differ for seed {seed}")
        values = [*treatment[seed].values(), *control[seed].values()]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite per-game result for seed {seed}")

    def seed_difference(seed: int, games: Sequence[str]) -> float:
        return sum(treatment[seed][game] - control[seed][game] for game in games) / len(games)

    estimate = sum(seed_difference(seed, game_ids) for seed in seed_ids) / len(seed_ids)
    rng = random.Random(rng_seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled_seeds = (
            [rng.choice(seed_ids) for _ in seed_ids] if resample_seeds else list(seed_ids)
        )
        # Seeds and evaluation games are crossed factors. Draw the game
        # clusters once per replicate and reuse them for every sampled seed;
        # independently drawing games inside each seed would break the shared
        # game effect and understate uncertainty.
        sampled_games = [rng.choice(game_ids) for _ in game_ids]
        seed_estimates = [seed_difference(seed, sampled_games) for seed in sampled_seeds]
        draws.append(sum(seed_estimates) / len(seed_estimates))
    draws.sort()
    alpha = 1.0 - confidence
    return BootstrapResult(
        estimate=estimate,
        lower=_quantile(draws, alpha / 2),
        upper=_quantile(draws, 1 - alpha / 2),
        replicates=replicates,
        confidence=confidence,
        resampled_seeds=resample_seeds,
        seed_count=len(seed_ids),
        game_count=len(game_ids),
    )
