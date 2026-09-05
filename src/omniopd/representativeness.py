from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence


def probability_mass(values: Iterable[Hashable]) -> dict[Hashable, float]:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("distribution cannot be empty")
    return {key: value / total for key, value in counts.items()}


def game_balanced_probability_mass(
    clusters: Sequence[Sequence[Hashable]],
) -> dict[Hashable, float]:
    """Average per-game empirical distributions so long games do not dominate."""

    if not clusters or any(len(cluster) == 0 for cluster in clusters):
        raise ValueError("every game cluster must contain at least one observation")
    per_game = [probability_mass(cluster) for cluster in clusters]
    keys = set().union(*(distribution.keys() for distribution in per_game))
    return {
        key: sum(distribution.get(key, 0.0) for distribution in per_game) / len(per_game)
        for key in keys
    }


def jensen_shannon(p: dict[Hashable, float], q: dict[Hashable, float]) -> float:
    keys = set(p) | set(q)
    midpoint = {key: (p.get(key, 0.0) + q.get(key, 0.0)) / 2 for key in keys}

    def kl(left: dict[Hashable, float], right: dict[Hashable, float]) -> float:
        return sum(
            value * math.log(value / right[key])
            for key, value in left.items()
            if value > 0
        )

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def fixed_width_bin(value: float, *, minimum: float, maximum: float, bins: int) -> int:
    if bins <= 0 or maximum <= minimum:
        raise ValueError("invalid bin specification")
    clipped = min(max(value, minimum), maximum)
    if clipped == maximum:
        return bins - 1
    return int((clipped - minimum) / (maximum - minimum) * bins)


@dataclass(frozen=True)
class JsInterval:
    estimate: float
    lower: float
    upper: float


def bootstrap_js(
    reference_clusters: Sequence[Sequence[Hashable]],
    selected_clusters: Sequence[Sequence[Hashable]],
    *,
    replicates: int = 10_000,
    rng_seed: int = 42,
) -> JsInterval:
    """Game-cluster bootstrap for a marginal diagnostic, not a causal test."""

    if not reference_clusters or not selected_clusters:
        raise ValueError("both cluster collections are required")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    def flatten(clusters: Sequence[Sequence[Hashable]]) -> list[Hashable]:
        return [item for cluster in clusters for item in cluster]

    estimate = jensen_shannon(
        probability_mass(flatten(reference_clusters)),
        probability_mass(flatten(selected_clusters)),
    )
    rng = random.Random(rng_seed)
    values = []
    for _ in range(replicates):
        ref = [rng.choice(reference_clusters) for _ in reference_clusters]
        sel = [rng.choice(selected_clusters) for _ in selected_clusters]
        values.append(jensen_shannon(probability_mass(flatten(ref)), probability_mass(flatten(sel))))
    values.sort()
    return JsInterval(estimate, values[int(0.025 * (replicates - 1))], values[int(0.975 * (replicates - 1))])


def paired_game_balanced_bootstrap_js(
    reference_by_game: Mapping[str, Sequence[Hashable]],
    selected_by_game: Mapping[str, Sequence[Hashable]],
    *,
    replicates: int = 10_000,
    rng_seed: int = 42,
) -> JsInterval:
    """Paired game-cluster JS diagnostic under the paper's GB population.

    The two samples must cover the same frozen games. This remains a marginal
    representativeness diagnostic rather than a test of causal utility.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    games = sorted(reference_by_game)
    if not games or set(games) != set(selected_by_game):
        raise ValueError("reference and selected data must cover the same non-empty game set")
    reference = [list(reference_by_game[game]) for game in games]
    selected = [list(selected_by_game[game]) for game in games]
    estimate = jensen_shannon(
        game_balanced_probability_mass(reference),
        game_balanced_probability_mass(selected),
    )
    rng = random.Random(rng_seed)
    draws = []
    for _ in range(replicates):
        sampled_games = [rng.choice(games) for _ in games]
        draws.append(
            jensen_shannon(
                game_balanced_probability_mass(
                    [reference_by_game[game] for game in sampled_games]
                ),
                game_balanced_probability_mass(
                    [selected_by_game[game] for game in sampled_games]
                ),
            )
        )
    draws.sort()
    return JsInterval(
        estimate,
        draws[int(0.025 * (replicates - 1))],
        draws[int(0.975 * (replicates - 1))],
    )
