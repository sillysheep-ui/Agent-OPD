from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

from .schema import RolloutTurn


@dataclass(frozen=True)
class SelectedTurn:
    turn: RolloutTurn
    policy: str
    inclusion_probability: float | None
    score: float | None = None


def select_uniform(
    turns: Sequence[RolloutTurn], count: int, rng: random.Random
) -> list[SelectedTurn]:
    """Uniform without replacement within one game, matching G_GB."""

    if count < 0:
        raise ValueError("count cannot be negative")
    total = len(turns)
    selected_count = min(count, total)
    if selected_count == 0:
        return []
    probability = selected_count / total
    indices = sorted(rng.sample(range(total), selected_count))
    return [SelectedTurn(turns[index], "uniform_per_game", probability) for index in indices]


def select_uniform_nested(
    turns: Sequence[RolloutTurn],
    count: int,
    *,
    seed: int,
    game_id: str,
) -> list[SelectedTurn]:
    """Select an SRSWOR prefix shared by every breadth/depth arm.

    A seed defines one hash-priority ordering per game. Thus the one-state
    depth sample is contained in the corresponding three-state breadth sample,
    and choices in one game do not depend on earlier games or on requested k.
    """

    if count < 0:
        raise ValueError("count cannot be negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    total = len(turns)
    selected_count = min(count, total)
    if selected_count == 0:
        return []

    def priority(turn: RolloutTurn) -> tuple[bytes, str]:
        payload = (
            f"omniopd-uniform-priority-v1\0{seed}\0{game_id}\0"
            f"{turn.state.state_hash}"
        ).encode("utf-8")
        return hashlib.sha256(payload).digest(), turn.state.state_hash

    chosen = sorted(turns, key=priority)[:selected_count]
    probability = selected_count / total
    return [
        SelectedTurn(turn, "uniform_per_game_nested_v1", probability)
        for turn in chosen
    ]


def admissible_entropy(action_log_scores: Sequence[float]) -> float:
    """Stable entropy over the admissible-action softmax."""

    if not action_log_scores:
        raise ValueError("at least one action score is required")
    maximum = max(action_log_scores)
    weights = [math.exp(score - maximum) for score in action_log_scores]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return -sum(p * math.log(p) for p in probabilities if p > 0.0)


def select_top_score(
    turns: Sequence[RolloutTurn], scores: Sequence[float], count: int, *, policy: str
) -> list[SelectedTurn]:
    if len(turns) != len(scores):
        raise ValueError("turns and scores must have equal length")
    if count < 0:
        raise ValueError("count cannot be negative")
    order = sorted(range(len(turns)), key=lambda index: (-scores[index], index))[:count]
    # Deterministic top-k does not have a design-based inclusion probability.
    return [SelectedTurn(turns[index], policy, None, scores[index]) for index in order]
