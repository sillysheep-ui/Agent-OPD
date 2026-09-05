from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def bag_of_words(text: str, stopwords: Iterable[str] = ()) -> Counter[str]:
    blocked = {word.lower() for word in stopwords}
    return Counter(
        token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in blocked
    )


def bow_cosine(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def set_jaccard(left: set[str], right: set[str]) -> float | None:
    """Return missing for two empty sets; callers must choose a policy explicitly."""

    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def combined_similarity(
    observation_left: str,
    observation_right: str,
    actions_left: set[str],
    actions_right: set[str],
    *,
    bow_weight: float = 0.5,
    empty_jaccard: float = 0.0,
) -> float:
    if not 0 <= bow_weight <= 1:
        raise ValueError("bow_weight must be in [0,1]")
    bow = bow_cosine(bag_of_words(observation_left), bag_of_words(observation_right))
    jac = set_jaccard(actions_left, actions_right)
    return bow_weight * bow + (1 - bow_weight) * (
        empty_jaccard if jac is None else jac
    )


@dataclass(frozen=True)
class Neighbor:
    state_id: str
    game_id: str
    similarity: float


def top_neighbors(
    candidates: Sequence[Neighbor], *, source_game: str, count: int, minimum_similarity: float
) -> list[Neighbor]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.game_id != source_game and candidate.similarity >= minimum_similarity
    ]
    return sorted(eligible, key=lambda item: (-item.similarity, item.state_id))[:count]
