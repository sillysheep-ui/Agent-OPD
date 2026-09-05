from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .dataset import TrainingRow


@dataclass(frozen=True)
class StateGroup:
    state_id: str
    game_id: str
    rows: tuple[TrainingRow, ...]
    state_weight: float


def group_training_rows(rows: Iterable[TrainingRow]) -> list[StateGroup]:
    """Keep all N Teacher actions for one state in the same sampling unit."""

    grouped: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        grouped[row.state_hash].append(row)
    result = []
    for state_id, state_rows in sorted(grouped.items()):
        state_weight = sum(row.state_weight for row in state_rows)
        games = {row.game_id for row in state_rows}
        if len(games) != 1:
            raise ValueError(f"state hash collision across games: {state_id}")
        result.append(StateGroup(state_id, next(iter(games)), tuple(state_rows), state_weight))
    return result


def state_group_batches(
    groups: Iterable[StateGroup], *, states_per_batch: int
) -> list[list[StateGroup]]:
    if states_per_batch <= 0:
        raise ValueError("states_per_batch must be positive")
    groups = list(groups)
    return [groups[index : index + states_per_batch] for index in range(0, len(groups), states_per_batch)]
