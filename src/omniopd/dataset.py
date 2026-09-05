from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable
import random

from .schema import CorrectionRecord, Message, WeightingMode
from .prompts import STUDENT_SYSTEM_PROMPT, replace_system


@dataclass(frozen=True)
class TrainingRow:
    messages: list[Message]
    enable_thinking: bool
    state_hash: str
    game_id: str
    task_type: str
    turn_index: int
    state_source: str
    teacher_sample_index: int
    teacher_action: str
    student_action: str
    disagreement: bool
    state_weight: float
    weighting_mode: WeightingMode
    selection_policy: str
    inclusion_probability: float | None
    protocol_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_training_rows(
    records: Iterable[CorrectionRecord],
    *,
    weighting: WeightingMode = "game_state_mean",
) -> list[TrainingRow]:
    """Convert valid sampled actions to action-only SFT rows.

    Weight contract before token normalization:
      sample_mean:      every accepted action has weight 1
      state_mean:       each state sums to 1 (1/K_s per accepted action)
      game_state_mean:  each game sums to 1 (1/(V_g K_s))

    game_state_mean preserves game balance conditional on Teacher acceptance. It
    does not pretend to remove acceptance bias; acceptance rates are audited
    separately.
    """

    records = list(records)
    valid_by_record = [record.valid_teacher_samples for record in records]
    valid_states_per_game: dict[str, int] = defaultdict(int)
    for record, valid_samples in zip(records, valid_by_record):
        if valid_samples:
            valid_states_per_game[record.state.game_id] += 1

    rows: list[TrainingRow] = []
    for record, valid_samples in zip(records, valid_by_record):
        count = len(valid_samples)
        if count == 0:
            continue
        if weighting == "sample_mean":
            base_weight = 1.0
        elif weighting == "state_mean":
            base_weight = 1.0 / count
        elif weighting == "game_state_mean":
            valid_states = valid_states_per_game[record.state.game_id]
            base_weight = 1.0 / (valid_states * count)
        else:
            raise ValueError(f"unknown weighting mode: {weighting}")

        for sample_index, sample in enumerate(record.teacher_samples):
            if not sample.valid:
                continue
            # Training always conditions on P_S. Teacher-state controls may have
            # been rolled out under another policy prompt, but prompt identity is
            # not allowed to become a training confound.
            messages = replace_system(record.state.messages, STUDENT_SYSTEM_PROMPT)
            messages.append(
                {"role": "assistant", "content": f"Action: {sample.executed_action}"}
            )
            rows.append(
                TrainingRow(
                    messages=messages,
                    enable_thinking=False,
                    state_hash=record.state.state_hash,
                    game_id=record.state.game_id,
                    task_type=record.state.task_type,
                    turn_index=record.state.turn_index,
                    state_source=record.state.state_source,
                    teacher_sample_index=sample_index,
                    teacher_action=sample.executed_action,
                    student_action=record.student.executed_action,
                    disagreement=sample.executed_action != record.student.executed_action,
                    state_weight=base_weight,
                    weighting_mode=weighting,
                    selection_policy=record.selection_policy,
                    inclusion_probability=record.inclusion_probability,
                    protocol_version=record.protocol_version,
                )
            )
    return rows


def audit_acceptance(records: Iterable[CorrectionRecord]) -> dict:
    records = list(records)
    by_game: dict[str, list[CorrectionRecord]] = defaultdict(list)
    for record in records:
        by_game[record.state.game_id].append(record)
    selected = len(records)
    valid_states = sum(bool(record.valid_teacher_samples) for record in records)
    calls = sum(record.teacher_calls for record in records)
    valid_samples = sum(len(record.valid_teacher_samples) for record in records)
    retained_games = sorted(
        game for game, group in by_game.items() if any(record.valid_teacher_samples for record in group)
    )
    dropped_games = sorted(set(by_game) - set(retained_games))
    return {
        "selected_states": selected,
        "valid_states": valid_states,
        "state_acceptance_rate": valid_states / selected if selected else None,
        "teacher_api_calls": calls,
        "valid_teacher_samples": valid_samples,
        "sample_acceptance_rate": valid_samples / calls if calls else None,
        "retained_games": retained_games,
        "dropped_games_with_no_valid_targets": dropped_games,
        "games": {
            game: {
                "selected_states": len(group),
                "valid_states": sum(bool(record.valid_teacher_samples) for record in group),
                "state_acceptance_rate": (
                    sum(bool(record.valid_teacher_samples) for record in group)
                    / len(group)
                ),
                "teacher_api_calls": sum(record.teacher_calls for record in group),
                "valid_teacher_samples": sum(
                    len(record.valid_teacher_samples) for record in group
                ),
                "sample_acceptance_rate": (
                    sum(len(record.valid_teacher_samples) for record in group)
                    / sum(record.teacher_calls for record in group)
                    if sum(record.teacher_calls for record in group)
                    else None
                ),
            }
            for game, group in sorted(by_game.items())
        },
        "acceptance_conditioning_disclosed": True,
    }


def split_rows_by_game(
    rows: Iterable[TrainingRow], *, val_fraction: float, rng_seed: int
) -> tuple[list[TrainingRow], list[TrainingRow], dict]:
    """Leakage-safe split: all states/actions from one game stay together."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    rows = list(rows)
    by_game: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        by_game[row.game_id].append(row)
    games = sorted(by_game)
    if len(games) < 2:
        raise ValueError("at least two games are required for a train/validation split")
    random.Random(rng_seed).shuffle(games)
    count = min(len(games) - 1, max(1, round(len(games) * val_fraction)))
    validation_games = set(games[:count])
    train = [row for game in games if game not in validation_games for row in by_game[game]]
    validation = [row for game in games if game in validation_games for row in by_game[game]]
    metadata = {
        "split_unit": "game",
        "rng_seed": rng_seed,
        "val_fraction": val_fraction,
        "train_games": sorted(set(games) - validation_games),
        "validation_games": sorted(validation_games),
    }
    return train, validation, metadata
