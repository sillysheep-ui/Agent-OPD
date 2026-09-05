from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import ActionSample, AgentState, CorrectionRecord, RolloutTurn


def record_game_id(data: dict[str, Any]) -> str | None:
    """Read a game id from canonical or legacy record nesting."""

    value = data.get("game_id") or data.get("gamefile")
    if value is None and isinstance(data.get("state"), dict):
        value = data["state"].get("game_id") or data["state"].get("gamefile")
    return str(value) if value is not None else None


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def state_from_dict(state_data: dict[str, Any]) -> AgentState:
    state = AgentState(
        task=state_data["task"],
        observation=state_data["observation"],
        admissible_actions=tuple(state_data["admissible_actions"]),
        messages=tuple(dict(message) for message in state_data["messages"]),
        game_id=state_data["game_id"],
        task_type=state_data["task_type"],
        turn_index=int(state_data["turn_index"]),
        full_messages=tuple(dict(message) for message in state_data.get("full_messages", ())),
        truncation=dict(state_data.get("truncation", {})),
        state_source=state_data.get("state_source", "unknown"),
    )
    recorded_hash = state_data.get("state_hash")
    if recorded_hash is not None and str(recorded_hash) != state.state_hash:
        raise ValueError(
            f"state hash mismatch for {state.game_id}/t{state.turn_index}: "
            "the serialized state was modified or produced by another schema"
        )
    return state


def action_sample_from_dict(data: dict[str, Any]) -> ActionSample:
    return ActionSample(**data)


def rollout_turn_from_dict(data: dict[str, Any]) -> RolloutTurn:
    return RolloutTurn(
        state=state_from_dict(data["state"]),
        student=action_sample_from_dict(data["student"]),
    )


def correction_from_dict(data: dict[str, Any]) -> CorrectionRecord:
    state = state_from_dict(data["state"])
    student = action_sample_from_dict(data["student"])
    return CorrectionRecord(
        state=state,
        student=student,
        teacher_samples=[action_sample_from_dict(sample) for sample in data["teacher_samples"]],
        selection_policy=data["selection_policy"],
        inclusion_probability=(
            None
            if data.get("inclusion_probability") is None
            else float(data["inclusion_probability"])
        ),
        teacher_calls=int(data["teacher_calls"]),
        protocol_version=data.get("protocol_version", "unknown"),
        metadata=dict(data.get("metadata", {})),
    )
