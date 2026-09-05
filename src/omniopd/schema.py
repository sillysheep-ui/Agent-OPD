from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Literal

Message = dict[str, str]


@dataclass(frozen=True)
class AgentState:
    """The observable state z_t=(Task,H_t,O_t,A_t) used by both agents."""

    task: str
    observation: str
    admissible_actions: tuple[str, ...]
    messages: tuple[Message, ...]
    game_id: str
    task_type: str
    turn_index: int
    full_messages: tuple[Message, ...] = ()
    truncation: dict[str, Any] = field(default_factory=dict)
    state_source: str = "student"

    @property
    def state_hash(self) -> str:
        # z_t excludes the policy system prompt P; otherwise Student/Teacher
        # views of the same environment state would receive different hashes.
        payload = {
            "task": self.task,
            "observation": self.observation,
            "admissible_actions": list(self.admissible_actions),
            "history_without_system": [dict(m) for m in (self.full_messages or self.messages)[1:]],
            "game_id": self.game_id,
            "turn_index": self.turn_index,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [dict(m) for m in self.messages]
        data["full_messages"] = [dict(m) for m in self.full_messages]
        data["admissible_actions"] = list(self.admissible_actions)
        data["state_hash"] = self.state_hash
        return data


@dataclass(frozen=True)
class ActionSample:
    raw: str
    executed_action: str
    valid: bool
    had_action_marker: bool
    failure_reason: str | None
    api_calls: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutTurn:
    state: AgentState
    student: ActionSample

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.to_dict(), "student": self.student.to_dict()}


@dataclass
class CorrectionRecord:
    state: AgentState
    student: ActionSample
    teacher_samples: list[ActionSample]
    selection_policy: str
    inclusion_probability: float | None
    teacher_calls: int
    protocol_version: str = "omniopd-v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_teacher_samples(self) -> list[ActionSample]:
        return [sample for sample in self.teacher_samples if sample.valid]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "state": self.state.to_dict(),
            "student": self.student.to_dict(),
            "teacher_samples": [sample.to_dict() for sample in self.teacher_samples],
            "selection_policy": self.selection_policy,
            "inclusion_probability": self.inclusion_probability,
            "teacher_calls": self.teacher_calls,
            "metadata": self.metadata,
        }


WeightingMode = Literal["sample_mean", "state_mean", "game_state_mean"]
