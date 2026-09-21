from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .context import TaskPreservingTruncator
from .history import ConversationHistory
from .parser import parse_action
from .prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, replace_system
from .provenance import sha256_json, sha256_text
from .schema import ActionSample, AgentState, CorrectionRecord, RolloutTurn


class ChatPolicy(Protocol):
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int,
        request_id: str,
    ) -> str: ...


@dataclass(frozen=True)
class ResetResult:
    observation: str
    admissible_actions: tuple[str, ...]
    game_id: str
    task_type: str
    task: str


@dataclass(frozen=True)
class StepResult:
    observation: str
    admissible_actions: tuple[str, ...]
    done: bool
    won: bool


class AgentEnvironment(Protocol):
    def reset(self) -> ResetResult: ...

    def step(self, action: str) -> StepResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float | None = 0.0
    max_tokens: int = 256
    fallback_action: str = "look"
    allow_unique_prefix: bool = False

    def __post_init__(self) -> None:
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be None or in [0,2]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not self.fallback_action.strip():
            raise ValueError("fallback_action cannot be empty")


@dataclass
class TeacherBudget:
    maximum_calls: int
    used_calls: int = 0

    def __post_init__(self) -> None:
        if self.maximum_calls < 0 or not 0 <= self.used_calls <= self.maximum_calls:
            raise ValueError("invalid Teacher budget state")

    @property
    def remaining(self) -> int:
        return self.maximum_calls - self.used_calls

    def consume(self) -> None:
        if self.remaining <= 0:
            raise RuntimeError("teacher API budget exhausted")
        self.used_calls += 1


def _sample_action(
    policy: ChatPolicy,
    state: AgentState,
    settings: GenerationSettings,
    *,
    request_id: str,
    require_executable_fallback: bool = True,
) -> ActionSample:
    raw = policy.generate(
        state.messages,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        request_id=request_id,
    )
    parsed = parse_action(
        raw,
        state.admissible_actions,
        allow_unique_prefix=settings.allow_unique_prefix,
    )
    executed = parsed.canonical_action if parsed.valid else settings.fallback_action
    if require_executable_fallback and executed not in state.admissible_actions:
        raise RuntimeError(
            f"fallback action {executed!r} is not admissible for {state.game_id}/t{state.turn_index}"
        )
    return ActionSample(
        raw=raw,
        executed_action=executed,
        valid=parsed.valid,
        had_action_marker=parsed.had_action_marker,
        failure_reason=parsed.failure_reason,
    )


def rollout_episode(
    env: AgentEnvironment,
    student: ChatPolicy,
    truncator: TaskPreservingTruncator,
    *,
    settings: GenerationSettings = GenerationSettings(),
    max_steps: int = 50,
    system_prompt: str = STUDENT_SYSTEM_PROMPT,
    state_source: str = "student",
    user_turn_style: str = "default",
    assistant_history: str = "action_only",
    demonstration: Sequence[tuple[str, str]] | None = None,
) -> tuple[list[RolloutTurn], bool]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    reset = env.reset()
    if not reset.admissible_actions:
        raise RuntimeError("environment reset returned an empty admissible-action set")
    if assistant_history not in {"action_only", "raw"}:
        raise ValueError(f"unsupported assistant_history={assistant_history!r}")
    history = ConversationHistory.start(
        system_prompt=system_prompt,
        task=reset.task,
        initial_observation=reset.observation,
        admissible_actions=reset.admissible_actions,
        game_id=reset.game_id,
        task_type=reset.task_type,
        user_turn_style=user_turn_style,
        demonstration=demonstration,
    )
    turns: list[RolloutTurn] = []
    won = False
    for _ in range(max_steps):
        history.assert_query_ready()
        query, truncation = truncator.truncate(history.messages)
        state = history.snapshot(query, truncation=truncation.__dict__, state_source=state_source)
        sample = _sample_action(
            student,
            state,
            settings,
            request_id=f"behavior:{state_source}:{state.state_hash}",
        )
        turns.append(RolloutTurn(state, sample))
        history.record_action(
            sample.executed_action,
            raw=sample.raw if assistant_history == "raw" else None,
        )
        transition = env.step(sample.executed_action)
        won = transition.won
        if not transition.done and not transition.admissible_actions:
            raise RuntimeError("nonterminal transition returned no admissible actions")
        history.record_observation(
            transition.observation,
            transition.admissible_actions,
            done=transition.done,
        )
        if transition.done:
            break
    return turns, won


def query_teacher(
    state: AgentState,
    student_sample: ActionSample,
    teacher: ChatPolicy,
    budget: TeacherBudget,
    *,
    samples_per_state: int,
    settings: GenerationSettings = GenerationSettings(),
    selection_policy: str,
    inclusion_probability: float | None,
) -> CorrectionRecord:
    """Draw exactly N budget-counted samples; invalid outputs remain observable."""

    if samples_per_state < 1:
        raise ValueError("samples_per_state must be positive")
    if budget.remaining < samples_per_state:
        raise RuntimeError(
            f"Teacher budget has {budget.remaining} calls left but this state requires "
            f"{samples_per_state}; refusing a partial annotation"
        )
    teacher_messages = replace_system(state.messages, TEACHER_SYSTEM_PROMPT)
    teacher_state = AgentState(
        task=state.task,
        observation=state.observation,
        admissible_actions=state.admissible_actions,
        messages=tuple(teacher_messages),
        game_id=state.game_id,
        task_type=state.task_type,
        turn_index=state.turn_index,
        full_messages=state.full_messages,
        truncation=state.truncation,
        state_source=state.state_source,
    )
    samples: list[ActionSample] = []
    request_ids: list[str] = []
    start_calls = budget.used_calls
    for draw in range(samples_per_state):
        request_id = f"teacher:{state.state_hash}:{draw}"
        request_ids.append(request_id)
        budget.consume()
        try:
            samples.append(
                _sample_action(
                    teacher,
                    teacher_state,
                    settings,
                    request_id=request_id,
                    require_executable_fallback=False,
                )
            )
        except Exception as error:
            # The attempt has already consumed black-box budget. Preserve it as
            # an invalid observation rather than silently retrying for free.
            fallback = (
                settings.fallback_action
                if settings.fallback_action in state.admissible_actions
                else (state.admissible_actions[0] if state.admissible_actions else "")
            )
            samples.append(
                ActionSample(
                    raw="",
                    executed_action=fallback,
                    valid=False,
                    had_action_marker=False,
                    failure_reason=f"teacher_request_error:{type(error).__name__}",
                )
            )
    return CorrectionRecord(
        state=state,
        student=student_sample,
        teacher_samples=samples,
        selection_policy=selection_policy,
        inclusion_probability=inclusion_probability,
        teacher_calls=budget.used_calls - start_calls,
        metadata={
            "teacher_prompt_replaced": True,
            "student_query_sha256": sha256_json(list(state.messages)),
            "teacher_query_sha256": sha256_json(teacher_messages),
            "non_system_query_sha256": sha256_json(teacher_messages[1:]),
            "teacher_system_prompt_sha256": sha256_text(TEACHER_SYSTEM_PROMPT),
            "student_teacher_non_system_identity": (
                teacher_messages[1:] == list(state.messages[1:])
            ),
            "budget_counts_invalid_calls": True,
            "samples_per_state": samples_per_state,
            "teacher_request_ids": request_ids,
        },
    )
