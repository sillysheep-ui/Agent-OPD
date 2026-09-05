from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .context import TaskPreservingTruncator
from .history import ConversationHistory
from .parser import parse_action
from .prompts import STUDENT_SYSTEM_PROMPT, replace_system
from .protocol import AgentEnvironment, ChatPolicy, GenerationSettings, ResetResult
from .schema import AgentState


@dataclass(frozen=True)
class BranchOutcome:
    intervention_action: str
    won: bool
    done_after_intervention: bool
    continuation_steps: int
    executed_actions: tuple[str, ...]


@dataclass(frozen=True)
class CounterfactualPair:
    student_branch: BranchOutcome
    teacher_branch: BranchOutcome

    @property
    def consequence(self) -> int:
        return int(self.teacher_branch.won) - int(self.student_branch.won)


def _replay_to_state(
    env_factory: Callable[[], AgentEnvironment],
    prefix_actions: Sequence[str],
    expected: AgentState,
) -> tuple[AgentEnvironment, ResetResult]:
    env = env_factory()
    reset = env.reset()
    if reset.game_id != expected.game_id or reset.task_type != expected.task_type:
        env.close()
        raise RuntimeError("state-fidelity gate failed: reset game identity mismatch")
    if reset.task != expected.task:
        env.close()
        raise RuntimeError("state-fidelity gate failed: task mismatch")
    current_observation = reset.observation
    current_actions = reset.admissible_actions
    for action in prefix_actions:
        transition = env.step(action)
        if transition.done:
            env.close()
            raise RuntimeError("prefix terminated before the intervention state")
        current_observation = transition.observation
        current_actions = transition.admissible_actions
    if current_observation != expected.observation:
        env.close()
        raise RuntimeError("state-fidelity gate failed: observation mismatch")
    if tuple(current_actions) != expected.admissible_actions:
        env.close()
        raise RuntimeError("state-fidelity gate failed: ordered admissible actions mismatch")
    return env, reset


def _run_branch(
    env_factory: Callable[[], AgentEnvironment],
    prefix_actions: Sequence[str],
    state: AgentState,
    intervention_action: str,
    policy: ChatPolicy,
    truncator: TaskPreservingTruncator,
    settings: GenerationSettings,
    max_steps: int,
    request_namespace: str,
    branch_label: str,
) -> BranchOutcome:
    if intervention_action not in state.admissible_actions:
        raise ValueError("intervention action is not admissible")
    env, _ = _replay_to_state(env_factory, prefix_actions, state)
    # Reconstruct the continuation from the full executed history. Reusing the
    # already-truncated query would permanently discard older pairs and can
    # change what a fresh policy query retains at later branch steps. The
    # continuation policy is always the Student, including for Teacher-state
    # source controls, so its system prompt must be P_S.
    full_history = state.full_messages or state.messages
    history = ConversationHistory(
        task=state.task,
        game_id=state.game_id,
        task_type=state.task_type,
        _messages=replace_system(full_history, STUDENT_SYSTEM_PROMPT),
        _observation=state.observation,
        _admissible_actions=state.admissible_actions,
        _turn_index=state.turn_index,
    )
    executed = [intervention_action]
    try:
        history.record_action(intervention_action)
        transition = env.step(intervention_action)
        history.record_observation(
            transition.observation, transition.admissible_actions, done=transition.done
        )
        done_after = transition.done
        steps = 0
        while not transition.done and len(prefix_actions) + 1 + steps < max_steps:
            history.assert_query_ready()
            query, _ = truncator.truncate(history.messages)
            raw = policy.generate(
                query,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                request_id=f"{request_namespace}:{branch_label}:{steps}",
            )
            parsed = parse_action(
                raw,
                transition.admissible_actions,
                allow_unique_prefix=settings.allow_unique_prefix,
            )
            action = parsed.canonical_action if parsed.valid else settings.fallback_action
            if action not in transition.admissible_actions:
                raise RuntimeError("counterfactual fallback is not admissible")
            executed.append(action)
            history.record_action(action)
            transition = env.step(action)
            history.record_observation(
                transition.observation, transition.admissible_actions, done=transition.done
            )
            steps += 1
        return BranchOutcome(
            intervention_action,
            transition.won,
            done_after,
            steps,
            tuple(executed),
        )
    finally:
        env.close()


def run_paired_counterfactual(
    *,
    env_factory: Callable[[], AgentEnvironment],
    prefix_actions: Sequence[str],
    state: AgentState,
    student_action: str,
    teacher_action: str,
    frozen_student: ChatPolicy,
    truncator: TaskPreservingTruncator,
    settings: GenerationSettings = GenerationSettings(),
    max_steps: int = 50,
    request_namespace: str | None = None,
) -> CounterfactualPair:
    """Replay two independent branches; the intervention action is the only change."""

    if max_steps <= 0 or len(prefix_actions) >= max_steps:
        raise ValueError("counterfactual intervention must occur within max_steps")
    if len(prefix_actions) != state.turn_index:
        raise ValueError("prefix length must equal the intervention state's turn index")
    namespace = request_namespace or f"cf:{state.state_hash}"
    if not namespace.strip() or any(character in namespace for character in "\r\n"):
        raise ValueError("request_namespace must be a non-empty single-line string")
    if state.full_messages:
        expected_length = 2 + 2 * state.turn_index
        if len(state.full_messages) != expected_length:
            raise ValueError("full state history length is inconsistent with turn_index")
        recorded_actions = []
        for message in state.full_messages[2::2]:
            content = message.get("content", "")
            if message.get("role") != "assistant" or not content.startswith("Action: "):
                raise ValueError("full state history contains a malformed executed action")
            recorded_actions.append(content[len("Action: ") :])
        if list(prefix_actions) != recorded_actions:
            raise ValueError("replay prefix does not match the recorded executed history")

    student = _run_branch(
        env_factory,
        prefix_actions,
        state,
        student_action,
        frozen_student,
        truncator,
        settings,
        max_steps,
        namespace,
        "student_action_branch",
    )
    teacher = _run_branch(
        env_factory,
        prefix_actions,
        state,
        teacher_action,
        frozen_student,
        truncator,
        settings,
        max_steps,
        namespace,
        "teacher_action_branch",
    )
    return CounterfactualPair(student, teacher)
