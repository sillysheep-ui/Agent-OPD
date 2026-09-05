from omniopd.context import TaskPreservingTruncator
from omniopd.counterfactual import run_paired_counterfactual
from omniopd.history import ConversationHistory
from omniopd.prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT
from omniopd.protocol import ResetResult, StepResult
from omniopd.schema import AgentState


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages) + 1))


class NeverCalledPolicy:
    def generate(self, messages, **kwargs):
        raise AssertionError("both intervention actions terminate immediately")


class BranchEnv:
    def reset(self):
        return ResetResult("fork", ("student", "teacher"), "g", "type", "task")

    def step(self, action):
        return StepResult("terminal", (), True, action == "teacher")

    def close(self):
        pass


def test_counterfactual_replays_both_branches_instead_of_reusing_observed_outcome():
    history = ConversationHistory.start(
        system_prompt="student prompt",
        task="task",
        initial_observation="fork",
        admissible_actions=["student", "teacher"],
        game_id="g",
        task_type="type",
    )
    pair = run_paired_counterfactual(
        env_factory=BranchEnv,
        prefix_actions=[],
        state=history.snapshot(),
        student_action="student",
        teacher_action="teacher",
        frozen_student=NeverCalledPolicy(),
        truncator=TaskPreservingTruncator(FakeTokenizer()),
    )
    assert not pair.student_branch.won
    assert pair.teacher_branch.won
    assert pair.consequence == 1


class RecordingPolicy:
    def __init__(self):
        self.request_ids = []
        self.messages = []

    def generate(self, messages, **kwargs):
        self.request_ids.append(kwargs["request_id"])
        self.messages.append([dict(message) for message in messages])
        return "Action: finish"


class ContinuingEnv:
    def reset(self):
        return ResetResult("fork", ("student", "teacher"), "g", "type", "task")

    def step(self, action):
        if action in {"student", "teacher"}:
            return StepResult("continue", ("finish",), False, False)
        return StepResult("terminal", (), True, True)

    def close(self):
        pass


def test_counterfactual_branch_request_ids_are_unique_and_auditable():
    history = ConversationHistory.start(
        system_prompt="student prompt",
        task="task",
        initial_observation="fork",
        admissible_actions=["student", "teacher"],
        game_id="g",
        task_type="type",
    )
    policy = RecordingPolicy()
    run_paired_counterfactual(
        env_factory=ContinuingEnv,
        prefix_actions=[],
        state=history.snapshot(),
        student_action="student",
        teacher_action="teacher",
        frozen_student=policy,
        truncator=TaskPreservingTruncator(FakeTokenizer()),
        request_namespace="pair:0",
    )
    assert policy.request_ids == [
        "pair:0:student_action_branch:0",
        "pair:0:teacher_action_branch:0",
    ]


class FullHistoryReplayEnv:
    def __init__(self):
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return ResetResult("o0", ("p0",), "g", "type", "task")

    def step(self, action):
        self.step_index += 1
        if self.step_index == 1:
            assert action == "p0"
            return StepResult("o1", ("p1",), False, False)
        if self.step_index == 2:
            assert action == "p1"
            return StepResult("fork", ("student", "teacher"), False, False)
        if self.step_index == 3:
            return StepResult("continue", ("finish",), False, False)
        return StepResult("terminal", (), True, True)

    def close(self):
        pass


def test_counterfactual_continuation_restores_full_history_and_student_prompt():
    full = (
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "Action: p0"},
        {"role": "user", "content": "o1"},
        {"role": "assistant", "content": "Action: p1"},
        {"role": "user", "content": "fork"},
    )
    truncated = (full[0], full[1], full[4], full[5])
    state = AgentState(
        task="task",
        observation="fork",
        admissible_actions=("student", "teacher"),
        messages=truncated,
        game_id="g",
        task_type="type",
        turn_index=2,
        full_messages=full,
        state_source="teacher",
    )
    policy = RecordingPolicy()
    run_paired_counterfactual(
        env_factory=FullHistoryReplayEnv,
        prefix_actions=["p0", "p1"],
        state=state,
        student_action="student",
        teacher_action="teacher",
        frozen_student=policy,
        truncator=TaskPreservingTruncator(FakeTokenizer()),
        request_namespace="restored",
    )
    assert policy.messages[0][0]["content"] == STUDENT_SYSTEM_PROMPT
    assert {message["content"] for message in policy.messages[0]} >= {
        "Action: p0",
        "Action: p1",
    }
