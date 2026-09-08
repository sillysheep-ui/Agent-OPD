from omniopd.context import TaskPreservingTruncator
from omniopd.counterfactual import (
    run_paired_counterfactual,
    summarize_position_counterfactuals,
)
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


def _position_row(
    state_hash,
    game_id,
    student_won,
    teacher_won,
    probability,
    *,
    disagreement=1,
):
    return {
        "state_hash": state_hash,
        "game_id": game_id,
        "inclusion_probability": probability,
        "student_action": "student",
        "teacher_action": "teacher" if disagreement else "student",
        "disagreement": disagreement,
        "student_branch": {"won": student_won},
        "teacher_branch": {"won": teacher_won},
        "consequence": int(teacher_won) - int(student_won),
    }


def test_position_summary_averages_draws_within_state_then_weights_games_equally():
    rows = [
        _position_row("s1", "g1", False, True, 0.5),
        _position_row("s1", "g1", False, True, 0.5),
        _position_row("s2", "g2", True, False, 0.5),
    ]
    summary = summarize_position_counterfactuals(
        rows,
        population_states_by_game={"g1": 2, "g2": 2},
        expected_state_hashes=["s1", "s2"],
        bootstrap_replicates=1_000,
        bootstrap_seed=7,
    )
    assert summary["population_identified"]
    assert len(summary["state_summaries"]) == 2
    assert summary["population_game_balanced_ht"]["consequence"] == 0.0
    assert summary["population_game_balanced_ht"]["per_game"]["g1"][
        "consequence"
    ] == 1.0
    assert summary["population_game_balanced_ht"]["per_game"]["g2"][
        "consequence"
    ] == -1.0
    assert summary["paired_game_bootstrap"]["resampling_unit"] == (
        "paired_game_cluster"
    )
    assert summary["paired_game_bootstrap"]["lower"] <= 0.0
    assert summary["paired_game_bootstrap"]["upper"] >= 0.0
    assert summary["primary_population_identified"]
    assert summary["population_game_balanced_ht_given_disagreement"][
        "consequence"
    ] == 0.0


def test_position_population_is_explicitly_unidentified_without_design_probability():
    summary = summarize_position_counterfactuals(
        [_position_row("s1", "g1", False, True, None)],
        population_states_by_game={"g1": 1},
        expected_state_hashes=["s1"],
        bootstrap_replicates=10,
    )
    assert not summary["population_identified"]
    assert summary["population_game_balanced_ht"] is None
    assert summary["paired_game_bootstrap"] is None
    assert "inclusion_probability" in summary["nonidentification_reasons"][0]


def test_position_primary_estimand_is_conditioned_on_disagreement():
    rows = [
        _position_row("s1", "g1", False, True, 1.0),
        _position_row("s2", "g1", False, False, 1.0, disagreement=0),
        _position_row("s3", "g2", False, True, 1.0),
        _position_row("s4", "g2", False, False, 1.0, disagreement=0),
    ]
    summary = summarize_position_counterfactuals(
        rows,
        population_states_by_game={"g1": 2, "g2": 2},
        expected_state_hashes=["s1", "s2", "s3", "s4"],
        bootstrap_replicates=100,
        bootstrap_seed=3,
    )
    assert summary["population_game_balanced_ht"]["consequence"] == 0.5
    assert summary["population_game_balanced_ht_given_disagreement"][
        "consequence"
    ] == 1.0


def test_position_population_is_unidentified_when_a_selected_state_has_no_valid_draw():
    summary = summarize_position_counterfactuals(
        [_position_row("s1", "g1", False, True, 1.0)],
        population_states_by_game={"g1": 2},
        expected_state_hashes=["s1", "s2"],
        bootstrap_replicates=10,
    )
    assert not summary["population_identified"]
    assert any(
        "replayable state set differs" in reason
        for reason in summary["nonidentification_reasons"]
    )
