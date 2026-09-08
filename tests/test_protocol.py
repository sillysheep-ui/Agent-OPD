from omniopd.context import TaskPreservingTruncator
from omniopd.prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT
from omniopd.protocol import GenerationSettings, ResetResult, StepResult, TeacherBudget, query_teacher, rollout_episode
from omniopd.validation import (
    audit_correction_records,
    validate_correction_manifest_against_records,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages) * 4 + 1))


class Policy:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self.answer


class Env:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        return ResetResult("initial room", ("look",), "game1", "type", "find object")

    def step(self, action):
        self.step_count += 1
        return StepResult("new room", ("look",), True, False)

    def close(self):
        pass


def test_rollout_and_teacher_share_state_but_use_distinct_policy_prompts():
    student = Policy("Action: look")
    turns, _ = rollout_episode(Env(), student, TaskPreservingTruncator(FakeTokenizer()))
    assert "initial room" in turns[0].state.messages[-1]["content"]
    teacher = Policy("Action: look")
    budget = TeacherBudget(1)
    record = query_teacher(
        turns[0].state,
        turns[0].student,
        teacher,
        budget,
        samples_per_state=1,
        selection_policy="uniform_per_game",
        inclusion_probability=1.0,
    )
    assert student.calls[0][0]["content"] == STUDENT_SYSTEM_PROMPT
    assert teacher.calls[0][0]["content"] == TEACHER_SYSTEM_PROMPT
    assert teacher.calls[0][1:] == student.calls[0][1:]
    assert record.teacher_calls == 1
    assert record.state.state_hash == turns[0].state.state_hash
    assert record.metadata["student_teacher_non_system_identity"] is True
    assert record.metadata["teacher_query_sha256"] != record.metadata["student_query_sha256"]
    assert audit_correction_records([record]) == []


def test_correction_audit_rejects_a_forged_teacher_query_attestation():
    student = Policy("Action: look")
    turns, _ = rollout_episode(Env(), student, TaskPreservingTruncator(FakeTokenizer()))
    record = query_teacher(
        turns[0].state,
        turns[0].student,
        Policy("Action: look"),
        TeacherBudget(1),
        samples_per_state=1,
        selection_policy="uniform_per_game_nested_v1",
        inclusion_probability=1.0,
    )
    record.metadata["teacher_query_sha256"] = "forged"
    codes = {issue.code for issue in audit_correction_records([record])}
    assert "TEACHER_QUERY_SHA256_MISMATCH" in codes


def test_correction_audit_rejects_missing_full_history_provenance():
    student = Policy("Action: look")
    turns, _ = rollout_episode(Env(), student, TaskPreservingTruncator(FakeTokenizer()))
    state = turns[0].state
    object.__setattr__(state, "full_messages", ())
    record = query_teacher(
        state,
        turns[0].student,
        Policy("Action: look"),
        TeacherBudget(1),
        samples_per_state=1,
        selection_policy="uniform_per_game_nested_v1",
        inclusion_probability=1.0,
    )
    codes = {issue.code for issue in audit_correction_records([record])}
    assert "FULL_HISTORY_TURN_MISMATCH" in codes


def test_correction_audit_rejects_state_fields_not_matching_the_query_message():
    turns, _ = rollout_episode(
        Env(), Policy("Action: look"), TaskPreservingTruncator(FakeTokenizer())
    )
    state = turns[0].state
    object.__setattr__(state, "observation", "forged observation")
    record = query_teacher(
        state,
        turns[0].student,
        Policy("Action: look"),
        TeacherBudget(1),
        samples_per_state=1,
        selection_policy="uniform_per_game_nested_v1",
        inclusion_probability=1.0,
    )
    codes = {issue.code for issue in audit_correction_records([record])}
    assert "CURRENT_STATE_MESSAGE_MISMATCH" in codes


def test_invalid_teacher_output_still_consumes_budget():
    student = Policy("Action: look")
    turns, _ = rollout_episode(Env(), student, TaskPreservingTruncator(FakeTokenizer()))
    budget = TeacherBudget(1)
    record = query_teacher(
        turns[0].state,
        turns[0].student,
        Policy("not an action"),
        budget,
        samples_per_state=1,
        settings=GenerationSettings(),
        selection_policy="random",
        inclusion_probability=1.0,
    )
    assert budget.used_calls == 1
    assert not record.teacher_samples[0].valid


def test_correction_manifest_counts_are_recomputed_from_records():
    turns, _ = rollout_episode(
        Env(), Policy("Action: look"), TaskPreservingTruncator(FakeTokenizer())
    )
    record = query_teacher(
        turns[0].state,
        turns[0].student,
        Policy("Action: look"),
        TeacherBudget(1),
        samples_per_state=1,
        selection_policy="uniform_per_game_nested_v1",
        inclusion_probability=1.0,
    )
    manifest = {
        "distinct_states_M": 1,
        "teacher_samples_per_state_N": 1,
        "declared_teacher_budget_B": 1,
        "actual_teacher_api_calls": 1,
        "valid_teacher_samples": 1,
        "invalid_teacher_samples": 0,
        "selected_state_hashes": [record.state.state_hash],
        "selected_counts_by_game": {record.state.game_id: 1},
        "selection_policies": [record.selection_policy],
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
    }
    identity = validate_correction_manifest_against_records(manifest, [record])
    assert identity["record_contract"]["actual_teacher_api_calls"] == 1

    manifest["selected_state_hashes"] = ["forged-state"]
    try:
        validate_correction_manifest_against_records(manifest, [record])
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("a manifest cannot invent state membership over fixed bytes")


class FailingPolicy:
    def generate(self, messages, **kwargs):
        raise TimeoutError("simulated timeout")


def test_teacher_request_error_is_ledgered_without_a_free_retry():
    student = Policy("Action: look")
    turns, _ = rollout_episode(Env(), student, TaskPreservingTruncator(FakeTokenizer()))
    budget = TeacherBudget(1)
    record = query_teacher(
        turns[0].state,
        turns[0].student,
        FailingPolicy(),
        budget,
        samples_per_state=1,
        selection_policy="random",
        inclusion_probability=1.0,
    )
    assert budget.used_calls == 1
    assert record.teacher_samples[0].failure_reason == "teacher_request_error:TimeoutError"


def test_teacher_budget_refuses_partial_state_annotation_before_any_call():
    student = Policy("Action: look")
    turns, _ = rollout_episode(Env(), student, TaskPreservingTruncator(FakeTokenizer()))
    teacher = Policy("Action: look")
    budget = TeacherBudget(1)
    try:
        query_teacher(
            turns[0].state,
            turns[0].student,
            teacher,
            budget,
            samples_per_state=2,
            selection_policy="random",
            inclusion_probability=1.0,
        )
    except RuntimeError as error:
        assert "partial annotation" in str(error)
    else:
        raise AssertionError("a state must receive all N samples or none")
    assert budget.used_calls == 0
    assert teacher.calls == []
