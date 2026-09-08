import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from omniopd.io import write_jsonl
from omniopd.prompts import STUDENT_SYSTEM_PROMPT, initial_user_message
from omniopd.provenance import sha256_file
from omniopd.schema import ActionSample, AgentState, CorrectionRecord
from scripts import judge_sage


def _state() -> AgentState:
    user = initial_user_message("put apple away", "room", ("look", "go"))
    messages = (
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )
    return AgentState(
        task="put apple away",
        observation="room",
        admissible_actions=("look", "go"),
        messages=messages,
        full_messages=messages,
        game_id="g1",
        task_type="pick_and_place",
        turn_index=0,
        truncation={"kept_pairs": 0, "dropped_pairs": 0, "token_count": 2, "budget": 8},
        state_source="student",
    )


def _write_source(tmp_path, experiment: str, teacher_action: str):
    student = ActionSample("Action: look", "look", True, True, None)
    teacher = ActionSample(
        f"Action: {teacher_action}", teacher_action, True, True, None
    )
    record = CorrectionRecord(
        state=_state(),
        student=student,
        teacher_samples=[teacher],
        selection_policy="uniform_per_game_nested_v1",
        inclusion_probability=0.5,
        teacher_calls=1,
    )
    corrections = tmp_path / f"{experiment}.jsonl"
    manifest = tmp_path / f"{experiment}.manifest.json"
    write_jsonl(corrections, [record.to_dict()])
    teacher_contract = {
        "teacher_model": "teacher",
        "teacher_model_revision": "snapshot",
        "teacher_url": "https://provider.invalid",
        "teacher_sampling": {
            "thinking_mode": "disabled",
            "temperature": 1.0,
            "reasoning_effort": None,
        },
        "teacher_max_tokens": 100,
        "teacher_context_window": 1000,
        "teacher_tokenizer": {"kind": "directory", "tree_sha256": "tok"},
        "teacher_profile": {"kind": "file", "sha256": "profile"},
        "teacher_prompt_sha256": "prompt",
        "teacher_context_preflight": {"passed": True},
        "provider_response_models": ["teacher"],
        "provider_system_fingerprints": [],
        "teacher_requests_sha256": "ledger",
        "declared_teacher_budget_B": 1,
        "valid_teacher_samples": 1,
        "invalid_teacher_samples": 0,
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
        "selected_state_hashes": [_state().state_hash],
        "selected_counts_by_game": {"g1": 1},
    }
    manifest.write_text(
        json.dumps(
            {
                "artifact": "teacher_corrections",
                "protocol_version": "omniopd-v1",
                "code": {"tree_sha256": "code"},
                "code_revision": "revision",
                "experiment": experiment,
                "state_pool_sha256": "pool",
                "corrections_sha256": sha256_file(corrections),
                "distinct_states_M": 1,
                "actual_teacher_api_calls": 1,
                "teacher_samples_per_state_N": 1,
                "selection_manifest_sha256": f"selection-{experiment}",
                "selection_policies": ["uniform_per_game_nested_v1"],
                **teacher_contract,
            }
        ),
        encoding="utf-8",
    )
    return corrections, manifest


def test_sage_union_judges_overlapping_state_once_and_keeps_memberships():
    with tempfile.TemporaryDirectory() as directory, patch.object(
        judge_sage, "audit_correction_records", lambda _records: []
    ):
        tmp_path = Path(directory)
        left = _write_source(tmp_path, "A1", "look")
        right = _write_source(tmp_path, "A3", "go")
        union, sources, pool = judge_sage._load_sage_union(
            [left[0], right[0]],
            [left[1], right[1]],
            current_code={"tree_sha256": "code"},
            current_revision="revision",
        )
    assert pool == "pool"
    assert len(union) == 1
    assert len(sources) == 2
    memberships = union[0]["memberships"]
    assert [row["experiment"] for row in memberships] == ["A1", "A3"]
    assert [row["disagreement"] for row in memberships] == [0, 1]


def test_sage_union_rejects_different_teacher_contracts():
    with tempfile.TemporaryDirectory() as directory, patch.object(
        judge_sage, "audit_correction_records", lambda _records: []
    ):
        tmp_path = Path(directory)
        left = _write_source(tmp_path, "A1", "look")
        right = _write_source(tmp_path, "A3", "go")
        right_manifest = json.loads(right[1].read_text(encoding="utf-8"))
        right_manifest["teacher_model_revision"] = "different"
        right[1].write_text(json.dumps(right_manifest), encoding="utf-8")
        try:
            judge_sage._load_sage_union(
                [left[0], right[0]],
                [left[1], right[1]],
                current_code={"tree_sha256": "code"},
                current_revision="revision",
            )
        except SystemExit as error:
            assert "Teacher contract" in str(error)
        else:
            raise AssertionError("SAGE union must reject mixed Teacher contracts")


def test_sage_union_recomputes_manifest_claims_from_correction_records():
    with tempfile.TemporaryDirectory() as directory, patch.object(
        judge_sage, "audit_correction_records", lambda _records: []
    ):
        tmp_path = Path(directory)
        source = _write_source(tmp_path, "A1", "look")
        manifest = json.loads(source[1].read_text(encoding="utf-8"))
        manifest["selected_counts_by_game"] = {"invented-game": 1}
        source[1].write_text(json.dumps(manifest), encoding="utf-8")
        try:
            judge_sage._load_sage_union(
                [source[0]],
                [source[1]],
                current_code={"tree_sha256": "code"},
                current_revision="revision",
            )
        except SystemExit as error:
            assert "contradicts its records" in str(error)
        else:
            raise AssertionError("SAGE must reject forged manifest count claims")
