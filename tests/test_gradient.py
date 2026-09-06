from omniopd.analysis.gradient import (
    assert_held_out_reference_games,
    assert_shared_teacher_contract,
    teacher_contract_from_manifest,
)


def test_m1_reference_games_must_be_held_out():
    assert_held_out_reference_games(["train-a"], ["probe-b"])
    try:
        assert_held_out_reference_games(["same"], ["same"])
    except ValueError as error:
        assert "held-out" in str(error)
    else:
        raise AssertionError("selection-complement gradients must not be called G_*")


def _teacher_manifest():
    return {
        "artifact": "teacher_corrections",
        "protocol_version": "omniopd-v1",
        "teacher_model": "teacher",
        "teacher_model_revision": "snapshot",
        "teacher_url": "https://provider",
        "teacher_prompt_sha256": "prompt",
        "teacher_sampling": {
            "thinking_mode": "disabled",
            "temperature": 1.0,
            "reasoning_effort": None,
        },
        "teacher_tokenizer": {"kind": "directory", "tree_sha256": "tok"},
        "teacher_profile": {"kind": "file", "sha256": "profile"},
        "teacher_max_tokens": 256,
        "teacher_context_window": 4096,
        "teacher_context_preflight": {"passed": True},
        "provider_response_models": ["teacher"],
        "provider_system_fingerprints": ["fp"],
        "teacher_requests_sha256": "ledger",
        "distinct_states_M": 2,
        "teacher_samples_per_state_N": 1,
        "declared_teacher_budget_B": 2,
        "actual_teacher_api_calls": 2,
        "valid_teacher_samples": 1,
        "invalid_teacher_samples": 1,
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
        "selected_state_hashes": ["s1", "s2"],
    }


def test_m1_teacher_contract_is_complete_and_shared():
    first = teacher_contract_from_manifest(_teacher_manifest())
    assert_shared_teacher_contract([first, dict(first)])
    changed = _teacher_manifest()
    changed["teacher_model_revision"] = "other"
    try:
        assert_shared_teacher_contract(
            [first, teacher_contract_from_manifest(changed)]
        )
    except ValueError as error:
        assert "exact Teacher contract" in str(error)
    else:
        raise AssertionError("M1 must not mix Teacher revisions")
