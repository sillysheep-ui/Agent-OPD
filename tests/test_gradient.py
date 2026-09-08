from omniopd.analysis.gradient import (
    assert_held_out_reference_games,
    assert_shared_teacher_contract,
    gradient_alignment,
    teacher_contract_from_manifest,
)
from scripts.analyze_m1_gradients import (
    _common_retained_game_support,
    _validate_m1_student_identity,
    _selected_game_support,
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
    second_manifest = _teacher_manifest()
    second_manifest["teacher_requests_sha256"] = "different-valid-ledger"
    second = teacher_contract_from_manifest(second_manifest)
    assert_shared_teacher_contract([first, second])
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

    wrong_response_model = _teacher_manifest()
    wrong_response_model["provider_response_models"] = ["another-model"]
    try:
        teacher_contract_from_manifest(wrong_response_model)
    except ValueError as error:
        assert "response.model" in str(error)
    else:
        raise AssertionError("M1 must reject a provider-reported model mismatch")


def test_m1_groups_are_restricted_to_common_teacher_valid_games():
    groups = {
        "A1": [{"game_id": "g1"}, {"game_id": "g2"}],
        "A3": [{"game_id": "g2"}, {"game_id": "g3"}],
    }
    restricted, common, excluded = _common_retained_game_support(groups)
    assert common == ["g2"]
    assert restricted == {
        "A1": [{"game_id": "g2"}],
        "A3": [{"game_id": "g2"}],
    }
    assert excluded == {"A1": ["g1"], "A3": ["g3"]}


def test_m1_held_out_support_uses_pre_validity_selected_games():
    manifest = {
        "distinct_states_M": 3,
        "selected_counts_by_game": {"retained": 1, "all-invalid": 2},
    }
    assert _selected_game_support(manifest) == {"retained", "all-invalid"}


def test_gradient_cosine_rejects_zero_norm_vectors():
    try:
        import torch
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional torch dependency is not installed")

    try:
        gradient_alignment(torch.zeros(2), torch.ones(2))
    except ValueError as error:
        assert "zero-norm" in str(error)
    else:
        raise AssertionError("a zero-norm gradient has no defined cosine")


def test_m1_model_and_reference_bind_to_pair_behavior_student():
    artifact = {
        "kind": "directory",
        "files": 2,
        "bytes": 20,
        "tree_sha256": "student-bytes",
    }
    behavior = {
        "behavior_artifact": artifact,
        "tokenizer": dict(artifact),
    }
    _validate_m1_student_identity(
        base_fingerprint={**artifact, "path": "/copied/model"},
        tokenizer_fingerprint={**artifact, "path": "/copied/model"},
        reference_manifest={"behavior_student": behavior},
        pair_binding={"shared_contract": {"behavior_student": behavior}},
    )
    try:
        _validate_m1_student_identity(
            base_fingerprint={**artifact, "tree_sha256": "other-model"},
            tokenizer_fingerprint=artifact,
            reference_manifest={"behavior_student": behavior},
            pair_binding={"shared_contract": {"behavior_student": behavior}},
        )
    except ValueError as error:
        assert "base/behavior Student" in str(error)
    else:
        raise AssertionError("M1 must not analyze a different Student parameter space")
