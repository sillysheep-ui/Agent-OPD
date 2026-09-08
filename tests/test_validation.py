import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from omniopd.provenance import sha256_json
from omniopd.selection import admissible_entropy
from omniopd.validation import (
    compare_control_protocols,
    validate_annotation_run_manifests,
    validate_budget,
    validate_fixed_budget_arms,
    validate_lora_checkpoint_directory,
    validate_selection_manifest_against_rows,
    validate_training_run_manifests,
    validate_uncertainty_score_rows,
)


def _write_test_lora_safetensors(path: Path, *, rank: int = 16) -> None:
    names_and_shapes = [
        ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [rank, 2]),
        ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, rank]),
        ("base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight", [rank, 2]),
        ("base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight", [2, rank]),
    ]
    header = {}
    offset = 0
    for name, shape in names_and_shapes:
        byte_count = shape[0] * shape[1] * 4
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        offset += byte_count
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(
        len(encoded).to_bytes(8, byteorder="little") + encoded + bytes(offset)
    )


def test_fixed_budget_contract():
    validate_budget(states=150, samples_per_state=1, declared_budget=150)
    try:
        validate_budget(states=150, samples_per_state=3, declared_budget=150)
    except ValueError as error:
        assert "fixed-budget violation" in str(error)
    else:
        raise AssertionError("M=150,N=3 must not be labeled B=150")


def test_selection_design_recomputes_per_game_counts_and_inclusion_probabilities():
    pool = {
        "g1": ["g1-s0", "g1-s1", "g1-s2", "g1-s3"],
        "g2": ["g2-s0", "g2-s1", "g2-s2", "g2-s3", "g2-s4"],
    }

    def chosen(game):
        return min(
            pool[game],
            key=lambda state_hash: (
                hashlib.sha256(
                    f"omniopd-uniform-priority-v1\0{42}\0{game}\0{state_hash}".encode()
                ).digest(),
                state_hash,
            ),
        )

    rows = [
        {
            "protocol_version": "omniopd-v1",
            "state_hash": chosen(game),
            "game_id": game,
            "turn_index": 0,
            "selection_policy": "uniform_per_game_nested_v1",
            "inclusion_probability": 1 / total,
            "score": None,
        }
        for game, total in (("g1", 4), ("g2", 5))
    ]
    selected_hashes = sorted(row["state_hash"] for row in rows)
    manifest = {
        "experiment": "depth",
        "policy": "uniform_per_game_nested_v1",
        "seed": 42,
        "games_G": 2,
        "states_per_game": 1,
        "distinct_states_M": 2,
        "population_inference_supported": True,
        "uniform_design": "hash_priority_srswor_nested_within_game_v1",
        "selected_state_hashes": selected_hashes,
        "selected_counts_by_game": {"g1": 1, "g2": 1},
    }
    result = validate_selection_manifest_against_rows(
        manifest,
        rows,
        pool_state_hashes_by_game=pool,
        expected_experiment="depth",
        expected_policy="uniform_per_game_nested_v1",
        expected_seed=42,
        expected_games=2,
        expected_states_per_game=1,
        expected_states=2,
    )
    assert result["selected_counts_by_game"] == {"g1": 1, "g2": 1}

    corrupted_probability = [dict(row) for row in rows]
    corrupted_probability[0]["inclusion_probability"] = 0.5
    try:
        validate_selection_manifest_against_rows(
            manifest,
            corrupted_probability,
            pool_state_hashes_by_game=pool,
            expected_experiment="depth",
            expected_policy="uniform_per_game_nested_v1",
            expected_seed=42,
            expected_games=2,
            expected_states_per_game=1,
            expected_states=2,
        )
    except ValueError as error:
        assert "m/T_g" in str(error)
    else:
        raise AssertionError("a forged design probability must be rejected")

    corrupted_manifest = {**manifest, "selected_counts_by_game": {"g1": 2}}
    try:
        validate_selection_manifest_against_rows(
            corrupted_manifest,
            rows,
            pool_state_hashes_by_game=pool,
            expected_experiment="depth",
            expected_policy="uniform_per_game_nested_v1",
            expected_seed=42,
            expected_games=2,
            expected_states_per_game=1,
            expected_states=2,
        )
    except ValueError as error:
        assert "claims" in str(error)
    else:
        raise AssertionError("manifest counts must be recomputed from selection rows")

    wrong_subset = [dict(row) for row in rows]
    wrong_subset[0]["state_hash"] = next(
        value for value in pool["g1"] if value != rows[0]["state_hash"]
    )
    wrong_subset_manifest = {
        **manifest,
        "selected_state_hashes": sorted(row["state_hash"] for row in wrong_subset),
    }
    try:
        validate_selection_manifest_against_rows(
            wrong_subset_manifest,
            wrong_subset,
            pool_state_hashes_by_game=pool,
            expected_experiment="depth",
            expected_policy="uniform_per_game_nested_v1",
            expected_seed=42,
            expected_games=2,
            expected_states_per_game=1,
            expected_states=2,
        )
    except ValueError as error:
        assert "configured" in str(error)
    else:
        raise AssertionError("correct counts cannot disguise the wrong seeded subset")


def test_top_score_selection_recomputes_full_score_table_and_tie_order():
    pool = {"g1": ["a", "b", "c"], "g2": ["d", "e", "f"]}
    scores = {"a": 1.0, "b": 2.0, "c": 2.0, "d": 0.0, "e": 4.0, "f": 3.0}
    rows = [
        {
            "protocol_version": "omniopd-v1",
            "state_hash": state_hash,
            "game_id": game,
            "turn_index": 0,
            "selection_policy": "top_score_per_game",
            "inclusion_probability": None,
            "score": scores[state_hash],
        }
        for game, state_hash in (("g1", "b"), ("g2", "e"))
    ]
    manifest = {
        "experiment": "top",
        "policy": "top_score_per_game",
        "seed": None,
        "games_G": 2,
        "states_per_game": 1,
        "distinct_states_M": 2,
        "population_inference_supported": False,
        "uniform_design": None,
        "selected_state_hashes": ["b", "e"],
        "selected_counts_by_game": {"g1": 1, "g2": 1},
    }
    validate_selection_manifest_against_rows(
        manifest,
        rows,
        pool_state_hashes_by_game=pool,
        score_by_state=scores,
        expected_experiment="top",
        expected_policy="top_score_per_game",
        expected_seed=None,
        expected_games=2,
        expected_states_per_game=1,
        expected_states=2,
    )
    wrong_tie = [dict(row) for row in rows]
    wrong_tie[0]["state_hash"] = "c"
    wrong_tie[0]["score"] = 2.0
    wrong_manifest = {**manifest, "selected_state_hashes": ["c", "e"]}
    try:
        validate_selection_manifest_against_rows(
            wrong_manifest,
            wrong_tie,
            pool_state_hashes_by_game=pool,
            score_by_state=scores,
            expected_experiment="top",
            expected_policy="top_score_per_game",
            expected_seed=None,
            expected_games=2,
            expected_states_per_game=1,
            expected_states=2,
        )
    except ValueError as error:
        assert "configured" in str(error)
    else:
        raise AssertionError("top-score tie-breaking must follow frozen pool turn order")


def test_uncertainty_scores_recompute_action_entropy_and_state_binding():
    action_scores = {"look": 0.0, "inventory": -1.0}
    row = {
        "protocol_version": "omniopd-v1",
        "state_hash": "s1",
        "game_id": "g1",
        "turn_index": 2,
        "score": admissible_entropy([0.0, -1.0]),
        "action_log_scores": action_scores,
        "score_definition": "entropy_of_softmax_sequence_action_logprob",
    }
    pool = {
        "s1": {
            "game_id": "g1",
            "turn_index": 2,
            "admissible_actions": ["look", "inventory"],
        }
    }
    assert validate_uncertainty_score_rows([row], pool_states=pool) == {
        "s1": row["score"]
    }
    corrupted = {**row, "score": row["score"] + 0.1}
    try:
        validate_uncertainty_score_rows([corrupted], pool_states=pool)
    except ValueError as error:
        assert "entropy" in str(error)
    else:
        raise AssertionError("reported entropy must be recomputed from action scores")
    wrong_metric = {**row, "score_definition": "maximum_log_probability"}
    try:
        validate_uncertainty_score_rows([wrong_metric], pool_states=pool)
    except ValueError as error:
        assert "definition" in str(error)
    else:
        raise AssertionError("a different score definition cannot drive entropy selection")


def test_lora_checkpoint_layout_binds_adapter_files_and_hyperparameters():
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": [
                        "model.layers.0.self_attn.q_proj",
                        "model.layers.0.self_attn.v_proj",
                    ],
                    "bias": "none",
                    "modules_to_save": None,
                    "use_dora": False,
                    "use_rslora": False,
                    "rank_pattern": {},
                    "alpha_pattern": {},
                }
            ),
            encoding="utf-8",
        )
        _write_test_lora_safetensors(checkpoint / "adapter_model.safetensors")
        (checkpoint / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        contract = validate_lora_checkpoint_directory(
            checkpoint, expected_rank=16, expected_alpha=32
        )
        assert contract["format"] == "peft_lora_adapter"
        assert contract["adapter_weights"]["filename"] == "adapter_model.safetensors"
        full_model_weight = checkpoint / "model.safetensors"
        full_model_weight.write_bytes(b"not-an-adapter")
        try:
            validate_lora_checkpoint_directory(
                checkpoint, expected_rank=16, expected_alpha=32
            )
        except ValueError as error:
            assert "full-model" in str(error)
        else:
            raise AssertionError("a LoRA checkpoint must reject full-model weights")
        full_model_weight.unlink()
        try:
            validate_lora_checkpoint_directory(
                checkpoint, expected_rank=8, expected_alpha=32
            )
        except ValueError as error:
            assert "shape" in str(error)
        else:
            raise AssertionError("a checkpoint with the wrong LoRA rank must be rejected")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"not-safetensors")
        try:
            validate_lora_checkpoint_directory(
                checkpoint, expected_rank=16, expected_alpha=32
            )
        except ValueError as error:
            assert "safetensors" in str(error)
        else:
            raise AssertionError("arbitrary non-empty bytes are not a LoRA checkpoint")


def test_control_comparison_allows_only_state_source():
    treatment = {
        "state_source": {
            "name": "student",
            "behavior_policy": "student",
            "behavior_temperature": 0.0,
        },
        "teacher": {"temperature": 0.0},
        "training": {"dtype": "bf16"},
    }
    control = {
        "state_source": {
            "name": "teacher",
            "behavior_policy": "teacher",
            "behavior_temperature": 0.5,
        },
        "teacher": {"temperature": 0.5},
        "training": {"dtype": "bf16"},
    }
    assert compare_control_protocols(treatment, control) == [
        "state_source.behavior_temperature",
        "teacher.temperature",
    ]


def test_fixed_budget_arms_also_freeze_sampling_profile_and_optimizer_steps():
    breadth = {
        "experiment": "breadth",
        "teacher_budget_definition": "annotation_api_attempts",
        "games": 50,
        "states_per_game": 3,
        "distinct_states_M": 150,
        "teacher_samples_per_state_N": 1,
        "teacher_budget_B": 150,
        "teacher_sampling_profile": "qT-v1",
        "invalid_policy": "retain_as_missing_no_free_retry",
        "selection": "uniform_per_game_nested_v1",
        "state_pool": {
            "state_source": "student",
            "games_G": 50,
            "seed": 42,
            "environment_seed": 314159,
        },
        "training": {
            "total_optimizer_steps": 100,
            "seeds": [7, 42],
            "data_split_seed": 42,
        },
    }
    depth = {
        "experiment": "depth",
        "teacher_budget_definition": "annotation_api_attempts",
        "games": 50,
        "states_per_game": 1,
        "distinct_states_M": 50,
        "teacher_samples_per_state_N": 3,
        "teacher_budget_B": 150,
        "teacher_sampling_profile": "qT-v1",
        "invalid_policy": "retain_as_missing_no_free_retry",
        "selection": "uniform_per_game_nested_v1",
        "state_pool": {
            "state_source": "student",
            "games_G": 50,
            "seed": 42,
            "environment_seed": 314159,
        },
        "training": {
            "total_optimizer_steps": 100,
            "seeds": [7, 42],
            "data_split_seed": 42,
        },
    }
    validate_fixed_budget_arms([breadth, depth])
    try:
        validate_fixed_budget_arms([breadth, breadth])
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("duplicating one config cannot establish a breadth/depth pair")
    depth["training"]["total_optimizer_steps"] = 300
    try:
        validate_fixed_budget_arms([breadth, depth])
    except ValueError as error:
        assert "optimizer-step budgets" in str(error)
    else:
        raise AssertionError("fixed Teacher calls alone do not ensure a fair training comparison")


def test_fixed_budget_config_rejects_misplaced_seed_fields():
    malformed = {
        "games": 2,
        "states_per_game": 1,
        "distinct_states_M": 2,
        "teacher_samples_per_state_N": 1,
        "teacher_budget_B": 2,
        "teacher_sampling_profile": "same",
        "state_pool": {"games_G": 2, "seeds": [7, 42]},
        "training": {"total_optimizer_steps": 1, "seed": 42},
    }
    try:
        validate_fixed_budget_arms([malformed, malformed])
    except ValueError as error:
        assert "not fully resolved" in str(error)
    else:
        raise AssertionError("pool, split, and replicate seeds must occupy distinct fields")


def test_fixed_budget_arms_reject_hidden_training_or_selection_changes():
    breadth = {
        "experiment": "breadth",
        "games": 50,
        "states_per_game": 3,
        "distinct_states_M": 150,
        "teacher_samples_per_state_N": 1,
        "teacher_budget_B": 150,
        "teacher_budget_definition": "annotation_api_attempts",
        "teacher_sampling_profile": "same",
        "invalid_policy": "retain_as_missing_no_free_retry",
        "selection": "uniform_per_game_nested_v1",
        "state_pool": {
            "state_source": "student",
            "games_G": 50,
            "seed": 42,
            "environment_seed": 314159,
        },
        "training": {
            "total_optimizer_steps": 100,
            "weighting": "state_mean",
            "seeds": [7, 42],
            "data_split_seed": 42,
        },
    }
    depth = {
        **breadth,
        "experiment": "depth",
        "states_per_game": 1,
        "distinct_states_M": 50,
        "teacher_samples_per_state_N": 3,
        "training": {
            "total_optimizer_steps": 100,
            "weighting": "sample_mean",
            "seeds": [7, 42],
            "data_split_seed": 42,
        },
    }
    try:
        validate_fixed_budget_arms([breadth, depth])
    except ValueError as error:
        assert "beyond breadth/depth" in str(error)
    else:
        raise AssertionError("hidden protocol changes must fail fixed-budget validation")


def test_realized_annotation_runs_share_pool_budget_and_sampling_contract():
    breadth_hashes = [f"s{index}" for index in range(150)]
    depth_hashes = breadth_hashes[:50]
    breadth = {
        "distinct_states_M": 150,
        "teacher_samples_per_state_N": 1,
        "declared_teacher_budget_B": 150,
        "actual_teacher_api_calls": 150,
        "state_pool_sha256": "pool",
        "teacher_model": "teacher",
        "teacher_model_revision": "snapshot",
        "teacher_url": "https://teacher.test",
        "code": {"tree_sha256": "code"},
        "code_revision": "revision",
        "teacher_sampling": {"thinking_mode": "disabled", "temperature": 1.0},
        "teacher_max_tokens": 256,
        "teacher_context_window": 8192,
        "teacher_tokenizer": {"tree_sha256": "tokenizer"},
        "teacher_prompt_sha256": "prompt",
        "provider_response_models": ["teacher"],
        "teacher_context_preflight": {"passed": True},
        "selection_policies": ["uniform_per_game_nested_v1"],
        "selected_state_hashes": breadth_hashes,
        "selected_counts_by_game": {f"g{index}": 3 for index in range(50)},
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
    }
    depth = {
        **breadth,
        "distinct_states_M": 50,
        "teacher_samples_per_state_N": 3,
        "selected_state_hashes": depth_hashes,
        "selected_counts_by_game": {f"g{index}": 1 for index in range(50)},
    }
    validate_annotation_run_manifests([breadth, depth])
    depth["state_pool_sha256"] = "other"
    try:
        validate_annotation_run_manifests([breadth, depth])
    except ValueError as error:
        assert "same frozen state pool" in str(error)
    else:
        raise AssertionError("realized fixed-B arms must share one state pool")


def test_realized_annotation_run_rejects_provider_model_alias_drift():
    hashes = ["s1", "s2"]
    common = {
        "distinct_states_M": 2,
        "teacher_samples_per_state_N": 1,
        "declared_teacher_budget_B": 2,
        "actual_teacher_api_calls": 2,
        "state_pool_sha256": "pool",
        "teacher_model": "teacher-snapshot",
        "teacher_model_revision": "revision",
        "teacher_url": "https://teacher.test",
        "code": {"tree_sha256": "code"},
        "code_revision": "revision",
        "teacher_sampling": {"thinking_mode": "disabled", "temperature": 1.0},
        "teacher_max_tokens": 256,
        "teacher_context_window": 8192,
        "teacher_tokenizer": {"tree_sha256": "tokenizer"},
        "teacher_prompt_sha256": "prompt",
        "teacher_context_preflight": {"passed": True},
        "selection_policies": ["uniform_per_game_nested_v1"],
        "selected_state_hashes": hashes,
        "selected_counts_by_game": {"g1": 1, "g2": 1},
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
        "provider_response_models": ["unexpected-alias"],
    }
    try:
        validate_annotation_run_manifests([common, common])
    except ValueError as error:
        assert "requested Teacher model" in str(error)
    else:
        raise AssertionError("a provider-reported model drift must invalidate annotation")


def test_realized_breadth_depth_runs_must_use_nested_state_sets():
    breadth_hashes = [f"s{index}" for index in range(6)]
    common = {
        "declared_teacher_budget_B": 6,
        "actual_teacher_api_calls": 6,
        "state_pool_sha256": "pool",
        "teacher_model": "teacher",
        "teacher_model_revision": "snapshot",
        "teacher_url": "https://teacher.test",
        "code": {"tree_sha256": "code"},
        "code_revision": "revision",
        "teacher_sampling": {"thinking_mode": "disabled", "temperature": 1.0},
        "teacher_max_tokens": 256,
        "teacher_context_window": 8192,
        "teacher_tokenizer": {"tree_sha256": "tokenizer"},
        "teacher_prompt_sha256": "prompt",
        "provider_response_models": ["teacher"],
        "teacher_context_preflight": {"passed": True},
        "selection_policies": ["uniform_per_game_nested_v1"],
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
    }
    breadth = {
        **common,
        "distinct_states_M": 6,
        "teacher_samples_per_state_N": 1,
        "selected_state_hashes": breadth_hashes,
        "selected_counts_by_game": {"g1": 3, "g2": 3},
    }
    depth = {
        **common,
        "distinct_states_M": 2,
        "teacher_samples_per_state_N": 3,
        "selected_state_hashes": ["outside1", "outside2"],
        "selected_counts_by_game": {"g1": 1, "g2": 1},
    }
    try:
        validate_annotation_run_manifests([breadth, depth])
    except ValueError as error:
        assert "not nested" in str(error)
    else:
        raise AssertionError("non-nested breadth/depth runs add avoidable state-sampling noise")


def _digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file(label):
    return {"kind": "file", "bytes": 10, "sha256": _digest(label)}


def _directory(label):
    return {
        "kind": "directory",
        "files": 2,
        "bytes": 20,
        "tree_sha256": _digest(label),
    }


def _training_arm(experiment, states, samples):
    return {
        "comparison_family": "fixed_teacher_budget",
        "experiment": experiment,
        "teacher_budget_definition": "annotation_api_attempts",
        "games": 2,
        "states_per_game": states // 2,
        "distinct_states_M": states,
        "teacher_samples_per_state_N": samples,
        "teacher_budget_B": states * samples,
        "teacher_sampling_profile": "teacher-profile",
        "teacher_max_tokens": 256,
        "invalid_policy": "retain_as_missing_no_free_retry",
        "selection": "uniform_per_game_nested_v1",
        "selection_seed": 42,
        "state_pool_games_G": 2,
        "state_pool_seed": 42,
        "state_pool_contract": {
            "state_source": "student",
            "games_G": 2,
            "seed": 42,
            "environment_seed": 314159,
        },
        "weighting": "game_state_mean",
        "sampling_unit": "uniform_action_row_with_state_weights",
        "data_split_seed": 42,
    }


def _training_launch_pair():
    breadth_arm = _training_arm("breadth", 6, 1)
    depth_arm = _training_arm("depth", 2, 3)

    def member(experiment, arm):
        return {
            "experiment": experiment,
            "annotation_manifest_sha256": _digest(f"annotation:{experiment}"),
            "corrections_sha256": _digest(f"corrections:{experiment}"),
            "experiment_config_sha256": _digest(f"config:{experiment}"),
            "state_pool_sha256": _digest("pool"),
            "state_pool_manifest_sha256": _digest("pool-manifest"),
            "selection_sha256": _digest(f"selection:{experiment}"),
            "selection_manifest_sha256": _digest(
                f"selection-manifest:{experiment}"
            ),
            "distinct_states_M": arm["distinct_states_M"],
            "teacher_samples_per_state_N": arm["teacher_samples_per_state_N"],
            "declared_teacher_budget_B": arm["teacher_budget_B"],
        }

    members = {
        "breadth": member("breadth", breadth_arm),
        "depth": member("depth", depth_arm),
    }
    pair_contract = {
        "comparison_contrast": "breadth_depth",
        "effect_direction": "breadth_minus_depth",
        "arm_roles": {"breadth": "breadth", "depth": "depth"},
        "shared_contract": {
            "code": {"tree_sha256": _digest("code"), "files": 5, "bytes": 50},
            "code_revision": "revision",
            "state_pool_sha256": _digest("pool"),
            "state_pool_manifest_sha256": _digest("pool-manifest"),
            "behavior_student": {
                "state_source": "student",
                "behavior_model": "served-student",
                "behavior_artifact": _directory("base-model"),
                "tokenizer": _directory("base-model"),
                "inference_runtime": "vllm:test",
                "behavior_sampling": {
                    "thinking_mode": "disabled",
                    "temperature": 0.0,
                    "reasoning_effort": None,
                },
                "prompt_sha256": _digest("student-prompt"),
                "provider_response_models": ["served-student"],
                "behavior_service_manifest_sha256": _digest(
                    "state-pool-service-manifest"
                ),
                "behavior_service_attestation": {
                    "attestation_scope": "local_wrapper_child_process",
                    "service_role": "student_state_pool",
                    "inference_runtime": "vllm:test",
                    "served_model": "served-student",
                    "max_model_len": 8192,
                    "model_artifact": _directory("base-model"),
                    "tokenizer_artifact": _directory("base-model"),
                    "launch_command_sha256": _digest("state-pool-launch"),
                    "trust_boundary": (
                        "host_local_process_observation_not_cryptographic_remote_attestation"
                    ),
                },
            },
            "teacher_model": "teacher",
            "teacher_model_revision": "teacher-revision",
            "teacher_url": "https://teacher.test/v1",
            "teacher_sampling": {"thinking_mode": "disabled", "temperature": 1.0},
            "teacher_max_tokens": 256,
            "teacher_context_window": 8192,
            "teacher_tokenizer": _directory("teacher-tokenizer"),
            "teacher_prompt_sha256": _digest("teacher-prompt"),
            "provider_response_models": ["teacher"],
            "provider_system_fingerprints": ["provider-build"],
            "selection_policies": ["uniform_per_game_nested_v1"],
            "teacher_budget_definition": "annotation_api_attempts",
            "invalid_policy": "retain_as_missing_no_free_retry",
            "invalid_calls_count_toward_budget": True,
            "free_retries": 0,
        },
        "members": members,
    }
    pair_sha = _digest("pair-file")

    def launch(experiment, arm):
        inputs = {
            "train_files": _file(f"train:{experiment}"),
            "val_files": _file(f"val:{experiment}"),
            "model_path": _directory("base-model"),
            "experiment_config": _file(f"config:{experiment}"),
            "train_audit": _file(f"audit:{experiment}"),
            "annotation_pair": {"kind": "file", "bytes": 10, "sha256": pair_sha},
        }
        binding = {
            "pair_schema_version": 2,
            "annotation_pair_manifest_sha256": pair_sha,
            "pair_contract_sha256": sha256_json(pair_contract),
            "comparison_contrast": "breadth_depth",
            "effect_direction": "breadth_minus_depth",
            "arm_roles": pair_contract["arm_roles"],
            "experiment": experiment,
            "member": members[experiment],
            "pair_contract": pair_contract,
        }
        return {
            "artifact": "omniopd_training_launch",
            "protocol_version": "omniopd-v1",
            "schema_version": 2,
            "experiment": experiment,
            "code": pair_contract["shared_contract"]["code"],
            "code_revision": "revision",
            "implementation": {"trainer": _file("trainer")},
            "training_runtime": {"torch": "test"},
            "git_worktrees_clean": {"omniopd": True, "verl": True},
            "training_contract": {
                "seed": 7,
                "total_optimizer_steps": 102,
                "global_train_batch_size": 4,
                "micro_batch_size_per_gpu": 1,
            },
            "hyperparameters": {
                "learning_rate": "2e-5",
                "max_length": 4096,
                "lora_rank": 16,
                "lora_alpha": 32,
                "target_modules": "all-linear",
                "experiment_name": experiment,
            },
            "inputs": inputs,
            "arm_contract": arm,
            "annotation_pair_binding": binding,
            "user_hydra_overrides": [],
            "python": "3.11",
            "verl_version": "0.4.1",
            "verl_version_declarations": ["0.4.1"],
            "verl_git_revision": "verl-revision",
        }

    return launch("breadth", breadth_arm), launch("depth", depth_arm)


def test_training_run_manifest_comparison_ignores_arm_data_and_run_name():
    baseline, depth = _training_launch_pair()
    validate_training_run_manifests([baseline, depth])
    depth["inputs"] = {
        **depth["inputs"],
        "val_files": _file("depth-specific-sanity-validation"),
    }
    validate_training_run_manifests([baseline, depth])
    depth["training_contract"] = {**depth["training_contract"], "total_optimizer_steps": 99}
    try:
        validate_training_run_manifests([baseline, depth])
    except ValueError as error:
        assert "training runs differ" in str(error)
    else:
        raise AssertionError("training exposure differences must be rejected")


def test_training_run_comparison_rejects_incomplete_or_same_arm_manifests():
    incomplete = {
        "artifact": "omniopd_training_launch",
        "protocol_version": "omniopd-v1",
    }
    try:
        validate_training_run_manifests([incomplete, incomplete])
    except ValueError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("two identical incomplete launches cannot validate")

    breadth, _ = _training_launch_pair()
    try:
        validate_training_run_manifests([breadth, breadth])
    except ValueError as error:
        assert "distinct experiment" in str(error)
    else:
        raise AssertionError("two copies of one arm cannot form a training comparison")
