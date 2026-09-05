from omniopd.validation import (
    compare_control_protocols,
    validate_annotation_run_manifests,
    validate_budget,
    validate_fixed_budget_arms,
    validate_training_run_manifests,
)


def test_fixed_budget_contract():
    validate_budget(states=150, samples_per_state=1, declared_budget=150)
    try:
        validate_budget(states=150, samples_per_state=3, declared_budget=150)
    except ValueError as error:
        assert "fixed-budget violation" in str(error)
    else:
        raise AssertionError("M=150,N=3 must not be labeled B=150")


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
        "games": 50,
        "states_per_game": 3,
        "distinct_states_M": 150,
        "teacher_samples_per_state_N": 1,
        "teacher_budget_B": 150,
        "teacher_sampling_profile": "qT-v1",
        "state_pool": {"games_G": 50, "seed": 42},
        "training": {
            "total_optimizer_steps": 100,
            "seeds": [7, 42],
            "data_split_seed": 42,
        },
    }
    depth = {
        "games": 50,
        "states_per_game": 1,
        "distinct_states_M": 50,
        "teacher_samples_per_state_N": 3,
        "teacher_budget_B": 150,
        "teacher_sampling_profile": "qT-v1",
        "state_pool": {"games_G": 50, "seed": 42},
        "training": {
            "total_optimizer_steps": 100,
            "seeds": [7, 42],
            "data_split_seed": 42,
        },
    }
    validate_fixed_budget_arms([breadth, depth])
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
        "teacher_sampling_profile": "same",
        "selection": "uniform_per_game",
        "state_pool": {"games_G": 50, "seed": 42},
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


def test_training_run_manifest_comparison_ignores_arm_data_and_run_name():
    baseline = {
        "artifact": "omniopd_training_launch",
        "protocol_version": "omniopd-v1",
        "implementation": {"trainer": {"sha256": "code"}},
        "training_contract": {"total_optimizer_steps": 102, "compute": "bf16"},
        "hyperparameters": {"learning_rate": "2e-5", "experiment_name": "breadth"},
        "inputs": {
            "train_files": {"sha256": "breadth-data"},
            "val_files": {"sha256": "same-val"},
            "model_path": {"tree_sha256": "same-base"},
        },
        "user_hydra_overrides": [],
        "verl_git_revision": "rev",
    }
    depth = {
        **baseline,
        "hyperparameters": {"learning_rate": "2e-5", "experiment_name": "depth"},
        "inputs": {**baseline["inputs"], "train_files": {"sha256": "depth-data"}},
    }
    validate_training_run_manifests([baseline, depth])
    depth["inputs"] = {
        **depth["inputs"],
        "val_files": {"sha256": "depth-specific-sanity-validation"},
    }
    validate_training_run_manifests([baseline, depth])
    depth["training_contract"] = {"total_optimizer_steps": 99, "compute": "bf16"}
    try:
        validate_training_run_manifests([baseline, depth])
    except ValueError as error:
        assert "training runs differ" in str(error)
    else:
        raise AssertionError("training exposure differences must be rejected")
