import hashlib
import json

from omniopd.evaluation import (
    aggregate_evaluation_results,
    validate_paired_aggregates,
    validate_position_service_attestation_manifest,
    validate_service_attestation_manifest,
    validate_state_pool_service_attestation_manifest,
    validate_training_completion_manifest,
    validate_training_launch_manifest,
)
from omniopd.environment_provenance import derive_environment_seed
from omniopd.provenance import sha256_json


def _arm(experiment="breadth_150x1", *, states=150, samples=1):
    return {
        "comparison_family": "fixed_teacher_budget",
        "experiment": experiment,
        "teacher_budget_definition": "annotation_api_attempts",
        "games": 50,
        "states_per_game": states // 50,
        "distinct_states_M": states,
        "teacher_samples_per_state_N": samples,
        "teacher_budget_B": 150,
        "teacher_sampling_profile": "same",
        "teacher_max_tokens": 256,
        "invalid_policy": "retain_as_missing_no_free_retry",
        "selection": "uniform_per_game_nested_v1",
        "selection_seed": 42,
        "state_pool_games_G": 50,
        "state_pool_seed": 42,
        "state_pool_contract": {
            "state_source": "student",
            "games_G": 50,
            "seed": 42,
            "environment_seed": 314159,
            "max_steps": 50,
            "max_context_tokens": 4096,
            "reserve_tokens": 256,
            "behavior_max_tokens": 256,
            "behavior_sampling": {
                "thinking_mode": "disabled",
                "temperature": 0.0,
                "reasoning_effort": None,
            },
        },
        "weighting": "game_state_mean",
        "sampling_unit": "uniform_action_row_with_state_weights",
        "data_split_seed": 42,
    }


def _file_fingerprint(name, digest):
    return {
        "path": f"/artifacts/{name}",
        "resolved_path": f"/artifacts/{name}",
        "kind": "file",
        "bytes": 10,
        "sha256": digest,
    }


def _directory_fingerprint(name, digest):
    return {
        "path": f"/artifacts/{name}",
        "resolved_path": f"/artifacts/{name}",
        "kind": "directory",
        "files": 2,
        "bytes": 20,
        "tree_sha256": digest,
    }


def _content_identity(value):
    if not isinstance(value, dict):
        return value
    return {
        key: _content_identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair_member(experiment, arm, config_sha=None):
    return {
        "experiment": experiment,
        "annotation_manifest_sha256": _sha(f"manifest:{experiment}"),
        "corrections_sha256": _sha(f"corrections:{experiment}"),
        "experiment_config_sha256": config_sha or _sha(experiment),
        "state_pool_sha256": _sha("state-pool"),
        "state_pool_manifest_sha256": _sha("state-pool-manifest"),
        "selection_sha256": _sha(f"selection:{experiment}"),
        "selection_manifest_sha256": _sha(f"selection-manifest:{experiment}"),
        "distinct_states_M": arm["distinct_states_M"],
        "teacher_samples_per_state_N": arm["teacher_samples_per_state_N"],
        "declared_teacher_budget_B": arm["teacher_budget_B"],
    }


def _pair_binding(experiment, arm, *, config_sha=None, comparison_contrast="breadth_depth"):
    if comparison_contrast == "breadth_depth":
        paired_arms = {
            "breadth_150x1": _arm(),
            "depth_50x3": _arm("depth_50x3", states=50, samples=3),
        }
    else:
        entropy = _arm("entropy_150x1")
        entropy["selection"] = "top_entropy_per_game_v1"
        paired_arms = {"breadth_150x1": _arm(), "entropy_150x1": entropy}
    members = {
        name: _pair_member(name, paired_arm)
        for name, paired_arm in paired_arms.items()
    }
    members[experiment] = _pair_member(experiment, arm, config_sha)
    if comparison_contrast == "breadth_depth":
        arm_roles = {"breadth": "breadth_150x1", "depth": "depth_50x3"}
        effect_direction = "breadth_minus_depth"
    else:
        arm_roles = {"treatment": "breadth_150x1", "control": "entropy_150x1"}
        effect_direction = "treatment_minus_control"
    pair_contract = {
        "comparison_contrast": comparison_contrast,
        "effect_direction": effect_direction,
        "arm_roles": arm_roles,
        "shared_contract": {
            "code": {"tree_sha256": _sha("code"), "files": 1, "bytes": 1},
            "code_revision": "revision",
            "state_pool_sha256": _sha("state-pool"),
            "state_pool_manifest_sha256": _sha("state-pool-manifest"),
            "behavior_student": {
                "state_source": "student",
                "behavior_model": "served-student",
                "behavior_artifact": _content_identity(
                    _directory_fingerprint("base", "base")
                ),
                "tokenizer": _content_identity(
                    _directory_fingerprint("base", "base")
                ),
                "inference_runtime": "vllm:test",
                "behavior_sampling": {
                    "thinking_mode": "disabled",
                    "temperature": 0.0,
                    "reasoning_effort": None,
                },
                "prompt_sha256": _sha("student-prompt"),
                "provider_response_models": ["served-student"],
                "behavior_service_manifest_sha256": _sha(
                    "state-pool-service-manifest"
                ),
                "behavior_service_attestation": {
                    "attestation_scope": "local_wrapper_child_process",
                    "service_role": "student_state_pool",
                    "inference_runtime": "vllm:test",
                    "served_model": "served-student",
                    "max_model_len": 8192,
                    "model_artifact": _content_identity(
                        _directory_fingerprint("base", "base")
                    ),
                    "tokenizer_artifact": _content_identity(
                        _directory_fingerprint("base", "base")
                    ),
                    "launch_command_sha256": _sha("state-pool-launch"),
                    "trust_boundary": (
                        "host_local_process_observation_not_cryptographic_remote_attestation"
                    ),
                },
            },
            "teacher_model": "teacher-snapshot",
            "teacher_model_revision": "teacher-revision",
            "teacher_url": "https://teacher.test/v1",
            "teacher_sampling": {"thinking_mode": "disabled", "temperature": 1.0},
            "teacher_max_tokens": 256,
            "teacher_context_window": 8192,
            "teacher_tokenizer": _content_identity(
                _directory_fingerprint("teacher-tokenizer", _sha("teacher-tokenizer"))
            ),
            "teacher_prompt_sha256": _sha("teacher-prompt"),
            "provider_response_models": ["teacher-snapshot"],
            "provider_system_fingerprints": ["provider-build"],
            "selection_policies": ["uniform_per_game_nested_v1"],
            "teacher_budget_definition": "annotation_api_attempts",
            "invalid_policy": "retain_as_missing_no_free_retry",
            "invalid_calls_count_toward_budget": True,
            "free_retries": 0,
        },
        "members": members,
    }
    return {
        "pair_schema_version": 2,
        "annotation_pair_manifest_sha256": _sha("annotation-pair"),
        "pair_contract_sha256": sha256_json(pair_contract),
        "comparison_contrast": comparison_contrast,
        "effect_direction": effect_direction,
        "arm_roles": arm_roles,
        "experiment": experiment,
        "member": members[experiment],
        "pair_contract": pair_contract,
    }


def _result(seed, outcomes):
    games = list(outcomes)
    experiment = "breadth_150x1"
    arm = _arm(experiment)
    training_inputs = {
        "train_files": _file_fingerprint("train", "train"),
        "val_files": _file_fingerprint("val", "val"),
        "model_path": _directory_fingerprint("base", "base"),
        "experiment_config": _file_fingerprint("config", _sha(experiment)),
        "train_audit": _file_fingerprint("audit", "audit"),
        "annotation_pair": _file_fingerprint(
            "annotation-pair", _sha("annotation-pair")
        ),
    }
    checkpoint = _directory_fingerprint(f"checkpoint-{seed}", f"checkpoint-{seed}")
    environment_seed = 31415
    environment_seeds = {
        game: derive_environment_seed(environment_seed, game) for game in games
    }
    return {
        "protocol_version": "omniopd-v1",
        "code": {"tree_sha256": "same"},
        "code_revision": "revision",
        "training_seed": seed,
        "model": "served-student",
        "experiment": experiment,
        "arm_contract": arm,
        "training_inputs": training_inputs,
        "rollout_seed": 2026,
        "student_decoding_seed": 2026,
        "environment_seed": environment_seed,
        "environment_rollout": {
            "master_seed": environment_seed,
            "derivation": "sha256_omniopd_env_seed_v1_uint32",
            "adapter_seed_method": "textworld_gym_env.seed_before_first_reset",
            "alfred_demangler_shuffle": False,
            "per_game": [
                {"game_id": game, "seed": environment_seeds[game]}
                for game in games
            ],
        },
        "runtime_dependencies": {
            "schema_version": 1,
            "python": {"implementation": "CPython", "version": "test"},
            "distributions": {"alfworld": "test", "textworld": "test"},
        },
        "game_artifacts_sha256": "games-content",
        "split": "eval_out_of_distribution",
        "games": games,
        "game_list_file_sha256": "games",
        "game_list_manifest_sha256": "games-manifest",
        "game_list_sha256": "canonical-games",
        "env_config_sha256": "env",
        "student_prompt_sha256": "prompt",
        "tokenizer": {"path": f"/seed-{seed}/tokenizer", "tree_sha256": "tok"},
        "base_model_artifact": training_inputs["model_path"],
        "checkpoint_artifact": checkpoint,
        "model_artifacts": [training_inputs["model_path"], checkpoint],
        "training_manifest_sha256": f"training-manifest-{seed}",
        "training_launch_manifest_sha256": f"training-manifest-{seed}",
        "training_completion_manifest_sha256": f"completion-{seed}",
        "service_manifest_sha256": f"service-{seed}",
        "training_completion": {
            "experiment": experiment,
            "training_seed": seed,
            "completion_status": "completed",
            "final_global_step": 102,
            "completion_manifest_sha256": f"completion-{seed}",
            "launch_manifest_sha256": f"training-manifest-{seed}",
            "inputs": _content_identity(training_inputs),
            "arm_contract": arm,
        "annotation_pair_binding": _pair_binding(experiment, arm),
            "final_checkpoint": _content_identity(checkpoint),
        },
        "service_attestation": {
            "attestation_scope": "local_wrapper_child_process",
            "inference_runtime": "vllm:test",
            "vllm_version": "test",
            "served_aliases": {"base": "served-base", "adapter": "served-student"},
            "launch_command_sha256": f"command-{seed}",
            "max_model_len": 8192,
            "base_model_artifact": _content_identity(training_inputs["model_path"]),
            "checkpoint_artifact": _content_identity(checkpoint),
            "training_completion_manifest_sha256": f"completion-{seed}",
            "training_launch_manifest_sha256": f"training-manifest-{seed}",
        },
        "training_protocol": {
            "training_contract": {"total_optimizer_steps": 102},
            "base_model": {"tree_sha256": "base"},
        },
        "inference_runtime": "vllm:test",
        "temperature": 0.0,
        "thinking_mode": "disabled",
        "thinking_control": "chat_template",
        "max_steps": 50,
        "max_context_tokens": 4096,
        "reserve_tokens": 256,
        "max_tokens": 256,
        "provider_response_models": ["served-student"],
        "provider_system_fingerprints": [],
        "request_count": len(outcomes),
        "request_ledger_sha256": f"ledger-{seed}",
        "traces": [
            {
                "game_id": game,
                "won": bool(outcomes[game]),
                "environment_rollout_seed": environment_seeds[game],
                "turns": [
                    {
                        "state": {
                            "game_id": game,
                            "state_hash": _sha(f"evaluation:{seed}:{game}"),
                        }
                    }
                ],
            }
            for game in games
        ],
        "per_game": outcomes,
        "success_rate": sum(outcomes.values()) / len(outcomes),
    }


def test_evaluation_aggregation_emits_bootstrap_schema_and_ignores_paths():
    bundle, contract = aggregate_evaluation_results(
        {1: _result(1, {"g1": 1, "g2": 0}), 2: _result(2, {"g1": 0, "g2": 1})}
    )
    assert bundle == {1: {"g1": 1.0, "g2": 0.0}, 2: {"g1": 0.0, "g2": 1.0}}
    assert contract["tokenizer"] == {"tree_sha256": "tok"}


def test_evaluation_aggregation_rejects_different_games():
    try:
        aggregate_evaluation_results(
            {1: _result(1, {"g1": 1}), 2: _result(2, {"g2": 1})}
        )
    except ValueError as error:
        assert "game list" in str(error)
    else:
        raise AssertionError("unpaired games cannot enter a paired bootstrap")


def test_evaluation_aggregation_rejects_different_training_exposure():
    first = _result(1, {"g1": 1})
    second = _result(2, {"g1": 0})
    second["training_protocol"] = {
        "training_contract": {"total_optimizer_steps": 999},
        "base_model": {"tree_sha256": "base"},
    }
    try:
        aggregate_evaluation_results({1: first, 2: second})
    except ValueError as error:
        assert "different rollout/game/code contracts" in str(error)
    else:
        raise AssertionError("different optimizer exposure cannot enter one seed aggregate")


def test_evaluation_aggregation_recomputes_outcomes_from_traces():
    forged = _result(1, {"g1": 1})
    forged["traces"][0]["won"] = False
    try:
        aggregate_evaluation_results({1: forged})
    except ValueError as error:
        assert "trace contradicts" in str(error)
    else:
        raise AssertionError("cached per_game outcomes cannot override the trace")

    malformed = _result(1, {"g1": 1})
    malformed["traces"] = [None]
    try:
        aggregate_evaluation_results({1: malformed})
    except ValueError as error:
        assert "traces" in str(error)
    else:
        raise AssertionError("a malformed trace row must fail cleanly")


def _manifest(
    games=("g1", "g2"),
    contract=None,
    *,
    experiment="breadth_150x1",
    arm=None,
):
    arm = arm or _arm(experiment)
    contract = contract or {
        "split": "ood",
        "temperature": 0.0,
        "code": {
            "tree_sha256": _sha("code"),
            "files": 1,
            "bytes": 1,
        },
        "code_revision": "revision",
        "experiment": experiment,
        "arm_contract": arm,
        "training_inputs": {
            "train_files": {"kind": "file", "bytes": 1, "sha256": experiment},
            "val_files": {"kind": "file", "bytes": 1, "sha256": f"val:{experiment}"},
            "model_path": {
                "kind": "directory",
                "files": 2,
                "bytes": 20,
                "tree_sha256": "base",
            },
            "experiment_config": {
                "kind": "file",
                "bytes": 1,
                "sha256": _sha(experiment),
            },
            "train_audit": {"kind": "file", "bytes": 1, "sha256": f"audit:{experiment}"},
            "annotation_pair": {
                "kind": "file",
                "bytes": 1,
                "sha256": _sha("annotation-pair"),
            },
        },
        "annotation_pair_binding": _pair_binding(
            experiment, arm, config_sha=_sha(experiment)
        ),
    }
    return {
        "protocol_version": "omniopd-v1",
        "artifact": "hierarchical_bootstrap_input",
        "schema_version": 2,
        "code_revision": "revision",
        "experiment": experiment,
        "arm_contract": arm,
        "seeds": [1, 2],
        "games": len(games),
        "game_ids": sorted(games),
        "code": contract.get("code"),
        "evaluation_contract": contract,
        "training_runs": {
            str(seed): {
                "completion_status": "completed",
                "training_seed": seed,
                "experiment": experiment,
                "training_completion_manifest_sha256": f"completion-{seed}",
                "service_manifest_sha256": f"service-{seed}",
                "annotation_pair_contract_sha256": contract.get(
                    "annotation_pair_binding", {}
                ).get("pair_contract_sha256"),
            }
            for seed in (1, 2)
        },
    }


def test_paired_aggregates_require_identical_seed_game_and_rollout_contracts():
    treatment = {1: {"g1": 1, "g2": 0}, 2: {"g1": 0, "g2": 1}}
    control = {1: {"g1": 0, "g2": 0}, 2: {"g1": 0, "g2": 1}}
    pairing = validate_paired_aggregates(
        treatment,
        control,
        _manifest(),
        _manifest(
            experiment="depth_50x3",
            arm=_arm("depth_50x3", states=50, samples=3),
        ),
    )
    assert pairing["seeds"] == [1, 2]
    assert pairing["games"] == ["g1", "g2"]
    assert pairing["comparison_contrast"] == "breadth_depth"
    assert pairing["effect_direction"] == "breadth_minus_depth"

    breadth_manifest = _manifest()
    depth_manifest = _manifest(
        experiment="depth_50x3",
        arm=_arm("depth_50x3", states=50, samples=3),
    )
    try:
        validate_paired_aggregates(
            control,
            treatment,
            depth_manifest,
            breadth_manifest,
        )
    except ValueError as error:
        assert "breadth-minus-depth" in str(error)
    else:
        raise AssertionError("CLI arm order must not be allowed to flip the effect sign")

    changed = _manifest(
        experiment="depth_50x3",
        arm=_arm("depth_50x3", states=50, samples=3),
    )
    changed["evaluation_contract"]["temperature"] = 0.7
    try:
        validate_paired_aggregates(treatment, control, _manifest(), changed)
    except ValueError as error:
        assert "different shared evaluation protocols" in str(error)
    else:
        raise AssertionError("rollout-contract drift must invalidate paired inference")


def test_paired_aggregates_reject_unimplemented_selection_training_contrast():
    treatment = {1: {"g1": 1, "g2": 0}, 2: {"g1": 0, "g2": 1}}
    control = {1: {"g1": 0, "g2": 0}, 2: {"g1": 0, "g2": 1}}
    selected_arm = _arm("entropy_150x1")
    selected_arm["selection"] = "top_entropy_per_game_v1"
    baseline = _manifest()
    baseline["evaluation_contract"]["annotation_pair_binding"] = _pair_binding(
        "breadth_150x1",
        _arm(),
        config_sha=_sha("breadth_150x1"),
        comparison_contrast="state_selection",
    )
    for identity in baseline["training_runs"].values():
        identity["annotation_pair_contract_sha256"] = baseline[
            "evaluation_contract"
        ]["annotation_pair_binding"]["pair_contract_sha256"]
    selected = _manifest(experiment="entropy_150x1", arm=selected_arm)
    selected["evaluation_contract"]["annotation_pair_binding"] = _pair_binding(
        "entropy_150x1",
        selected_arm,
        config_sha=_sha("entropy_150x1"),
        comparison_contrast="state_selection",
    )
    for identity in selected["training_runs"].values():
        identity["annotation_pair_contract_sha256"] = selected[
            "evaluation_contract"
        ]["annotation_pair_binding"]["pair_contract_sha256"]
    try:
        validate_paired_aggregates(
            treatment,
            control,
            baseline,
            selected,
        )
    except ValueError as error:
        assert "not implemented" in str(error)
    else:
        raise AssertionError(
            "an unreachable state-selection training contrast must fail closed"
        )


def test_evaluation_aggregation_rejects_mixed_experiment_arms():
    first = _result(1, {"g1": 1})
    second = _result(2, {"g1": 0})
    second["experiment"] = "depth_50x3"
    second["arm_contract"] = _arm("depth_50x3", states=50, samples=3)
    second["training_completion"]["experiment"] = "depth_50x3"
    second["training_completion"]["arm_contract"] = second["arm_contract"]
    second["training_inputs"]["experiment_config"] = _file_fingerprint(
        "config-depth", _sha("depth_50x3")
    )
    second["training_completion"]["inputs"] = _content_identity(
        second["training_inputs"]
    )
    second["training_completion"]["annotation_pair_binding"] = _pair_binding(
        "depth_50x3", second["arm_contract"]
    )
    try:
        aggregate_evaluation_results({1: first, 2: second})
    except ValueError as error:
        assert "different rollout/game/code contracts" in str(error)
    else:
        raise AssertionError("seeds from different experiment arms cannot be aggregated")


def _launch_and_completion():
    inputs = {
        "train_files": _file_fingerprint("train", "train"),
        "val_files": _file_fingerprint("val", "val"),
        "model_path": _directory_fingerprint("base", "base"),
        "experiment_config": _file_fingerprint("config", _sha("breadth_150x1")),
        "train_audit": _file_fingerprint("audit", "audit"),
        "annotation_pair": _file_fingerprint(
            "annotation-pair", _sha("annotation-pair")
        ),
    }
    launch = {
        "protocol_version": "omniopd-v1",
        "artifact": "omniopd_training_launch",
        "schema_version": 2,
        "experiment": "breadth_150x1",
        "code": {
            "tree_sha256": _sha("code"),
            "files": 1,
            "bytes": 1,
        },
        "code_revision": "revision",
        "inputs": inputs,
        "arm_contract": _arm(),
        "annotation_pair_binding": _pair_binding("breadth_150x1", _arm()),
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
        },
    }
    checkpoint = _directory_fingerprint("checkpoint", "checkpoint")
    completion = {
        "protocol_version": "omniopd-v1",
        "artifact": "omniopd_training_completion",
        "schema_version": 2,
        "completion_status": "completed",
        "experiment": "breadth_150x1",
        "training_seed": 7,
        "final_global_step": 102,
        "code": launch["code"],
        "code_revision": "revision",
        "launch_manifest": _file_fingerprint("launch", "launch-sha"),
        "final_checkpoint": checkpoint,
        "checkpoint_format": {
            "format": "peft_lora_adapter",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "rank": 16,
            "lora_alpha": 32,
            "target_modules_policy": "all-linear",
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "modules_to_save": [],
            "use_dora": False,
            "use_rslora": False,
            "rank_pattern": {},
            "alpha_pattern": {},
            "adapter_config_sha256": _sha("adapter-config"),
            "adapter_weights": {
                "filename": "adapter_model.safetensors",
                "bytes": 10,
                "sha256": _sha("adapter-weights"),
            },
            "adapter_tensors": {
                "pair_count": 2,
                "tensor_count": 4,
                "elements": 128,
                "dtypes": ["F32"],
                "names_sha256": _sha("adapter-tensor-names"),
            },
            "tokenizer_config_sha256": _sha("tokenizer-config"),
        },
        "resolved_trainer_config": _file_fingerprint("resolved", "resolved-sha"),
        "training_inputs_sha256": sha256_json(_content_identity(inputs)),
        "training_contract_sha256": sha256_json(launch["training_contract"]),
        "hyperparameters_sha256": sha256_json(launch["hyperparameters"]),
        "annotation_pair_binding": launch["annotation_pair_binding"],
    }
    return launch, completion, checkpoint


def test_completion_manifest_binds_success_step_inputs_and_checkpoint_bytes():
    launch, completion, checkpoint = _launch_and_completion()
    identity = validate_training_completion_manifest(
        completion,
        launch,
        launch_manifest_sha256="launch-sha",
        completion_manifest_sha256="completion-sha",
        checkpoint_fingerprint=checkpoint,
        current_checkpoint_format=completion["checkpoint_format"],
        resolved_config_fingerprint=_file_fingerprint("resolved", "resolved-sha"),
    )
    assert identity["final_global_step"] == 102
    changed = {**checkpoint, "tree_sha256": "different-bytes"}
    try:
        validate_training_completion_manifest(
            completion,
            launch,
            launch_manifest_sha256="launch-sha",
            completion_manifest_sha256="completion-sha",
            checkpoint_fingerprint=changed,
            current_checkpoint_format=completion["checkpoint_format"],
        )
    except ValueError as error:
        assert "final checkpoint content fingerprint" in str(error)
    else:
        raise AssertionError("a path-correct but byte-different checkpoint must be rejected")

    launch, completion, checkpoint = _launch_and_completion()
    current_checkpoint_format = json.loads(
        json.dumps(completion["checkpoint_format"])
    )
    completion["checkpoint_format"] = {
        **completion["checkpoint_format"],
        "rank": 8,
    }
    try:
        validate_training_completion_manifest(
            completion,
            launch,
            launch_manifest_sha256="launch-sha",
            completion_manifest_sha256="completion-sha",
            checkpoint_fingerprint=checkpoint,
            current_checkpoint_format=current_checkpoint_format,
        )
    except ValueError as error:
        assert "LoRA checkpoint" in str(error)
    else:
        raise AssertionError("completion cannot relabel a rank-8 adapter as rank 16")


def test_launch_and_completion_reject_annotation_pair_substitution():
    launch, completion, checkpoint = _launch_and_completion()
    launch["annotation_pair_binding"] = {
        **launch["annotation_pair_binding"],
        "annotation_pair_manifest_sha256": _sha("another-pair-file"),
    }
    try:
        validate_training_launch_manifest(launch)
    except ValueError as error:
        assert "contract digest" in str(error)
    else:
        raise AssertionError("launch cannot substitute a different pair projection")

    launch, completion, checkpoint = _launch_and_completion()
    completion["annotation_pair_binding"] = {
        **completion["annotation_pair_binding"],
        "experiment": "depth_50x3",
    }
    try:
        validate_training_completion_manifest(
            completion,
            launch,
            launch_manifest_sha256="launch-sha",
            completion_manifest_sha256="completion-sha",
            checkpoint_fingerprint=checkpoint,
            current_checkpoint_format=completion["checkpoint_format"],
        )
    except ValueError as error:
        assert "contradicts" in str(error)
    else:
        raise AssertionError("completion cannot contradict its launch annotation pair")


def test_service_attestation_binds_command_alias_artifacts_and_parent():
    _, _, checkpoint = _launch_and_completion()
    base = _directory_fingerprint("base", "base")
    tokenizer = _directory_fingerprint("tokenizer", "tokenizer")
    wrapper = _file_fingerprint("wrapper", "wrapper")
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base["path"],
        "--tokenizer",
        tokenizer["path"],
        "--served-model-name",
        "served-base",
        "--max-model-len",
        "8192",
        "--port",
        "8000",
        "--enable-lora",
        "--lora-modules",
        f"served-student={checkpoint['path']}",
    ]
    attestation = {
        "protocol_version": "omniopd-v1",
        "artifact": "omniopd_local_vllm_service_attestation",
        "schema_version": 2,
        "attestation_scope": "local_wrapper_child_process",
        "service_role": "checkpoint_evaluation",
        "launch_command": command,
        "launch_command_sha256": sha256_json(command),
        "port": 8000,
        "base_url": "http://127.0.0.1:8000/v1",
        "server_pid": 99,
        "wrapper_pid": 88,
        "served_aliases": {"base": "served-base", "adapter": "served-student"},
        "base_model_artifact": base,
        "tokenizer_artifact": tokenizer,
        "checkpoint_artifact": checkpoint,
        "wrapper": wrapper,
        "training_completion_manifest_sha256": "completion-sha",
        "training_launch_manifest_sha256": "launch-sha",
        "vllm_version": "0.10.2",
        "inference_runtime": "vllm:0.10.2",
        "process_observation": {
            "server_pid": 99,
            "parent_pid": 88,
            "port": 8000,
            "argv": command,
            "argv_sha256": sha256_json(command),
            "listening_owner_pids": [99],
            "trust_boundary": (
                "host_local_process_observation_not_cryptographic_remote_attestation"
            ),
        },
    }
    identity = validate_service_attestation_manifest(
        attestation,
        requested_model="served-student",
        base_url="http://127.0.0.1:8000/v1",
        inference_runtime="vllm:0.10.2",
        base_fingerprint=base,
        checkpoint_fingerprint=checkpoint,
        tokenizer_fingerprint=tokenizer,
        completion_manifest_sha256="completion-sha",
        launch_manifest_sha256="launch-sha",
        wrapper_fingerprint=wrapper,
        expected_parent_pid=88,
    )
    assert identity["served_aliases"]["adapter"] == "served-student"
    attestation["wrapper_pid"] = 87
    try:
        validate_service_attestation_manifest(
            attestation,
            requested_model="served-student",
            base_url="http://127.0.0.1:8000/v1",
            inference_runtime="vllm:0.10.2",
            base_fingerprint=base,
            checkpoint_fingerprint=checkpoint,
            tokenizer_fingerprint=tokenizer,
            completion_manifest_sha256="completion-sha",
            launch_manifest_sha256="launch-sha",
            wrapper_fingerprint=wrapper,
            expected_parent_pid=88,
        )
    except ValueError as error:
        assert "direct child" in str(error)
    else:
        raise AssertionError("a detached/fabricated service attestation must be rejected")

    attestation["wrapper_pid"] = 88
    wrong_tokenizer = {**tokenizer, "tree_sha256": "wrong-tokenizer"}
    try:
        validate_service_attestation_manifest(
            attestation,
            requested_model="served-student",
            base_url="http://127.0.0.1:8000/v1",
            inference_runtime="vllm:0.10.2",
            base_fingerprint=base,
            checkpoint_fingerprint=checkpoint,
            tokenizer_fingerprint=wrong_tokenizer,
            completion_manifest_sha256="completion-sha",
            launch_manifest_sha256="launch-sha",
            wrapper_fingerprint=wrapper,
            expected_parent_pid=88,
        )
    except ValueError as error:
        assert "tokenizer" in str(error)
    else:
        raise AssertionError("a service launched with a different tokenizer must be rejected")


def test_state_pool_service_rejects_wrong_process_owner_and_model_artifact():
    model = _directory_fingerprint("student", "student")
    tokenizer = _directory_fingerprint("tokenizer", "tokenizer")
    wrapper = _file_fingerprint("state-wrapper", "wrapper")
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["path"],
        "--tokenizer",
        tokenizer["path"],
        "--served-model-name",
        "student-alias",
        "--max-model-len",
        "8192",
        "--port",
        "8010",
    ]
    attestation = {
        "protocol_version": "omniopd-v1",
        "artifact": "omniopd_local_vllm_service_attestation",
        "schema_version": 2,
        "attestation_scope": "local_wrapper_child_process",
        "service_role": "student_state_pool",
        "launch_command": command,
        "launch_command_sha256": sha256_json(command),
        "port": 8010,
        "base_url": "http://127.0.0.1:8010/v1",
        "server_pid": 109,
        "wrapper_pid": 108,
        "model_artifact": model,
        "tokenizer_artifact": tokenizer,
        "wrapper": wrapper,
        "vllm_version": "0.10.2",
        "inference_runtime": "vllm:0.10.2",
        "process_observation": {
            "server_pid": 109,
            "parent_pid": 108,
            "port": 8010,
            "argv": command,
            "argv_sha256": sha256_json(command),
            "listening_owner_pids": [109],
            "trust_boundary": (
                "host_local_process_observation_not_cryptographic_remote_attestation"
            ),
        },
    }
    identity = validate_state_pool_service_attestation_manifest(
        attestation,
        requested_model="student-alias",
        base_url="http://127.0.0.1:8010/v1",
        inference_runtime="vllm:0.10.2",
        model_fingerprint=model,
        tokenizer_fingerprint=tokenizer,
        wrapper_fingerprint=wrapper,
        expected_parent_pid=108,
    )
    assert identity["served_model"] == "student-alias"

    attestation["process_observation"]["listening_owner_pids"] = [999]
    try:
        validate_state_pool_service_attestation_manifest(
            attestation,
            requested_model="student-alias",
            base_url="http://127.0.0.1:8010/v1",
            inference_runtime="vllm:0.10.2",
            model_fingerprint=model,
            tokenizer_fingerprint=tokenizer,
            wrapper_fingerprint=wrapper,
            expected_parent_pid=108,
        )
    except ValueError as error:
        assert "port owner" in str(error)
    else:
        raise AssertionError("an attestation naming another port owner must be rejected")

    attestation["process_observation"]["listening_owner_pids"] = [109]
    wrong_model = {**model, "tree_sha256": "wrong-model"}
    try:
        validate_state_pool_service_attestation_manifest(
            attestation,
            requested_model="student-alias",
            base_url="http://127.0.0.1:8010/v1",
            inference_runtime="vllm:0.10.2",
            model_fingerprint=wrong_model,
            tokenizer_fingerprint=tokenizer,
            wrapper_fingerprint=wrapper,
            expected_parent_pid=108,
        )
    except ValueError as error:
        assert "model" in str(error)
    else:
        raise AssertionError("a different Student model artifact must be rejected")


def test_position_service_requires_its_own_stage_role_and_wrapper():
    model = _directory_fingerprint("student", "student")
    tokenizer = _directory_fingerprint("tokenizer", "tokenizer")
    wrapper = _file_fingerprint("position-wrapper", "position-wrapper")
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["path"],
        "--tokenizer",
        tokenizer["path"],
        "--served-model-name",
        "student-alias",
        "--max-model-len",
        "8192",
        "--port",
        "8020",
    ]
    attestation = {
        "protocol_version": "omniopd-v1",
        "artifact": "omniopd_local_vllm_service_attestation",
        "schema_version": 2,
        "attestation_scope": "local_wrapper_child_process",
        "service_role": "position_counterfactual",
        "launch_command": command,
        "launch_command_sha256": sha256_json(command),
        "port": 8020,
        "base_url": "http://127.0.0.1:8020/v1",
        "server_pid": 119,
        "wrapper_pid": 118,
        "model_artifact": model,
        "tokenizer_artifact": tokenizer,
        "wrapper": wrapper,
        "vllm_version": "0.10.2",
        "inference_runtime": "vllm:0.10.2",
        "process_observation": {
            "server_pid": 119,
            "parent_pid": 118,
            "port": 8020,
            "argv": command,
            "argv_sha256": sha256_json(command),
            "listening_owner_pids": [119],
            "trust_boundary": (
                "host_local_process_observation_not_cryptographic_remote_attestation"
            ),
        },
    }
    identity = validate_position_service_attestation_manifest(
        attestation,
        requested_model="student-alias",
        base_url="http://127.0.0.1:8020/v1",
        inference_runtime="vllm:0.10.2",
        model_fingerprint=model,
        tokenizer_fingerprint=tokenizer,
        wrapper_fingerprint=wrapper,
        expected_parent_pid=118,
    )
    assert identity["service_role"] == "position_counterfactual"

    attestation["service_role"] = "student_state_pool"
    try:
        validate_position_service_attestation_manifest(
            attestation,
            requested_model="student-alias",
            base_url="http://127.0.0.1:8020/v1",
            inference_runtime="vllm:0.10.2",
            model_fingerprint=model,
            tokenizer_fingerprint=tokenizer,
            wrapper_fingerprint=wrapper,
            expected_parent_pid=118,
        )
    except ValueError as error:
        assert "position_counterfactual" in str(error)
    else:
        raise AssertionError("a stopped state-pool service cannot be reused for Position")
