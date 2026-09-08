from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .environment_provenance import validate_environment_seed_contract
from .provenance import sha256_json
from .service_attestation import one_command_option, verify_recorded_live_snapshot
from .validation import validate_lora_checkpoint_contract


def _content_identity(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: _content_identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


def _artifact_digest(value: Any, *, role: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} is not a content fingerprint")
    kind = value.get("kind")
    if kind == "file":
        digest = value.get("sha256")
    elif kind == "directory":
        digest = value.get("tree_sha256")
    else:
        raise ValueError(f"{role} must resolve to a local file or directory")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{role} has no content digest")
    try:
        size = int(value["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{role} has no valid byte count") from error
    if size < 0:
        raise ValueError(f"{role} has a negative byte count")
    if kind == "directory":
        try:
            files = int(value["files"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{role} has no valid file count") from error
        if files <= 0:
            raise ValueError(f"{role} directory is empty")
    return digest


def _same_artifact(left: Any, right: Any, *, role: str) -> None:
    _artifact_digest(left, role=f"recorded {role}")
    _artifact_digest(right, role=f"current {role}")
    if _content_identity(left) != _content_identity(right):
        raise ValueError(f"{role} content fingerprint does not match")


def validate_artifact_fingerprint(
    recorded: Mapping[str, Any], current: Mapping[str, Any], *, role: str
) -> None:
    """Public fail-closed check for a recorded path-independent artifact identity."""

    _same_artifact(recorded, current, role=role)


def _require_sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{role} is not a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty string")
    return value


def validate_annotation_pair_contract(
    pair_contract: Mapping[str, Any], *, comparison_contrast: str
) -> dict[str, Any]:
    """Validate both realized arms and their immutable shared annotation protocol."""

    if comparison_contrast != "breadth_depth":
        raise ValueError("annotation pair has an unsupported comparison contrast")
    if pair_contract.get("comparison_contrast") != comparison_contrast:
        raise ValueError("annotation pair contract names a different contrast")
    shared = pair_contract.get("shared_contract")
    members = pair_contract.get("members")
    roles = pair_contract.get("arm_roles")
    if (
        not isinstance(shared, Mapping)
        or not isinstance(members, Mapping)
        or len(members) != 2
        or not isinstance(roles, Mapping)
    ):
        raise ValueError("annotation pair must contain one shared contract and exactly two arms")

    required_shared = {
        "code",
        "code_revision",
        "state_pool_sha256",
        "state_pool_manifest_sha256",
        "behavior_student",
        "teacher_model",
        "teacher_model_revision",
        "teacher_url",
        "teacher_sampling",
        "teacher_max_tokens",
        "teacher_context_window",
        "teacher_tokenizer",
        "teacher_prompt_sha256",
        "provider_response_models",
        "provider_system_fingerprints",
        "selection_policies",
        "teacher_budget_definition",
        "invalid_policy",
        "invalid_calls_count_toward_budget",
        "free_retries",
    }
    if set(shared) != required_shared:
        raise ValueError("annotation pair shared Teacher/state-pool contract is incomplete")
    code = shared.get("code")
    behavior_student = shared.get("behavior_student")
    sampling = shared.get("teacher_sampling")
    provider_models = shared.get("provider_response_models")
    provider_fingerprints = shared.get("provider_system_fingerprints")
    selection_policies = shared.get("selection_policies")
    teacher_model = _require_nonempty_string(
        shared.get("teacher_model"), role="shared Teacher model"
    )
    _require_nonempty_string(
        shared.get("teacher_model_revision"), role="shared Teacher revision"
    )
    teacher_url = urlparse(
        _require_nonempty_string(shared.get("teacher_url"), role="shared Teacher URL")
    )
    try:
        free_retries = int(shared["free_retries"])
        code_files = int(code["files"]) if isinstance(code, Mapping) else -1
        code_bytes = int(code["bytes"]) if isinstance(code, Mapping) else -1
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shared code/retry provenance is incomplete") from error
    if (
        not isinstance(code, Mapping)
        or not isinstance(code.get("tree_sha256"), str)
        or not code.get("tree_sha256")
        or code_files <= 0
        or code_bytes <= 0
        or not _require_nonempty_string(
            shared.get("code_revision"), role="shared code revision"
        )
        or teacher_url.scheme not in {"http", "https"}
        or not teacher_url.hostname
        or not isinstance(sampling, Mapping)
        or not sampling
        or not isinstance(provider_models, list)
        or provider_models != [teacher_model]
        or not isinstance(provider_fingerprints, list)
        or not all(isinstance(value, str) and value for value in provider_fingerprints)
        or not isinstance(selection_policies, list)
        or not selection_policies
        or len(selection_policies) != len(set(selection_policies))
        or not all(isinstance(value, str) and value for value in selection_policies)
        or shared.get("teacher_budget_definition") != "annotation_api_attempts"
        or shared.get("invalid_policy") != "retain_as_missing_no_free_retry"
        or shared.get("invalid_calls_count_toward_budget") is not True
        or free_retries != 0
    ):
        raise ValueError("annotation pair shared Teacher/state-pool contract is invalid")
    _require_sha256(code.get("tree_sha256"), role="shared code tree")
    _require_sha256(shared.get("state_pool_sha256"), role="shared state pool")
    _require_sha256(
        shared.get("state_pool_manifest_sha256"), role="shared state-pool manifest"
    )
    _require_sha256(shared.get("teacher_prompt_sha256"), role="shared Teacher prompt")
    _artifact_digest(shared.get("teacher_tokenizer"), role="shared Teacher tokenizer")
    if not isinstance(behavior_student, Mapping) or set(behavior_student) != {
        "state_source",
        "behavior_model",
        "behavior_artifact",
        "tokenizer",
        "inference_runtime",
        "behavior_sampling",
        "prompt_sha256",
        "provider_response_models",
        "behavior_service_manifest_sha256",
        "behavior_service_attestation",
    }:
        raise ValueError("annotation pair has no complete behavior Student contract")
    behavior_model = _require_nonempty_string(
        behavior_student.get("behavior_model"), role="behavior Student model"
    )
    behavior_artifact = behavior_student.get("behavior_artifact")
    behavior_tokenizer = behavior_student.get("tokenizer")
    behavior_sampling = behavior_student.get("behavior_sampling")
    behavior_service = behavior_student.get("behavior_service_attestation")
    _artifact_digest(behavior_artifact, role="behavior Student artifact")
    _artifact_digest(behavior_tokenizer, role="behavior Student tokenizer")
    if (
        behavior_student.get("state_source") != "student"
        or behavior_artifact.get("kind") != "directory"
        or _content_identity(behavior_artifact) != _content_identity(behavior_tokenizer)
        or not isinstance(behavior_student.get("inference_runtime"), str)
        or not behavior_student["inference_runtime"].strip()
        or behavior_sampling
        != {
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "reasoning_effort": None,
        }
        or behavior_student.get("provider_response_models") != [behavior_model]
        or not isinstance(behavior_service, Mapping)
        or behavior_service.get("attestation_scope")
        != "local_wrapper_child_process"
        or behavior_service.get("service_role") != "student_state_pool"
        or behavior_service.get("inference_runtime")
        != behavior_student.get("inference_runtime")
        or behavior_service.get("served_model") != behavior_model
        or behavior_service.get("model_artifact") != behavior_artifact
        or behavior_service.get("tokenizer_artifact") != behavior_tokenizer
        or behavior_service.get("trust_boundary")
        != "host_local_process_observation_not_cryptographic_remote_attestation"
    ):
        raise ValueError("annotation pair behavior Student identity is invalid")
    _require_sha256(
        behavior_student.get("prompt_sha256"), role="behavior Student prompt"
    )
    _require_sha256(
        behavior_student.get("behavior_service_manifest_sha256"),
        role="behavior Student service manifest",
    )
    _require_sha256(
        behavior_service.get("launch_command_sha256"),
        role="behavior Student service launch",
    )
    try:
        if int(behavior_service["max_model_len"]) <= 0:
            raise ValueError("behavior Student service context limit is invalid")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("behavior Student service context limit is invalid") from error
    try:
        teacher_max_tokens = int(shared["teacher_max_tokens"])
        teacher_context_window = int(shared["teacher_context_window"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shared Teacher token limits are incomplete") from error
    if teacher_max_tokens <= 0 or teacher_context_window <= teacher_max_tokens:
        raise ValueError("shared Teacher token limits are invalid")

    required_member_fields = {
        "experiment",
        "annotation_manifest_sha256",
        "corrections_sha256",
        "experiment_config_sha256",
        "state_pool_sha256",
        "state_pool_manifest_sha256",
        "selection_sha256",
        "selection_manifest_sha256",
        "distinct_states_M",
        "teacher_samples_per_state_N",
        "declared_teacher_budget_B",
    }
    normalized_members: dict[str, dict[str, Any]] = {}
    for experiment, raw_member in members.items():
        if (
            not isinstance(experiment, str)
            or not experiment
            or not isinstance(raw_member, Mapping)
            or set(raw_member) != required_member_fields
            or raw_member.get("experiment") != experiment
        ):
            raise ValueError("annotation pair contains a malformed arm member")
        for digest_role in required_member_fields - {
            "experiment",
            "distinct_states_M",
            "teacher_samples_per_state_N",
            "declared_teacher_budget_B",
        }:
            _require_sha256(
                raw_member.get(digest_role),
                role=f"annotation member {experiment} {digest_role}",
            )
        try:
            states = int(raw_member["distinct_states_M"])
            samples = int(raw_member["teacher_samples_per_state_N"])
            budget = int(raw_member["declared_teacher_budget_B"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("annotation pair member budget is incomplete") from error
        if (
            states <= 0
            or samples <= 0
            or states * samples != budget
            or raw_member.get("state_pool_sha256") != shared.get("state_pool_sha256")
            or raw_member.get("state_pool_manifest_sha256")
            != shared.get("state_pool_manifest_sha256")
        ):
            raise ValueError("annotation pair member violates its shared fixed-B contract")
        normalized_members[experiment] = dict(raw_member)

    if (
        set(roles) != {"breadth", "depth"}
        or set(roles.values()) != set(normalized_members)
        or pair_contract.get("effect_direction") != "breadth_minus_depth"
    ):
        raise ValueError("breadth/depth annotation roles or effect direction are invalid")
    breadth = normalized_members[str(roles["breadth"])]
    depth = normalized_members[str(roles["depth"])]
    if (
        int(breadth["distinct_states_M"]) <= int(depth["distinct_states_M"])
        or int(breadth["teacher_samples_per_state_N"])
        >= int(depth["teacher_samples_per_state_N"])
        or int(breadth["declared_teacher_budget_B"])
        != int(depth["declared_teacher_budget_B"])
    ):
        raise ValueError("breadth/depth annotation roles contradict M, N, or B")
    for role in (
        "annotation_manifest_sha256",
        "corrections_sha256",
        "experiment_config_sha256",
        "selection_sha256",
        "selection_manifest_sha256",
    ):
        if breadth[role] == depth[role]:
            raise ValueError(f"breadth/depth annotation members reuse the same {role}")
    return {
        "comparison_contrast": comparison_contrast,
        "effect_direction": pair_contract["effect_direction"],
        "arm_roles": dict(roles),
        "shared_contract": dict(shared),
        "members": normalized_members,
    }


def validate_annotation_pair_manifest(
    manifest: Mapping[str, Any], *, manifest_sha256: str | None = None
) -> dict[str, Any]:
    """Validate the complete realized annotation-pair artifact."""

    if (
        manifest.get("artifact") != "omniopd_annotation_pair"
        or manifest.get("protocol_version") != "omniopd-v1"
        or int(manifest.get("schema_version", -1)) != 2
        or manifest.get("pair_kind") != "fixed_budget_annotation_runs"
    ):
        raise ValueError("annotation-pair manifest has an unsupported schema")
    comparison_contrast = manifest.get("comparison_contrast")
    contract = manifest.get("pair_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("annotation-pair manifest is missing its contract")
    contract_sha256 = _require_sha256(
        manifest.get("pair_contract_sha256"), role="annotation pair contract"
    )
    if sha256_json(contract) != contract_sha256:
        raise ValueError("annotation-pair contract digest is invalid")
    normalized = validate_annotation_pair_contract(
        contract, comparison_contrast=str(comparison_contrast)
    )
    entries = manifest.get("annotation_manifests")
    members = normalized["members"]
    if (
        not isinstance(entries, list)
        or len(entries) != 2
        or not all(isinstance(entry, Mapping) for entry in entries)
        or {entry.get("experiment") for entry in entries} != set(members)
    ):
        raise ValueError("annotation-pair manifest inventory does not contain both arms")
    for entry in entries:
        experiment = str(entry["experiment"])
        if (
            set(entry) != {"experiment", "kind", "bytes", "sha256"}
            or entry.get("kind") != "file"
            or int(entry.get("bytes", -1)) <= 0
            or _require_sha256(
                entry.get("sha256"), role=f"annotation manifest {experiment}"
            )
            != members[experiment]["annotation_manifest_sha256"]
        ):
            raise ValueError("annotation-pair manifest inventory has an invalid arm entry")
    if manifest_sha256 is not None:
        _require_sha256(manifest_sha256, role="annotation-pair manifest file")
    return {
        "schema_version": 2,
        "pair_contract_sha256": contract_sha256,
        "annotation_pair_manifest_sha256": manifest_sha256,
        **normalized,
    }


def validate_annotation_pair_binding(
    binding: Mapping[str, Any],
    inputs: Mapping[str, Any],
    arm_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the launch-side projection of the realized annotation pair."""

    annotation_pair_sha256 = _artifact_digest(
        inputs.get("annotation_pair"), role="annotation pair manifest"
    )
    comparison_contrast = binding.get("comparison_contrast")
    if comparison_contrast != "breadth_depth":
        raise ValueError("training launch has an unsupported annotation-pair contrast")
    experiment = arm_contract.get("experiment")
    member = binding.get("member")
    if (
        not isinstance(experiment, str)
        or not experiment
        or binding.get("experiment") != experiment
        or not isinstance(member, Mapping)
        or member.get("experiment") != experiment
    ):
        raise ValueError("annotation-pair member and training arm identities disagree")
    pair_contract_sha256 = _require_sha256(
        binding.get("pair_contract_sha256"), role="annotation pair contract"
    )
    pair_contract = binding.get("pair_contract")
    if (
        int(binding.get("pair_schema_version", -1)) != 2
        or binding.get("annotation_pair_manifest_sha256") != annotation_pair_sha256
        or
        not isinstance(pair_contract, Mapping)
        or sha256_json(pair_contract) != pair_contract_sha256
    ):
        raise ValueError("annotation-pair binding does not reproduce its contract digest")
    normalized_contract = validate_annotation_pair_contract(
        pair_contract, comparison_contrast=str(comparison_contrast)
    )
    behavior_student = normalized_contract["shared_contract"]["behavior_student"]
    if _content_identity(inputs.get("model_path")) != _content_identity(
        behavior_student["behavior_artifact"]
    ):
        raise ValueError(
            "training base model is not the frozen state-pool behavior Student"
        )
    if (
        normalized_contract["members"].get(experiment) != member
        or binding.get("effect_direction") != normalized_contract["effect_direction"]
        or binding.get("arm_roles") != normalized_contract["arm_roles"]
    ):
        raise ValueError("annotation-pair binding does not identify its realized arm")
    for role in (
        "annotation_manifest_sha256",
        "corrections_sha256",
        "experiment_config_sha256",
        "state_pool_sha256",
        "state_pool_manifest_sha256",
        "selection_sha256",
        "selection_manifest_sha256",
    ):
        _require_sha256(member.get(role), role=f"annotation member {role}")
    try:
        states = int(member["distinct_states_M"])
        samples = int(member["teacher_samples_per_state_N"])
        budget = int(member["declared_teacher_budget_B"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"annotation-pair member is incomplete: {error}") from error
    if (
        states <= 0
        or samples <= 0
        or states * samples != budget
        or states != int(arm_contract.get("distinct_states_M", -1))
        or samples != int(arm_contract.get("teacher_samples_per_state_N", -1))
        or budget != int(arm_contract.get("teacher_budget_B", -1))
        or member.get("experiment_config_sha256")
        != _artifact_digest(
            inputs.get("experiment_config"), role="training experiment config"
        )
    ):
        raise ValueError("annotation-pair member does not bind this fixed-budget arm")
    return {
        "pair_schema_version": 2,
        "annotation_pair_manifest_sha256": annotation_pair_sha256,
        "pair_contract_sha256": pair_contract_sha256,
        "comparison_contrast": comparison_contrast,
        "effect_direction": normalized_contract["effect_direction"],
        "arm_roles": normalized_contract["arm_roles"],
        "experiment": experiment,
        "member": dict(member),
        "pair_contract": dict(pair_contract),
    }


def validate_training_launch_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete launch-side identity needed by evaluation.

    This intentionally checks redundant fields.  A path or experiment alias is not
    accepted as evidence that a run used the declared data and optimizer contract.
    """

    if (
        manifest.get("artifact") != "omniopd_training_launch"
        or manifest.get("protocol_version") != "omniopd-v1"
        or int(manifest.get("schema_version", -1)) != 2
    ):
        raise ValueError("training launch manifest has an unsupported schema")
    experiment = manifest.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("training launch manifest has no experiment identity")
    if not isinstance(manifest.get("code"), Mapping) or not manifest.get(
        "code_revision"
    ):
        raise ValueError("training launch manifest has no immutable code identity")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("training launch manifest is missing its inputs")
    required_inputs = (
        "train_files",
        "val_files",
        "model_path",
        "experiment_config",
        "train_audit",
        "annotation_pair",
    )
    for role in required_inputs:
        _artifact_digest(inputs.get(role), role=f"training input {role}")
    if len({_artifact_digest(inputs[role], role=role) for role in required_inputs}) != len(
        required_inputs
    ):
        raise ValueError("training input roles unexpectedly reuse identical content")

    contract = manifest.get("training_contract")
    hyperparameters = manifest.get("hyperparameters")
    arm_contract = manifest.get("arm_contract")
    if not isinstance(contract, Mapping) or not isinstance(hyperparameters, Mapping):
        raise ValueError("training launch manifest is missing its training contract")
    if not isinstance(arm_contract, Mapping):
        raise ValueError("training launch manifest is missing its experiment-arm contract")
    try:
        seed = int(contract["seed"])
        steps = int(contract["total_optimizer_steps"])
        batch_size = int(contract["global_train_batch_size"])
        micro_batch_size = int(contract["micro_batch_size_per_gpu"])
        learning_rate = float(hyperparameters["learning_rate"])
        max_length = int(hyperparameters["max_length"])
        lora_rank = int(hyperparameters["lora_rank"])
        lora_alpha = int(hyperparameters["lora_alpha"])
        target_modules = str(hyperparameters["target_modules"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"training launch contract is incomplete: {error}") from error
    if (
        seed < 0
        or steps <= 0
        or batch_size <= 0
        or micro_batch_size <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or max_length <= 0
        or lora_rank <= 0
        or lora_alpha <= 0
        or target_modules != "all-linear"
    ):
        raise ValueError("training launch contract contains invalid values")
    if arm_contract.get("experiment") != experiment:
        raise ValueError("experiment and arm-contract identities disagree")
    if arm_contract.get("comparison_family") == "fixed_teacher_budget":
        try:
            states = int(arm_contract["distinct_states_M"])
            samples = int(arm_contract["teacher_samples_per_state_N"])
            budget = int(arm_contract["teacher_budget_B"])
            state_pool = arm_contract["state_pool_contract"]
            environment_seed = int(state_pool["environment_seed"])
            state_pool_games = int(state_pool["games_G"])
            state_pool_seed = int(state_pool["seed"])
            games = int(arm_contract["games"])
            teacher_max_tokens = int(arm_contract["teacher_max_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"fixed-budget arm contract is incomplete: {error}") from error
        if (
            not isinstance(state_pool, Mapping)
            or states <= 0
            or samples <= 0
            or states * samples != budget
            or state_pool_games <= 0
            or state_pool_games != games
            or state_pool_games != int(arm_contract.get("state_pool_games_G", -1))
            or state_pool_seed != int(arm_contract.get("state_pool_seed", -1))
            or environment_seed < 0
            or teacher_max_tokens <= 0
            or arm_contract.get("teacher_budget_definition")
            != "annotation_api_attempts"
            or arm_contract.get("invalid_policy")
            != "retain_as_missing_no_free_retry"
            or state_pool.get("state_source") != "student"
        ):
            raise ValueError("fixed-budget arm does not satisfy M*N=B")
    else:
        raise ValueError("training launch uses an unsupported experiment family")
    annotation_pair_binding = manifest.get("annotation_pair_binding")
    if not isinstance(annotation_pair_binding, Mapping):
        raise ValueError("training launch is missing its annotation-pair binding")
    validated_pair_binding = validate_annotation_pair_binding(
        annotation_pair_binding, inputs, arm_contract
    )
    pair_shared = validated_pair_binding["pair_contract"]["shared_contract"]
    if (
        pair_shared.get("code") != _content_identity(manifest.get("code"))
        or pair_shared.get("code_revision") != manifest.get("code_revision")
    ):
        raise ValueError("training launch code identity differs from its annotation pair")
    return {
        "experiment": experiment,
        "training_seed": seed,
        "total_optimizer_steps": steps,
        "inputs": _content_identity(inputs),
        "arm_contract": dict(arm_contract),
        "annotation_pair_binding": validated_pair_binding,
    }


def validate_training_completion_manifest(
    completion: Mapping[str, Any],
    launch: Mapping[str, Any],
    *,
    launch_manifest_sha256: str,
    completion_manifest_sha256: str,
    checkpoint_fingerprint: Mapping[str, Any],
    current_checkpoint_format: Mapping[str, Any],
    resolved_config_fingerprint: Mapping[str, Any] | None = None,
    training_log_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a successful run to the bytes of its final checkpoint and launch."""

    launch_identity = validate_training_launch_manifest(launch)
    if (
        completion.get("artifact") != "omniopd_training_completion"
        or completion.get("protocol_version") != "omniopd-v1"
        or int(completion.get("schema_version", -1)) != 2
        or completion.get("completion_status") != "completed"
    ):
        raise ValueError("training completion manifest is not a successful canonical run")
    try:
        final_step = int(completion["final_global_step"])
        completion_seed = int(completion["training_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"training completion manifest is incomplete: {error}") from error
    if (
        final_step != launch_identity["total_optimizer_steps"]
        or completion_seed != launch_identity["training_seed"]
        or completion.get("experiment") != launch_identity["experiment"]
        or completion.get("code") != launch.get("code")
        or completion.get("code_revision") != launch.get("code_revision")
    ):
        raise ValueError("training completion identity disagrees with its launch")
    launch_record = completion.get("launch_manifest")
    if (
        _artifact_digest(launch_record, role="completion launch manifest")
        != launch_manifest_sha256
    ):
        raise ValueError("training completion does not bind the supplied launch manifest")
    _same_artifact(
        completion.get("final_checkpoint"), checkpoint_fingerprint, role="final checkpoint"
    )
    if completion.get("training_inputs_sha256") != sha256_json(
        _content_identity(launch.get("inputs"))
    ):
        raise ValueError("training completion does not bind the launch input identities")
    if completion.get("training_contract_sha256") != sha256_json(
        launch.get("training_contract")
    ):
        raise ValueError("training completion does not bind the optimizer contract")
    if completion.get("hyperparameters_sha256") != sha256_json(
        launch.get("hyperparameters")
    ):
        raise ValueError("training completion does not bind the hyperparameters")
    if completion.get("annotation_pair_binding") != launch_identity[
        "annotation_pair_binding"
    ]:
        raise ValueError("training completion contradicts the launch annotation pair")
    try:
        checkpoint_contract = validate_lora_checkpoint_contract(
            completion.get("checkpoint_format", {}),
            expected_rank=int(launch["hyperparameters"]["lora_rank"]),
            expected_alpha=float(launch["hyperparameters"]["lora_alpha"]),
            expected_target_modules_policy=str(
                launch["hyperparameters"]["target_modules"]
            ),
        )
        current_checkpoint_contract = validate_lora_checkpoint_contract(
            current_checkpoint_format,
            expected_rank=int(launch["hyperparameters"]["lora_rank"]),
            expected_alpha=float(launch["hyperparameters"]["lora_alpha"]),
            expected_target_modules_policy=str(
                launch["hyperparameters"]["target_modules"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"training completion has an invalid LoRA checkpoint: {error}") from error
    if checkpoint_contract != current_checkpoint_contract:
        raise ValueError("training completion LoRA contract differs from the current checkpoint")
    if resolved_config_fingerprint is not None:
        _same_artifact(
            completion.get("resolved_trainer_config"),
            resolved_config_fingerprint,
            role="resolved trainer config",
        )
    if training_log_fingerprint is not None:
        _same_artifact(
            completion.get("train_log"),
            training_log_fingerprint,
            role="completed training log",
        )
    if not isinstance(completion_manifest_sha256, str) or not completion_manifest_sha256:
        raise ValueError("training completion manifest has no external content digest")
    return {
        **launch_identity,
        "completion_manifest_sha256": completion_manifest_sha256,
        "launch_manifest_sha256": launch_manifest_sha256,
        "completion_status": "completed",
        "final_global_step": final_step,
        "final_checkpoint": _content_identity(checkpoint_fingerprint),
        "checkpoint_format": checkpoint_contract,
    }


def _validate_recorded_process_observation(
    attestation: Mapping[str, Any],
    command: list[str],
    *,
    server_pid: int,
    wrapper_pid: int,
    port: int,
) -> Mapping[str, Any]:
    observation = attestation.get("process_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("service attestation has no host-local process observation")
    argv = observation.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
        raise ValueError("service process observation has no exact live argv")
    if (
        observation.get("server_pid") != server_pid
        or observation.get("parent_pid") != wrapper_pid
        or observation.get("port") != port
        or observation.get("argv_sha256") != sha256_json(argv)
        or observation.get("trust_boundary")
        != "host_local_process_observation_not_cryptographic_remote_attestation"
        or Path(argv[0]).name != Path(command[0]).name
        or argv[1:] != command[1:]
    ):
        raise ValueError("recorded process observation contradicts the service launch identity")
    owners = observation.get("listening_owner_pids")
    if not isinstance(owners, list) or server_pid not in owners:
        raise ValueError("recorded service PID is not the listening-port owner")
    return observation


def validate_service_attestation_manifest(
    attestation: Mapping[str, Any],
    *,
    requested_model: str,
    base_url: str,
    inference_runtime: str,
    base_fingerprint: Mapping[str, Any],
    checkpoint_fingerprint: Mapping[str, Any],
    tokenizer_fingerprint: Mapping[str, Any],
    completion_manifest_sha256: str,
    launch_manifest_sha256: str,
    wrapper_fingerprint: Mapping[str, Any],
    expected_parent_pid: int | None = None,
    require_live_process: bool = False,
) -> dict[str, Any]:
    """Verify a local wrapper's vLLM launch attestation and loaded artifacts."""

    if (
        attestation.get("artifact") != "omniopd_local_vllm_service_attestation"
        or attestation.get("protocol_version") != "omniopd-v1"
        or int(attestation.get("schema_version", -1)) != 2
        or attestation.get("attestation_scope") != "local_wrapper_child_process"
        or attestation.get("service_role") != "checkpoint_evaluation"
    ):
        raise ValueError("evaluation requires a canonical local-service attestation")
    command = attestation.get("launch_command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError("service attestation has no exact launch argv")
    if attestation.get("launch_command_sha256") != sha256_json(command):
        raise ValueError("service launch argv digest is invalid")
    if len(command) < 3 or command[1:3] != [
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]:
        raise ValueError("service was not launched through the audited vLLM entrypoint")
    if command.count("--enable-lora") != 1:
        raise ValueError("service command must enable exactly one LoRA serving mode")

    normalized_url = base_url.rstrip("/")
    parsed = urlparse(normalized_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("confirmatory evaluation only accepts a loopback service")
    try:
        port = int(attestation["port"])
        server_pid = int(attestation["server_pid"])
        wrapper_pid = int(attestation["wrapper_pid"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"service process identity is incomplete: {error}") from error
    if (
        port <= 0
        or server_pid <= 0
        or wrapper_pid <= 0
        or parsed.port != port
        or attestation.get("base_url", "").rstrip("/") != normalized_url
        or int(one_command_option(command, "--port")) != port
    ):
        raise ValueError("service URL, port, and launch command disagree")
    aliases = attestation.get("served_aliases")
    if (
        not isinstance(aliases, Mapping)
        or aliases.get("adapter") != requested_model
        or aliases.get("base") == requested_model
        or one_command_option(command, "--served-model-name") != aliases.get("base")
    ):
        raise ValueError("served base/adapter aliases are invalid")
    lora_spec = one_command_option(command, "--lora-modules")
    if "=" not in lora_spec:
        raise ValueError("service LoRA command does not name an adapter path")
    lora_alias, lora_path = lora_spec.split("=", 1)
    if lora_alias != requested_model:
        raise ValueError("service LoRA alias disagrees with the requested model")
    if Path(lora_path).resolve() != Path(str(checkpoint_fingerprint.get("path"))).resolve():
        raise ValueError("service command loaded a different checkpoint path")
    if Path(one_command_option(command, "--model")).resolve() != Path(
        str(base_fingerprint.get("path"))
    ).resolve():
        raise ValueError("service command loaded a different base-model path")
    if Path(one_command_option(command, "--tokenizer")).resolve() != Path(
        str(tokenizer_fingerprint.get("path"))
    ).resolve():
        raise ValueError("service command loaded a different tokenizer path")

    _same_artifact(attestation.get("base_model_artifact"), base_fingerprint, role="base model")
    _same_artifact(
        attestation.get("checkpoint_artifact"), checkpoint_fingerprint, role="served checkpoint"
    )
    _same_artifact(
        attestation.get("tokenizer_artifact"), tokenizer_fingerprint, role="served tokenizer"
    )
    _same_artifact(attestation.get("wrapper"), wrapper_fingerprint, role="evaluation wrapper")
    if (
        attestation.get("training_completion_manifest_sha256")
        != completion_manifest_sha256
        or attestation.get("training_launch_manifest_sha256") != launch_manifest_sha256
    ):
        raise ValueError("service is not bound to the supplied training run")
    vllm_version = attestation.get("vllm_version")
    if (
        not isinstance(vllm_version, str)
        or not vllm_version
        or inference_runtime != f"vllm:{vllm_version}"
        or attestation.get("inference_runtime") != inference_runtime
    ):
        raise ValueError("service vLLM version and evaluation runtime disagree")
    if expected_parent_pid is not None and wrapper_pid != expected_parent_pid:
        raise ValueError("evaluation is not a direct child of the attested local wrapper")
    observation = _validate_recorded_process_observation(
        attestation, command, server_pid=server_pid, wrapper_pid=wrapper_pid, port=port
    )
    if require_live_process:
        verify_recorded_live_snapshot(
            observation,
            server_pid=server_pid,
            wrapper_pid=wrapper_pid,
            port=port,
            launch_command=command,
        )
    return {
        "attestation_scope": attestation["attestation_scope"],
        "inference_runtime": inference_runtime,
        "vllm_version": vllm_version,
        "served_aliases": dict(aliases),
        "launch_command_sha256": attestation["launch_command_sha256"],
        "max_model_len": int(one_command_option(command, "--max-model-len")),
        "base_model_artifact": _content_identity(base_fingerprint),
        "checkpoint_artifact": _content_identity(checkpoint_fingerprint),
        "tokenizer_artifact": _content_identity(tokenizer_fingerprint),
        "training_completion_manifest_sha256": completion_manifest_sha256,
        "training_launch_manifest_sha256": launch_manifest_sha256,
    }


def validate_complete_student_service_attestation_manifest(
    attestation: Mapping[str, Any],
    *,
    service_role: str,
    requested_model: str,
    base_url: str,
    inference_runtime: str,
    model_fingerprint: Mapping[str, Any],
    tokenizer_fingerprint: Mapping[str, Any],
    wrapper_fingerprint: Mapping[str, Any],
    expected_parent_pid: int | None = None,
    require_live_process: bool = False,
) -> dict[str, Any]:
    """Validate one complete-model Student service for a named protocol stage."""

    if service_role not in {"student_state_pool", "position_counterfactual"}:
        raise ValueError("unsupported complete-model Student service role")

    if (
        attestation.get("artifact") != "omniopd_local_vllm_service_attestation"
        or attestation.get("protocol_version") != "omniopd-v1"
        or int(attestation.get("schema_version", -1)) != 2
        or attestation.get("attestation_scope") != "local_wrapper_child_process"
        or attestation.get("service_role") != service_role
    ):
        raise ValueError(
            f"{service_role} requires a canonical local-service attestation"
        )
    command = attestation.get("launch_command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError("service attestation has no exact launch argv")
    if attestation.get("launch_command_sha256") != sha256_json(command):
        raise ValueError("service launch argv digest is invalid")
    if len(command) < 3 or command[1:3] != ["-m", "vllm.entrypoints.openai.api_server"]:
        raise ValueError("service was not launched through the audited vLLM entrypoint")
    if "--enable-lora" in command or "--lora-modules" in command:
        raise ValueError("canonical Student service must load one complete model artifact")
    normalized_url = base_url.rstrip("/")
    parsed = urlparse(normalized_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("complete-model Student inference only accepts a loopback service")
    try:
        port = int(attestation["port"])
        server_pid = int(attestation["server_pid"])
        wrapper_pid = int(attestation["wrapper_pid"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"service process identity is incomplete: {error}") from error
    if (
        port <= 0
        or server_pid <= 0
        or wrapper_pid <= 0
        or parsed.port != port
        or attestation.get("base_url", "").rstrip("/") != normalized_url
        or int(one_command_option(command, "--port")) != port
        or one_command_option(command, "--served-model-name") != requested_model
        or Path(one_command_option(command, "--model")).resolve()
        != Path(str(model_fingerprint.get("path"))).resolve()
        or Path(one_command_option(command, "--tokenizer")).resolve()
        != Path(str(tokenizer_fingerprint.get("path"))).resolve()
    ):
        raise ValueError("Student service URL/model/tokenizer/alias command is inconsistent")
    _same_artifact(attestation.get("model_artifact"), model_fingerprint, role="Student model")
    _same_artifact(
        attestation.get("tokenizer_artifact"), tokenizer_fingerprint, role="Student tokenizer"
    )
    _same_artifact(attestation.get("wrapper"), wrapper_fingerprint, role="service wrapper")
    version = attestation.get("vllm_version")
    if (
        not isinstance(version, str)
        or not version
        or inference_runtime != f"vllm:{version}"
        or attestation.get("inference_runtime") != inference_runtime
    ):
        raise ValueError("Student service vLLM version and declared runtime disagree")
    if expected_parent_pid is not None and wrapper_pid != expected_parent_pid:
        raise ValueError("Student inference runner is not a direct child of the attested wrapper")
    observation = _validate_recorded_process_observation(
        attestation, command, server_pid=server_pid, wrapper_pid=wrapper_pid, port=port
    )
    if require_live_process:
        verify_recorded_live_snapshot(
            observation,
            server_pid=server_pid,
            wrapper_pid=wrapper_pid,
            port=port,
            launch_command=command,
        )
    return {
        "attestation_scope": attestation["attestation_scope"],
        "service_role": service_role,
        "inference_runtime": inference_runtime,
        "served_model": requested_model,
        "max_model_len": int(one_command_option(command, "--max-model-len")),
        "model_artifact": _content_identity(model_fingerprint),
        "tokenizer_artifact": _content_identity(tokenizer_fingerprint),
        "launch_command_sha256": attestation["launch_command_sha256"],
        "trust_boundary": observation["trust_boundary"],
    }


def validate_state_pool_service_attestation_manifest(
    attestation: Mapping[str, Any],
    *,
    requested_model: str,
    base_url: str,
    inference_runtime: str,
    model_fingerprint: Mapping[str, Any],
    tokenizer_fingerprint: Mapping[str, Any],
    wrapper_fingerprint: Mapping[str, Any],
    expected_parent_pid: int | None = None,
    require_live_process: bool = False,
) -> dict[str, Any]:
    """Validate the live local Student service used to create a state pool."""

    return validate_complete_student_service_attestation_manifest(
        attestation,
        service_role="student_state_pool",
        requested_model=requested_model,
        base_url=base_url,
        inference_runtime=inference_runtime,
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        wrapper_fingerprint=wrapper_fingerprint,
        expected_parent_pid=expected_parent_pid,
        require_live_process=require_live_process,
    )


def validate_position_service_attestation_manifest(
    attestation: Mapping[str, Any],
    *,
    requested_model: str,
    base_url: str,
    inference_runtime: str,
    model_fingerprint: Mapping[str, Any],
    tokenizer_fingerprint: Mapping[str, Any],
    wrapper_fingerprint: Mapping[str, Any],
    expected_parent_pid: int | None = None,
    require_live_process: bool = False,
) -> dict[str, Any]:
    """Validate the live frozen Student service used by Position replay."""

    return validate_complete_student_service_attestation_manifest(
        attestation,
        service_role="position_counterfactual",
        requested_model=requested_model,
        base_url=base_url,
        inference_runtime=inference_runtime,
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        wrapper_fingerprint=wrapper_fingerprint,
        expected_parent_pid=expected_parent_pid,
        require_live_process=require_live_process,
    )


def aggregate_evaluation_results(
    evaluations: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    """Validate evaluation comparability and build bootstrap's seed→game schema."""

    if not evaluations:
        raise ValueError("at least one evaluation result is required")
    reference_contract = None
    expected_games = None
    bundle: dict[int, dict[str, float]] = {}
    for declared_seed, result in sorted(evaluations.items()):
        if result.get("protocol_version") != "omniopd-v1":
            raise ValueError("evaluation has an unsupported protocol version")
        if int(result.get("training_seed", -1)) != int(declared_seed):
            raise ValueError(f"evaluation seed mismatch for declared seed {declared_seed}")
        games = [str(value) for value in result.get("games", [])]
        if not games or len(games) != len(set(games)):
            raise ValueError("every evaluation requires one non-empty unique frozen game list")
        if expected_games is None:
            expected_games = games
        elif games != expected_games:
            raise ValueError("evaluation seeds do not use the exact same ordered game list")
        raw_per_game = result.get("per_game")
        if not isinstance(raw_per_game, Mapping) or set(map(str, raw_per_game)) != set(games):
            raise ValueError("evaluation per_game keys do not match its frozen game list")
        per_game = {str(game): float(value) for game, value in raw_per_game.items()}
        if not all(math.isfinite(value) and value in {0.0, 1.0} for value in per_game.values()):
            raise ValueError("ALFWorld per-game outcomes must be finite binary values")
        reported = float(result.get("success_rate", float("nan")))
        realized = sum(per_game.values()) / len(per_game)
        if not math.isclose(reported, realized, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("evaluation success_rate disagrees with per_game outcomes")
        traces = result.get("traces")
        if (
            not isinstance(traces, list)
            or len(traces) != len(games)
            or not all(isinstance(trace, Mapping) for trace in traces)
            or [str(trace.get("game_id")) for trace in traces] != games
        ):
            raise ValueError("evaluation traces do not reproduce the ordered game list")
        try:
            environment_seeds = validate_environment_seed_contract(
                result.get("environment_rollout", {}),
                game_ids=games,
                expected_master_seed=int(result["environment_seed"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"evaluation environment seed contract is invalid: {error}") from error
        trace_state_hashes: list[str] = []
        recomputed_request_count = 0
        for trace in traces:
            game = str(trace["game_id"])
            turns = trace.get("turns")
            if (
                not isinstance(trace.get("won"), bool)
                or int(trace["won"]) != int(per_game[game])
                or int(trace.get("environment_rollout_seed", -1))
                != environment_seeds[game]
                or not isinstance(turns, list)
                or not turns
            ):
                raise ValueError("evaluation trace contradicts its outcome, seed, or turns")
            recomputed_request_count += len(turns)
            for turn in turns:
                state = turn.get("state") if isinstance(turn, Mapping) else None
                if (
                    not isinstance(state, Mapping)
                    or str(state.get("game_id")) != game
                    or not isinstance(state.get("state_hash"), str)
                    or not state["state_hash"]
                ):
                    raise ValueError("evaluation trace contains a malformed state turn")
                trace_state_hashes.append(str(state["state_hash"]))
        if len(trace_state_hashes) != len(set(trace_state_hashes)):
            raise ValueError("evaluation traces contain duplicate request/state identities")
        try:
            rollout_seed = int(result["rollout_seed"])
            environment_seed = int(result["environment_seed"])
            max_steps = int(result["max_steps"])
            max_context_tokens = int(result["max_context_tokens"])
            reserve_tokens = int(result["reserve_tokens"])
            max_tokens = int(result["max_tokens"])
            request_count = int(result["request_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"evaluation rollout contract is incomplete: {error}") from error
        required_strings = [
            result.get("code_revision"),
            result.get("game_list_file_sha256"),
            result.get("game_list_manifest_sha256"),
            result.get("game_list_sha256"),
            result.get("env_config_sha256"),
            result.get("student_prompt_sha256"),
            result.get("request_ledger_sha256"),
            result.get("inference_runtime"),
            result.get("training_manifest_sha256"),
            result.get("training_launch_manifest_sha256"),
            result.get("training_completion_manifest_sha256"),
            result.get("service_manifest_sha256"),
            result.get("game_artifacts_sha256"),
        ]
        experiment = result.get("experiment")
        arm_contract = result.get("arm_contract")
        training_inputs = result.get("training_inputs")
        completion = result.get("training_completion")
        service = result.get("service_attestation")
        if (
            not all(isinstance(value, str) and value for value in required_strings)
            or not isinstance(experiment, str)
            or not experiment
            or not isinstance(arm_contract, Mapping)
            or not isinstance(training_inputs, Mapping)
            or not isinstance(completion, Mapping)
            or not isinstance(service, Mapping)
            or not isinstance(result.get("checkpoint_artifact"), Mapping)
            or not isinstance(result.get("base_model_artifact"), Mapping)
            or not isinstance(result.get("code"), Mapping)
            or not isinstance(result.get("tokenizer"), Mapping)
            or not result.get("model_artifacts")
            or not isinstance(result.get("training_protocol"), Mapping)
            or not isinstance(result.get("runtime_dependencies"), Mapping)
            or not isinstance(result.get("environment_rollout"), Mapping)
            or result.get("split")
            not in {"eval_in_distribution", "eval_out_of_distribution"}
            or float(result.get("temperature", float("nan"))) != 0.0
            or result.get("thinking_mode") != "disabled"
            or result.get("thinking_control") != "chat_template"
            or rollout_seed < 0
            or environment_seed < 0
            or int(result.get("student_decoding_seed", -1)) != rollout_seed
            or int(
                result.get("environment_rollout", {}).get("master_seed", -1)
            )
            != environment_seed
            or max_steps <= 0
            or max_context_tokens <= reserve_tokens
            or reserve_tokens < max_tokens
            or max_tokens <= 0
            or request_count <= 0
            or request_count != recomputed_request_count
        ):
            raise ValueError("evaluation does not satisfy the canonical rollout contract")
        if (
            completion.get("completion_status") != "completed"
            or completion.get("experiment") != experiment
            or int(completion.get("training_seed", -1)) != int(declared_seed)
            or completion.get("completion_manifest_sha256")
            != result.get("training_completion_manifest_sha256")
            or completion.get("launch_manifest_sha256")
            != result.get("training_launch_manifest_sha256")
            or result.get("training_manifest_sha256")
            != result.get("training_launch_manifest_sha256")
            or completion.get("inputs") != _content_identity(training_inputs)
            or completion.get("arm_contract") != arm_contract
            or completion.get("annotation_pair_binding")
            != validate_annotation_pair_binding(
                completion.get("annotation_pair_binding", {}),
                training_inputs,
                arm_contract,
            )
            or completion.get("final_checkpoint")
            != _content_identity(result.get("checkpoint_artifact"))
            or int(completion.get("final_global_step", -1)) <= 0
            or service.get("served_aliases", {}).get("adapter")
            != result.get("model")
            or result.get("provider_response_models") != [result.get("model")]
            or service.get("inference_runtime") != result.get("inference_runtime")
            or service.get("training_completion_manifest_sha256")
            != result.get("training_completion_manifest_sha256")
            or service.get("training_launch_manifest_sha256")
            != result.get("training_launch_manifest_sha256")
            or service.get("checkpoint_artifact")
            != _content_identity(result.get("checkpoint_artifact"))
            or service.get("base_model_artifact")
            != _content_identity(result.get("base_model_artifact"))
        ):
            raise ValueError("evaluation training/service identity chain is inconsistent")
        contract = {
            "protocol_version": result.get("protocol_version"),
            "code": result.get("code"),
            "code_revision": result.get("code_revision"),
            "split": result.get("split"),
            "game_list_file_sha256": result.get("game_list_file_sha256"),
            "game_list_manifest_sha256": result.get(
                "game_list_manifest_sha256"
            ),
            "game_list_sha256": result.get("game_list_sha256"),
            "env_config_sha256": result.get("env_config_sha256"),
            "student_prompt_sha256": result.get("student_prompt_sha256"),
            "tokenizer": _content_identity(result.get("tokenizer")),
            "inference_runtime": result.get("inference_runtime"),
            "temperature": result.get("temperature"),
            "thinking_mode": result.get("thinking_mode"),
            "thinking_control": result.get("thinking_control"),
            "max_steps": max_steps,
            "max_context_tokens": max_context_tokens,
            "reserve_tokens": reserve_tokens,
            "max_tokens": max_tokens,
            "rollout_seed": rollout_seed,
            "student_decoding_seed": rollout_seed,
            "environment_seed": environment_seed,
            "environment_rollout": result.get("environment_rollout"),
            "runtime_dependencies": result.get("runtime_dependencies"),
            "game_artifacts_sha256": result.get("game_artifacts_sha256"),
            "provider_response_models": result.get("provider_response_models", []),
            "provider_system_fingerprints": result.get(
                "provider_system_fingerprints", []
            ),
            "experiment": experiment,
            "arm_contract": arm_contract,
            "training_inputs": _content_identity(training_inputs),
            "annotation_pair_binding": completion.get("annotation_pair_binding"),
            "training_protocol": _content_identity(
                result.get("training_protocol")
            ),
            "service_protocol": {
                "attestation_scope": service.get("attestation_scope"),
                "inference_runtime": service.get("inference_runtime"),
                "vllm_version": service.get("vllm_version"),
                "served_aliases": service.get("served_aliases"),
                "max_model_len": service.get("max_model_len"),
                "base_model_artifact": service.get("base_model_artifact"),
            },
        }
        serialized = json.dumps(contract, sort_keys=True, allow_nan=False)
        if reference_contract is None:
            reference_contract = serialized
        elif serialized != reference_contract:
            raise ValueError("evaluation seeds use different rollout/game/code contracts")
        bundle[int(declared_seed)] = per_game
    return bundle, json.loads(reference_contract)


def validate_aggregate_manifest(
    bundle: Mapping[int, Mapping[str, float]], manifest: Mapping[str, Any]
) -> tuple[list[int], list[str], dict[str, Any]]:
    """Validate one immutable aggregate before it enters confirmatory inference."""

    if (
        manifest.get("protocol_version") != "omniopd-v1"
        or int(manifest.get("schema_version", -1)) != 2
    ):
        raise ValueError("aggregate manifest has an unsupported protocol version")
    if manifest.get("artifact") != "hierarchical_bootstrap_input":
        raise ValueError("manifest does not describe a hierarchical-bootstrap input")
    seeds = sorted(int(seed) for seed in bundle)
    if not seeds or seeds != sorted(int(seed) for seed in manifest.get("seeds", [])):
        raise ValueError("aggregate data and manifest identify different training seeds")
    first_games = sorted(str(game) for game in bundle[seeds[0]])
    if not first_games:
        raise ValueError("aggregate contains no evaluation games")
    for seed in seeds:
        games = sorted(str(game) for game in bundle[seed])
        if games != first_games:
            raise ValueError("aggregate seeds do not contain the same paired games")
        values = [float(value) for value in bundle[seed].values()]
        if not all(math.isfinite(value) and value in {0.0, 1.0} for value in values):
            raise ValueError("aggregate ALFWorld outcomes must be finite and binary")
    manifest_games = [str(game) for game in manifest.get("game_ids", [])]
    if manifest_games != first_games or int(manifest.get("games", -1)) != len(first_games):
        raise ValueError("aggregate data and manifest identify different games")
    contract = manifest.get("evaluation_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("aggregate manifest is missing its evaluation contract")
    if not manifest.get("code_revision"):
        raise ValueError("aggregate manifest must record a non-empty code revision")
    if (
        manifest.get("code") != contract.get("code")
        or manifest.get("code_revision") != contract.get("code_revision")
    ):
        raise ValueError("aggregation and evaluation used different code revisions")
    if (
        not isinstance(manifest.get("experiment"), str)
        or manifest.get("experiment") != contract.get("experiment")
        or manifest.get("arm_contract") != contract.get("arm_contract")
    ):
        raise ValueError("aggregate manifest has a missing or inconsistent arm identity")
    training_runs = manifest.get("training_runs")
    if not isinstance(training_runs, Mapping) or sorted(
        int(seed) for seed in training_runs
    ) != seeds:
        raise ValueError("aggregate manifest does not identify every completed training run")
    for seed in seeds:
        identity = training_runs.get(str(seed), training_runs.get(seed))
        if (
            not isinstance(identity, Mapping)
            or identity.get("completion_status") != "completed"
            or int(identity.get("training_seed", -1)) != seed
            or identity.get("experiment") != manifest.get("experiment")
            or not identity.get("training_completion_manifest_sha256")
            or not identity.get("service_manifest_sha256")
            or identity.get("annotation_pair_contract_sha256")
            != contract.get("annotation_pair_binding", {}).get(
                "pair_contract_sha256"
            )
        ):
            raise ValueError("aggregate contains an invalid completed-run identity")
    return seeds, first_games, dict(contract)


def _shared_evaluation_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    shared = dict(contract)
    shared.pop("experiment", None)
    shared.pop("arm_contract", None)
    shared.pop("annotation_pair_binding", None)
    inputs = shared.pop("training_inputs", None)
    expected_roles = {
        "train_files",
        "val_files",
        "model_path",
        "experiment_config",
        "train_audit",
        "annotation_pair",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_roles:
        raise ValueError("evaluation contract has incomplete or extra training-input roles")
    # Arm-specific data/config/audit bytes may differ, but the base model and the
    # realized annotation-pair manifest must be literally identical across arms.
    shared["shared_training_inputs"] = {
        role: inputs[role] for role in ("model_path", "annotation_pair")
    }
    return shared


def _validate_annotation_pair_across_arms(
    treatment_contract: Mapping[str, Any],
    control_contract: Mapping[str, Any],
    treatment_arm: Mapping[str, Any],
    control_arm: Mapping[str, Any],
) -> tuple[str, str, str]:
    treatment_inputs = treatment_contract.get("training_inputs")
    control_inputs = control_contract.get("training_inputs")
    treatment_binding = treatment_contract.get("annotation_pair_binding")
    control_binding = control_contract.get("annotation_pair_binding")
    if not all(
        isinstance(value, Mapping)
        for value in (
            treatment_inputs,
            control_inputs,
            treatment_binding,
            control_binding,
        )
    ):
        raise ValueError("paired aggregates are missing their annotation-source identity")
    left = validate_annotation_pair_binding(
        treatment_binding, treatment_inputs, treatment_arm
    )
    right = validate_annotation_pair_binding(control_binding, control_inputs, control_arm)
    if (
        left["pair_contract_sha256"] != right["pair_contract_sha256"]
        or left["comparison_contrast"] != right["comparison_contrast"]
        or _content_identity(treatment_inputs.get("annotation_pair"))
        != _content_identity(control_inputs.get("annotation_pair"))
    ):
        raise ValueError("paired arms were not trained from one annotation-pair manifest")
    left_member = left["member"]
    right_member = right["member"]
    if (
        left_member["annotation_manifest_sha256"]
        == right_member["annotation_manifest_sha256"]
        or left_member["corrections_sha256"] == right_member["corrections_sha256"]
        or left_member["state_pool_sha256"] != right_member["state_pool_sha256"]
        or left_member["state_pool_manifest_sha256"]
        != right_member["state_pool_manifest_sha256"]
    ):
        raise ValueError("annotation-pair members do not encode two arms on one frozen pool")
    roles = left["arm_roles"]
    if (
        left.get("effect_direction") != "breadth_minus_depth"
        or roles.get("breadth") != treatment_arm.get("experiment")
        or roles.get("depth") != control_arm.get("experiment")
    ):
        raise ValueError(
            "paired inference direction must be the preregistered breadth-minus-depth contrast"
        )
    return (
        str(left["pair_contract_sha256"]),
        str(left["comparison_contrast"]),
        str(left["effect_direction"]),
    )


def _validate_arm_pair(
    treatment: Mapping[str, Any], control: Mapping[str, Any]
) -> str:
    if treatment.get("experiment") == control.get("experiment"):
        raise ValueError("treatment and control cannot identify the same experiment arm")
    if (
        treatment.get("comparison_family") != "fixed_teacher_budget"
        or control.get("comparison_family") != "fixed_teacher_budget"
    ):
        raise ValueError("paired aggregates use unsupported or mismatched arm families")
    try:
        left_games = int(treatment["games"])
        left_states_per_game = int(treatment["states_per_game"])
        left_m = int(treatment["distinct_states_M"])
        left_n = int(treatment["teacher_samples_per_state_N"])
        left_b = int(treatment["teacher_budget_B"])
        right_games = int(control["games"])
        right_states_per_game = int(control["states_per_game"])
        right_m = int(control["distinct_states_M"])
        right_n = int(control["teacher_samples_per_state_N"])
        right_b = int(control["teacher_budget_B"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"paired aggregate arm contract is incomplete: {error}") from error
    if (
        min(
            left_games,
            left_states_per_game,
            left_m,
            left_n,
            left_b,
            right_games,
            right_states_per_game,
            right_m,
            right_n,
            right_b,
        )
        <= 0
        or left_m != left_games * left_states_per_game
        or right_m != right_games * right_states_per_game
        or left_m * left_n != left_b
        or right_m * right_n != right_b
        or left_b != right_b
    ):
        raise ValueError("paired arms do not each encode the same valid fixed Teacher budget")

    if left_m != right_m and left_n != right_n:
        if (left_m - right_m) * (left_n - right_n) >= 0:
            raise ValueError("breadth/depth arms must trade more states for fewer samples")
        contrast = "breadth_depth"
        ignored = {
            "experiment",
            "distinct_states_M",
            "teacher_samples_per_state_N",
            "states_per_game",
        }
    elif left_m == right_m and left_n == right_n:
        raise ValueError(
            "state-selection training comparisons are not implemented by the canonical "
            "annotation-pair builder"
        )
    else:
        raise ValueError(
            "paired fixed-budget arms must change both M and N, or only the selection rule"
        )

    left_shared = {key: value for key, value in treatment.items() if key not in ignored}
    right_shared = {key: value for key, value in control.items() if key not in ignored}
    if json.dumps(left_shared, sort_keys=True, allow_nan=False) != json.dumps(
        right_shared, sort_keys=True, allow_nan=False
    ):
        raise ValueError(
            f"paired arms change protocol fields beyond the {contrast} intervention"
        )
    return contrast


def validate_aggregate_pair_contracts(
    treatment_manifest: Mapping[str, Any], control_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate arm roles and the shared protocol without loading outcome data."""

    for manifest in (treatment_manifest, control_manifest):
        if (
            manifest.get("artifact") != "hierarchical_bootstrap_input"
            or manifest.get("protocol_version") != "omniopd-v1"
            or int(manifest.get("schema_version", -1)) != 2
            or not isinstance(manifest.get("evaluation_contract"), Mapping)
            or manifest.get("experiment")
            != manifest.get("evaluation_contract", {}).get("experiment")
            or manifest.get("arm_contract")
            != manifest.get("evaluation_contract", {}).get("arm_contract")
        ):
            raise ValueError("aggregate pair contains an invalid arm manifest")
    treatment_arm = treatment_manifest["arm_contract"]
    control_arm = control_manifest["arm_contract"]
    comparison_contrast = _validate_arm_pair(treatment_arm, control_arm)
    (
        pair_contract_sha256,
        annotation_contrast,
        effect_direction,
    ) = _validate_annotation_pair_across_arms(
        treatment_manifest["evaluation_contract"],
        control_manifest["evaluation_contract"],
        treatment_arm,
        control_arm,
    )
    if annotation_contrast != comparison_contrast:
        raise ValueError("annotation pair and aggregate arm contrast disagree")
    treatment_shared = _shared_evaluation_contract(
        treatment_manifest["evaluation_contract"]
    )
    control_shared = _shared_evaluation_contract(control_manifest["evaluation_contract"])
    if json.dumps(treatment_shared, sort_keys=True, allow_nan=False) != json.dumps(
        control_shared, sort_keys=True, allow_nan=False
    ):
        raise ValueError("treatment and control use different shared evaluation protocols")
    return {
        "treatment_experiment": treatment_manifest["experiment"],
        "control_experiment": control_manifest["experiment"],
        "treatment_arm": dict(treatment_arm),
        "control_arm": dict(control_arm),
        "comparison_contrast": comparison_contrast,
        "effect_direction": effect_direction,
        "annotation_pair_contract_sha256": pair_contract_sha256,
        "evaluation_contract": treatment_shared,
    }


def validate_paired_aggregates(
    treatment: Mapping[int, Mapping[str, float]],
    control: Mapping[int, Mapping[str, float]],
    treatment_manifest: Mapping[str, Any],
    control_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact seed, game, and rollout-contract pairing across two arms."""

    treatment_seeds, treatment_games, treatment_contract = validate_aggregate_manifest(
        treatment, treatment_manifest
    )
    control_seeds, control_games, control_contract = validate_aggregate_manifest(
        control, control_manifest
    )
    if treatment_seeds != control_seeds:
        raise ValueError("treatment and control use different training-seed sets")
    if treatment_games != control_games:
        raise ValueError("treatment and control use different paired evaluation games")
    pairing = validate_aggregate_pair_contracts(treatment_manifest, control_manifest)
    return {
        "seeds": treatment_seeds,
        "games": treatment_games,
        **pairing,
    }
