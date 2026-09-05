from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping

from .schema import CorrectionRecord
from .prompts import initial_user_message, turn_user_message
from .prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, replace_system
from .provenance import sha256_json, sha256_text


@dataclass(frozen=True)
class ProtocolIssue:
    severity: str
    code: str
    message: str


def validate_budget(*, states: int, samples_per_state: int, declared_budget: int) -> None:
    if states <= 0 or samples_per_state <= 0:
        raise ValueError("M and N must be positive")
    if states * samples_per_state != declared_budget:
        raise ValueError(
            f"fixed-budget violation: M*N={states * samples_per_state}, declared B={declared_budget}"
        )


def validate_fixed_budget_arms(arms: Iterable[Mapping[str, Any]]) -> None:
    """Fail unless all resolved breadth/depth arms share budget and training protocol."""

    arms = list(arms)
    if len(arms) < 2:
        raise ValueError("at least two budget arms are required")
    budgets = set()
    sampling_profiles = set()
    optimizer_steps = set()
    training_seed_sets = set()
    for index, arm in enumerate(arms):
        try:
            games = int(arm["games"])
            states_per_game = int(arm["states_per_game"])
            states = int(arm["distinct_states_M"])
            samples = int(arm["teacher_samples_per_state_N"])
            budget = int(arm["teacher_budget_B"])
            steps = int(arm["training"]["total_optimizer_steps"])
            data_split_seed = int(arm["training"]["data_split_seed"])
            training_seeds = tuple(int(value) for value in arm["training"]["seeds"])
            pool_games = int(arm["state_pool"]["games_G"])
            pool_seed = int(arm["state_pool"]["seed"])
            profile = str(arm["teacher_sampling_profile"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"budget arm {index} is not fully resolved: {error}") from error
        if (
            games <= 0
            or states_per_game <= 0
            or states != games * states_per_game
            or pool_games != games
        ):
            raise ValueError(
                f"budget arm {index} has inconsistent G, states/game, M, or state-pool G"
            )
        if (
            not training_seeds
            or len(training_seeds) != len(set(training_seeds))
            or any(seed < 0 for seed in training_seeds)
            or data_split_seed < 0
            or pool_seed < 0
        ):
            raise ValueError(f"budget arm {index} has invalid pool/split/training seeds")
        validate_budget(states=states, samples_per_state=samples, declared_budget=budget)
        budgets.add(budget)
        sampling_profiles.add(profile)
        optimizer_steps.add(steps)
        training_seed_sets.add(training_seeds)
    if len(budgets) != 1:
        raise ValueError(f"Teacher budgets differ across arms: {sorted(budgets)}")
    if len(sampling_profiles) != 1:
        raise ValueError("Teacher sampling profiles differ across arms")
    if len(optimizer_steps) != 1:
        raise ValueError(f"optimizer-step budgets differ across arms: {sorted(optimizer_steps)}")
    if len(training_seed_sets) != 1:
        raise ValueError("training-seed sets differ across fixed-budget arms")
    allowed = frozenset(
        {
            "experiment",
            "states_per_game",
            "distinct_states_M",
            "teacher_samples_per_state_N",
        }
    )
    baseline = arms[0]
    for index, arm in enumerate(arms[1:], 1):
        differences = compare_control_protocols(
            baseline, arm, allowed_differences=allowed
        )
        if differences:
            raise ValueError(
                f"budget arm {index} changes fields beyond breadth/depth: {differences}"
            )


def validate_actual_calls(records: Iterable[CorrectionRecord], declared_budget: int) -> None:
    actual = sum(record.teacher_calls for record in records)
    if actual != declared_budget:
        raise ValueError(f"actual Teacher API calls={actual}, declared B={declared_budget}")


def validate_annotation_run_manifests(manifests: Iterable[Mapping[str, Any]]) -> None:
    """Validate realized fixed-B annotation runs, not only planned configs."""

    manifests = list(manifests)
    if len(manifests) < 2:
        raise ValueError("at least two annotation manifests are required")
    budgets = set()
    state_pools = set()
    sampling_contracts = set()
    selection_contracts = set()
    selected_state_sets: list[set[str]] = []
    selected_counts_by_game: list[dict[str, int]] = []
    distinct_state_counts: list[int] = []

    def artifact_identity(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            key: artifact_identity(item)
            for key, item in value.items()
            if key not in {"path", "resolved_path", "value"}
        }
    for index, manifest in enumerate(manifests):
        try:
            states = int(manifest["distinct_states_M"])
            samples = int(manifest["teacher_samples_per_state_N"])
            budget = int(manifest["declared_teacher_budget_B"])
            actual = int(manifest["actual_teacher_api_calls"])
            state_pool = str(manifest["state_pool_sha256"])
            sampling = json.dumps(
                {
                    "teacher_model": manifest["teacher_model"],
                    "teacher_model_revision": manifest["teacher_model_revision"],
                    "teacher_url": manifest["teacher_url"],
                    "code": manifest["code"],
                    "code_revision": manifest.get("code_revision"),
                    "sampling": manifest["teacher_sampling"],
                    "max_tokens": int(manifest["teacher_max_tokens"]),
                    "context_window": int(manifest["teacher_context_window"]),
                    "tokenizer": artifact_identity(manifest["teacher_tokenizer"]),
                    "prompt_sha256": manifest["teacher_prompt_sha256"],
                    "provider_response_models": manifest.get(
                        "provider_response_models", []
                    ),
                    "provider_system_fingerprints": manifest.get(
                        "provider_system_fingerprints", []
                    ),
                },
                sort_keys=True,
                allow_nan=False,
            )
            selection = json.dumps(
                manifest["selection_policies"], sort_keys=True, allow_nan=False
            )
            selected_hashes = [str(value) for value in manifest["selected_state_hashes"]]
            counts_by_game = {
                str(game): int(count)
                for game, count in manifest["selected_counts_by_game"].items()
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"annotation manifest {index} is incomplete: {error}"
            ) from error
        validate_budget(
            states=states, samples_per_state=samples, declared_budget=budget
        )
        if actual != budget:
            raise ValueError(
                f"annotation manifest {index} realized {actual} calls, expected {budget}"
            )
        if manifest.get("invalid_calls_count_toward_budget") is not True:
            raise ValueError("invalid Teacher attempts must count toward B")
        if int(manifest.get("free_retries", 0)) != 0:
            raise ValueError("fixed-B annotation runs cannot use free retries")
        if manifest.get("teacher_context_preflight", {}).get("passed") is not True:
            raise ValueError("Teacher context preflight must pass before annotation")
        if len(selected_hashes) != states or len(set(selected_hashes)) != states:
            raise ValueError(
                f"annotation manifest {index} does not identify exactly M unique states"
            )
        if (
            not counts_by_game
            or any(count <= 0 for count in counts_by_game.values())
            or sum(counts_by_game.values()) != states
        ):
            raise ValueError(
                f"annotation manifest {index} has invalid selected_counts_by_game"
            )
        budgets.add(budget)
        state_pools.add(state_pool)
        sampling_contracts.add(sampling)
        selection_contracts.add(selection)
        selected_state_sets.append(set(selected_hashes))
        selected_counts_by_game.append(counts_by_game)
        distinct_state_counts.append(states)
    if len(budgets) != 1:
        raise ValueError(f"realized Teacher budgets differ: {sorted(budgets)}")
    if len(state_pools) != 1:
        raise ValueError("annotation arms did not use the same frozen state pool")
    if len(sampling_contracts) != 1:
        raise ValueError("annotation arms used different Teacher sampling contracts")
    if len(selection_contracts) != 1:
        raise ValueError("annotation arms used different selection-policy families")
    # For a breadth/depth comparison, use one common random priority ordering:
    # every lower-M arm must be a subset of every higher-M arm. This eliminates
    # avoidable between-arm state-selection noise while preserving each arm's
    # stated per-game inclusion probability.
    for left in range(len(manifests)):
        for right in range(left + 1, len(manifests)):
            if distinct_state_counts[left] == distinct_state_counts[right]:
                continue
            smaller, larger = (
                (left, right)
                if distinct_state_counts[left] < distinct_state_counts[right]
                else (right, left)
            )
            if not selected_state_sets[smaller] <= selected_state_sets[larger]:
                raise ValueError(
                    "breadth/depth annotation arms are not nested within the shared pool"
                )
            if set(selected_counts_by_game[smaller]) != set(
                selected_counts_by_game[larger]
            ):
                raise ValueError("breadth/depth arms cover different game sets")
            if any(
                selected_counts_by_game[smaller][game]
                > selected_counts_by_game[larger][game]
                for game in selected_counts_by_game[smaller]
            ):
                raise ValueError("breadth/depth per-game state selections are not nested")


def validate_training_run_manifests(manifests: Iterable[Mapping[str, Any]]) -> None:
    """Require equal realized optimizer exposure, precision, code, and base inputs."""

    manifests = list(manifests)
    if len(manifests) < 2:
        raise ValueError("at least two training launch manifests are required")
    if any(
        manifest.get("artifact") != "omniopd_training_launch"
        or manifest.get("protocol_version") != "omniopd-v1"
        for manifest in manifests
    ):
        raise ValueError("training manifests have an unsupported artifact or protocol")

    def content_identity(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        ignored = {"path", "resolved_path", "value"}
        return {
            key: content_identity(item)
            for key, item in value.items()
            if key not in ignored
        }

    def projection(manifest: Mapping[str, Any]) -> str:
        hyperparameters = dict(manifest.get("hyperparameters", {}))
        hyperparameters.pop("experiment_name", None)
        inputs = manifest.get("inputs", {})
        projected = {
            "code": manifest.get("code"),
            "code_revision": manifest.get("code_revision"),
            "implementation": content_identity(manifest.get("implementation")),
            "training_contract": manifest.get("training_contract"),
            "hyperparameters": hyperparameters,
            "model_path": content_identity(inputs.get("model_path")),
            "user_hydra_overrides": manifest.get("user_hydra_overrides", []),
            "python": manifest.get("python"),
            "verl_version": manifest.get("verl_version"),
            "verl_version_declarations": manifest.get(
                "verl_version_declarations"
            ),
            "verl_git_revision": manifest.get("verl_git_revision"),
        }
        return json.dumps(projected, sort_keys=True, allow_nan=False)

    projections = {projection(manifest) for manifest in manifests}
    if len(projections) != 1:
        raise ValueError(
            "training runs differ in code, base input, optimizer exposure, "
            "precision, or hyperparameters"
        )


def validate_game_disjoint(train_games: Iterable[str], evaluation_games: Iterable[str]) -> None:
    overlap = set(train_games) & set(evaluation_games)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"train/evaluation game leakage ({len(overlap)} games): {preview}")


def compare_control_protocols(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    allowed_differences: frozenset[str] = frozenset(
        {"state_source.name", "state_source.behavior_policy"}
    ),
) -> list[str]:
    """Return dotted fields that differ outside the preregistered intervention."""

    differences: list[str] = []

    def walk(left: Any, right: Any, prefix: str) -> None:
        if any(prefix == allowed or prefix.startswith(allowed + ".") for allowed in allowed_differences):
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                child = f"{prefix}.{key}" if prefix else str(key)
                if key not in left or key not in right:
                    differences.append(child)
                else:
                    walk(left[key], right[key], child)
            return
        if left != right:
            differences.append(prefix)

    walk(treatment, control, "")
    return differences


def audit_correction_records(records: Iterable[CorrectionRecord]) -> list[ProtocolIssue]:
    records = list(records)
    issues: list[ProtocolIssue] = []
    for record in records:
        roles = [message.get("role") for message in record.state.messages]
        expected = ["system", "user"] + [
            "assistant" if index % 2 == 0 else "user" for index in range(2, len(roles))
        ]
        location = f"{record.state.game_id}/t{record.state.turn_index}"
        if roles != expected or not roles or roles[-1] != "user":
            issues.append(ProtocolIssue("error", "HISTORY_ROLE_ORDER", location))
            continue
        if not record.state.admissible_actions:
            issues.append(ProtocolIssue("error", "EMPTY_ADMISSIBLE_SET", location))
        if record.state.observation not in record.state.messages[-1].get("content", ""):
            issues.append(ProtocolIssue("error", "OBSERVATION_NOT_IN_QUERY", location))
        for action in record.state.admissible_actions:
            if action not in record.state.messages[-1].get("content", ""):
                issues.append(ProtocolIssue("error", "ADMISSIBLE_NOT_IN_QUERY", location))
                break
        if record.student.executed_action not in record.state.admissible_actions:
            issues.append(ProtocolIssue("error", "STUDENT_ACTION_NOT_EXECUTABLE", location))
        for sample in record.teacher_samples:
            if sample.valid and sample.executed_action not in record.state.admissible_actions:
                issues.append(ProtocolIssue("error", "VALID_TEACHER_ACTION_NOT_EXECUTABLE", location))
            if sample.api_calls != 1:
                issues.append(ProtocolIssue("error", "NONATOMIC_TEACHER_ATTEMPT", location))
        if record.teacher_calls != sum(sample.api_calls for sample in record.teacher_samples):
            issues.append(ProtocolIssue("error", "BUDGET_CALL_MISMATCH", location))
        expected_teacher_messages = replace_system(
            record.state.messages, TEACHER_SYSTEM_PROMPT
        )
        expected_metadata = {
            "student_query_sha256": sha256_json(list(record.state.messages)),
            "teacher_query_sha256": sha256_json(expected_teacher_messages),
            "non_system_query_sha256": sha256_json(expected_teacher_messages[1:]),
            "teacher_system_prompt_sha256": sha256_text(TEACHER_SYSTEM_PROMPT),
        }
        if record.protocol_version == "omniopd-v1":
            full_messages = list(record.state.full_messages)
            full_roles = [message.get("role") for message in full_messages]
            expected_full_roles = ["system", "user"] + [
                "assistant" if index % 2 == 0 else "user"
                for index in range(2, 2 + 2 * record.state.turn_index)
            ]
            if full_roles != expected_full_roles:
                issues.append(
                    ProtocolIssue("error", "FULL_HISTORY_TURN_MISMATCH", location)
                )
            else:
                expected_current_content = (
                    initial_user_message(
                        record.state.task,
                        record.state.observation,
                        record.state.admissible_actions,
                    )
                    if record.state.turn_index == 0
                    else turn_user_message(
                        record.state.observation,
                        record.state.admissible_actions,
                    )
                )
                if full_messages[-1] != {
                    "role": "user",
                    "content": expected_current_content,
                }:
                    issues.append(
                        ProtocolIssue("error", "CURRENT_STATE_MESSAGE_MISMATCH", location)
                    )
                if any(
                    message.get("role") != "assistant"
                    or not str(message.get("content", "")).startswith("Action: ")
                    for message in full_messages[2::2]
                ):
                    issues.append(
                        ProtocolIssue("error", "FULL_HISTORY_ACTION_FORMAT", location)
                    )
                truncation = record.state.truncation
                try:
                    kept_pairs = int(truncation["kept_pairs"])
                    dropped_pairs = int(truncation["dropped_pairs"])
                    token_count = int(truncation["token_count"])
                    budget = int(truncation["budget"])
                except (KeyError, TypeError, ValueError):
                    issues.append(
                        ProtocolIssue("error", "TRUNCATION_ATTESTATION_MISSING", location)
                    )
                else:
                    expected_query = full_messages[:2]
                    if kept_pairs:
                        expected_query += full_messages[-2 * kept_pairs :]
                    if (
                        kept_pairs < 0
                        or dropped_pairs < 0
                        or kept_pairs + dropped_pairs != record.state.turn_index
                        or token_count <= 0
                        or budget <= 0
                        or list(record.state.messages) != expected_query
                    ):
                        issues.append(
                            ProtocolIssue("error", "TRUNCATED_HISTORY_MISMATCH", location)
                        )
            expected_prompt = (
                STUDENT_SYSTEM_PROMPT
                if record.state.state_source == "student"
                else TEACHER_SYSTEM_PROMPT
                if record.state.state_source == "teacher"
                else None
            )
            if (
                expected_prompt is None
                or record.state.messages[0].get("content") != expected_prompt
            ):
                issues.append(ProtocolIssue("error", "STATE_SOURCE_PROMPT_MISMATCH", location))
            if record.metadata.get("teacher_prompt_replaced") is not True:
                issues.append(ProtocolIssue("error", "TEACHER_PROMPT_NOT_ATTESTED", location))
            if record.metadata.get("student_teacher_non_system_identity") is not True:
                issues.append(ProtocolIssue("error", "CONTEXT_IDENTITY_NOT_ATTESTED", location))
            if record.metadata.get("budget_counts_invalid_calls") is not True:
                issues.append(ProtocolIssue("error", "BUDGET_POLICY_NOT_ATTESTED", location))
            if record.metadata.get("samples_per_state") != len(record.teacher_samples):
                issues.append(ProtocolIssue("error", "TEACHER_SAMPLE_COUNT_MISMATCH", location))
            for key, expected_value in expected_metadata.items():
                if record.metadata.get(key) != expected_value:
                    issues.append(
                        ProtocolIssue(
                            "error", f"{key.upper()}_MISMATCH", location
                        )
                    )
        probability = record.inclusion_probability
        if probability is not None and (
            not math.isfinite(probability) or not 0.0 < probability <= 1.0
        ):
            issues.append(ProtocolIssue("error", "INVALID_INCLUSION_PROBABILITY", location))
    return issues
