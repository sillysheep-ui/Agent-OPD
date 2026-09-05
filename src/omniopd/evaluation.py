from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


def _content_identity(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: _content_identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


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
        try:
            rollout_seed = int(result["rollout_seed"])
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
        ]
        if (
            not all(isinstance(value, str) and value for value in required_strings)
            or not isinstance(result.get("code"), Mapping)
            or not isinstance(result.get("tokenizer"), Mapping)
            or not result.get("model_artifacts")
            or not isinstance(result.get("training_protocol"), Mapping)
            or result.get("split")
            not in {"eval_in_distribution", "eval_out_of_distribution"}
            or float(result.get("temperature", float("nan"))) != 0.0
            or result.get("thinking_mode") != "disabled"
            or result.get("thinking_control") != "chat_template"
            or rollout_seed < 0
            or max_steps <= 0
            or max_context_tokens <= reserve_tokens
            or reserve_tokens < max_tokens
            or max_tokens <= 0
            or request_count <= 0
            or not result.get("provider_response_models")
        ):
            raise ValueError("evaluation does not satisfy the canonical rollout contract")
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
            "provider_response_models": result.get("provider_response_models", []),
            "provider_system_fingerprints": result.get(
                "provider_system_fingerprints", []
            ),
            "training_protocol": _content_identity(
                result.get("training_protocol")
            ),
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

    if manifest.get("protocol_version") != "omniopd-v1":
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
    return seeds, first_games, dict(contract)


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
    if json.dumps(treatment_contract, sort_keys=True, allow_nan=False) != json.dumps(
        control_contract, sort_keys=True, allow_nan=False
    ):
        raise ValueError("treatment and control use different evaluation contracts")
    return {
        "seeds": treatment_seeds,
        "games": treatment_games,
        "evaluation_contract": treatment_contract,
    }
