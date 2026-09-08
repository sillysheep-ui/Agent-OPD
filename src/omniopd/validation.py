from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .schema import CorrectionRecord
from .selection import admissible_entropy
from .prompts import initial_user_message, turn_user_message
from .prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, replace_system
from .provenance import sha256_file, sha256_json, sha256_text


@dataclass(frozen=True)
class ProtocolIssue:
    severity: str
    code: str
    message: str


def _path_independent_fingerprint(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: _path_independent_fingerprint(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


def state_pool_behavior_student_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the complete Student identity that must initialize training.

    The canonical fixed-budget protocol supports one complete local Student
    artifact.  Its tokenizer must be loaded from that same content-addressed
    directory; silently composing a different tokenizer or adapter would make
    the collected states come from a different policy than the trained model.
    """

    artifacts = manifest.get("behavior_artifacts")
    tokenizer = manifest.get("tokenizer")
    model = manifest.get("behavior_model")
    runtime = manifest.get("inference_runtime")
    sampling = manifest.get("behavior_sampling")
    service_manifest_sha256 = manifest.get("behavior_service_manifest_sha256")
    service = manifest.get("behavior_service_attestation")
    if (
        manifest.get("artifact") != "immutable_state_pool"
        or manifest.get("protocol_version") != "omniopd-v1"
        or manifest.get("state_source") != "student"
        or not isinstance(artifacts, list)
        or len(artifacts) != 1
        or not isinstance(artifacts[0], Mapping)
        or not isinstance(tokenizer, Mapping)
        or not isinstance(model, str)
        or not model.strip()
        or not isinstance(runtime, str)
        or not runtime.strip()
        or not isinstance(sampling, Mapping)
        or not sampling
        or manifest.get("provider_response_models") != [model]
        or not isinstance(service, Mapping)
    ):
        raise ValueError("state pool has no canonical complete behavior Student identity")
    artifact_identity = _path_independent_fingerprint(artifacts[0])
    tokenizer_identity = _path_independent_fingerprint(tokenizer)
    if (
        artifact_identity.get("kind") != "directory"
        or int(artifact_identity.get("files", 0)) <= 0
        or artifact_identity != tokenizer_identity
    ):
        raise ValueError(
            "canonical behavior Student must use one non-empty complete model directory "
            "as its tokenizer"
        )
    for role, value in {
        "behavior Student artifact": artifact_identity,
        "behavior Student tokenizer": tokenizer_identity,
    }.items():
        kind = value.get("kind") if isinstance(value, Mapping) else None
        digest = (
            value.get("sha256") if kind == "file" else value.get("tree_sha256")
            if kind == "directory"
            else None
        )
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"{role} has no content digest")
    if (
        not isinstance(service_manifest_sha256, str)
        or len(service_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in service_manifest_sha256
        )
    ):
        raise ValueError(
            "state pool has no content digest for its live-service attestation"
        )
    service_identity = _path_independent_fingerprint(service)
    required_service_fields = {
        "attestation_scope",
        "service_role",
        "inference_runtime",
        "served_model",
        "max_model_len",
        "model_artifact",
        "tokenizer_artifact",
        "launch_command_sha256",
        "trust_boundary",
    }
    try:
        max_model_len = int(service_identity["max_model_len"])
        state_context = int(manifest["max_context_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"state-pool service context contract is incomplete: {error}"
        ) from error
    launch_digest = service_identity.get("launch_command_sha256")
    if (
        set(service_identity) != required_service_fields
        or service_identity.get("attestation_scope")
        != "local_wrapper_child_process"
        or service_identity.get("service_role") != "student_state_pool"
        or service_identity.get("inference_runtime") != runtime
        or service_identity.get("served_model") != model
        or service_identity.get("model_artifact") != artifact_identity
        or service_identity.get("tokenizer_artifact") != tokenizer_identity
        or service_identity.get("trust_boundary")
        != "host_local_process_observation_not_cryptographic_remote_attestation"
        or max_model_len < state_context
        or not isinstance(launch_digest, str)
        or len(launch_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in launch_digest
        )
        or dict(sampling)
        != {
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "reasoning_effort": None,
        }
        or manifest.get("prompt_sha256") != sha256_text(STUDENT_SYSTEM_PROMPT)
    ):
        raise ValueError("state-pool live Student service identity is inconsistent")
    return {
        "state_source": "student",
        "behavior_model": model,
        "behavior_artifact": artifact_identity,
        "tokenizer": tokenizer_identity,
        "inference_runtime": runtime,
        "behavior_sampling": dict(sampling),
        "prompt_sha256": manifest.get("prompt_sha256"),
        "provider_response_models": [model],
        "behavior_service_manifest_sha256": service_manifest_sha256,
        "behavior_service_attestation": service_identity,
    }


def validate_selection_manifest_against_rows(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    *,
    pool_state_hashes_by_game: Mapping[str, Iterable[str]],
    score_by_state: Mapping[str, float] | None = None,
    expected_experiment: str,
    expected_policy: str,
    expected_seed: int | None,
    expected_games: int,
    expected_states_per_game: int,
    expected_states: int,
) -> dict[str, Any]:
    """Recompute the within-game selection design before spending Teacher calls."""

    rows = list(rows)
    try:
        population_hashes = {
            str(game): [str(state_hash) for state_hash in state_hashes]
            for game, state_hashes in pool_state_hashes_by_game.items()
        }
        state_hashes = [str(row["state_hash"]) for row in rows]
        games = [str(row["game_id"]) for row in rows]
        policies = [str(row["selection_policy"]) for row in rows]
        reported_hashes = sorted(
            str(value) for value in manifest["selected_state_hashes"]
        )
        reported_counts = {
            str(game): int(count)
            for game, count in manifest["selected_counts_by_game"].items()
        }
        manifest_games = int(manifest["games_G"])
        manifest_states_per_game = int(manifest["states_per_game"])
        manifest_states = int(manifest["distinct_states_M"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"selection design contract is incomplete: {error}") from error

    population = {
        game: len(state_hashes) for game, state_hashes in population_hashes.items()
    }
    population_sets = {
        game: set(state_hashes) for game, state_hashes in population_hashes.items()
    }
    flattened_population = [
        state_hash
        for state_hashes in population_hashes.values()
        for state_hash in state_hashes
    ]
    realized_counts = dict(sorted(Counter(games).items()))
    if (
        expected_policy not in {"uniform_per_game_nested_v1", "top_score_per_game"}
        or expected_games <= 0
        or expected_states_per_game <= 0
        or expected_states != expected_games * expected_states_per_game
        or len(population) != expected_games
        or any(
            not game
            or count < expected_states_per_game
            or len(set(population_hashes[game])) != count
            for game, count in population.items()
        )
        or len(flattened_population) != len(set(flattened_population))
        or len(rows) != expected_states
        or len(set(state_hashes)) != expected_states
        or set(realized_counts) != set(population)
        or any(count != expected_states_per_game for count in realized_counts.values())
        or policies != [expected_policy] * expected_states
        or any(row.get("protocol_version") != "omniopd-v1" for row in rows)
        or any(
            game not in population_sets or state_hash not in population_sets[game]
            for state_hash, game in zip(state_hashes, games, strict=True)
        )
    ):
        raise ValueError(
            "selection rows do not realize exactly m states in every frozen-pool game"
        )

    expected_population_support = expected_policy == "uniform_per_game_nested_v1"
    if expected_population_support:
        if (
            isinstance(expected_seed, bool)
            or not isinstance(expected_seed, int)
            or expected_seed < 0
        ):
            raise ValueError("uniform nested selection requires a non-negative integer seed")
        expected_manifest_seed = expected_seed
    else:
        expected_manifest_seed = None
    expected_uniform_design = (
        "hash_priority_srswor_nested_within_game_v1"
        if expected_population_support
        else None
    )
    manifest_projection = {
        "experiment": manifest.get("experiment"),
        "policy": manifest.get("policy"),
        "seed": manifest.get("seed"),
        "games_G": manifest_games,
        "states_per_game": manifest_states_per_game,
        "distinct_states_M": manifest_states,
        "population_inference_supported": manifest.get(
            "population_inference_supported"
        ),
        "uniform_design": manifest.get("uniform_design"),
        "selected_state_hashes": reported_hashes,
        "selected_counts_by_game": reported_counts,
    }
    expected_projection = {
        "experiment": expected_experiment,
        "policy": expected_policy,
        "seed": expected_manifest_seed,
        "games_G": expected_games,
        "states_per_game": expected_states_per_game,
        "distinct_states_M": expected_states,
        "population_inference_supported": expected_population_support,
        "uniform_design": expected_uniform_design,
        "selected_state_hashes": sorted(state_hashes),
        "selected_counts_by_game": realized_counts,
    }
    if manifest_projection != expected_projection:
        raise ValueError(
            "selection manifest game/count/state claims do not match its rows or config"
        )

    selected_by_game: dict[str, list[str]] = {game: [] for game in population_hashes}
    for state_hash, game in zip(state_hashes, games, strict=True):
        selected_by_game[game].append(state_hash)

    normalized_scores: dict[str, float] | None = None
    if expected_population_support:
        if score_by_state is not None:
            raise ValueError("uniform nested selection cannot consume a score table")

        def uniform_priority(game: str, state_hash: str) -> tuple[bytes, str]:
            payload = (
                f"omniopd-uniform-priority-v1\0{expected_seed}\0{game}\0{state_hash}"
            ).encode("utf-8")
            return hashlib.sha256(payload).digest(), state_hash

        expected_by_game = {
            game: sorted(
                state_hashes_for_game,
                key=lambda state_hash, game=game: uniform_priority(game, state_hash),
            )[:expected_states_per_game]
            for game, state_hashes_for_game in population_hashes.items()
        }
    else:
        if not isinstance(score_by_state, Mapping):
            raise ValueError("top-score selection requires the complete bound score table")
        try:
            normalized_scores = {
                str(state_hash): float(score)
                for state_hash, score in score_by_state.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError(f"top-score table contains a non-numeric score: {error}") from error
        if (
            set(normalized_scores) != set(flattened_population)
            or not all(math.isfinite(score) for score in normalized_scores.values())
        ):
            raise ValueError("top-score table must map the complete frozen pool to finite scores")
        expected_by_game = {}
        for game, state_hashes_for_game in population_hashes.items():
            order = sorted(
                range(len(state_hashes_for_game)),
                key=lambda index: (-normalized_scores[state_hashes_for_game[index]], index),
            )[:expected_states_per_game]
            expected_by_game[game] = [state_hashes_for_game[index] for index in order]
    if any(
        set(selected_by_game[game]) != set(expected_by_game[game])
        for game in population_hashes
    ):
        raise ValueError(
            "selection rows are not the configured nested-uniform/top-score subset"
        )

    for row in rows:
        game = str(row["game_id"])
        state_hash = str(row["state_hash"])
        probability = row.get("inclusion_probability")
        score = row.get("score")
        if expected_population_support:
            expected_probability = expected_states_per_game / population[game]
            try:
                numeric_probability = float(probability)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "uniform selection probability is not numeric"
                ) from error
            if (
                isinstance(probability, bool)
                or probability is None
                or not math.isfinite(numeric_probability)
                or not math.isclose(
                    numeric_probability,
                    expected_probability,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or score is not None
            ):
                raise ValueError(
                    "uniform selection probability must equal realized m/T_g and have no score"
                )
        else:
            try:
                numeric_score = float(score)
            except (TypeError, ValueError) as error:
                raise ValueError("top-score row has a non-numeric score") from error
            if probability is not None or (
                isinstance(score, bool)
                or score is None
                or not math.isfinite(numeric_score)
                or not math.isclose(
                    numeric_score,
                    normalized_scores[state_hash],
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(
                    "top-score selection must have a finite score and no design probability"
                )
    return {
        "selected_state_hashes": sorted(state_hashes),
        "selected_counts_by_game": realized_counts,
        "population_states_by_game": dict(sorted(population.items())),
        "population_inference_supported": expected_population_support,
    }


def validate_uncertainty_score_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    pool_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Recompute entropy scores and their state/action binding from score rows."""

    rows = list(rows)
    try:
        normalized_pool = {
            str(state_hash): {
                "game_id": str(state["game_id"]),
                "turn_index": int(state["turn_index"]),
                "admissible_actions": [
                    str(action) for action in state["admissible_actions"]
                ],
            }
            for state_hash, state in pool_states.items()
        }
        row_hashes = [str(row["state_hash"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"uncertainty score state contract is malformed: {error}") from error
    if (
        not normalized_pool
        or len(row_hashes) != len(set(row_hashes))
        or set(row_hashes) != set(normalized_pool)
        or any(
            not state["game_id"]
            or state["turn_index"] < 0
            or not state["admissible_actions"]
            or len(state["admissible_actions"])
            != len(set(state["admissible_actions"]))
            for state in normalized_pool.values()
        )
    ):
        raise ValueError("uncertainty scores must map one-to-one onto the frozen state pool")

    score_map: dict[str, float] = {}
    for row in rows:
        state_hash = str(row["state_hash"])
        state = normalized_pool[state_hash]
        action_scores = row.get("action_log_scores")
        try:
            numeric_scores = {
                str(action): float(value) for action, value in action_scores.items()
            }
            reported_score = float(row["score"])
            reported_turn = int(row["turn_index"])
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"uncertainty score row is malformed: {error}") from error
        actions = state["admissible_actions"]
        if (
            row.get("protocol_version") != "omniopd-v1"
            or row.get("score_definition")
            != "entropy_of_softmax_sequence_action_logprob"
            or str(row.get("game_id")) != state["game_id"]
            or reported_turn != state["turn_index"]
            or set(numeric_scores) != set(actions)
            or any(
                isinstance(action_scores[action], bool)
                or not math.isfinite(numeric_scores[action])
                for action in actions
            )
            or isinstance(row.get("score"), bool)
            or not math.isfinite(reported_score)
        ):
            raise ValueError(
                "uncertainty row state/action/definition fields contradict the pool"
            )
        recomputed = admissible_entropy([numeric_scores[action] for action in actions])
        if not math.isclose(reported_score, recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("uncertainty entropy does not match its action log scores")
        score_map[state_hash] = reported_score
    return score_map


def validate_lora_checkpoint_contract(
    contract: Mapping[str, Any],
    *,
    expected_rank: int,
    expected_alpha: float,
    expected_target_modules_policy: str = "all-linear",
) -> dict[str, Any]:
    """Validate the path-independent PEFT adapter contract in a completion record."""

    required = {
        "format",
        "peft_type",
        "task_type",
        "rank",
        "lora_alpha",
        "target_modules_policy",
        "target_modules",
        "bias",
        "modules_to_save",
        "use_dora",
        "use_rslora",
        "rank_pattern",
        "alpha_pattern",
        "adapter_config_sha256",
        "adapter_weights",
        "adapter_tensors",
        "tokenizer_config_sha256",
    }
    if not isinstance(contract, Mapping) or set(contract) != required:
        raise ValueError("LoRA checkpoint contract is incomplete")
    weights = contract.get("adapter_weights")
    tensors = contract.get("adapter_tensors")
    targets = contract.get("target_modules")
    try:
        rank = int(contract["rank"])
        alpha = float(contract["lora_alpha"])
        weight_bytes = int(weights["bytes"])
        pair_count = int(tensors["pair_count"])
        tensor_count = int(tensors["tensor_count"])
        tensor_elements = int(tensors["elements"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"LoRA checkpoint numeric contract is invalid: {error}") from error

    def require_sha256(value: Any, role: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{role} is not a lowercase SHA-256 digest")
        return value

    if (
        contract.get("format") != "peft_lora_adapter"
        or str(contract.get("peft_type", "")).upper() != "LORA"
        or str(contract.get("task_type", "")).upper() != "CAUSAL_LM"
        or rank <= 0
        or rank != int(expected_rank)
        or not math.isfinite(alpha)
        or alpha <= 0
        or not math.isclose(alpha, float(expected_alpha), rel_tol=0.0, abs_tol=0.0)
        or contract.get("target_modules_policy") != expected_target_modules_policy
        or expected_target_modules_policy != "all-linear"
        or not isinstance(targets, list)
        or not targets
        or not all(isinstance(value, str) and value for value in targets)
        or targets != sorted(set(targets))
        or not isinstance(weights, Mapping)
        or set(weights) != {"filename", "bytes", "sha256"}
        or weights.get("filename") != "adapter_model.safetensors"
        or weight_bytes <= 0
        or contract.get("bias") != "none"
        or contract.get("modules_to_save") != []
        or contract.get("use_dora") is not False
        or contract.get("use_rslora") is not False
        or contract.get("rank_pattern") != {}
        or contract.get("alpha_pattern") != {}
        or not isinstance(tensors, Mapping)
        or set(tensors)
        != {"pair_count", "tensor_count", "elements", "dtypes", "names_sha256"}
        or pair_count <= 0
        or tensor_count != 2 * pair_count
        or tensor_elements <= 0
        or not isinstance(tensors.get("dtypes"), list)
        or not tensors["dtypes"]
        or not all(
            dtype in {"F16", "BF16", "F32", "F64"}
            for dtype in tensors["dtypes"]
        )
        or tensors["dtypes"] != sorted(set(tensors["dtypes"]))
    ):
        raise ValueError("checkpoint is not the configured non-empty PEFT LoRA adapter")
    require_sha256(contract.get("adapter_config_sha256"), "adapter config")
    require_sha256(weights.get("sha256"), "adapter weights")
    require_sha256(tensors.get("names_sha256"), "adapter tensor names")
    require_sha256(contract.get("tokenizer_config_sha256"), "tokenizer config")
    return {
        "format": "peft_lora_adapter",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "rank": rank,
        "lora_alpha": alpha,
        "target_modules_policy": expected_target_modules_policy,
        "target_modules": list(targets),
        "bias": "none",
        "modules_to_save": [],
        "use_dora": False,
        "use_rslora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "adapter_config_sha256": contract["adapter_config_sha256"],
        "adapter_weights": {
            "filename": weights["filename"],
            "bytes": weight_bytes,
            "sha256": weights["sha256"],
        },
        "adapter_tensors": {
            "pair_count": pair_count,
            "tensor_count": tensor_count,
            "elements": tensor_elements,
            "dtypes": list(tensors["dtypes"]),
            "names_sha256": tensors["names_sha256"],
        },
        "tokenizer_config_sha256": contract["tokenizer_config_sha256"],
    }


def _inspect_lora_safetensors(path: Path, *, expected_rank: int) -> dict[str, Any]:
    """Parse the safe, non-pickle tensor index and verify paired LoRA matrices."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("adapter safetensors file has no complete header length")
        header_size = int.from_bytes(prefix, byteorder="little", signed=False)
        if header_size <= 2 or header_size > file_size - 8:
            raise ValueError("adapter safetensors header length is invalid")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"adapter safetensors header is invalid JSON: {error}") from error
    if not isinstance(header, Mapping):
        raise ValueError("adapter safetensors header must be an object")
    tensors = {
        str(name): metadata
        for name, metadata in header.items()
        if name != "__metadata__"
    }
    if not tensors:
        raise ValueError("adapter safetensors file contains no tensors")

    dtype_bytes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
    data_size = file_size - 8 - header_size
    ranges: list[tuple[int, int]] = []
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    tensor_dtypes: dict[str, str] = {}
    total_elements = 0
    for name, metadata in tensors.items():
        if not name or not isinstance(metadata, Mapping):
            raise ValueError("adapter safetensors contains malformed tensor metadata")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            dtype not in dtype_bytes
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise ValueError(f"adapter tensor metadata is invalid for {name}")
        start, end = offsets
        elements = math.prod(shape)
        if (
            start < 0
            or end <= start
            or end > data_size
            or end - start != elements * dtype_bytes[dtype]
        ):
            raise ValueError(f"adapter tensor byte range is invalid for {name}")
        ranges.append((start, end))
        tensor_shapes[name] = tuple(shape)
        tensor_dtypes[name] = dtype
        total_elements += elements
    ordered_ranges = sorted(ranges)
    if (
        ordered_ranges[0][0] != 0
        or ordered_ranges[-1][1] != data_size
        or any(
            left[1] != right[0]
            for left, right in zip(ordered_ranges, ordered_ranges[1:])
        )
    ):
        raise ValueError(
            "adapter safetensors ranges overlap or do not cover the data section"
        )

    pattern = re.compile(
        r"^(?P<prefix>.+)\.lora_(?P<side>A|B)(?P<adapter>\.[^.]+)?\.weight$"
    )
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    for name in tensor_shapes:
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"adapter contains a non-LoRA tensor: {name}")
        pair = (match.group("prefix"), match.group("adapter") or "")
        side = match.group("side")
        if side in pairs.setdefault(pair, {}):
            raise ValueError(f"adapter repeats LoRA side {side} for {pair[0]}")
        pairs[pair][side] = name
    if not pairs:
        raise ValueError("adapter contains no LoRA A/B tensor pairs")
    for (prefix_name, _), pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"adapter has an unpaired LoRA tensor for {prefix_name}")
        a_shape = tensor_shapes[pair["A"]]
        b_shape = tensor_shapes[pair["B"]]
        if (
            len(a_shape) != 2
            or len(b_shape) != 2
            or a_shape[0] != expected_rank
            or b_shape[1] != expected_rank
            or tensor_dtypes[pair["A"]] != tensor_dtypes[pair["B"]]
        ):
            raise ValueError(f"LoRA A/B shape or dtype mismatch for {prefix_name}")
    return {
        "pair_count": len(pairs),
        "tensor_count": len(tensors),
        "elements": total_elements,
        "dtypes": sorted(set(tensor_dtypes.values())),
        "names_sha256": sha256_json(sorted(tensor_shapes)),
        "module_paths": sorted({prefix for prefix, _ in pairs}),
    }


def validate_lora_checkpoint_directory(
    path: str | Path,
    *,
    expected_rank: int,
    expected_alpha: float,
    expected_target_modules_policy: str = "all-linear",
) -> dict[str, Any]:
    """Inspect one saved checkpoint and return its canonical PEFT LoRA contract."""

    checkpoint = Path(path)
    config_path = checkpoint / "adapter_config.json"
    tokenizer_config_path = checkpoint / "tokenizer_config.json"
    weight_path = checkpoint / "adapter_model.safetensors"
    full_model_patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "pytorch_model.bin.index.json",
        "consolidated*.pth",
    )
    full_model_artifacts = sorted(
        {
            candidate.name
            for pattern in full_model_patterns
            for candidate in checkpoint.glob(pattern)
            if candidate.is_file()
        }
    )
    if (
        not checkpoint.is_dir()
        or not config_path.is_file()
        or not tokenizer_config_path.is_file()
        or not weight_path.is_file()
        or (checkpoint / "adapter_model.bin").exists()
        or full_model_artifacts
    ):
        raise ValueError(
            "checkpoint must contain one PEFT adapter config, one adapter weight file, "
            "and its tokenizer config, without a full-model weight payload"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_targets = config["target_modules"]
        targets = (
            [raw_targets]
            if isinstance(raw_targets, str)
            else sorted(str(value) for value in raw_targets)
        )
        tensor_contract = _inspect_lora_safetensors(
            weight_path, expected_rank=expected_rank
        )
        actual_module_paths = tensor_contract.pop("module_paths")
        if targets != ["all-linear"]:
            def target_matches(module_path: str, target: str) -> bool:
                return module_path == target or module_path.endswith("." + target)

            if any(
                not any(target_matches(module_path, target) for target in targets)
                for module_path in actual_module_paths
            ) or any(
                not any(
                    target_matches(module_path, target)
                    for module_path in actual_module_paths
                )
                for target in targets
            ):
                raise ValueError(
                    "PEFT target_modules do not cover the saved LoRA tensor modules"
                )
        contract = {
            "format": "peft_lora_adapter",
            "peft_type": config["peft_type"],
            "task_type": config["task_type"],
            "rank": config["r"],
            "lora_alpha": config["lora_alpha"],
            "target_modules_policy": expected_target_modules_policy,
            "target_modules": targets,
            "bias": config.get("bias", "none"),
            "modules_to_save": sorted(config.get("modules_to_save") or []),
            "use_dora": bool(config.get("use_dora", False)),
            "use_rslora": bool(config.get("use_rslora", False)),
            "rank_pattern": dict(config.get("rank_pattern") or {}),
            "alpha_pattern": dict(config.get("alpha_pattern") or {}),
            "adapter_config_sha256": sha256_file(config_path),
            "adapter_weights": {
                "filename": weight_path.name,
                "bytes": weight_path.stat().st_size,
                "sha256": sha256_file(weight_path),
            },
            "adapter_tensors": tensor_contract,
            "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"PEFT adapter config is malformed: {error}") from error
    return validate_lora_checkpoint_contract(
        contract,
        expected_rank=expected_rank,
        expected_alpha=expected_alpha,
        expected_target_modules_policy=expected_target_modules_policy,
    )


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
    experiments = []
    breadth_depth_pairs = []
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
            environment_seed = int(arm["state_pool"]["environment_seed"])
            profile = str(arm["teacher_sampling_profile"])
            experiment = str(arm["experiment"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"budget arm {index} is not fully resolved: {error}") from error
        if (
            games <= 0
            or states_per_game <= 0
            or states != games * states_per_game
            or pool_games != games
            or arm["state_pool"].get("state_source") != "student"
            or arm.get("selection") != "uniform_per_game_nested_v1"
            or arm.get("teacher_budget_definition") != "annotation_api_attempts"
            or arm.get("invalid_policy") != "retain_as_missing_no_free_retry"
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
            or environment_seed < 0
        ):
            raise ValueError(f"budget arm {index} has invalid pool/split/training seeds")
        validate_budget(states=states, samples_per_state=samples, declared_budget=budget)
        budgets.add(budget)
        sampling_profiles.add(profile)
        optimizer_steps.add(steps)
        training_seed_sets.add(training_seeds)
        experiments.append(experiment)
        breadth_depth_pairs.append((states, samples))
    if any(not value for value in experiments) or len(set(experiments)) != len(experiments):
        raise ValueError("fixed-budget arms must have distinct non-empty experiment identities")
    if len(set(breadth_depth_pairs)) < 2:
        raise ValueError("fixed-budget validation requires a realized breadth/depth trade-off")
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


def annotation_shared_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project the fields that must be identical across realized annotation arms."""

    def content_identity(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            key: content_identity(item)
            for key, item in value.items()
            if key not in {"path", "resolved_path", "value"}
        }

    return {
        "code": content_identity(manifest.get("code")),
        "code_revision": manifest.get("code_revision"),
        "state_pool_sha256": manifest.get("state_pool_sha256"),
        "state_pool_manifest_sha256": manifest.get("state_pool_manifest_sha256"),
        "behavior_student": manifest.get("behavior_student"),
        "teacher_model": manifest.get("teacher_model"),
        "teacher_model_revision": manifest.get("teacher_model_revision"),
        "teacher_url": manifest.get("teacher_url"),
        "teacher_sampling": manifest.get("teacher_sampling"),
        "teacher_max_tokens": manifest.get("teacher_max_tokens"),
        "teacher_context_window": manifest.get("teacher_context_window"),
        "teacher_tokenizer": content_identity(manifest.get("teacher_tokenizer")),
        "teacher_prompt_sha256": manifest.get("teacher_prompt_sha256"),
        "provider_response_models": manifest.get("provider_response_models"),
        "provider_system_fingerprints": manifest.get(
            "provider_system_fingerprints"
        ),
        "selection_policies": manifest.get("selection_policies"),
        "teacher_budget_definition": "annotation_api_attempts",
        "invalid_policy": "retain_as_missing_no_free_retry",
        "invalid_calls_count_toward_budget": manifest.get(
            "invalid_calls_count_toward_budget"
        ),
        "free_retries": manifest.get("free_retries"),
    }


def annotation_member_contract(
    manifest: Mapping[str, Any], *, annotation_manifest_sha256: str
) -> dict[str, Any]:
    """Project one realized annotation arm into the pair/training contract."""

    config = manifest.get("experiment_config")
    if not isinstance(config, Mapping):
        raise ValueError("annotation manifest has no experiment-config fingerprint")
    return {
        "experiment": manifest.get("experiment"),
        "annotation_manifest_sha256": annotation_manifest_sha256,
        "corrections_sha256": manifest.get("corrections_sha256"),
        "experiment_config_sha256": config.get("sha256"),
        "state_pool_sha256": manifest.get("state_pool_sha256"),
        "state_pool_manifest_sha256": manifest.get("state_pool_manifest_sha256"),
        "selection_sha256": manifest.get("selection_sha256"),
        "selection_manifest_sha256": manifest.get("selection_manifest_sha256"),
        "distinct_states_M": manifest.get("distinct_states_M"),
        "teacher_samples_per_state_N": manifest.get("teacher_samples_per_state_N"),
        "declared_teacher_budget_B": manifest.get("declared_teacher_budget_B"),
    }


def validate_correction_manifest_against_records(
    manifest: Mapping[str, Any], records: Iterable[CorrectionRecord]
) -> dict[str, Any]:
    """Recompute every count/set claim that the training pipeline trusts."""

    records = list(records)
    if not records:
        raise ValueError("correction artifact is empty")
    try:
        states = int(manifest["distinct_states_M"])
        samples = int(manifest["teacher_samples_per_state_N"])
        budget = int(manifest["declared_teacher_budget_B"])
        actual_calls = int(manifest["actual_teacher_api_calls"])
        reported_valid = int(manifest["valid_teacher_samples"])
        reported_invalid = int(manifest["invalid_teacher_samples"])
        reported_hashes = sorted(str(value) for value in manifest["selected_state_hashes"])
        reported_counts = {
            str(game): int(count)
            for game, count in manifest["selected_counts_by_game"].items()
        }
        reported_policies = sorted(str(value) for value in manifest["selection_policies"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"correction manifest count contract is incomplete: {error}") from error
    validate_budget(states=states, samples_per_state=samples, declared_budget=budget)
    realized_hashes = sorted(record.state.state_hash for record in records)
    realized_counts = dict(sorted(Counter(record.state.game_id for record in records).items()))
    realized_policies = sorted({record.selection_policy for record in records})
    realized_calls = sum(record.teacher_calls for record in records)
    realized_draws = sum(len(record.teacher_samples) for record in records)
    realized_valid = sum(len(record.valid_teacher_samples) for record in records)
    if (
        len(records) != states
        or len(set(realized_hashes)) != states
        or any(
            record.teacher_calls != samples
            or len(record.teacher_samples) != samples
            for record in records
        )
        or realized_calls != budget
        or realized_draws != budget
        or actual_calls != budget
        or reported_valid != realized_valid
        or reported_invalid != budget - realized_valid
        or reported_valid + reported_invalid != budget
        or reported_hashes != realized_hashes
        or reported_counts != realized_counts
        or reported_policies != realized_policies
        or manifest.get("invalid_calls_count_toward_budget") is not True
        or int(manifest.get("free_retries", -1)) != 0
    ):
        raise ValueError(
            "correction manifest M/N/B, validity, state, game, or selection claims "
            "do not match the correction records"
        )
    return {
        "shared_contract": annotation_shared_contract(manifest),
        "record_contract": {
            "distinct_states_M": states,
            "teacher_samples_per_state_N": samples,
            "declared_teacher_budget_B": budget,
            "actual_teacher_api_calls": realized_calls,
            "valid_teacher_samples": realized_valid,
            "invalid_teacher_samples": budget - realized_valid,
            "selected_state_hashes": realized_hashes,
            "selected_counts_by_game": realized_counts,
            "selection_policies": realized_policies,
        },
    }


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
        if manifest.get("provider_response_models") != [manifest.get("teacher_model")]:
            raise ValueError(
                "annotation provider must report exactly the requested Teacher model"
            )
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
    if len(manifests) != 2:
        raise ValueError("exactly two training launch manifests are required")
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

    from .evaluation import validate_training_launch_manifest

    identities = [validate_training_launch_manifest(manifest) for manifest in manifests]
    experiments = [str(identity["experiment"]) for identity in identities]
    if len(set(experiments)) != 2:
        raise ValueError("training comparison requires two distinct experiment arms")
    pair_hashes = {
        str(identity["annotation_pair_binding"]["pair_contract_sha256"])
        for identity in identities
    }
    pair_inputs = {
        json.dumps(
            content_identity(manifest["inputs"]["annotation_pair"]),
            sort_keys=True,
            allow_nan=False,
        )
        for manifest in manifests
    }
    roles = identities[0]["annotation_pair_binding"]["arm_roles"]
    if (
        len(pair_hashes) != 1
        or len(pair_inputs) != 1
        or set(roles.values()) != set(experiments)
        or identities[1]["annotation_pair_binding"]["arm_roles"] != roles
        or identities[0]["annotation_pair_binding"]["effect_direction"]
        != "breadth_minus_depth"
        or identities[1]["annotation_pair_binding"]["effect_direction"]
        != "breadth_minus_depth"
    ):
        raise ValueError(
            "training runs do not bind the same complete breadth/depth annotation pair"
        )

    def projection(manifest: Mapping[str, Any]) -> str:
        hyperparameters = dict(manifest.get("hyperparameters", {}))
        hyperparameters.pop("experiment_name", None)
        inputs = manifest.get("inputs", {})
        projected = {
            "code": manifest.get("code"),
            "code_revision": manifest.get("code_revision"),
            "implementation": content_identity(manifest.get("implementation")),
            "training_runtime": content_identity(manifest.get("training_runtime")),
            "git_worktrees_clean": manifest.get("git_worktrees_clean"),
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
