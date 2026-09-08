#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import (
    AlfworldEnvironment,
    OpenAIChatPolicy,
    list_alfworld_games,
    validate_provider_response_model_identity,
)
from omniopd.context import TaskPreservingTruncator
from omniopd.environment_provenance import (
    capture_runtime_dependencies,
    game_artifacts_digest,
    validate_environment_seed_contract,
    verify_game_artifacts,
    verify_runtime_dependencies,
)
from omniopd.evaluation import (
    validate_artifact_fingerprint,
    validate_service_attestation_manifest,
    validate_training_completion_manifest,
)
from omniopd.protocol import GenerationSettings, rollout_episode
from omniopd.prompts import STUDENT_SYSTEM_PROMPT
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
    sha256_json,
)
from omniopd.validation import validate_lora_checkpoint_directory

def _training_protocol(manifest: dict) -> dict:
    """Project a launch manifest onto fairness-relevant, seed-free fields."""

    contract = dict(manifest["training_contract"])
    contract.pop("seed", None)
    dataset_contract = dict(contract.get("dataset_contract", {}))
    dataset_contract.pop("seed", None)
    contract["dataset_contract"] = dataset_contract
    hyperparameters = dict(manifest.get("hyperparameters", {}))
    hyperparameters.pop("experiment_name", None)
    return {
        "implementation": manifest.get("implementation"),
        "training_contract": contract,
        "hyperparameters": hyperparameters,
        "base_model": manifest.get("inputs", {}).get("model_path"),
        "python": manifest.get("python"),
        "verl_version": manifest.get("verl_version"),
        "verl_version_declarations": manifest.get("verl_version_declarations"),
        "verl_git_revision": manifest.get("verl_git_revision"),
        "user_hydra_overrides": manifest.get("user_hydra_overrides", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic paired-game ALFWorld evaluation")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model-artifact", required=True)
    parser.add_argument("--checkpoint-artifact", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--training-completion-manifest", required=True)
    parser.add_argument("--service-manifest", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--inference-runtime",
        required=True,
        help="auditable engine/version label, e.g. vllm:0.10.2",
    )
    parser.add_argument(
        "--split",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
        required=True,
    )
    parser.add_argument("--limit-games", type=int, required=True)
    parser.add_argument("--game-list", required=True, help="immutable JSON list")
    parser.add_argument("--game-list-manifest", required=True)
    parser.add_argument(
        "--training-seed",
        type=int,
        required=True,
        help="identity of the trained checkpoint; not a decoding control",
    )
    parser.add_argument(
        "--rollout-seed",
        type=int,
        required=True,
        help="fixed Student decoding seed shared across compared checkpoints",
    )
    parser.add_argument(
        "--environment-seed",
        type=int,
        required=True,
        help="frozen TextWorld master seed from the game-list manifest",
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.output)
    game_list_path = Path(args.game_list)
    game_list_manifest_path = Path(args.game_list_manifest)
    requests_output = Path(str(output) + ".requests.jsonl")
    requests_partial = Path(str(output) + ".requests.partial.jsonl")
    if output.exists() or requests_output.exists() or requests_partial.exists():
        raise SystemExit("refusing to overwrite an evaluation artifact or request ledger")
    if (
        args.limit_games <= 0
        or args.max_steps <= 0
        or args.max_tokens <= 0
        or args.reserve_tokens < args.max_tokens
        or args.max_context_tokens <= args.reserve_tokens
    ):
        raise SystemExit(
            "games/steps/tokens must be positive, reserve must cover generation, "
            "and context must exceed reserve"
        )
    if args.training_seed < 0 or args.rollout_seed < 0 or args.environment_seed < 0:
        raise SystemExit("training, decoding, and environment seeds must be non-negative")
    if not args.inference_runtime.strip():
        raise SystemExit("--inference-runtime must be non-empty")
    if not Path(args.env_config).is_file() or not game_list_path.is_file():
        raise SystemExit("environment config and frozen game list must exist")
    if not game_list_manifest_path.is_file():
        raise SystemExit("--game-list-manifest must identify the frozen game list")
    if not Path(args.tokenizer).exists():
        raise SystemExit("--tokenizer must be a local immutable path that can be fingerprinted")
    base_model_artifact = Path(args.base_model_artifact)
    checkpoint_artifact = Path(args.checkpoint_artifact)
    training_manifest_path = Path(args.training_manifest)
    training_completion_manifest_path = Path(args.training_completion_manifest)
    service_manifest_path = Path(args.service_manifest)
    missing_artifacts = [
        str(path)
        for path in [
            base_model_artifact,
            checkpoint_artifact,
            training_manifest_path,
            training_completion_manifest_path,
            service_manifest_path,
        ]
        if not path.exists()
    ]
    if missing_artifacts:
        raise SystemExit(f"model artifacts do not exist: {missing_artifacts}")

    from transformers import AutoTokenizer

    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    training_completion_manifest = json.loads(
        training_completion_manifest_path.read_text(encoding="utf-8")
    )
    service_manifest = json.loads(service_manifest_path.read_text(encoding="utf-8"))
    base_fingerprint = fingerprint_path(base_model_artifact)
    checkpoint_fingerprint = fingerprint_path(checkpoint_artifact)
    tokenizer_fingerprint = fingerprint_path(args.tokenizer)
    try:
        completion_identity = validate_training_completion_manifest(
            training_completion_manifest,
            training_manifest,
            launch_manifest_sha256=sha256_file(training_manifest_path),
            completion_manifest_sha256=sha256_file(
                training_completion_manifest_path
            ),
            checkpoint_fingerprint=checkpoint_fingerprint,
            current_checkpoint_format=validate_lora_checkpoint_directory(
                checkpoint_artifact,
                expected_rank=int(training_manifest["hyperparameters"]["lora_rank"]),
                expected_alpha=float(training_manifest["hyperparameters"]["lora_alpha"]),
                expected_target_modules_policy=str(
                    training_manifest["hyperparameters"]["target_modules"]
                ),
            ),
            resolved_config_fingerprint=fingerprint_path(
                training_manifest_path.parent / "resolved_config.yaml"
            ),
            training_log_fingerprint=fingerprint_path(
                training_manifest_path.parent / "train.log"
            ),
        )
    except ValueError as error:
        raise SystemExit(f"invalid training completion chain: {error}") from error
    manifest_training_seed = int(completion_identity["training_seed"])
    total_training_steps = int(completion_identity["total_optimizer_steps"])
    expected_checkpoint = training_manifest_path.parent / (
        f"global_step_{total_training_steps}"
    )
    if (
        not current_revision
        or training_manifest.get("code") != current_code
        or training_manifest.get("code_revision") != current_revision
        or manifest_training_seed != args.training_seed
        or checkpoint_artifact.resolve() != expected_checkpoint.resolve()
    ):
        raise SystemExit(
            "evaluation checkpoint/seed/code must match a successfully completed canonical run"
        )
    try:
        validate_artifact_fingerprint(
            training_manifest["inputs"]["model_path"],
            base_fingerprint,
            role="training/evaluation base model",
        )
        validate_artifact_fingerprint(
            base_fingerprint,
            tokenizer_fingerprint,
            role="complete base-model tokenizer",
        )
        for role, recorded in training_manifest["inputs"].items():
            recorded_path = recorded.get("resolved_path")
            if not isinstance(recorded_path, str) or not recorded_path:
                raise ValueError(f"training input {role} has no resolved local path")
            validate_artifact_fingerprint(
                recorded,
                fingerprint_path(recorded_path),
                role=f"training input {role}",
            )
        service_identity = validate_service_attestation_manifest(
            service_manifest,
            requested_model=args.model,
            base_url=args.base_url,
            inference_runtime=args.inference_runtime,
            base_fingerprint=base_fingerprint,
            checkpoint_fingerprint=checkpoint_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            completion_manifest_sha256=sha256_file(
                training_completion_manifest_path
            ),
            launch_manifest_sha256=sha256_file(training_manifest_path),
            wrapper_fingerprint=fingerprint_path(ROOT / "scripts/run_vllm_eval.sh"),
            expected_parent_pid=os.getppid(),
            require_live_process=True,
        )
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid training/service artifact binding: {error}") from error
    if int(service_identity["max_model_len"]) < args.max_context_tokens:
        raise SystemExit("attested vLLM context limit is smaller than evaluation context")
    game_list_manifest = json.loads(
        game_list_manifest_path.read_text(encoding="utf-8")
    )
    if (
        not current_revision
        or game_list_manifest.get("artifact") != "frozen_game_list"
        or game_list_manifest.get("protocol_version") != "omniopd-v1"
        or game_list_manifest.get("game_list_sha256") != sha256_file(game_list_path)
        or game_list_manifest.get("env_config_sha256") != sha256_file(args.env_config)
        or game_list_manifest.get("split") != args.split
        or int(game_list_manifest.get("games", -1)) != args.limit_games
        or game_list_manifest.get("code") != current_code
        or game_list_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit(
            "evaluation game list must match a frozen manifest from this code revision"
        )
    try:
        runtime_dependencies = capture_runtime_dependencies()
        verify_runtime_dependencies(
            game_list_manifest.get("runtime_dependencies", {}),
            actual=runtime_dependencies,
        )
        game_artifacts = verify_game_artifacts(
            game_list_manifest.get("game_artifacts", [])
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"evaluation environment provenance failed: {error}") from error
    if game_list_manifest.get("game_artifacts_sha256") != game_artifacts_digest(
        game_artifacts
    ):
        raise SystemExit("frozen game-artifact digest is missing or inconsistent")
    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    games = json.loads(game_list_path.read_text(encoding="utf-8"))
    if len(games) != args.limit_games:
        raise SystemExit(f"expected exactly {args.limit_games} games, got {len(games)}")
    games = [str(game) for game in games]
    if len(set(games)) != len(games):
        raise SystemExit("evaluation game list contains duplicates")
    if games != [str(row["game_id"]) for row in game_artifacts]:
        raise SystemExit("frozen game list and byte-level game artifacts disagree")
    try:
        environment_rollout = game_list_manifest["environment_rollout"]
        environment_seeds = validate_environment_seed_contract(
            environment_rollout,
            game_ids=games,
            expected_master_seed=args.environment_seed,
        )
    except (KeyError, ValueError) as error:
        raise SystemExit(f"evaluation environment seed contract failed: {error}") from error
    eligible_games = set(list_alfworld_games(config, args.split))
    outside_split = sorted(set(games) - eligible_games)
    if outside_split:
        raise SystemExit(
            f"frozen game list contains games outside split {args.split}: {outside_split[:5]}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = OpenAIChatPolicy(
        model=args.model,
        base_url=args.base_url,
        api_key="EMPTY",
        thinking_mode="disabled",
        thinking_control="chat_template",
        seed=args.rollout_seed,
        request_ledger_path=requests_partial,
    )
    per_game = {}
    traces = []
    for game in games:
        env = AlfworldEnvironment(
            config,
            game,
            rollout_seed=environment_seeds[game],
        )
        try:
            turns, won = rollout_episode(
                env,
                policy,
                truncator,
                settings=GenerationSettings(
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                ),
                max_steps=args.max_steps,
            )
        finally:
            env.close()
        per_game[game] = int(won)
        traces.append(
            {
                "game_id": game,
                "environment_rollout_seed": environment_seeds[game],
                "won": won,
                "turns": [turn.to_dict() for turn in turns],
            }
        )
    expected_requests = sum(len(trace["turns"]) for trace in traces)
    if len(policy.request_ledger) != expected_requests:
        raise RuntimeError("evaluation request ledger length does not match rollout turns")
    ledger_by_id = {str(row["request_id"]): row for row in policy.request_ledger}
    if len(ledger_by_id) != expected_requests:
        raise RuntimeError("evaluation request ledger contains duplicate request IDs")
    for trace in traces:
        for serialized_turn in trace["turns"]:
            state = serialized_turn["state"]
            request_id = f"behavior:student:{state['state_hash']}"
            if (
                request_id not in ledger_by_id
                or ledger_by_id[request_id]["messages_sha256"]
                != sha256_json(state["messages"])
            ):
                raise RuntimeError("evaluation ledger context does not match its state trace")
    provider_response_models = validate_provider_response_model_identity(
        policy.request_ledger, args.model
    )
    try:
        verify_game_artifacts(game_artifacts)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise RuntimeError("ALFWorld game content changed during evaluation") from error
    requests_partial.rename(requests_output)
    result = {
        "protocol_version": "omniopd-v1",
        "code": current_code,
        "code_revision": current_revision,
        "experiment": completion_identity["experiment"],
        "arm_contract": completion_identity["arm_contract"],
        "model": args.model,
        "inference_runtime": args.inference_runtime,
        "split": args.split,
        "training_seed": args.training_seed,
        "rollout_seed": args.rollout_seed,
        "student_decoding_seed": args.rollout_seed,
        "environment_seed": args.environment_seed,
        "environment_rollout": environment_rollout,
        "runtime_dependencies": runtime_dependencies,
        "game_artifacts_sha256": game_artifacts_digest(game_artifacts),
        "temperature": 0.0,
        "games": games,
        "game_list_sha256": hashlib.sha256(
            json.dumps(games, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "game_list_file_sha256": sha256_file(game_list_path),
        "game_list_manifest_sha256": sha256_file(game_list_manifest_path),
        "env_config_sha256": sha256_file(args.env_config),
        "student_prompt_sha256": hashlib.sha256(
            STUDENT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "tokenizer": tokenizer_fingerprint,
        "base_model_artifact": base_fingerprint,
        "checkpoint_artifact": checkpoint_fingerprint,
        "model_artifacts": [base_fingerprint, checkpoint_fingerprint],
        "training_inputs": training_manifest["inputs"],
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "training_launch_manifest_sha256": sha256_file(training_manifest_path),
        "training_completion_manifest_sha256": sha256_file(
            training_completion_manifest_path
        ),
        "service_manifest_sha256": sha256_file(service_manifest_path),
        "training_completion": completion_identity,
        "service_attestation": service_identity,
        "training_protocol": _training_protocol(training_manifest),
        "request_ledger_sha256": sha256_file(requests_output),
        "request_count": expected_requests,
        "provider_response_models": provider_response_models,
        "provider_system_fingerprints": sorted(
            {
                str(row["system_fingerprint"])
                for row in policy.request_ledger
                if row.get("system_fingerprint") is not None
            }
        ),
        "thinking_mode": "disabled",
        "thinking_control": "chat_template",
        "max_steps": args.max_steps,
        "max_context_tokens": args.max_context_tokens,
        "reserve_tokens": args.reserve_tokens,
        "max_tokens": args.max_tokens,
        "success_rate": sum(per_game.values()) / len(per_game),
        "per_game": per_game,
        "traces": traces,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
