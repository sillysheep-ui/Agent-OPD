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
    content_fingerprint_identity,
    derive_environment_seed,
    fingerprint_game_artifacts,
    game_artifacts_digest,
    verify_game_artifacts,
)
from omniopd.io import read_jsonl, record_game_id, write_jsonl
from omniopd.evaluation import validate_state_pool_service_attestation_manifest
from omniopd.prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT
from omniopd.protocol import GenerationSettings, rollout_episode
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
    sha256_json,
)
from omniopd.sampling import SamplingProtocol, stable_shuffled


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one immutable behavior-policy state pool")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--behavior-model", required=True)
    parser.add_argument(
        "--behavior-model-revision",
        help="required immutable provider snapshot label for Teacher-state collection",
    )
    parser.add_argument(
        "--behavior-artifact",
        action="append",
        default=[],
        help="local base/checkpoint/adapter path; repeat for composed models",
    )
    parser.add_argument("--behavior-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--inference-runtime",
        required=True,
        help="immutable serving-runtime identifier, e.g. vllm:0.8.5",
    )
    parser.add_argument("--behavior-api-key")
    parser.add_argument(
        "--behavior-service-manifest",
        help="required live local-service attestation for Student-state collection",
    )
    parser.add_argument(
        "--behavior-thinking-mode",
        choices=["enabled", "disabled", "provider_default"],
        required=True,
    )
    parser.add_argument("--behavior-temperature", type=float)
    parser.add_argument(
        "--behavior-reasoning-effort", choices=["low", "high", "max"]
    )
    parser.add_argument("--state-source", choices=["student", "teacher"], required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--limit-games", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--behavior-max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--environment-seed",
        type=int,
        required=True,
        help="master RNG seed deterministically expanded to one seed per game",
    )
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    if not Path(args.env_config).is_file():
        raise SystemExit("--env-config must exist")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("state-pool collection requires an immutable Git revision")
    if not Path(args.tokenizer).exists():
        raise SystemExit("--tokenizer must be a local immutable path that can be fingerprinted")
    if args.state_source == "student" and len(args.behavior_artifact) != 1:
        raise SystemExit("Student-state collection requires exactly one --behavior-artifact")
    if args.state_source == "student" and not args.behavior_service_manifest:
        raise SystemExit("Student-state collection requires --behavior-service-manifest")
    if args.state_source == "teacher" and args.behavior_service_manifest:
        raise SystemExit("remote Teacher-state collection must not claim a local service manifest")
    if args.state_source == "teacher" and not (args.behavior_model_revision or "").strip():
        raise SystemExit("Teacher-state collection requires --behavior-model-revision")
    missing_artifacts = [path for path in args.behavior_artifact if not Path(path).exists()]
    if missing_artifacts:
        raise SystemExit(f"behavior artifacts do not exist: {missing_artifacts}")
    if (
        args.behavior_max_tokens <= 0
        or args.reserve_tokens < args.behavior_max_tokens
        or args.seed < 0
        or args.environment_seed < 0
        or args.limit_games <= 0
        or args.max_steps <= 0
        or args.max_context_tokens <= args.reserve_tokens
        or not args.inference_runtime.strip()
    ):
        raise SystemExit(
            "token limits must be coherent, seeds non-negative, and inference runtime non-empty"
        )
    try:
        runtime_dependencies = capture_runtime_dependencies()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    tokenizer_fingerprint = fingerprint_path(args.tokenizer)
    behavior_artifacts = [fingerprint_path(path) for path in args.behavior_artifact]
    if (
        args.state_source == "student"
        and (
            behavior_artifacts[0].get("kind") != "directory"
            or int(behavior_artifacts[0].get("files", 0)) <= 0
            or content_fingerprint_identity(behavior_artifacts[0])
            != content_fingerprint_identity(tokenizer_fingerprint)
        )
    ):
        raise SystemExit(
            "canonical Student-state collection requires one complete model directory "
            "that is also the tokenizer artifact"
        )
    behavior_service_identity = None
    behavior_service_manifest_sha256 = None
    if args.state_source == "student":
        service_path = Path(str(args.behavior_service_manifest))
        if not service_path.is_file():
            raise SystemExit("--behavior-service-manifest must exist for Student-state collection")
        service_manifest = json.loads(service_path.read_text(encoding="utf-8"))
        try:
            behavior_service_identity = validate_state_pool_service_attestation_manifest(
                service_manifest,
                requested_model=args.behavior_model,
                base_url=args.behavior_url,
                inference_runtime=args.inference_runtime,
                model_fingerprint=behavior_artifacts[0],
                tokenizer_fingerprint=tokenizer_fingerprint,
                wrapper_fingerprint=fingerprint_path(ROOT / "scripts/run_vllm_state_pool.sh"),
                expected_parent_pid=os.getppid(),
                require_live_process=True,
            )
        except (KeyError, ValueError) as error:
            raise SystemExit(f"invalid Student behavior-service binding: {error}") from error
        if int(behavior_service_identity["max_model_len"]) < args.max_context_tokens:
            raise SystemExit("attested Student service context is smaller than state-pool context")
        behavior_service_manifest_sha256 = sha256_file(service_path)

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for path in args.exclude_jsonl:
        for record in read_jsonl(path):
            game = record_game_id(record)
            if game is not None:
                excluded.add(game)
    games = stable_shuffled(
        (
            game
            for game in list_alfworld_games(config, args.split)
            if game not in excluded
        ),
        seed=args.seed,
    )
    games = [str(game) for game in games[: args.limit_games]]
    if len(games) != args.limit_games or len(games) != len(set(games)):
        raise SystemExit(
            f"expected {args.limit_games} unique eligible games after exclusions, got "
            f"{len(set(games))}"
        )
    try:
        game_artifacts = fingerprint_game_artifacts(games)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    environment_seeds = {
        game: derive_environment_seed(args.environment_seed, game) for game in games
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
        enable_thinking=args.behavior_thinking_mode == "enabled",
    )
    behavior_key = args.behavior_api_key or os.environ.get("BEHAVIOR_API_KEY")
    if args.state_source == "teacher" and not behavior_key:
        raise SystemExit(
            "teacher-state collection requires --behavior-api-key or BEHAVIOR_API_KEY"
        )
    sampling = SamplingProtocol(
        args.behavior_thinking_mode,
        args.behavior_temperature,
        args.behavior_reasoning_effort,
    )
    sampling.validate()
    if args.state_source == "teacher" and sampling.thinking_mode == "provider_default":
        raise SystemExit("teacher-state collection requires an explicit thinking mode")
    if args.state_source == "student" and sampling.thinking_mode != "disabled":
        raise SystemExit(
            "canonical Qwen Student-state collection requires explicit non-thinking mode"
        )
    output.mkdir(parents=True, exist_ok=True)
    requests_partial = output / "behavior_requests.partial.jsonl"
    behavior = OpenAIChatPolicy(
        model=args.behavior_model,
        base_url=args.behavior_url,
        api_key=behavior_key or "EMPTY",
        thinking_mode=sampling.policy_thinking_mode,
        thinking_control=(
            "chat_template" if args.state_source == "student" else "deepseek"
        ),
        reasoning_effort=sampling.reasoning_effort,
        seed=args.seed if args.state_source == "student" else None,
        request_ledger_path=requests_partial,
    )
    prompt = STUDENT_SYSTEM_PROMPT if args.state_source == "student" else TEACHER_SYSTEM_PROMPT
    episodes = []
    for game in games:
        game_environment_seed = environment_seeds[game]
        env = AlfworldEnvironment(
            config,
            game,
            rollout_seed=game_environment_seed,
        )
        try:
            turns, won = rollout_episode(
                env,
                behavior,
                truncator,
                settings=GenerationSettings(
                    temperature=sampling.temperature,
                    max_tokens=args.behavior_max_tokens,
                ),
                max_steps=args.max_steps,
                system_prompt=prompt,
                state_source=args.state_source,
            )
        finally:
            env.close()
        episodes.append(
            {
                "game_id": game,
                "environment_rollout_seed": game_environment_seed,
                "won": won,
                "turns": [turn.to_dict() for turn in turns],
            }
        )

    state_count = sum(len(episode["turns"]) for episode in episodes)
    if state_count == 0:
        raise RuntimeError("the frozen state pool is empty")
    state_hashes = [
        turn["state"]["state_hash"] for episode in episodes for turn in episode["turns"]
    ]
    if len(state_hashes) != len(set(state_hashes)):
        raise RuntimeError("duplicate state hashes in the newly collected pool")
    games_path = output / "games.json"
    episodes_path = output / "episodes.jsonl"
    state_pool_path = output / "state_pool.jsonl"
    requests_path = output / "behavior_requests.jsonl"
    games_path.write_text(json.dumps(games, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(episodes_path, episodes)
    write_jsonl(
        state_pool_path,
        (turn for episode in episodes for turn in episode["turns"]),
    )
    if len(behavior.request_ledger) != state_count:
        raise RuntimeError("behavior request ledger length does not match collected states")
    ledger_by_id = {
        str(row["request_id"]): row for row in behavior.request_ledger
    }
    if len(ledger_by_id) != state_count:
        raise RuntimeError("behavior request ledger contains duplicate request IDs")
    for episode in episodes:
        for serialized_turn in episode["turns"]:
            state = serialized_turn["state"]
            request_id = f"behavior:{args.state_source}:{state['state_hash']}"
            if (
                request_id not in ledger_by_id
                or ledger_by_id[request_id]["messages_sha256"]
                != sha256_json(state["messages"])
            ):
                raise RuntimeError("behavior ledger context does not match the frozen state")
    provider_response_models = validate_provider_response_model_identity(
        behavior.request_ledger, args.behavior_model
    )
    try:
        verify_game_artifacts(game_artifacts)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise RuntimeError(
            "ALFWorld game content changed during state-pool collection"
        ) from error
    requests_partial.rename(requests_path)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "immutable_state_pool",
        "code": current_code,
        "code_revision": current_revision,
        "state_source": args.state_source,
        "behavior_model": args.behavior_model,
        "behavior_model_revision": args.behavior_model_revision,
        "behavior_url": args.behavior_url,
        "inference_runtime": args.inference_runtime,
        "runtime_dependencies": runtime_dependencies,
        "behavior_sampling": sampling.to_dict(),
        "behavior_api_calls": state_count,
        "provider_response_models": provider_response_models,
        "provider_system_fingerprints": sorted(
            {
                str(row["system_fingerprint"])
                for row in behavior.request_ledger
                if row.get("system_fingerprint") is not None
            }
        ),
        "teacher_rollout_calls": state_count if args.state_source == "teacher" else 0,
        "games_G": len(games),
        "states": state_count,
        "seed": args.seed,
        "game_order_seed": args.seed,
        "student_decoding_seed": args.seed if args.state_source == "student" else None,
        "environment_rollout": {
            "master_seed": args.environment_seed,
            "derivation": "sha256_omniopd_env_seed_v1_uint32",
            "per_game": [
                {"game_id": game, "seed": environment_seeds[game]}
                for game in games
            ],
            "adapter_seed_method": "textworld_gym_env.seed_before_first_reset",
            "alfred_demangler_shuffle": False,
        },
        "max_steps": args.max_steps,
        "max_context_tokens": args.max_context_tokens,
        "reserve_tokens": args.reserve_tokens,
        "behavior_max_tokens": args.behavior_max_tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tokenizer": tokenizer_fingerprint,
        "behavior_artifacts": behavior_artifacts,
        "behavior_service_manifest_sha256": behavior_service_manifest_sha256,
        "behavior_service_attestation": behavior_service_identity,
        "service_trust_boundary": (
            "local Student: host-local argv/parent/port/content observation; remote Teacher: "
            "provider response.model plus operator-supplied revision, not cryptographic attestation"
        ),
        "game_artifacts": game_artifacts,
        "game_artifacts_sha256": game_artifacts_digest(game_artifacts),
        "env_config_sha256": sha256_file(args.env_config),
        "exclusion_jsonl_sha256": {
            str(path): sha256_file(path) for path in args.exclude_jsonl
        },
        "outputs": {
            "games_sha256": sha256_file(games_path),
            "episodes_sha256": sha256_file(episodes_path),
            "state_pool_sha256": sha256_file(state_pool_path),
            "behavior_requests_sha256": sha256_file(requests_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
