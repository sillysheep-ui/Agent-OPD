#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import AlfworldEnvironment, OpenAIChatPolicy, list_alfworld_games
from omniopd.context import TaskPreservingTruncator
from omniopd.protocol import GenerationSettings, rollout_episode
from omniopd.prompts import STUDENT_SYSTEM_PROMPT
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
    sha256_json,
)


def _artifact_digest(value: dict) -> str | None:
    return value.get("sha256") or value.get("tree_sha256")


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
        help="fixed decoding/environment seed shared across compared checkpoints",
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
    if args.max_tokens <= 0 or args.reserve_tokens < args.max_tokens:
        raise SystemExit(
            "--reserve-tokens must be at least --max-tokens and both must be positive"
        )
    if args.training_seed < 0 or args.rollout_seed < 0:
        raise SystemExit("training and rollout seeds must be non-negative")
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
    missing_artifacts = [
        str(path)
        for path in [
            base_model_artifact,
            checkpoint_artifact,
            training_manifest_path,
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
    training_contract = training_manifest.get("training_contract", {})
    base_fingerprint = fingerprint_path(base_model_artifact)
    checkpoint_fingerprint = fingerprint_path(checkpoint_artifact)
    try:
        manifest_training_seed = int(training_contract["seed"])
        total_training_steps = int(training_contract["total_optimizer_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"training launch manifest is incomplete: {error}") from error
    expected_checkpoint = training_manifest_path.parent / (
        f"global_step_{total_training_steps}"
    )
    if (
        not current_revision
        or training_manifest.get("artifact") != "omniopd_training_launch"
        or training_manifest.get("protocol_version") != "omniopd-v1"
        or training_manifest.get("code") != current_code
        or training_manifest.get("code_revision") != current_revision
        or manifest_training_seed != args.training_seed
        or checkpoint_artifact.resolve() != expected_checkpoint.resolve()
        or _artifact_digest(
            training_manifest.get("inputs", {}).get("model_path", {})
        )
        != _artifact_digest(base_fingerprint)
    ):
        raise SystemExit(
            "evaluation checkpoint/base/seed must match the final checkpoint declared "
            "by a canonical training launch manifest"
        )
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
    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    games = json.loads(game_list_path.read_text(encoding="utf-8"))
    if len(games) != args.limit_games:
        raise SystemExit(f"expected exactly {args.limit_games} games, got {len(games)}")
    games = [str(game) for game in games]
    if len(set(games)) != len(games):
        raise SystemExit("evaluation game list contains duplicates")
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
        env = AlfworldEnvironment(config, game)
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
        traces.append({"game_id": game, "won": won, "turns": [turn.to_dict() for turn in turns]})
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
    provider_response_models = sorted(
        {
            str(row["response_model"])
            for row in policy.request_ledger
            if row.get("response_model") is not None
        }
    )
    if provider_response_models != [args.model]:
        raise RuntimeError(
            "evaluation provider must report exactly the requested stable served-model alias; "
            f"requested={args.model!r}, reported={provider_response_models}"
        )
    requests_partial.rename(requests_output)
    result = {
        "protocol_version": "omniopd-v1",
        "code": current_code,
        "code_revision": current_revision,
        "model": args.model,
        "inference_runtime": args.inference_runtime,
        "split": args.split,
        "training_seed": args.training_seed,
        "rollout_seed": args.rollout_seed,
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
        "tokenizer": fingerprint_path(args.tokenizer),
        "base_model_artifact": base_fingerprint,
        "checkpoint_artifact": checkpoint_fingerprint,
        "model_artifacts": [base_fingerprint, checkpoint_fingerprint],
        "training_manifest_sha256": sha256_file(training_manifest_path),
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
