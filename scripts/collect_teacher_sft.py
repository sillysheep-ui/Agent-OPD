#!/usr/bin/env python3
"""Collect demonstration trajectories from a served teacher policy.

Route A of the agreed plan: a capable teacher is rolled out on the train split
with a reference prompt, and only the episodes it actually solves become
cold-start supervision for the Student.  The output schema matches the scripted
expert collector, so the same data builder and trainer consume both.

Reused machinery: the ALFWorld game enumeration and per-task-type sampling from
``collect_expert_sft``, the shared ``rollout_episode`` loop, the
task-preserving truncator, and the provenance helpers.  Nothing here trains.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The repository root has to be importable for the shared scripts package.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import (  # noqa: E402
    AlfworldEnvironment,
    OpenAIChatPolicy,
    extract_task_type,
    validate_provider_response_model_identity,
)
from omniopd.context import TaskPreservingTruncator  # noqa: E402
from omniopd.environment_provenance import (  # noqa: E402
    capture_runtime_dependencies,
    derive_environment_seed,
    fingerprint_game_artifacts,
    game_artifacts_digest,
)
from omniopd.prompts import build_reference_prompt  # noqa: E402
from omniopd.protocol import GenerationSettings, rollout_episode  # noqa: E402
from omniopd.provenance import (  # noqa: E402
    fingerprint_code_tree,
    git_revision,
    sha256_file,
    sha256_json,
    sha256_text,
)
from scripts.collect_expert_sft import (  # noqa: E402
    CANONICAL_TASK_TYPES,
    resolve_game_list,
    select_games,
    task_family_from_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--game-list-cache", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-types", default=",".join(CANONICAL_TASK_TYPES))
    parser.add_argument("--solved-per-task-type", type=int, default=20)
    parser.add_argument("--max-attempts-per-task-type", type=int, default=60)
    parser.add_argument("--game-order-seed", type=int, default=42)
    parser.add_argument("--environment-master-seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config_path = args.env_config.resolve()
    tokenizer_path = args.tokenizer.resolve()
    prompt_json_path = args.prompt_json.resolve()
    output = args.output.resolve()
    task_types = tuple(item.strip() for item in args.task_types.split(",") if item.strip())
    if (
        args.split != "train"
        or args.solved_per_task_type <= 0
        or args.max_attempts_per_task_type < args.solved_per_task_type
        or args.game_order_seed < 0
        or args.environment_master_seed < 0
        or args.max_steps <= 0
        or args.max_tokens <= 0
        or args.reserve_tokens < args.max_tokens
        or args.max_context_tokens <= args.reserve_tokens
        or output.exists()
        or not env_config_path.is_file()
        or not prompt_json_path.is_file()
        or not (tokenizer_path / "tokenizer_config.json").is_file()
        or not task_types
        or len(set(task_types)) != len(task_types)
    ):
        raise SystemExit("invalid teacher trajectory collection input or output")

    import yaml
    from transformers import AutoTokenizer

    reference = json.loads(prompt_json_path.read_text(encoding="utf-8"))
    system_prompt = build_reference_prompt(
        str(reference["instruction"]), reference.get("examples") or []
    )
    config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    cache_path = args.game_list_cache.resolve() if args.game_list_cache else None
    games, game_list_source = resolve_game_list(
        config=config, split=args.split, cache_path=cache_path, task_types=task_types
    )
    candidates = select_games(
        games,
        task_types=task_types,
        games_per_task_type=args.max_attempts_per_task_type,
        seed=args.game_order_seed,
    )
    candidates_by_type: dict[str, list[str]] = defaultdict(list)
    for game in candidates:
        candidates_by_type[task_family_from_path(game)].append(game)

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
        enable_thinking=False,
    )
    policy = OpenAIChatPolicy(
        model=args.model,
        base_url=args.base_url,
        api_key="EMPTY",
        thinking_mode="disabled",
        thinking_control="chat_template",
        max_retries=0,
    )

    output.mkdir(parents=True)
    episodes_path = output / "episodes.jsonl"
    environment_seeds: dict[str, int] = {}
    tasks_by_game: dict[str, str] = {}
    per_task_type = {
        task_type: {"attempts": 0, "solved": 0} for task_type in task_types
    }
    counts = {"games_attempted": 0, "games_won": 0, "turns": 0, "invalid_turns": 0}
    attempted: list[str] = []
    with episodes_path.open("x", encoding="utf-8") as handle:
        for family in task_types:
            for game in candidates_by_type[family]:
                if per_task_type[family]["solved"] >= args.solved_per_task_type:
                    break
                if per_task_type[family]["attempts"] >= args.max_attempts_per_task_type:
                    break
                task_type = extract_task_type(game)
                if task_type != family:
                    raise SystemExit(
                        f"task family {family!r} disagrees with traj_data.json for {game}"
                    )
                environment_seed = derive_environment_seed(args.environment_master_seed, game)
                environment_seeds[game] = environment_seed
                tasks_by_game[game] = task_type
                attempted.append(game)
                before = len(policy.request_ledger)
                env = AlfworldEnvironment(config, game, rollout_seed=environment_seed)
                try:
                    turns, won = rollout_episode(
                        env,
                        policy,
                        truncator,
                        settings=GenerationSettings(
                            temperature=0.0, max_tokens=args.max_tokens
                        ),
                        max_steps=args.max_steps,
                        system_prompt=system_prompt,
                        state_source="teacher",
                    )
                    seed_attestation = env.seed_attestation
                finally:
                    env.close()
                calls = len(policy.request_ledger) - before
                if calls != len(turns):
                    raise SystemExit(f"teacher call accounting drifted for {game}")
                invalid_turns = sum(1 for turn in turns if not turn.student.valid)
                per_task_type[family]["attempts"] += 1
                per_task_type[family]["solved"] += int(bool(won))
                counts["games_attempted"] += 1
                counts["games_won"] += int(bool(won))
                counts["turns"] += len(turns)
                counts["invalid_turns"] += invalid_turns
                handle.write(
                    json.dumps(
                        {
                            "game_id": game,
                            "task_type": task_type,
                            "environment_seed": environment_seed,
                            "seed_attestation": seed_attestation,
                            "won": bool(won),
                            "steps": len(turns),
                            "invalid_turns": invalid_turns,
                            "turns": [turn.to_dict() for turn in turns],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
                print(
                    f"[{counts['games_attempted']}] {task_type}: won={bool(won)} "
                    f"steps={len(turns)} invalid={invalid_turns} "
                    f"solved={per_task_type[family]['solved']}/{args.solved_per_task_type}",
                    flush=True,
                )
            if per_task_type[family]["solved"] < args.solved_per_task_type:
                raise SystemExit(
                    f"task family {family!r} solved only "
                    f"{per_task_type[family]['solved']} of {args.solved_per_task_type} "
                    f"within {per_task_type[family]['attempts']} attempts"
                )

    served_models = validate_provider_response_model_identity(
        policy.request_ledger, args.model
    )
    manifest = {
        "artifact": "expert_trajectory_collection",
        "protocol_version": "omniopd-v1",
        "training_ready": False,
        "supervision_source": f"{args.model}_reference_prompt",
        "demonstrator": {
            "kind": "served_model",
            "alias": args.model,
            "base_url": args.base_url,
            "response_model_attestation": served_models,
            "request_ledger_calls": len(policy.request_ledger),
            "request_ledger_sha256": sha256_json(policy.request_ledger),
        },
        "split": args.split,
        "task_types": list(task_types),
        "solved_per_task_type": args.solved_per_task_type,
        "max_attempts_per_task_type": args.max_attempts_per_task_type,
        "per_task_type": per_task_type,
        "game_order_seed": args.game_order_seed,
        "environment_master_seed": args.environment_master_seed,
        "max_steps": args.max_steps,
        "context": {
            "max_context_tokens": args.max_context_tokens,
            "reserve_tokens": args.reserve_tokens,
            "enable_thinking": False,
        },
        "student_system_prompt_sha256": sha256_text(system_prompt),
        "prompt_json": {
            "path": str(prompt_json_path),
            "sha256": sha256_file(prompt_json_path),
        },
        "game_list": {
            "source": game_list_source,
            "cache_path": None if cache_path is None else str(cache_path),
            "cache_sha256": (
                None
                if cache_path is None or not cache_path.is_file()
                else sha256_file(cache_path)
            ),
            "games_in_split": len(games),
        },
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "script_sha256": sha256_file(Path(__file__)),
        "env_config": {"path": str(env_config_path), "sha256": sha256_file(env_config_path)},
        "tokenizer": {
            "path": str(tokenizer_path),
            "tokenizer_config_sha256": sha256_file(tokenizer_path / "tokenizer_config.json"),
        },
        "runtime_dependencies": capture_runtime_dependencies(),
        "game_artifacts": fingerprint_game_artifacts(attempted),
        "game_artifacts_digest": game_artifacts_digest(fingerprint_game_artifacts(attempted)),
        "environment_seeds": environment_seeds,
        "tasks_by_game": tasks_by_game,
        "counts": counts,
        "episodes": {"path": str(episodes_path), "sha256": sha256_file(episodes_path)},
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"collected {counts['games_attempted']} teacher episodes "
        f"({counts['games_won']} won) into {episodes_path}"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
