#!/usr/bin/env python3
"""Collect ALFWorld handcoded-expert trajectories for the shared SFT cold start.

Policy decisions this script enforces, so the corpus cannot drift:

- only the named ALFWorld split (``train`` by default) is read;
- the expert acts through the shared ``rollout_episode`` path, so history,
  task-preserving truncation and state hashing are the same code used by the
  Student and Teacher pipelines;
- every selected game gets an explicit TextWorld seed derived from the game
  identifier, and the seed attestation is recorded;
- every attempted episode is written, including the ones the expert failed to
  solve, because dropping failures silently would hide coverage.  Consumers
  filter ``won`` themselves.

The output directory must not exist.  Nothing here trains or calls an API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import (  # noqa: E402
    AlfworldEnvironment,
    ExpertPolicy,
    extract_task_type,
)
from omniopd.context import TaskPreservingTruncator  # noqa: E402
from omniopd.environment_provenance import (  # noqa: E402
    capture_runtime_dependencies,
    derive_environment_seed,
    fingerprint_game_artifacts,
    game_artifacts_digest,
)
from omniopd.prompts import STUDENT_SYSTEM_PROMPT  # noqa: E402
from omniopd.protocol import GenerationSettings, rollout_episode  # noqa: E402
from omniopd.provenance import (  # noqa: E402
    fingerprint_code_tree,
    git_revision,
    sha256_file,
    sha256_text,
)
from omniopd.sampling import stable_shuffled  # noqa: E402

CANONICAL_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def task_family_from_path(game: str) -> str:
    """Read the task family from the trial directory name without opening metadata."""

    trial = Path(game).parent
    family_directory = trial.parent.name if trial.name.startswith("trial_") else trial.name
    return family_directory.split("-", 1)[0]


def enumerate_split_games(
    *, data_path: Path, task_types: tuple[str, ...]
) -> list[str]:
    """List the split's executable games with single-level directory reads.

    ALFWorld's own ``AlfredTWEnv.collect_game_files`` builds
    ``list(os.walk(root, topdown=False))`` first, which walks thousands of
    directories and on the FUSE-backed shared filesystem has been observed to
    block inside ``fuse_readdir`` for minutes.  Listing one family directory at
    a time is cheap; the ``solvable`` metadata check is skipped here and the
    environment itself rejects an unusable game when it is loaded.
    """

    games: list[str] = []
    for family_directory in sorted(os.listdir(data_path)):
        if "movable" in family_directory or "Sliced" in family_directory:
            continue
        if family_directory.split("-", 1)[0] not in task_types:
            continue
        family_path = data_path / family_directory
        if not family_path.is_dir():
            continue
        for trial in sorted(os.listdir(family_path)):
            if not trial.startswith("trial_"):
                continue
            game = family_path / trial / "game.tw-pddl"
            if game.is_file():
                games.append(str(game))
    if not games:
        raise SystemExit(f"no executable games found under {data_path}")
    return games


def select_games(
    games: list[str],
    *,
    task_types: tuple[str, ...],
    games_per_task_type: int,
    seed: int,
) -> list[str]:
    """Take a stable per-task-type sample without scanning every traj_data.json."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for game in games:
        family = task_family_from_path(game)
        if family in task_types:
            grouped[family].append(game)
    selected: list[str] = []
    for task_type in task_types:
        candidates = grouped.get(task_type, [])
        if len(candidates) < games_per_task_type:
            raise SystemExit(
                f"split has only {len(candidates)} games for task type {task_type!r}, "
                f"but {games_per_task_type} were requested"
            )
        selected.extend(
            stable_shuffled(candidates, seed=seed)[:games_per_task_type]
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--task-types",
        default=",".join(CANONICAL_TASK_TYPES),
        help="comma-separated ALFWorld task families to cover",
    )
    parser.add_argument(
        "--solved-per-task-type",
        type=int,
        default=10,
        help="solved episodes required from every task family",
    )
    parser.add_argument(
        "--game-list-cache",
        type=Path,
        help=(
            "optional JSON cache of the split's game list; a cache miss walks "
            "the ALFWorld tree once and writes the cache"
        ),
    )
    parser.add_argument(
        "--max-attempts-per-task-type",
        type=int,
        default=50,
        help="attempts allowed per family before the collection fails closed",
    )
    parser.add_argument("--game-order-seed", type=int, default=42)
    parser.add_argument("--environment-master-seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    return parser.parse_args()


def resolve_game_list(
    *,
    config: dict,
    split: str,
    cache_path: Path | None,
    task_types: tuple[str, ...],
) -> tuple[list[str], str]:
    """Return the split's game list, reusing a cache when one is supplied.

    Walking the ALFWorld tree costs minutes on a FUSE-backed shared
    filesystem and has been observed to block indefinitely, so the walk is
    paid once and persisted instead of running at every collection.
    """

    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("split") != split or not isinstance(cached.get("games"), list):
            raise SystemExit(f"game list cache does not describe split {split!r}")
        games = [str(item) for item in cached["games"]]
        if not games or len(set(games)) != len(games):
            raise SystemExit("game list cache is empty or contains duplicates")
        return games, "cache"
    data_path = Path(
        os.path.expandvars(str(config["dataset"]["data_path"]))
    )
    games = enumerate_split_games(data_path=data_path, task_types=task_types)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"split": split, "games": games}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return games, "alfworld_environment"


def main() -> None:
    args = parse_args()
    env_config_path = args.env_config.resolve()
    tokenizer_path = args.tokenizer.resolve()
    output = args.output.resolve()
    task_types = tuple(item.strip() for item in args.task_types.split(",") if item.strip())
    if (
        args.split != "train"
        or args.solved_per_task_type <= 0
        or args.max_attempts_per_task_type < args.solved_per_task_type
        or args.game_order_seed < 0
        or args.environment_master_seed < 0
        or args.max_steps <= 0
        or args.reserve_tokens < 0
        or args.max_context_tokens <= args.reserve_tokens
        or output.exists()
        or not env_config_path.is_file()
        or not (tokenizer_path / "tokenizer_config.json").is_file()
        or not task_types
        or len(set(task_types)) != len(task_types)
    ):
        raise SystemExit("invalid expert trajectory collection input or output")

    import yaml
    from transformers import AutoTokenizer

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

    output.mkdir(parents=True)
    episodes_path = output / "episodes.jsonl"
    environment_seeds: dict[str, int] = {}
    tasks_by_game: dict[str, str] = {}
    per_task_type: dict[str, dict[str, int]] = {
        task_type: {"attempts": 0, "solved": 0} for task_type in task_types
    }
    counts = {
        "games_attempted": 0,
        "games_won": 0,
        "games_with_environment_error": 0,
        "turns": 0,
        "invalid_turns": 0,
    }
    attempted_games: list[str] = []
    total_budget = len(task_types) * args.max_attempts_per_task_type
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
                        f"task family {family!r} disagrees with traj_data.json "
                        f"task_type {task_type!r} for {game}"
                    )
                environment_seed = derive_environment_seed(args.environment_master_seed, game)
                environment_seeds[game] = environment_seed
                tasks_by_game[game] = task_type
                attempted_games.append(game)
                # A game the environment cannot even load is recorded as a
                # rejected attempt instead of aborting the whole collection:
                # the attempt still consumes budget, and the reason is kept.
                failure_reason: str | None = None
                seed_attestation: dict | None = None
                turns: list = []
                won = False
                try:
                    env = AlfworldEnvironment(
                        config, game, rollout_seed=environment_seed, expert_type="handcoded"
                    )
                except Exception as error:  # noqa: BLE001 - recorded, not hidden
                    failure_reason = f"environment_load_error:{type(error).__name__}"
                else:
                    policy = ExpertPolicy(env.expert_action)
                    try:
                        turns, won = rollout_episode(
                            env,
                            policy,
                            truncator,
                            settings=GenerationSettings(temperature=0.0, max_tokens=64),
                            max_steps=args.max_steps,
                            system_prompt=STUDENT_SYSTEM_PROMPT,
                            state_source="student",
                        )
                        seed_attestation = env.seed_attestation
                    except Exception as error:  # noqa: BLE001 - recorded, not hidden
                        failure_reason = f"expert_rollout_error:{type(error).__name__}"
                        turns, won = [], False
                    finally:
                        env.close()
                    if failure_reason is None and len(policy.request_ids) != len(turns):
                        raise SystemExit(f"expert turn accounting drifted for {game}")
                invalid_turns = sum(1 for turn in turns if not turn.student.valid)
                per_task_type[family]["attempts"] += 1
                per_task_type[family]["solved"] += int(bool(won))
                counts["games_attempted"] += 1
                counts["games_won"] += int(bool(won))
                counts["games_with_environment_error"] += int(failure_reason is not None)
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
                            "failure_reason": failure_reason,
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
                    f"[{counts['games_attempted']}/{total_budget}] {task_type}: "
                    f"won={bool(won)} steps={len(turns)} invalid={invalid_turns} "
                    f"solved={per_task_type[family]['solved']}/{args.solved_per_task_type}"
                    + (f" failure={failure_reason}" if failure_reason else ""),
                    flush=True,
                )
            if per_task_type[family]["solved"] < args.solved_per_task_type:
                raise SystemExit(
                    f"task family {family!r} solved only "
                    f"{per_task_type[family]['solved']} of the requested "
                    f"{args.solved_per_task_type} episodes within "
                    f"{per_task_type[family]['attempts']} attempts; the corpus is "
                    "incomplete, so the run fails closed instead of shipping an "
                    "imbalanced corpus"
                )

    manifest = {
        "artifact": "expert_trajectory_collection",
        "protocol_version": "omniopd-v1",
        "training_ready": False,
        "supervision_source": "alfworld_handcoded_expert",
        "expert_type": "handcoded",
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
        "student_system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
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
        "selection": {
            "rule": "per_task_type_stable_shuffle_attempt_until_solved_target",
            "task_family_source": "trial_directory_name_cross_checked_with_traj_data",
            "game_order_seed": args.game_order_seed,
        },
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "script_sha256": sha256_file(Path(__file__)),
        "env_config": {
            "path": str(env_config_path),
            "sha256": sha256_file(env_config_path),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "tokenizer_config_sha256": sha256_file(tokenizer_path / "tokenizer_config.json"),
        },
        "runtime_dependencies": capture_runtime_dependencies(),
        "game_artifacts": fingerprint_game_artifacts(attempted_games),
        "game_artifacts_digest": game_artifacts_digest(
            fingerprint_game_artifacts(attempted_games)
        ),
        "environment_seeds": environment_seeds,
        "tasks_by_game": tasks_by_game,
        "counts": counts,
        "episodes": {
            "path": str(episodes_path),
            "sha256": sha256_file(episodes_path),
        },
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"collected {counts['games_attempted']} episodes "
        f"({counts['games_won']} won) into {episodes_path}"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
