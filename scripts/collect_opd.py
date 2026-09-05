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

from omniopd.adapters import AlfworldEnvironment, OpenAIChatPolicy, list_alfworld_games
from omniopd.context import TaskPreservingTruncator
from omniopd.io import read_jsonl, record_game_id, write_jsonl
from omniopd.protocol import GenerationSettings, TeacherBudget, query_teacher, rollout_episode
from omniopd.prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
    sha256_json,
)
from omniopd.selection import select_uniform_nested
from omniopd.sampling import SamplingProtocol, stable_shuffled
from omniopd.validation import validate_actual_calls, validate_budget
from omniopd.tokenization import apply_chat_template_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a small, nonconfirmatory budget-accounted OPD diagnostic"
    )
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--behavior-model", "--student-model", dest="behavior_model", required=True
    )
    parser.add_argument(
        "--behavior-artifact",
        action="append",
        default=[],
        help="local base/checkpoint/adapter path; repeat for composed models",
    )
    parser.add_argument(
        "--behavior-url",
        "--student-url",
        dest="behavior_url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--behavior-api-key")
    parser.add_argument(
        "--behavior-thinking-mode",
        choices=["disabled"],
        default="disabled",
    )
    parser.add_argument("--behavior-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-model-revision", required=True)
    parser.add_argument("--teacher-url", default="https://api.deepseek.com")
    parser.add_argument("--teacher-api-key", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--teacher-context-window", type=int, required=True)
    parser.add_argument("--limit-games", type=int, default=50)
    parser.add_argument("--states-per-game", type=int, default=3)
    parser.add_argument("--samples-per-state", type=int, default=1)
    parser.add_argument("--teacher-budget", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--behavior-max-tokens", type=int, default=256)
    parser.add_argument("--teacher-max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument(
        "--teacher-thinking-mode", choices=["enabled", "disabled"], required=True
    )
    parser.add_argument("--teacher-temperature", type=float)
    parser.add_argument(
        "--teacher-reasoning-effort", choices=["low", "high", "max"]
    )
    parser.add_argument(
        "--state-source", choices=["student", "teacher"], default="student"
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(
            f"refusing to reuse output directory: {output}; "
            "use a new run directory so provenance remains immutable"
        )
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("diagnostic collection requires an immutable Git revision")
    if not Path(args.env_config).is_file() or not args.teacher_model_revision.strip():
        raise SystemExit("environment config and a non-empty Teacher revision are required")
    if not Path(args.tokenizer).exists():
        raise SystemExit("--tokenizer must be a local immutable path that can be fingerprinted")
    if not Path(args.teacher_tokenizer).exists():
        raise SystemExit("--teacher-tokenizer must be a local immutable path")
    if args.state_source == "student" and not args.behavior_artifact:
        raise SystemExit("Student-state collection requires at least one --behavior-artifact")
    missing_artifacts = [path for path in args.behavior_artifact if not Path(path).exists()]
    if missing_artifacts:
        raise SystemExit(f"behavior artifacts do not exist: {missing_artifacts}")
    if (
        args.behavior_max_tokens <= 0
        or args.teacher_max_tokens <= 0
        or args.reserve_tokens < max(
            args.behavior_max_tokens, args.teacher_max_tokens
        )
        or args.max_context_tokens <= args.reserve_tokens
        or args.teacher_context_window <= args.teacher_max_tokens
    ):
        raise SystemExit(
            "generation limits must be positive and reserve-tokens must cover both limits"
        )

    from transformers import AutoTokenizer

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    excluded = set()
    for path in args.exclude_jsonl:
        for record in read_jsonl(path):
            game = record_game_id(record)
            if game:
                excluded.add(str(game))
    games = stable_shuffled(
        (
            game
            for game in list_alfworld_games(config, "train")
            if game not in excluded
        ),
        seed=args.seed,
    )
    games = games[: args.limit_games]
    if len(games) != args.limit_games:
        raise SystemExit(
            f"expected exactly {args.limit_games} eligible games after exclusions, got {len(games)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_tokenizer, trust_remote_code=True
    )
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
    )
    teacher_key = args.teacher_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not teacher_key:
        raise SystemExit("provide --teacher-api-key or DEEPSEEK_API_KEY")
    teacher_sampling = SamplingProtocol(
        args.teacher_thinking_mode,
        args.teacher_temperature,
        args.teacher_reasoning_effort,
    )
    teacher_sampling.validate(samples_per_state=args.samples_per_state)
    behavior_sampling = SamplingProtocol(
        args.behavior_thinking_mode, args.behavior_temperature, None
    )
    behavior_sampling.validate()
    if args.state_source == "teacher":
        if (
            args.behavior_model != args.teacher_model
            or args.behavior_url.rstrip("/") != args.teacher_url.rstrip("/")
        ):
            raise SystemExit(
                "teacher-state collection requires behavior model and endpoint to match "
                "the declared Teacher exactly"
            )
        if behavior_sampling.thinking_mode != "disabled":
            raise SystemExit(
                "teacher-state behavior requires explicit --behavior-thinking-mode disabled"
            )
        behavior_key = teacher_key
    else:
        behavior_key = args.behavior_api_key or os.environ.get("BEHAVIOR_API_KEY", "EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    behavior_requests_partial = output / "behavior_requests.partial.jsonl"
    teacher_requests_partial = output / "teacher_requests.partial.jsonl"
    behavior_policy = OpenAIChatPolicy(
        model=args.behavior_model,
        base_url=args.behavior_url,
        api_key=behavior_key,
        thinking_mode=behavior_sampling.policy_thinking_mode,
        thinking_control=(
            "chat_template" if args.state_source == "student" else "deepseek"
        ),
        reasoning_effort=behavior_sampling.reasoning_effort,
        seed=args.seed if args.state_source == "student" else None,
        request_ledger_path=behavior_requests_partial,
    )
    teacher = OpenAIChatPolicy(
        model=args.teacher_model,
        base_url=args.teacher_url,
        api_key=teacher_key,
        thinking_mode=teacher_sampling.policy_thinking_mode,
        reasoning_effort=teacher_sampling.reasoning_effort,
        request_ledger_path=teacher_requests_partial,
    )
    distinct_states = len(games) * args.states_per_game
    maximum_calls = args.teacher_budget
    validate_budget(
        states=distinct_states,
        samples_per_state=args.samples_per_state,
        declared_budget=maximum_calls,
    )
    budget = TeacherBudget(maximum_calls)
    episodes = []
    selected_states = []
    behavior_prompt = (
        STUDENT_SYSTEM_PROMPT if args.state_source == "student" else TEACHER_SYSTEM_PROMPT
    )

    # Freeze and validate the complete behavior-policy state pool before
    # spending any Teacher *annotation* calls.
    for game in games:
        env = AlfworldEnvironment(config, game)
        try:
            turns, won = rollout_episode(
                env,
                behavior_policy,
                truncator,
                settings=GenerationSettings(
                    temperature=behavior_sampling.temperature,
                    max_tokens=args.behavior_max_tokens,
                ),
                max_steps=args.max_steps,
                system_prompt=behavior_prompt,
                state_source=args.state_source,
            )
        finally:
            env.close()
        episodes.append(
            {
                "game_id": game,
                "won": won,
                "turns": [turn.to_dict() for turn in turns],
            }
        )
        selected = select_uniform_nested(
            turns,
            args.states_per_game,
            seed=args.seed,
            game_id=game,
        )
        if len(selected) != args.states_per_game:
            raise RuntimeError(
                f"game {game!r} yielded {len(turns)} states, fewer than the "
                f"preregistered {args.states_per_game}; no annotation calls were made"
            )
        selected_states.extend(selected)

    if len(selected_states) != distinct_states:
        raise AssertionError(
            f"internal M mismatch: selected={len(selected_states)}, declared={distinct_states}"
        )

    teacher_prompt_lengths = {
        selected.turn.state.state_hash: len(
            apply_chat_template_ids(
                teacher_tokenizer,
                [
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    *[dict(message) for message in selected.turn.state.messages[1:]],
                ],
                add_generation_prompt=True,
                enable_thinking=args.teacher_thinking_mode == "enabled",
            )
        )
        for selected in selected_states
    }
    overlength = [
        state_hash
        for state_hash, length in teacher_prompt_lengths.items()
        if length + args.teacher_max_tokens > args.teacher_context_window
    ]
    if overlength:
        raise RuntimeError(
            "Teacher context preflight failed before annotation calls; "
            f"overlength states={overlength[:5]}"
        )

    # Spend exactly B=M*N annotation attempts. Invalid responses remain in the
    # ledger and never receive an unbudgeted retry.
    corrections = []
    for selected in selected_states:
        corrections.append(
            query_teacher(
                selected.turn.state,
                selected.turn.student,
                teacher,
                budget,
                samples_per_state=args.samples_per_state,
                settings=GenerationSettings(
                    temperature=teacher_sampling.temperature,
                    max_tokens=args.teacher_max_tokens,
                ),
                selection_policy=selected.policy,
                inclusion_probability=selected.inclusion_probability,
            )
        )
    if budget.used_calls != maximum_calls:
        raise RuntimeError(
            "exact budget was not reached, usually because an episode had fewer states than requested; "
            "do not label this run as fixed-B"
        )
    validate_actual_calls(corrections, maximum_calls)
    episodes_path = output / "episodes.jsonl"
    state_pool_path = output / "state_pool.jsonl"
    corrections_path = output / "corrections.jsonl"
    games_path = output / "games.json"
    behavior_requests_path = output / "behavior_requests.jsonl"
    teacher_requests_path = output / "teacher_requests.jsonl"
    write_jsonl(episodes_path, episodes)
    write_jsonl(
        state_pool_path,
        (turn for episode in episodes for turn in episode["turns"]),
    )
    write_jsonl(corrections_path, (record.to_dict() for record in corrections))
    behavior_calls = sum(len(episode["turns"]) for episode in episodes)
    if len(behavior_policy.request_ledger) != behavior_calls:
        raise RuntimeError("behavior request ledger length does not match collected states")
    if len(teacher.request_ledger) != maximum_calls:
        raise RuntimeError("Teacher request ledger length does not equal declared budget B")
    behavior_ledger = {
        str(row["request_id"]): row for row in behavior_policy.request_ledger
    }
    teacher_ledger = {str(row["request_id"]): row for row in teacher.request_ledger}
    if len(behavior_ledger) != behavior_calls or len(teacher_ledger) != maximum_calls:
        raise RuntimeError("diagnostic request ledgers contain duplicate request IDs")
    for episode in episodes:
        for turn in episode["turns"]:
            state = turn["state"]
            request_id = f"behavior:{args.state_source}:{state['state_hash']}"
            if (
                request_id not in behavior_ledger
                or behavior_ledger[request_id]["messages_sha256"]
                != sha256_json(state["messages"])
            ):
                raise RuntimeError("behavior request ledger does not match state pool")
    for record in corrections:
        for request_id in record.metadata["teacher_request_ids"]:
            if (
                request_id not in teacher_ledger
                or teacher_ledger[request_id]["messages_sha256"]
                != record.metadata["teacher_query_sha256"]
            ):
                raise RuntimeError("Teacher request ledger does not match corrections")
    behavior_requests_partial.rename(behavior_requests_path)
    teacher_requests_partial.rename(teacher_requests_path)
    games_path.write_text(json.dumps(games, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "nonconfirmatory_single_process_opd_diagnostic",
        "confirmatory_use_allowed": False,
        "code": current_code,
        "code_revision": current_revision,
        "seed": args.seed,
        "games_G": len(games),
        "states_per_game": args.states_per_game,
        "distinct_states_M": distinct_states,
        "teacher_samples_per_state_N": args.samples_per_state,
        "declared_teacher_budget_B": maximum_calls,
        "actual_teacher_api_calls": budget.used_calls,
        "invalid_calls_count_toward_budget": True,
        "teacher_prompt_is_distinct": True,
        "behavior_model": args.behavior_model,
        "behavior_url": args.behavior_url,
        "state_source": args.state_source,
        "teacher_model": args.teacher_model,
        "teacher_model_revision": args.teacher_model_revision,
        "teacher_url": args.teacher_url,
        "tokenizer": fingerprint_path(args.tokenizer),
        "teacher_tokenizer": fingerprint_path(args.teacher_tokenizer),
        "behavior_artifacts": [
            fingerprint_path(path) for path in args.behavior_artifact
        ],
        "behavior_sampling": behavior_sampling.to_dict(),
        "teacher_sampling": teacher_sampling.to_dict(),
        "max_steps": args.max_steps,
        "max_context_tokens": args.max_context_tokens,
        "reserve_tokens": args.reserve_tokens,
        "behavior_max_tokens": args.behavior_max_tokens,
        "teacher_max_tokens": args.teacher_max_tokens,
        "teacher_context_window": args.teacher_context_window,
        "teacher_context_preflight": {
            "passed": True,
            "minimum_prompt_tokens": min(teacher_prompt_lengths.values()),
            "maximum_prompt_tokens": max(teacher_prompt_lengths.values()),
        },
        "behavior_provider_response_models": sorted(
            {
                str(row["response_model"])
                for row in behavior_policy.request_ledger
                if row.get("response_model") is not None
            }
        ),
        "teacher_provider_response_models": sorted(
            {
                str(row["response_model"])
                for row in teacher.request_ledger
                if row.get("response_model") is not None
            }
        ),
        "student_prompt_sha256": hashlib.sha256(
            STUDENT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "teacher_prompt_sha256": hashlib.sha256(
            TEACHER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "state_pool_validated_before_teacher_annotation_calls": True,
        "teacher_rollout_calls": (
            sum(len(episode["turns"]) for episode in episodes)
            if args.state_source == "teacher"
            else 0
        ),
        "excluded_games": len(excluded),
        "inputs": {
            "env_config_sha256": sha256_file(args.env_config),
            "exclusion_jsonl_sha256": {
                str(path): sha256_file(path) for path in args.exclude_jsonl
            },
        },
        "outputs": {
            "games_sha256": sha256_file(games_path),
            "episodes_sha256": sha256_file(episodes_path),
            "state_pool_sha256": sha256_file(state_pool_path),
            "corrections_sha256": sha256_file(corrections_path),
            "behavior_requests_sha256": sha256_file(behavior_requests_path),
            "teacher_requests_sha256": sha256_file(teacher_requests_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
