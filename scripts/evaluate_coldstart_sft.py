#!/usr/bin/env python3
"""Closed-loop ALFWorld evaluation of a cold-start SFT checkpoint.

Nonconfirmatory pilot evaluator.  It replays exactly the games held out of the
cold-start corpus, with the same Student system prompt and the same 50-step
protocol the corpus was collected under, and reports per-game wins plus the
action validity the environment actually observed.

The confirmatory comparison arms use the audited evaluation chain
(``run_vllm_eval.sh`` + ``evaluate.py``), which requires schema-v2 launch and
completion manifests with an annotation-pair binding.  A cold-start run has no
annotation pair, so this pilot deliberately does not claim that chain's
guarantees and records ``confirmatory_use_allowed: false``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import (  # noqa: E402
    AdmissibleChoicePolicy,
    AlfworldEnvironment,
    OpenAIChatPolicy,
    validate_provider_response_model_identity,
)
from omniopd.context import TaskPreservingTruncator  # noqa: E402
from omniopd.environment_provenance import (  # noqa: E402
    derive_environment_seed,
    fingerprint_game_artifacts,
    game_artifacts_digest,
)
from omniopd.prompts import build_reference_prompt, resolve_student_prompt  # noqa: E402
from omniopd.protocol import GenerationSettings, rollout_episode  # noqa: E402
from omniopd.provenance import (  # noqa: E402
    fingerprint_code_tree,
    git_revision,
    sha256_file,
    sha256_json,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-master-seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--student-prompt", default="v1")
    parser.add_argument(
        "--demonstration-json",
        type=Path,
        default=None,
        help="one-shot demonstration artifact with 'user' and 'assistant' blocks",
    )
    parser.add_argument(
        "--user-turn-style",
        default="default",
        choices=["default", "sage_opd"],
        help="how each observation/admissible turn is rendered",
    )
    parser.add_argument(
        "--assistant-history",
        default="action_only",
        choices=["action_only", "raw"],
        help="keep only the executed action or the model's own raw turn",
    )
    parser.add_argument(
        "--prompt-json",
        type=Path,
        default=None,
        help=(
            "path to a reference prompt JSON with 'instruction' and 'examples' "
            "(for example AgentBoard's prompts/VanillaAgent/alfworld_base.json)"
        ),
    )
    parser.add_argument(
        "--constrain-admissible",
        action="store_true",
        help="sample only from the current admissible action set",
    )
    parser.add_argument(
        "--constraint-field",
        default="guided_choice",
        choices=["guided_choice", "structured_outputs"],
        help="request field the served vLLM honours for choice constraints",
    )
    return parser.parse_args()


def held_out_games(val_path: Path) -> dict[str, list[str]]:
    """List the held-out games per task family, preserving first-seen order."""

    games_by_task: dict[str, list[str]] = defaultdict(list)
    with val_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            game_id = str(row["game_id"])
            if game_id not in games_by_task[str(row["task_type"])]:
                games_by_task[str(row["task_type"])].append(game_id)
    return games_by_task


def main() -> None:
    args = parse_args()
    env_config_path = args.env_config.resolve()
    tokenizer_path = args.tokenizer.resolve()
    val_path = args.val_data.resolve()
    output_path = args.output.resolve()
    if (
        args.environment_master_seed < 0
        or args.max_steps <= 0
        or args.max_tokens <= 0
        or args.reserve_tokens < args.max_tokens
        or args.max_context_tokens <= args.reserve_tokens
        or output_path.exists()
        or not env_config_path.is_file()
        or not val_path.is_file()
        or not (tokenizer_path / "tokenizer_config.json").is_file()
    ):
        raise SystemExit("invalid cold-start evaluation input or output")

    import yaml
    from transformers import AutoTokenizer

    games_by_task = held_out_games(val_path)
    games = [game for task in sorted(games_by_task) for game in games_by_task[task]]
    if not games:
        raise SystemExit("validation data contains no held-out games")

    config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    demonstration: list[tuple[str, str]] | None = None
    if args.demonstration_json is not None:
        demo = json.loads(args.demonstration_json.read_text(encoding="utf-8"))
        demonstration = [
            (str(turn["user"]), str(turn["assistant"])) for turn in demo["turns"]
        ]
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
        enable_thinking=False,
        anchor_messages=2 + 2 * len(demonstration) if demonstration else 2,
    )
    if args.prompt_json is not None:
        reference = json.loads(args.prompt_json.read_text(encoding="utf-8"))
        prompt_name = f"reference:{args.prompt_json.name}"
        prompt_text = build_reference_prompt(
            str(reference["instruction"]), reference.get("examples") or []
        )
    else:
        prompt_name, prompt_text = resolve_student_prompt(args.student_prompt)
    policy_kwargs = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": "EMPTY",
        "thinking_mode": "disabled",
        "thinking_control": "chat_template",
        "max_retries": 0,
    }
    if args.constrain_admissible:
        policy = AdmissibleChoicePolicy(
            constraint_field=args.constraint_field, **policy_kwargs
        )
    else:
        policy = OpenAIChatPolicy(**policy_kwargs)

    episodes = []
    environment_seeds: dict[str, int] = {}
    for game in games:
        environment_seed = derive_environment_seed(args.environment_master_seed, game)
        environment_seeds[game] = environment_seed
        before = len(policy.request_ledger)
        env = AlfworldEnvironment(config, game, rollout_seed=environment_seed)
        try:
            turns, won = rollout_episode(
                env,
                policy,
                truncator,
                settings=GenerationSettings(temperature=0.0, max_tokens=args.max_tokens),
                max_steps=args.max_steps,
                system_prompt=prompt_text,
                state_source="student",
                user_turn_style=args.user_turn_style,
                assistant_history=args.assistant_history,
                demonstration=demonstration,
            )
        finally:
            env.close()
        calls = len(policy.request_ledger) - before
        if calls != len(turns):
            raise SystemExit(f"policy call accounting drifted for {game}")
        episodes.append(
            {
                "game_id": game,
                "task_type": next(
                    task for task, values in games_by_task.items() if game in values
                ),
                "environment_seed": environment_seed,
                "won": bool(won),
                "steps": len(turns),
                "invalid_turns": sum(1 for turn in turns if not turn.student.valid),
                "actions": [turn.student.executed_action for turn in turns],
                "turns": [turn.to_dict() for turn in turns],
            }
        )
        print(
            f"{episodes[-1]['task_type']}: won={episodes[-1]['won']} "
            f"steps={episodes[-1]['steps']} invalid={episodes[-1]['invalid_turns']}",
            flush=True,
        )

    served_models = validate_provider_response_model_identity(
        policy.request_ledger, args.model
    )
    by_task = defaultdict(list)
    for episode in episodes:
        by_task[episode["task_type"]].append(episode)
    payload = {
        "artifact": "nonconfirmatory_coldstart_closed_loop_evaluation",
        "confirmatory_use_allowed": False,
        "training_performed": False,
        "student_prompt": {"name": prompt_name, "sha256": sha256_text(prompt_text)},
        "prompt_json": (
            None
            if args.prompt_json is None
            else {"path": str(args.prompt_json.resolve()), "sha256": sha256_file(args.prompt_json)}
        ),
        "user_turn_style": args.user_turn_style,
        "demonstration": (
            None
            if args.demonstration_json is None
            else {
                "path": str(args.demonstration_json.resolve()),
                "sha256": sha256_file(args.demonstration_json),
            }
        ),
        "assistant_history": args.assistant_history,
        "constraint_mode": (
            "admissible_choice" if args.constrain_admissible else "free_generation"
        ),
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "script_sha256": sha256_file(Path(__file__)),
        "validation_rows": {
            "path": str(val_path),
            "sha256": sha256_file(val_path),
        },
        "env_config": {
            "path": str(env_config_path),
            "sha256": sha256_file(env_config_path),
        },
        "served_model": args.model,
        "served_model_attestation": served_models,
        "base_url": args.base_url,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "environment_master_seed": args.environment_master_seed,
        "game_artifacts": fingerprint_game_artifacts(games),
        "game_artifacts_digest": game_artifacts_digest(fingerprint_game_artifacts(games)),
        "environment_seeds": environment_seeds,
        "request_ledger": {
            "calls": len(policy.request_ledger),
            "sha256": sha256_json(policy.request_ledger),
        },
        "summary": {
            "games": len(episodes),
            "wins": sum(1 for episode in episodes if episode["won"]),
            "invalid_turns": sum(episode["invalid_turns"] for episode in episodes),
            "by_task_type": {
                task: {
                    "games": len(values),
                    "wins": sum(1 for episode in values if episode["won"]),
                    "steps": sum(episode["steps"] for episode in values),
                }
                for task, values in sorted(by_task.items())
            },
        },
        "episodes": episodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    summary = payload["summary"]
    print(f"closed-loop: {summary['wins']}/{summary['games']} wins")
    print(output_path)


if __name__ == "__main__":
    main()
