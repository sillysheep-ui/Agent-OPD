#!/usr/bin/env python3
"""Drive one real ALFWorld game with the handcoded expert through rollout_episode.

This is a nonconfirmatory technical check.  It exists to prove that the shared
rollout path (history state machine, task-preserving truncation, state hashing)
can be driven by the environment expert instead of a served policy, so expert
cold-start data does not need a second rollout implementation.  It never
trains and never calls an API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import AlfworldEnvironment, ExpertPolicy  # noqa: E402
from omniopd.context import TaskPreservingTruncator  # noqa: E402
from omniopd.environment_provenance import (  # noqa: E402
    derive_environment_seed,
    fingerprint_game_artifacts,
)
from omniopd.prompts import STUDENT_SYSTEM_PROMPT  # noqa: E402
from omniopd.protocol import GenerationSettings, rollout_episode  # noqa: E402
from omniopd.provenance import (  # noqa: E402
    fingerprint_code_tree,
    git_revision,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-master-seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    args = parser.parse_args()

    env_config_path = args.env_config.resolve()
    game_path = args.game.resolve()
    tokenizer_path = args.tokenizer.resolve()
    output_path = args.output.resolve()
    if (
        args.environment_master_seed < 0
        or args.max_steps <= 0
        or args.max_context_tokens <= args.reserve_tokens
        or args.reserve_tokens < 0
        or output_path.exists()
        or not env_config_path.is_file()
        or not game_path.is_file()
        or not (tokenizer_path / "tokenizer_config.json").is_file()
    ):
        raise SystemExit("invalid expert rollout smoke input or output")

    import yaml
    from transformers import AutoTokenizer

    config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
        enable_thinking=False,
    )
    game_id = str(game_path)
    environment_seed = derive_environment_seed(args.environment_master_seed, game_id)
    game_artifacts = fingerprint_game_artifacts([game_id])

    env = AlfworldEnvironment(
        config,
        game_id,
        rollout_seed=environment_seed,
        expert_type="handcoded",
    )
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
    finally:
        env.close()

    if len(policy.request_ids) != len(turns):
        raise SystemExit("expert turn accounting does not match the recorded turns")
    invalid_turns = [turn for turn in turns if not turn.student.valid]
    payload = {
        "artifact": "nonconfirmatory_expert_rollout_smoke",
        "training_performed": False,
        "model_used": False,
        "api_calls": 0,
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
        "expert_type": "handcoded",
        "game_id": game_id,
        "game_artifacts": game_artifacts,
        "environment_master_seed": args.environment_master_seed,
        "environment_seed": environment_seed,
        "seed_attestation": seed_attestation,
        "max_steps": args.max_steps,
        "max_context_tokens": args.max_context_tokens,
        "reserve_tokens": args.reserve_tokens,
        "won": bool(won),
        "steps": len(turns),
        "invalid_turns": len(invalid_turns),
        "turns": [
            {
                "turn_index": turn.state.turn_index,
                "state_hash": turn.state.state_hash,
                "task_type": turn.state.task_type,
                "admissible_count": len(turn.state.admissible_actions),
                "truncation": turn.state.truncation,
                "raw": turn.student.raw,
                "executed_action": turn.student.executed_action,
                "valid": turn.student.valid,
                "failure_reason": turn.student.failure_reason,
            }
            for turn in turns
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"expert rollout: steps={payload['steps']}, won={payload['won']}, "
        f"invalid_turns={payload['invalid_turns']}, task_type={turns[0].state.task_type}"
    )
    print(output_path)


if __name__ == "__main__":
    main()
