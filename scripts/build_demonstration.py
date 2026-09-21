#!/usr/bin/env python3
"""Build the one-shot demonstration the reference prompt expects.

SAGE-OPD's Table 6 says the ALFWorld prompt carries "a demonstration before the
user turn" but the paper never publishes its content.  This script therefore
builds one from our own train split and records that fact, so a reproduction
attempt can be honest about what is ours and what came from the paper.

Design: the demonstration is a complete successful episode, not a fragment,
because the pilot's failures are as much about never terminating as about the
first action.  It is drawn from a task family that needs a transformation
(heat in the default), since that is where invalid commands concentrate, and
among the successful candidates the shortest episode wins so the prompt stays
compact.  The generator is a served model using the same reference prompt the
student will see.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import (  # noqa: E402
    AlfworldEnvironment,
    OpenAIChatPolicy,
    extract_task_type,
    validate_provider_response_model_identity,
)
from omniopd.context import TaskPreservingTruncator  # noqa: E402
from omniopd.environment_provenance import derive_environment_seed  # noqa: E402
from omniopd.prompts import (  # noqa: E402
    SAGE_OPD_ALFWORLD_SYSTEM_PROMPT,
    resolve_student_prompt,
)
from omniopd.protocol import GenerationSettings, rollout_episode  # noqa: E402
from omniopd.provenance import (  # noqa: E402
    fingerprint_code_tree,
    git_revision,
    sha256_file,
    sha256_json,
    sha256_text,
)
from scripts.collect_expert_sft import task_family_from_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--game-list-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", default="pick_heat_then_place_in_recep")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--environment-master-seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--reserve-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config_path = args.env_config.resolve()
    tokenizer_path = args.tokenizer.resolve()
    cache_path = args.game_list_cache.resolve()
    output = args.output.resolve()
    if (
        args.candidates <= 0
        or args.max_steps <= 0
        or args.reserve_tokens < args.max_tokens
        or args.max_context_tokens <= args.reserve_tokens
        or output.exists()
        or not env_config_path.is_file()
        or not cache_path.is_file()
        or not (tokenizer_path / "tokenizer_config.json").is_file()
    ):
        raise SystemExit("invalid demonstration build input or output")

    import yaml
    from transformers import AutoTokenizer

    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if cached.get("split") != "train":
        raise SystemExit("the demonstration must come from the train split")
    candidates = [
        str(game)
        for game in cached["games"]
        if task_family_from_path(str(game)) == args.family
    ][: args.candidates]
    if not candidates:
        raise SystemExit(f"no train games found for family {args.family!r}")

    config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
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

    attempts = []
    chosen = None
    for game in candidates:
        environment_seed = derive_environment_seed(args.environment_master_seed, game)
        before = len(policy.request_ledger)
        env = AlfworldEnvironment(config, game, rollout_seed=environment_seed)
        try:
            turns, won = rollout_episode(
                env,
                policy,
                truncator,
                settings=GenerationSettings(temperature=0.0, max_tokens=args.max_tokens),
                max_steps=args.max_steps,
                system_prompt=SAGE_OPD_ALFWORLD_SYSTEM_PROMPT,
                state_source="teacher",
                user_turn_style="sage_opd",
                assistant_history="raw",
            )
        finally:
            env.close()
        attempts.append(
            {
                "game_id": game,
                "task_type": extract_task_type(game),
                "environment_seed": environment_seed,
                "won": bool(won),
                "steps": len(turns),
                "calls": len(policy.request_ledger) - before,
            }
        )
        print(f"{game.split('/')[-3]}: won={bool(won)} steps={len(turns)}", flush=True)
        if won and (chosen is None or len(turns) < chosen["steps"]):
            chosen = {
                "game_id": game,
                "environment_seed": environment_seed,
                "steps": len(turns),
                "turns": [
                    {
                        "user": str(turn.state.messages[-1]["content"]),
                        "assistant": str(turn.student.raw),
                        "state_hash": turn.state.state_hash,
                    }
                    for turn in turns
                ],
            }
    if chosen is None:
        raise SystemExit(
            f"the generator solved none of {len(candidates)} candidates for {args.family!r}"
        )

    payload = {
        "artifact": "oneshot_demonstration",
        "source": "built_by_this_project; the reference paper does not publish its demonstration",
        "family": args.family,
        "generator": {
            "model": args.model,
            "base_url": args.base_url,
            "response_model_attestation": validate_provider_response_model_identity(
                policy.request_ledger, args.model
            ),
            "request_ledger_calls": len(policy.request_ledger),
            "request_ledger_sha256": sha256_json(policy.request_ledger),
        },
        "game_id": chosen["game_id"],
        "game_sha256": sha256_file(chosen["game_id"]),
        "environment_seed": chosen["environment_seed"],
        "steps": chosen["steps"],
        "system_prompt_sha256": sha256_text(SAGE_OPD_ALFWORLD_SYSTEM_PROMPT),
        "student_prompt_name": resolve_student_prompt("sage_opd")[0],
        "attempts": attempts,
        "turns": chosen["turns"],
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "script_sha256": sha256_file(Path(__file__)),
        "game_list_cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"demonstration: {chosen['steps']} turns from {chosen['game_id']}")
    print(output)


if __name__ == "__main__":
    main()
