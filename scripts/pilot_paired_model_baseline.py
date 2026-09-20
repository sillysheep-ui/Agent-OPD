#!/usr/bin/env python3
"""Nonconfirmatory, paired ALFWorld baseline for two untrained-on-task models.

This pilot intentionally uses the same constrained action protocol for both
models. It is not an SFT dataset or a reproduction of SAGE-OPD training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from omniopd.adapters import (
    AlfworldEnvironment,
    OpenAIChatPolicy,
    extract_task_type,
    list_alfworld_games,
    validate_provider_response_model_identity,
)
from omniopd.context import TaskPreservingTruncator
from omniopd.environment_provenance import derive_environment_seed
from omniopd.protocol import GenerationSettings, rollout_episode
from omniopd.provenance import sha256_file, sha256_json
from omniopd.sampling import stable_shuffled


PROMPT = """You are an expert ALFWorld household agent.

Complete the task one action at a time using only the task, executed interaction history, current observation, and current admissible actions.

Keep track of the target object or objects, their count and last observed locations, what you carry, the required clean/heat/cool state, the destination, and which subgoals are already completed. Update this understanding from actual observations. Do not invent object identities, locations, or state changes.

Complete any required clean/heat/cool transformation before final placement. For a two-object task, track two distinct target objects. If recent actions made no progress, avoid repeating the same cycle and choose a different admissible action that gathers information or advances the task.

Before responding, check that your chosen command exactly matches one current admissible action.

Return exactly one line:
Action: <command>

Do not output explanations, reasoning, multiple actions, or predicted future observations."""

GAME_ORDER_SEED = 42
ENVIRONMENT_MASTER_SEED = 314159
GAME_INDEX = 1  # index 0 was repeatedly used for prompt development


class AdmissibleChoicePolicy(OpenAIChatPolicy):
    """Constrain each response to the *current* admissible action set."""

    def generate(self, messages, *, temperature, max_tokens, request_id):
        marker = "\n\nAdmissible actions:\n"
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("query must end with the current user message")
        parts = messages[-1]["content"].rsplit(marker, 1)
        if len(parts) != 2:
            raise ValueError("current admissible-action block is missing")
        lines = [line for line in parts[1].splitlines() if line.strip()]
        if not lines or any(not line.startswith("- ") for line in lines):
            raise ValueError("malformed admissible-action block")
        actions = [line[2:] for line in lines]
        if len(actions) != len(set(actions)):
            raise ValueError("duplicate admissible actions")
        choices = ["Action: " + action for action in actions]
        self._choice_metadata = {
            "constraint_mode": "admissible_choice",
            "choice_count": len(choices),
            "choices_sha256": sha256_json(choices),
        }
        previous_extra_body = self.extra_body
        self.extra_body = {
            **previous_extra_body,
            "structured_outputs": {"choice": choices},
        }
        try:
            return super().generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                request_id=request_id,
            )
        finally:
            self.extra_body = previous_extra_body
            self._choice_metadata = None

    def _record_request(self, entry):
        metadata = getattr(self, "_choice_metadata", None)
        if metadata is None:
            raise RuntimeError("choice metadata missing for request ledger")
        super()._record_request({**entry, **metadata})


def _run_arm(*, config, selected, arm, output):
    arm_dir = output / arm["label"]
    arm_dir.mkdir(exist_ok=False)
    model_dir = Path(arm["model_dir"])
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"model config is missing: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=4096,
        reserve_tokens=256,
        enable_thinking=False,
    )
    policy = AdmissibleChoicePolicy(
        model=arm["alias"],
        base_url=arm["base_url"],
        api_key="EMPTY",
        thinking_mode="disabled",
        thinking_control="chat_template",
        max_retries=0,
        request_ledger_path=arm_dir / "requests.jsonl",
    )
    episodes = []
    with (arm_dir / "episodes.jsonl").open("x", encoding="utf-8") as handle:
        for index, (task_type, game) in enumerate(selected, 1):
            environment_seed = derive_environment_seed(
                ENVIRONMENT_MASTER_SEED, game
            )
            before_calls = len(policy.request_ledger)
            env = AlfworldEnvironment(config, game, rollout_seed=environment_seed)
            try:
                turns, won = rollout_episode(
                    env,
                    policy,
                    truncator,
                    settings=GenerationSettings(temperature=0.0, max_tokens=64),
                    max_steps=50,
                    system_prompt=PROMPT,
                    state_source="paired_baseline",
                )
            finally:
                env.close()
            calls = len(policy.request_ledger) - before_calls
            if calls != len(turns):
                raise RuntimeError("request count does not match turns")
            record = {
                "artifact": "nonconfirmatory_paired_model_baseline_episode",
                "game_id": game,
                "game_sha256": sha256_file(game),
                "task_type": task_type,
                "environment_seed": environment_seed,
                "won": won,
                "steps": len(turns),
                "valid_turns": sum(turn.student.valid for turn in turns),
                "invalid_turns": sum(not turn.student.valid for turn in turns),
                "request_calls": calls,
                "turns": [turn.to_dict() for turn in turns],
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            episodes.append(record)
            print(
                f"{arm['label']} {index}/{len(selected)} {task_type}: "
                f"won={won}, steps={len(turns)}, invalid={record['invalid_turns']}",
                flush=True,
            )
    validate_provider_response_model_identity(policy.request_ledger, arm["alias"])
    token_counts = [row.get("total_tokens") for row in policy.request_ledger]
    summary = {
        "artifact": "nonconfirmatory_paired_model_baseline_summary",
        "training_ready": False,
        "model_alias": arm["alias"],
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "constraint_mode": "admissible_choice",
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "max_context_tokens": 4096,
        "max_tokens_per_action": 64,
        "max_steps_per_game": 50,
        "game_order_seed": GAME_ORDER_SEED,
        "game_index": GAME_INDEX,
        "environment_master_seed": ENVIRONMENT_MASTER_SEED,
        "games": len(episodes),
        "successes": sum(row["won"] for row in episodes),
        "request_calls": len(policy.request_ledger),
        "total_tokens": (
            sum(token_counts)
            if all(isinstance(value, int) for value in token_counts)
            else None
        ),
        "episodes_sha256": sha256_file(arm_dir / "episodes.jsonl"),
        "requests_sha256": sha256_file(arm_dir / "requests.jsonl"),
    }
    (arm_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--game-list",
        help="reuse an existing six-game JSON list instead of rescanning ALFWorld",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    arms = [
        {
            "label": "qwen3_14b",
            "model_dir": "/models/qwen3-14b",
            "alias": "qwen3-14b-sft-pilot",
            "base_url": "http://127.0.0.1:18081/v1",
        },
        {
            "label": "qwen3_4b",
            "model_dir": "/models/qwen3-4b",
            "alias": "qwen3-4b-baseline",
            "base_url": "http://127.0.0.1:18082/v1",
        },
    ]
    for arm in arms:
        if not (Path(arm["model_dir"]) / "config.json").is_file():
            raise SystemExit(f"Missing model directory: {arm['model_dir']}")
    config = yaml.safe_load(
        Path("/app/configs/alfworld_textworld.yaml").read_text(encoding="utf-8")
    )
    if args.game_list:
        selected = json.loads(Path(args.game_list).read_text(encoding="utf-8"))
        if (
            not isinstance(selected, list)
            or len(selected) != 6
            or any(not isinstance(row, list) or len(row) != 2 for row in selected)
            or len({row[0] for row in selected}) != 6
            or any(
                not isinstance(row[0], str)
                or not isinstance(row[1], str)
                or not row[1].startswith("/alfworld_data/json_2.1.1/train/")
                or not Path(row[1]).is_file()
                for row in selected
            )
        ):
            raise SystemExit("--game-list must contain six distinct training games")
    else:
        by_type = {}
        for game in list_alfworld_games(config, "train"):
            by_type.setdefault(extract_task_type(game), []).append(str(game))
        if len(by_type) != 6:
            raise SystemExit(f"Expected six ALFWorld task types, found {sorted(by_type)}")
        selected = []
        for task_type in sorted(by_type):
            ordered = stable_shuffled(by_type[task_type], seed=GAME_ORDER_SEED)
            selected.append((task_type, ordered[GAME_INDEX]))
    output.mkdir(parents=True)
    (output / "game_list.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summaries = [
        _run_arm(config=config, selected=selected, arm=arm, output=output)
        for arm in arms
    ]
    (output / "paired_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
