#!/usr/bin/env python3
"""Six-game, nonconfirmatory Qwen3-14B SFT demonstration pilot."""

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
)
from omniopd.context import TaskPreservingTruncator
from omniopd.environment_provenance import derive_environment_seed
from omniopd.prompts import TEACHER_SYSTEM_PROMPT
from omniopd.protocol import GenerationSettings, rollout_episode
from omniopd.provenance import sha256_file, sha256_json
from omniopd.sampling import stable_shuffled

OUTPUT = Path("/runs/sft_teacher_pilot_20260918_choice_01")
MODEL_DIR = Path("/model")
MODEL_ALIAS = "qwen3-14b-sft-pilot"
GAME_ORDER_SEED = 42
ENVIRONMENT_MASTER_SEED = 314159
class AdmissibleChoicePolicy(OpenAIChatPolicy):
    """Sequential exploratory policy; preserve the parent request ledger."""

    def generate(self, messages, *, temperature, max_tokens, request_id):
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("current query must end with a user message")

        marker = "\n\nAdmissible actions:\n"
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

def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUTPUT}")
    if not (MODEL_DIR / "config.json").is_file():
        raise SystemExit("Qwen3-14B model configuration is missing")

    config = yaml.safe_load(
        Path("/app/configs/alfworld_textworld.yaml").read_text(encoding="utf-8")
    )
    by_type = {}
    for game in list_alfworld_games(config, "train"):
        by_type.setdefault(extract_task_type(game), []).append(str(game))

    if len(by_type) != 6:
        raise SystemExit(f"Expected six task types, found: {sorted(by_type)}")

    selected = [
        (task_type, stable_shuffled(by_type[task_type], seed=GAME_ORDER_SEED)[0])
        for task_type in sorted(by_type)
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=4096,
        reserve_tokens=256,
        enable_thinking=False,
    )

    OUTPUT.mkdir(parents=True)
    teacher = AdmissibleChoicePolicy(
        model=MODEL_ALIAS,
        base_url="http://127.0.0.1:18081/v1",
        api_key="EMPTY",
        thinking_mode="disabled",
        thinking_control="chat_template",
        max_retries=0,
        request_ledger_path=OUTPUT / "requests.jsonl",
    )

    results = []
    with (OUTPUT / "episodes.jsonl").open("x", encoding="utf-8") as handle:
        for index, (task_type, game) in enumerate(selected, 1):
            environment_seed = derive_environment_seed(
                ENVIRONMENT_MASTER_SEED, game
            )
            before_calls = len(teacher.request_ledger)
            env = AlfworldEnvironment(
                config, game, rollout_seed=environment_seed
            )
            try:
                turns, won = rollout_episode(
                    env,
                    teacher,
                    truncator,
                    settings=GenerationSettings(
                        temperature=0.0,
                        max_tokens=64,
                    ),
                    max_steps=50,
                    system_prompt=TEACHER_SYSTEM_PROMPT,
                    state_source="teacher",
                )
            finally:
                env.close()

            calls = len(teacher.request_ledger) - before_calls
            if calls != len(turns):
                raise RuntimeError("Teacher call count does not match turns")

            record = {
                "artifact": "nonconfirmatory_sft_teacher_pilot_episode",
                "game_id": game,
                "game_sha256": sha256_file(game),
                "task_type": task_type,
                "environment_seed": environment_seed,
                "won": won,
                "steps": len(turns),
                "valid_turns": sum(turn.student.valid for turn in turns),
                "invalid_turns": sum(not turn.student.valid for turn in turns),
                "teacher_calls": calls,
                "turns": [turn.to_dict() for turn in turns],
            }
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            results.append(record)
            print(
                f"{index}/6 {task_type}: won={won}, "
                f"steps={len(turns)}, invalid={record['invalid_turns']}",
                flush=True,
            )

    ledger = teacher.request_ledger
    response_models = sorted({
        str(row.get("response_model"))
        for row in ledger
        if row.get("status") == "ok"
    })
    if response_models != [MODEL_ALIAS]:
        raise RuntimeError(f"Unexpected response models: {response_models}")

    token_counts = [row.get("total_tokens") for row in ledger]
    summary = {
        "artifact": "nonconfirmatory_sft_teacher_pilot_summary",
        "training_ready": False,
        "model_alias": MODEL_ALIAS,
        "model_config_sha256": sha256_file(MODEL_DIR / "config.json"),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "max_tokens_per_action": 64,
        "max_steps_per_game": 50,
        "game_order_seed": GAME_ORDER_SEED,
        "environment_master_seed": ENVIRONMENT_MASTER_SEED,
        "games": len(results),
        "successes": sum(item["won"] for item in results),
        "teacher_calls": len(ledger),
        "total_tokens": (
            sum(token_counts)
            if all(isinstance(value, int) for value in token_counts)
            else None
        ),
        "episodes_sha256": sha256_file(OUTPUT / "episodes.jsonl"),
        "requests_sha256": sha256_file(OUTPUT / "requests.jsonl"),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
