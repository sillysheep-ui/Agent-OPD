#!/usr/bin/env python3
"""Pre-validate the correction mechanism before the full four-arm study.

Measures the *paired counterfactual value* of a single Teacher correction:

  Y(a_S)  the Student's own rollout on a game (baseline)
  Y(a_T)  same game, same seed, same prefix, but at the FIRST state where the
          Teacher (black box, P_T, temperature 0) gives a valid action that
          differs from the Student's action, the Teacher's action is executed
          and the SAME frozen Student continues to the end.

主报 E[C | D=1, V_T=1], C = Y(a_T) - Y(a_S)，并按分歧位置分层（早期 vs 后期）。

Decision rule (pre-registered): >= +10 points of corrected-vs-baseline win rate
over ~40 games means the mechanism has room and the four-arm study is worth it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import AlfworldEnvironment, OpenAIChatPolicy, extract_task_type  # noqa: E402
from omniopd.context import TaskPreservingTruncator  # noqa: E402
from omniopd.counterfactual import run_paired_counterfactual  # noqa: E402
from omniopd.environment_provenance import derive_environment_seed  # noqa: E402
from omniopd.parser import parse_action  # noqa: E402
from omniopd.prompts import TEACHER_SYSTEM_PROMPT, replace_system  # noqa: E402
from omniopd.protocol import GenerationSettings, rollout_episode  # noqa: E402

EARLY_TURN_CUTOFF = 5


def student_prompt_from_pool(pool: Path) -> str:
    first = json.loads(pool.read_text(encoding="utf-8").splitlines()[0])
    return first["turns"][0]["state"]["messages"][0]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True, help="JSONL list of games")
    parser.add_argument("--pool", type=Path, required=True, help="for the frozen Student system prompt")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--student-url", default="http://127.0.0.1:18121/v1")
    parser.add_argument("--student-model", default="base4b")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:18111/v1")
    parser.add_argument("--teacher-model", default="teacher")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--master-seed", type=int, default=314159)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    from transformers import AutoTokenizer

    config = yaml.safe_load(args.env_config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
        enable_thinking=False,
    )
    system_prompt = student_prompt_from_pool(args.pool)
    student = OpenAIChatPolicy(
        model=args.student_model, base_url=args.student_url, api_key="EMPTY",
        thinking_mode="disabled", thinking_control="chat_template", max_retries=0,
        request_ledger_path=args.output_dir / "student_requests.jsonl",
    )
    teacher = OpenAIChatPolicy(
        model=args.teacher_model, base_url=args.teacher_url, api_key="EMPTY",
        thinking_mode="disabled", thinking_control="chat_template", max_retries=0,
        request_ledger_path=args.output_dir / "teacher_requests.jsonl",
    )
    settings = GenerationSettings(temperature=0.0, max_tokens=args.max_tokens)

    games = []
    for line in args.val_data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        games.append(row.get("game_id") or row.get("game") or row.get("file"))
    if not games:
        raise SystemExit(f"no games in {args.val_data}")
    games = games[: args.games]

    records = []
    for index, game in enumerate(games, 1):
        environment_seed = derive_environment_seed(args.master_seed, game)
        env = AlfworldEnvironment(config, game, rollout_seed=environment_seed)
        try:
            turns, won = rollout_episode(
                env, student, truncator, settings=settings, max_steps=args.max_steps,
                system_prompt=system_prompt, state_source="student",
            )
        finally:
            env.close()

        # first state where the Teacher's action is valid and differs from the Student's
        correction = None
        prefix: list[str] = []
        for turn in turns:
            teacher_reply = teacher.generate(
                replace_system(list(turn.state.messages), TEACHER_SYSTEM_PROMPT),
                temperature=0.0, max_tokens=args.max_tokens,
                request_id=f"preval:{index}:{turn.state.turn_index}",
            )
            parsed = parse_action(teacher_reply, turn.state.admissible_actions)
            if parsed.valid and parsed.canonical_action != turn.student.executed_action:
                correction = (prefix[:], turn, parsed.canonical_action)
                break
            prefix.append(turn.student.executed_action)

        row = {
            "game_id": game,
            "task_type": extract_task_type(game),
            "baseline_won": bool(won),
            "baseline_steps": len(turns),
            "disagreement_found": correction is not None,
        }
        if correction is not None:
            prefix_actions, turn, teacher_action = correction

            def env_factory():
                return AlfworldEnvironment(config, game, rollout_seed=environment_seed)

            pair = run_paired_counterfactual(
                env_factory=env_factory,
                prefix_actions=prefix_actions,
                state=turn.state,
                student_action=turn.student.executed_action,
                teacher_action=teacher_action,
                frozen_student=student,
                truncator=truncator,
                settings=settings,
                max_steps=args.max_steps,
                request_namespace=f"preval:{index}:{turn.state.turn_index}",
                student_system_prompt=system_prompt,
            )
            row.update({
                "turn_index": turn.state.turn_index,
                "student_action": turn.student.executed_action,
                "teacher_action": teacher_action,
                "replayed_won": bool(pair.student_branch.won),
                "corrected_won": bool(pair.teacher_branch.won),
                "consequence": int(pair.consequence),
                "early": turn.state.turn_index < EARLY_TURN_CUTOFF,
            })
        records.append(row)
        print(json.dumps(row, ensure_ascii=False)[:300], flush=True)

    (args.output_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8"
    )
    with_correction = [row for row in records if row["disagreement_found"]]
    baseline_wins = sum(row["baseline_won"] for row in records)
    replay_wins = sum(row.get("replayed_won", row["baseline_won"]) for row in records)
    corrected_wins = sum(row.get("corrected_won", row["baseline_won"]) for row in records)
    flips = Counter(str(row.get("consequence")) for row in with_correction)
    early = [row for row in with_correction if row["early"]]
    summary = {
        "games": len(records),
        "states_with_correction": len(with_correction),
        "correction_rate": len(with_correction) / len(records),
        "baseline_wins": baseline_wins,
        "replayed_wins": replay_wins,
        "corrected_wins": corrected_wins,
        "baseline_rate": baseline_wins / len(records),
        "corrected_rate": corrected_wins / len(records),
        "delta_points": 100 * (corrected_wins - baseline_wins) / len(records),
        "consequences": dict(flips),
        "early_states": len(early),
        "early_positive": sum(row["consequence"] > 0 for row in early),
        "verdict": ("MECHANISM HAS ROOM" if 100 * (corrected_wins - baseline_wins) / len(records) >= 10
                    else "no measurable single-step gain at this scale"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
