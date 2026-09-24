#!/usr/bin/env python3
"""MVP of the black-box Agent OPD data path, with a feasibility check per step.

Steps (each prints its own verdict so a failure is localised):

  1. states    : read the collected Student pool and rebuild AgentState objects
  2. selection : deterministic G games x m turns (nested hash order, seed)
  3. teacher    : text-only query to an OpenAI-compatible Teacher endpoint,
                  N attempts per state, parse + admissible validation, ledger
  4. rows       : CorrectionRecord -> build_training_rows(game_state_mean)
                  -> action-only token masks for the SFT trainer

The Teacher is queried exactly as a black box: only the conversation is sent and
only the text answer is used. The endpoint applies the Teacher's own chat
template, so no project-side template hook is needed on this path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import OpenAIChatPolicy  # noqa: E402
from omniopd.dataset import build_training_rows  # noqa: E402
from omniopd.parser import parse_action  # noqa: E402
from omniopd.provenance import sha256_file  # noqa: E402
from omniopd.sampling import stable_shuffled  # noqa: E402
from omniopd.schema import ActionSample, AgentState, CorrectionRecord  # noqa: E402
from omniopd.tokenization import encode_final_assistant_example  # noqa: E402


def load_episodes(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def student_sample(turn: dict) -> ActionSample:
    student = turn["student"]
    return ActionSample(
        raw=str(student.get("raw") or ""),
        executed_action=str(student.get("executed_action") or ""),
        valid=bool(student.get("valid")),
        had_action_marker=bool(student.get("had_action_marker")),
        failure_reason=student.get("failure_reason"),
    )


def agent_state(turn: dict, episode: dict) -> AgentState:
    state = turn["state"]
    return AgentState(
        task=state["task"],
        observation=state["observation"],
        admissible_actions=tuple(state["admissible_actions"]),
        messages=tuple(dict(message) for message in state["messages"]),
        game_id=state["game_id"],
        task_type=state["task_type"],
        turn_index=int(state["turn_index"]),
        full_messages=tuple(dict(message) for message in state.get("full_messages", state["messages"])),
        truncation=dict(state.get("truncation", {})),
        state_source="student",
    )


def select(episodes: list[dict], games: int, per_game: int, seed: int) -> list[tuple[dict, dict]]:
    usable = [
        episode
        for episode in episodes
        if any(turn["state"].get("admissible_actions") for turn in episode.get("turns", []))
    ]
    if len(usable) < games:
        raise SystemExit(f"pool has {len(usable)} usable games, need {games}")
    chosen_games = stable_shuffled([episode["game_id"] for episode in usable], seed=seed)[:games]
    by_id = {episode["game_id"]: episode for episode in usable}
    selected: list[tuple[dict, dict]] = []
    for game_id in chosen_games:
        episode = by_id[game_id]
        turns = [turn for turn in episode["turns"] if turn["state"].get("admissible_actions")]
        ordered = stable_shuffled([turn["state"]["state_hash"] for turn in turns],
                                  seed=seed + int(episode.get("environment_seed", 0)))
        by_hash = {turn["state"]["state_hash"]: turn for turn in turns}
        ordered = [by_hash[state_hash] for state_hash in ordered]
        for turn in ordered[:per_game]:
            selected.append((episode, turn))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--states-per-game", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--teacher-model", default="teacher")
    parser.add_argument("--teacher-max-tokens", type=int, default=64)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    # ---- step 1: states -------------------------------------------------
    episodes = load_episodes(args.pool)
    turns_total = sum(len(episode.get("turns", [])) for episode in episodes)
    print(f"[1] states      : {len(episodes)} episodes, {turns_total} turns from {args.pool.name}")

    # ---- step 2: selection ---------------------------------------------
    selected = select(episodes, args.games, args.states_per_game, args.seed)
    games_hit = {episode["game_id"] for episode, _ in selected}
    print(f"[2] selection   : {len(selected)} states over {len(games_hit)} games "
          f"({args.games}x{args.states_per_game}, seed {args.seed})")

    # ---- step 3: teacher (black box) -----------------------------------
    policy = OpenAIChatPolicy(
        model=args.teacher_model,
        base_url=args.teacher_url,
        api_key="EMPTY",
        thinking_mode="disabled",
        thinking_control="chat_template",
        max_retries=0,
        request_ledger_path=args.output_dir / "requests.jsonl",
    )
    records: list[CorrectionRecord] = []
    ledger_rows: list[dict] = []
    for index, (episode, turn) in enumerate(selected):
        state = agent_state(turn, episode)
        samples: list[ActionSample] = []
        for attempt in range(args.attempts):
            request_id = f"mvp-{index:04d}-{attempt}"
            try:
                raw = policy.generate(
                    list(state.messages),
                    temperature=0.0,
                    max_tokens=args.teacher_max_tokens,
                    request_id=request_id,
                )
            except Exception as error:  # noqa: BLE001 -记录并继续，预算照算
                samples.append(
                    ActionSample(raw="", executed_action="", valid=False, had_action_marker=False,
                                 failure_reason=f"request_error:{type(error).__name__}")
                )
                ledger_rows.append({"request_id": request_id, "status": "error", "error": type(error).__name__})
                continue
            parsed = parse_action(raw, state.admissible_actions)
            samples.append(
                ActionSample(
                    raw=raw,
                    executed_action=parsed.canonical_action or parsed.normalized_action or "",
                    valid=parsed.valid,
                    had_action_marker=parsed.had_action_marker,
                    failure_reason=parsed.failure_reason,
                )
            )
            ledger_rows.append(
                {
                    "request_id": request_id,
                    "game_id": state.game_id,
                    "state_hash": state.state_hash,
                    "attempt": attempt,
                    "valid": parsed.valid,
                    "failure_reason": parsed.failure_reason,
                    "raw": raw[:400],
                }
            )
        records.append(
            CorrectionRecord(
                state=state,
                student=student_sample(turn),
                teacher_samples=samples,
                selection_policy="mvp_uniform_per_game",
                inclusion_probability=None,
                teacher_calls=args.attempts,
            )
        )
    total_attempts = len(selected) * args.attempts
    valid_attempts = sum(1 for row in ledger_rows if row.get("valid"))
    valid_states = sum(1 for record in records if record.valid_teacher_samples)
    print(f"[3] teacher     : {total_attempts} attempts, sample acceptance "
          f"{valid_attempts}/{total_attempts}, state acceptance {valid_states}/{len(records)}")
    (args.output_dir / "records.jsonl").write_text(
        "\n".join(json.dumps({"state": record.state.to_dict(), "student": record.student.to_dict(),
                              "teacher_samples": [s.to_dict() for s in record.teacher_samples],
                              "selection_policy": record.selection_policy,
                              "teacher_calls": record.teacher_calls}, ensure_ascii=False)
                  for record in records) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "attempt_ledger.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ledger_rows) + "\n", encoding="utf-8"
    )

    # ---- step 4: rows ---------------------------------------------------
    rows = build_training_rows(records, weighting="game_state_mean")
    if not rows:
        raise SystemExit("no valid Teacher action produced any training row")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.student_tokenizer), local_files_only=True)
    encoded_rows = []
    masked_tokens = defaultdict(int)
    for row in rows:
        messages = [*row.messages, {"role": "assistant", "content": f"Action: {row.teacher_action}"}]
        example = encode_final_assistant_example(
            tokenizer, messages, max_length=args.max_length, truncation="error",
            enable_thinking=row.enable_thinking,
        )
        encoded_rows.append(
            {
                "input_ids": example.input_ids,
                "target_token_mask": example.target_token_mask,
                "state_weight": row.state_weight,
                "game_id": row.game_id,
                "state_hash": row.state_hash,
                "teacher_action": row.teacher_action,
                "student_action": row.student_action,
            }
        )
        masked_tokens[row.task_type] += sum(example.target_token_mask)
    weight_by_game = defaultdict(float)
    for row in encoded_rows:
        weight_by_game[row["game_id"]] += row["state_weight"]
    (args.output_dir / "sft_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in encoded_rows) + "\n", encoding="utf-8"
    )
    print(f"[4] rows        : {len(encoded_rows)} rows, target tokens "
          f"{sum(masked_tokens.values())} (min {min(len(r['input_ids']) for r in encoded_rows)} / "
          f"max {max(len(r['input_ids']) for r in encoded_rows)} tokens)")
    print("    weight sums per game: " + ", ".join(
        f"{g.split('/')[-1][:22]}={w:.3f}" for g, w in sorted(weight_by_game.items())
    ))
    print(f"    pool sha256: {sha256_file(args.pool)[:16]}…  output: {args.output_dir}")


if __name__ == "__main__":
    main()
