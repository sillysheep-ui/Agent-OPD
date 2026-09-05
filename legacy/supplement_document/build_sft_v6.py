#!/usr/bin/env python3
"""build_sft_v6.py — Trajectory → SFT dataset conversion (v6 protocol, 7 hard rules).
 
Rule 1: only won=True trajectories become samples.
Rule 2: Teacher technical-failure turns (empty -> fallback look) are NOT targets,
        but the executed 'look' stays in factual history.
Rule 3: history stores only actually executed actions (executed_action), never raw.
Rule 4: no reasoning anywhere; targets are always "Action: X".
Rule 5: admissible actions are the ones of that exact turn (observation_t + A_t).
Rule 6: System + Task are never dropped (task-preserving truncation applied here).
Rule 7: every assistant history message and every target is "Action: X".
"""
from __future__ import annotations
 
import argparse
import json
import random
from pathlib import Path
 
import pandas as pd
import yaml
from transformers import AutoTokenizer
 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_harness.context import TaskPreservingTruncator  # noqa: E402
 
STUDENT_SYSTEM_PROMPT = """You are an ALFWorld household agent.
 
Complete the given household task by interacting with the environment one step at a time.
 
At each turn, use the task description, interaction history, current observation, and current admissible actions to choose the next action.
 
Return exactly one action from the current admissible actions in this format:
Action: <command>
 
Use the action exactly as written in the admissible action list. Do not change object names, object numbers, or command syntax.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="run2/trajectories.jsonl")
    p.add_argument("--output-prefix", required=True, help="e.g. .../alfworld_teacher_sft_v6")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer-path", default="/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft")
    p.add_argument("--context-max-tokens", type=int, default=4096)
    p.add_argument("--context-reserve-tokens", type=int, default=256)
    p.add_argument("--student-base", default="Qwen3-4B-Instruct-2507")
    p.add_argument("--teacher-model", default="deepseek-v4-flash")
    return p.parse_args()
 
 
def admissible_block(admissible) -> str:
    return "\n".join(f"- {a}" for a in admissible)
 
 
def first_user_message(desc: str, observation: str, admissible) -> str:
    return (f"Task:\n{desc}\n\nObservation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def turn_user_message(observation: str, admissible) -> str:
    return (f"Observation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
 
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    truncator = TaskPreservingTruncator(tokenizer, max_context_tokens=args.context_max_tokens,
                                        reserve_tokens=args.context_reserve_tokens,
                                        enable_thinking=False)
 
    recs = [json.loads(l) for l in Path(args.input).open(encoding="utf-8") if l.strip()]
    wins = [r for r in recs if r.get("won")]
    n_games = len(recs)
    n_wins = len(wins)
 
    rows = []
    skipped_technical = 0
    for r in wins:
        desc = str(r.get("task") or "")
        turns = r["turns"]
        # history grows as [system, user0, a0, u1, a1, ..., u_i] — ends on a
        # complete (assistant, user) pair, exactly what the truncator expects.
        history = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": first_user_message(desc, desc, turns[0]["admissible_actions"])},
        ]
        for i, t in enumerate(turns):
            executed = str(t.get("executed_action") or "look")
            # Rule 2: technical-failure turns are not targets, but stay in history
            if t.get("valid"):
                truncated, _ = truncator.truncate(history)
                messages = truncated + [{"role": "assistant", "content": f"Action: {executed}"}]
                rows.append({
                    "messages": messages,
                    "enable_thinking": False,
                    "gamefile": r.get("game_id"),
                    "task_type": r.get("task_type"),
                    "turn_index": i,
                    "teacher_action": executed,
                })
            else:
                skipped_technical += 1
            # factual history continues regardless (Rule 2/3)
            history.append({"role": "assistant", "content": f"Action: {executed}"})
            if i + 1 < len(turns):
                nxt = turns[i + 1]
                history.append({"role": "user",
                                "content": turn_user_message(str(nxt.get("observation") or ""),
                                                             nxt["admissible_actions"])})
 
    if not rows:
        raise RuntimeError("No samples generated; check input and validity flags.")
 
    # Game-level split (MUST be by trajectory, not by step): same game's steps are
    # highly correlated (shared history), so G_train ∩ G_val = ∅.
    by_game = {}
    for row in rows:
        by_game.setdefault(row["gamefile"], []).append(row)
    games = list(by_game.keys())
    rng.shuffle(games)
    n_val_games = max(1, round(len(games) * args.val_fraction))
    val_games = set(games[:n_val_games])
    val_rows = [r for g in val_games for r in by_game[g]]
    train_rows = [r for g in games if g not in val_games for r in by_game[g]]
    print(f"game-level split: {len(games)} games -> train {len(games)-n_val_games} / val {n_val_games}")
 
    train_path = Path(args.output_prefix + "_train.parquet")
    val_path = Path(args.output_prefix + "_val.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
 
    manifest = {
        "protocol": "v6",
        "num_games": n_games,
        "successful_games": n_wins,
        "num_samples": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "skipped_technical_failure_turns": skipped_technical,
        "teacher": args.teacher_model,
        "student_base": args.student_base,
        "with_admissible": True,
        "history_format": "action_observation",
        "target_format": "Action:<cmd>",
        "reasoning_in_target": False,
        "context_max_tokens": args.context_max_tokens,
        "context_reserve_tokens": args.context_reserve_tokens,
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "files": {"train": str(train_path), "val": str(val_path)},
    }
    Path(args.output_prefix + "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
