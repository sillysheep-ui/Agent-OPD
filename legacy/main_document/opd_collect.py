#!/usr/bin/env python3
"""Collect Route-A on-policy distillation data from ALFWorld (v5 protocol).
 
v5 alignment:
  - Student rolls out on ALFWorld TRAIN games.
  - Full interaction history H_t^S is distinguished from model-visible context
    c_t = T(H_t^S), where T is task-preserving truncation.
  - Student and Teacher see the exact same c_t.
  - Shared deterministic parser/canonicalizer is imported from agent_harness.
  - History stores only the action actually executed + real env observation.
  - Teacher sampling supports <=2 retries per planned sample.
  - Query-level success, attempt-level validity, retry overhead, and A0-v2
    diagnostics are reported separately.
  - Default target is action-only to match the formal Route-A objective.
 
This script performs collection only; no parameter update happens here.
"""
from __future__ import annotations
 
import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
 
import yaml
from openai import OpenAI
from transformers import AutoTokenizer
 
from agent_harness.context import TaskPreservingTruncator
from agent_harness.parser import (
    build_target_response,
    canonicalize_step_response,
    parse_and_canonicalize,
)
 
 
SYSTEM_PROMPT = """You are an ALFWorld household agent. Solve the current task one environment turn at a time.
Reply in this format:
Thought: <brief reasoning>
Action: <one executable ALFWorld command>
Use the command grammar shown by the environment (for example go to, take, move, open, close, use, clean, heat, cool).
To place objects use "move X to Y" (NOT "put X in/on Y"). Object and receptacle names include their numbers, e.g. "fridge 1", "drawer 2".
Do not simulate future observations or future turns."""
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect Route-A black-box OPD samples on ALFWorld train games (v5)"
    )
    p.add_argument("--env-config", required=True)
    p.add_argument("--output", required=True, help="Selected-turn JSONL output")
    p.add_argument(
        "--trajectory-output",
        default=None,
        help="Raw episode trajectory JSONL; default: <output>.trajectories.jsonl",
    )
    p.add_argument(
        "--summary-output",
        default=None,
        help="A0-v2 summary JSON; default: <output>.summary.json",
    )
    p.add_argument("--limit-games", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--turns-per-episode", type=int, default=3)
    p.add_argument(
        "--selection",
        choices=["random", "mean_nll"],
        default="random",
        help="A0/A1 should use random. mean_nll is an optional later ablation.",
    )
 
    p.add_argument("--student-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--student-api-key", default="EMPTY")
    p.add_argument("--student-model", required=True)
    p.add_argument("--student-temperature", type=float, default=0.0)
    p.add_argument("--student-max-tokens", type=int, default=256)
 
    p.add_argument(
        "--tokenizer-path",
        default="/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft",
        help="HF tokenizer path used for exact task-preserving chat-template token counting",
    )
    p.add_argument("--context-max-tokens", type=int, default=4096)
    p.add_argument("--context-reserve-tokens", type=int, default=256)
 
    p.add_argument("--teacher-base-url", default="https://api.deepseek.com")
    p.add_argument("--teacher-api-key", default=None)
    p.add_argument("--teacher-key-file", default="/root/.deepseek_key")
    p.add_argument("--teacher-model", default="deepseek-v4-flash")
    p.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.5,
        help="Part of the Teacher policy definition. Calibrate once, then freeze across A1/A2.",
    )
    p.add_argument("--teacher-max-tokens", type=int, default=3000)
    p.add_argument("--teacher-samples", type=int, default=1)
    p.add_argument(
        "--teacher-max-retries",
        type=int,
        default=2,
        help="Retries AFTER the first failed attempt; max attempts = 1 + this value",
    )
 
    p.add_argument(
        "--target-mode",
        choices=["action", "full"],
        default="action",
        help="v5 default is action-only. 'full' is an explicit ablation.",
    )
    p.add_argument("--action-marker", choices=["first", "last"], default="first")
    return p.parse_args()
 
 
def read_secret(cli_value: str | None, key_file: str | None, env_name: str) -> str:
    if cli_value:
        return cli_value.strip()
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()
    if key_file and Path(key_file).exists():
        raw = Path(key_file).read_text(encoding="utf-8").strip()
        if "=" in raw and not raw.startswith("sk-"):
            raw = raw.split("=", 1)[1]
        return raw.strip()
    raise RuntimeError(
        f"Missing API key: pass CLI value, set {env_name}, or provide a readable key file"
    )
 
 
def task_description(initial_observation: str) -> str:
    parts = str(initial_observation).split("\n\n")
    return "\n".join(parts[1:]).strip() if len(parts) > 1 else str(initial_observation).strip()
 
 
def completion_mean_nll(choice: Any) -> float | None:
    """Optional Student-only uncertainty proxy; NOT full action entropy."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    vals: list[float] = []
    if content:
        for tok in content:
            v = getattr(tok, "logprob", None)
            if v is not None and math.isfinite(float(v)):
                vals.append(-float(v))
    return sum(vals) / len(vals) if vals else None
 
 
@dataclass
class GenerationResult:
    content: str
    mean_nll: float | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    reasoning_present: bool
 
 
def chat_generate(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    want_logprobs: bool = False,
) -> GenerationResult:
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if want_logprobs:
        kwargs.update(logprobs=True, top_logprobs=5)
 
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
 
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None:
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = extra.get("reasoning_content")
 
    usage = getattr(resp, "usage", None)
    return GenerationResult(
        content=content,
        mean_nll=completion_mean_nll(choice) if want_logprobs else None,
        finish_reason=getattr(choice, "finish_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage is not None else None,
        reasoning_present=bool(reasoning),
    )
 
 
@dataclass
class TurnSnapshot:
    turn_index: int
    # H_t^S: complete, untruncated Student-induced interaction history before this action.
    full_history: list[dict[str, str]]
    # c_t = T(H_t^S): exact model-visible context shared by Student and Teacher.
    model_context: list[dict[str, str]]
    context_token_count: int
    dropped_history_pairs: int
    observation: str
    admissible_commands: list[str]
    student_raw_response: str
    student_history_response: str
    student_parsed_action: str
    student_normalized_action: str
    student_executed_action: str
    student_action_valid: bool
    student_had_action_marker: bool
    student_used_put_to_move: bool
    student_used_unique_prefix: bool
    uncertainty_mean_nll: float | None
 
 
def select_turns(
    turns: list[TurnSnapshot],
    m: int,
    mode: str,
    rng: random.Random,
) -> list[TurnSnapshot]:
    if not turns or m <= 0:
        return []
    m = min(m, len(turns))
    if mode == "random":
        return sorted(rng.sample(turns, k=m), key=lambda x: x.turn_index)
 
    missing = [t.turn_index for t in turns if t.uncertainty_mean_nll is None]
    if missing:
        raise RuntimeError(f"mean_nll requested but logprobs missing for turns: {missing}")
 
    top = sorted(
        turns,
        key=lambda x: float(x.uncertainty_mean_nll),
        reverse=True,
    )[:m]
    return sorted(top, key=lambda x: x.turn_index)
 
 
def list_train_games(env_config: dict[str, Any]) -> list[str]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
 
    controller = AlfredTWEnv(env_config, train_eval="train")
    return list(controller.game_files)
 
 
def build_one_game_env(env_config: dict[str, Any], game: str):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
 
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        extras=["gamefile"],
    )
    method = env_config["general"]["training_method"]
    if method == "dagger":
        max_episode_steps = env_config["dagger"]["training"]["max_nb_steps_per_episode"]
    elif method == "dqn":
        max_episode_steps = env_config["rl"]["training"]["max_nb_steps_per_episode"]
    else:
        raise ValueError(
            f"Unsupported ALFWorld training_method={method!r}; use dqn for collection"
        )
 
    env_id = textworld.gym.register_games(
        [game],
        request_infos,
        batch_size=1,
        auto_reset=False,
        asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)
 
 
def unwrap_info(info: dict[str, Any], index: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in info.items():
        try:
            out[key] = value[index]
        except Exception:
            out[key] = value
    return out
 
 
def count_repeated_state_turns(observations: list[str], min_run: int = 3) -> int:
    """Count observations belonging to exact-equality runs of length >= min_run."""
    if not observations:
        return 0
 
    values = [str(x).strip() for x in observations]
    total = 0
    run = 1
    for i in range(1, len(values) + 1):
        if i < len(values) and values[i] == values[i - 1]:
            run += 1
            continue
        if run >= min_run:
            total += run
        run = 1
    return total
 
 
def looks_like_early_none(parsed_action: str, raw_response: str) -> bool:
    value = str(parsed_action or "").strip().lower()
    if value in {"none", "done", "finish", "finished", "complete", "completed"}:
        return True
    return bool(
        __import__("re").search(
            r"(?im)^\s*Action\s*:\s*(?:None|Done|Finish(?:ed)?|Complete(?:d)?)\s*$",
            str(raw_response or ""),
        )
    )
 
 
def collect_episode(
    *,
    game: str,
    env_config: dict[str, Any],
    student: OpenAI,
    student_model: str,
    student_temperature: float,
    student_max_tokens: int,
    max_steps: int,
    selection: str,
    action_marker: str,
    truncator: TaskPreservingTruncator,
) -> tuple[list[TurnSnapshot], dict[str, Any]]:
    env = build_one_game_env(env_config, game)
    try:
        observations, info = env.reset()
        initial_obs = str(observations[0])
        info0 = unwrap_info(info)
        desc = task_description(initial_obs)
 
        # Full interaction history H_t^S.  At query time it always has:
        # System + Task + zero or more complete (executed action, real observation) pairs.
        full_history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Now solve this new task.\n{desc}"},
        ]
 
        current_observation = desc
        turns: list[TurnSnapshot] = []
        success = False
        done = False
        early_none_count = 0
        student_put_to_move_count = 0
        student_prefix_count = 0
 
        for step in range(max_steps):
            admissible = list(info0.get("admissible_commands") or [])
 
            model_context, trunc_info = truncator.truncate(full_history)
            generated = chat_generate(
                student,
                model=student_model,
                messages=model_context,
                temperature=student_temperature,
                max_tokens=student_max_tokens,
                want_logprobs=(selection == "mean_nll"),
            )
 
            parsed = parse_and_canonicalize(
                generated.content,
                admissible,
                marker_strategy=action_marker,
            )
            canonical = parsed.canonical_action
            execute_action = canonical if canonical is not None else "look"
 
            if looks_like_early_none(parsed.raw_candidate, generated.content):
                early_none_count += 1
            student_put_to_move_count += int(parsed.used_put_to_move)
            student_prefix_count += int(parsed.used_unique_prefix)
 
            # v5.1 M22: history assistant messages are plain "Action: X" only
            # (no Thought prefix), matching eval harness and target-mode=action.
            history_response = f"Action: {execute_action}"
 
            turns.append(
                TurnSnapshot(
                    turn_index=step,
                    full_history=copy.deepcopy(full_history),
                    model_context=copy.deepcopy(model_context),
                    context_token_count=trunc_info.token_count,
                    dropped_history_pairs=trunc_info.dropped_pairs,
                    observation=current_observation,
                    admissible_commands=admissible,
                    student_raw_response=generated.content,
                    student_history_response=history_response,
                    student_parsed_action=parsed.raw_candidate,
                    student_normalized_action=parsed.normalized_candidate,
                    student_executed_action=execute_action,
                    student_action_valid=parsed.valid,
                    student_had_action_marker=parsed.had_action_marker,
                    student_used_put_to_move=parsed.used_put_to_move,
                    student_used_unique_prefix=parsed.used_unique_prefix,
                    uncertainty_mean_nll=generated.mean_nll,
                )
            )
 
            full_history.append({"role": "assistant", "content": history_response})
 
            observations, rewards, dones, infos = env.step([execute_action])
            current_observation = str(observations[0])
            info0 = unwrap_info(infos)
 
            done = bool(dones[0])
            won = bool(info0.get("won", False))
            # EnvInfos(won=True) is requested above; use the authoritative task-completion
            # flag. Do not treat a possibly shaped/intermediate positive reward as success.
            success = won
 
            if done:
                break
 
            # Only the REAL environment observation enters H_{t+1}^S.
            full_history.append(
                {
                    "role": "user",
                    "content": f"Observation: {current_observation.strip()}",
                }
            )
 
        observation_trace = [t.observation for t in turns]
        repeated_turns = count_repeated_state_turns(observation_trace, min_run=3)
        invalid_actions = sum(not t.student_action_valid for t in turns)
 
        max_step_exhausted = (
            not success and not done and len(turns) >= max_steps
        ) or (
            not success and len(turns) >= max_steps
        )
 
        meta = {
            "gamefile": str(info0.get("extra.gamefile") or game),
            "task_description": desc,
            "student_success": bool(success),
            "student_steps": len(turns),
            "student_invalid_actions": invalid_actions,
            "student_invalid_rate": invalid_actions / len(turns) if turns else 0.0,
            "student_repeated_state_turns": repeated_turns,
            "student_repeated_state_rate": repeated_turns / len(turns) if turns else 0.0,
            "student_early_none_count": early_none_count,
            "student_max_step_exhausted": bool(max_step_exhausted),
            "student_put_to_move_count": student_put_to_move_count,
            "student_unique_prefix_count": student_prefix_count,
            "max_context_token_count": max(
                (t.context_token_count for t in turns),
                default=0,
            ),
            "max_dropped_history_pairs": max(
                (t.dropped_history_pairs for t in turns),
                default=0,
            ),
        }
        return turns, meta
    finally:
        try:
            env.close()
        except Exception:
            pass
 
 
def classify_teacher_attempt(raw: str, parsed) -> str | None:
    if not str(raw or "").strip():
        return "empty_content"
    if not parsed.had_action_marker:
        return "no_action_marker"
    if not parsed.normalized_candidate:
        return "empty_action"
    if not parsed.valid:
        return "not_admissible_after_canonicalization"
    return None
 
 
def teacher_samples_for_turn(
    *,
    teacher: OpenAI,
    model: str,
    turn: TurnSnapshot,
    temperature: float,
    max_tokens: int,
    n: int,
    max_retries: int,
    action_marker: str,
    target_mode: str,
) -> list[dict[str, Any]]:
    """Collect N planned accepted-action samples, each with bounded retries.
 
    Each planned sample is one query-level target. Each API request is an
    attempt. The first valid canonical action is accepted; otherwise the query
    fails after 1 + max_retries attempts.
    """
    samples: list[dict[str, Any]] = []
 
    for sample_index in range(n):
        attempts: list[dict[str, Any]] = []
        accepted_attempt: dict[str, Any] | None = None
 
        for attempt_index in range(max_retries + 1):
            generated = chat_generate(
                teacher,
                model=model,
                messages=turn.model_context,
                temperature=temperature,
                max_tokens=max_tokens,
                want_logprobs=False,
            )
            parsed = parse_and_canonicalize(
                generated.content,
                turn.admissible_commands,
                marker_strategy=action_marker,
            )
            failure_reason = classify_teacher_attempt(generated.content, parsed)
 
            attempt = {
                "attempt_index": attempt_index,
                "raw_response": generated.content,
                "parsed_action": parsed.raw_candidate,
                "normalized_action": parsed.normalized_candidate,
                "canonical_action": parsed.canonical_action,
                "valid": bool(parsed.valid),
                "failure_reason": failure_reason,
                "had_action_marker": parsed.had_action_marker,
                "used_put_to_move": parsed.used_put_to_move,
                "used_unique_prefix": parsed.used_unique_prefix,
                "finish_reason": generated.finish_reason,
                "reasoning_present": generated.reasoning_present,
                "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens,
                "total_tokens": generated.total_tokens,
            }
            attempts.append(attempt)
 
            if parsed.valid:
                accepted_attempt = attempt
                break
 
            # v5 §4.2: only technical/format failures (empty content, no Action marker,
            # empty action) are retriable. A parsed Action that is not admissible is a
            # DECISION-level failure: fail the query immediately instead of
            # rejection-sampling the Teacher until something legal happens.
            if failure_reason == "not_admissible_after_canonicalization":
                break
 
        if accepted_attempt is not None:
            canonical = str(accepted_attempt["canonical_action"])
            raw = str(accepted_attempt["raw_response"])
            samples.append(
                {
                    "sample_index": sample_index,
                    "valid": True,
                    "accepted": True,
                    "attempt_count": len(attempts),
                    "accepted_attempt_index": accepted_attempt["attempt_index"],
                    "raw_response": raw,
                    "parsed_action": accepted_attempt["parsed_action"],
                    "normalized_action": accepted_attempt["normalized_action"],
                    "canonical_action": canonical,
                    "target_response": build_target_response(
                        raw,
                        canonical,
                        target_mode,
                    ),
                    "attempts": attempts,
                }
            )
        else:
            last = attempts[-1] if attempts else {}
            samples.append(
                {
                    "sample_index": sample_index,
                    "valid": False,
                    "accepted": False,
                    "attempt_count": len(attempts),
                    "accepted_attempt_index": None,
                    "raw_response": last.get("raw_response", ""),
                    "parsed_action": last.get("parsed_action", ""),
                    "normalized_action": last.get("normalized_action", ""),
                    "canonical_action": None,
                    "target_response": None,
                    "attempts": attempts,
                }
            )
 
    return samples
 
 
def depth_bucket(turn_index: int) -> str:
    if turn_index < 10:
        return "00-09"
    if turn_index < 20:
        return "10-19"
    if turn_index < 30:
        return "20-29"
    if turn_index < 40:
        return "30-39"
    return "40+"
 
 
def inc(d: dict[str, int], key: str, value: int = 1) -> None:
    d[key] = d.get(key, 0) + value
 
 
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
 
    env_cfg = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    method = env_cfg.get("general", {}).get("training_method")
    if method != "dqn":
        print(
            f"WARNING: env general.training_method={method!r}; "
            "the current protocol recommends dqn during collection/eval."
        )
 
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.context_max_tokens,
        reserve_tokens=args.context_reserve_tokens,
        enable_thinking=False,
    )
 
    student = OpenAI(
        api_key=args.student_api_key,
        base_url=args.student_base_url,
    )
    teacher_key = read_secret(
        args.teacher_api_key,
        args.teacher_key_file,
        "DEEPSEEK_API_KEY",
    )
    teacher = OpenAI(
        api_key=teacher_key,
        base_url=args.teacher_base_url,
    )
 
    games = list_train_games(env_cfg)
    rng.shuffle(games)
    games = games[: max(0, args.limit_games)]
 
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    trajectory_path = (
        Path(args.trajectory_output)
        if args.trajectory_output
        else out_path.with_suffix(out_path.suffix + ".trajectories.jsonl")
    )
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else out_path.with_suffix(out_path.suffix + ".summary.json")
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
 
    counters: dict[str, int] = {
        "episodes": 0,
        "student_successes": 0,
        "student_turns": 0,
        "student_invalid_actions": 0,
        "student_repeated_state_turns": 0,
        "student_early_none_count": 0,
        "student_max_step_exhausted": 0,
        "student_put_to_move_count": 0,
        "student_unique_prefix_count": 0,
        "selected_turns": 0,
        "teacher_planned_samples": 0,
        "teacher_accepted_samples": 0,
        "teacher_api_attempts": 0,
        "teacher_valid_attempts": 0,
        "teacher_nonempty_attempts": 0,
        "teacher_parsed_action_attempts": 0,
        "teacher_empty_content_attempts": 0,
        "teacher_no_action_marker_attempts": 0,
        "teacher_empty_action_attempts": 0,
        "teacher_not_admissible_attempts": 0,
        "teacher_put_to_move_attempts": 0,
        "teacher_unique_prefix_attempts": 0,
    }
    depth_stats: dict[str, dict[str, int]] = {}
 
    with (
        out_path.open("w", encoding="utf-8") as selected_f,
        trajectory_path.open("w", encoding="utf-8") as traj_f,
    ):
        for ep_idx, game in enumerate(games):
            turns, meta = collect_episode(
                game=game,
                env_config=env_cfg,
                student=student,
                student_model=args.student_model,
                student_temperature=args.student_temperature,
                student_max_tokens=args.student_max_tokens,
                max_steps=args.max_steps,
                selection=args.selection,
                action_marker=args.action_marker,
                truncator=truncator,
            )
 
            counters["episodes"] += 1
            counters["student_successes"] += int(meta["student_success"])
            counters["student_turns"] += int(meta["student_steps"])
            counters["student_invalid_actions"] += int(meta["student_invalid_actions"])
            counters["student_repeated_state_turns"] += int(meta["student_repeated_state_turns"])
            counters["student_early_none_count"] += int(meta["student_early_none_count"])
            counters["student_max_step_exhausted"] += int(meta["student_max_step_exhausted"])
            counters["student_put_to_move_count"] += int(meta["student_put_to_move_count"])
            counters["student_unique_prefix_count"] += int(meta["student_unique_prefix_count"])
 
            traj_f.write(
                json.dumps(
                    {
                        "episode_index": ep_idx,
                        **meta,
                        "turns": [asdict(t) for t in turns],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            traj_f.flush()
 
            selected = select_turns(
                turns,
                args.turns_per_episode,
                args.selection,
                rng,
            )
 
            for turn in selected:
                samples = teacher_samples_for_turn(
                    teacher=teacher,
                    model=args.teacher_model,
                    turn=turn,
                    temperature=args.teacher_temperature,
                    max_tokens=args.teacher_max_tokens,
                    n=args.teacher_samples,
                    max_retries=args.teacher_max_retries,
                    action_marker=args.action_marker,
                    target_mode=args.target_mode,
                )
 
                counters["selected_turns"] += 1
                counters["teacher_planned_samples"] += len(samples)
 
                bucket = depth_bucket(turn.turn_index)
                depth_stats.setdefault(
                    bucket,
                    {
                        "planned_samples": 0,
                        "accepted_samples": 0,
                        "api_attempts": 0,
                    },
                )
 
                for sample in samples:
                    depth_stats[bucket]["planned_samples"] += 1
                    counters["teacher_accepted_samples"] += int(sample["accepted"])
                    depth_stats[bucket]["accepted_samples"] += int(sample["accepted"])
 
                    for attempt in sample["attempts"]:
                        counters["teacher_api_attempts"] += 1
                        depth_stats[bucket]["api_attempts"] += 1
                        counters["teacher_valid_attempts"] += int(attempt["valid"])
                        counters["teacher_nonempty_attempts"] += int(
                            bool(str(attempt["raw_response"] or "").strip())
                        )
                        counters["teacher_parsed_action_attempts"] += int(
                            bool(attempt["had_action_marker"])
                            and bool(str(attempt["normalized_action"] or "").strip())
                        )
                        counters["teacher_put_to_move_attempts"] += int(
                            attempt["used_put_to_move"]
                        )
                        counters["teacher_unique_prefix_attempts"] += int(
                            attempt["used_unique_prefix"]
                        )
 
                        reason = attempt["failure_reason"]
                        if reason == "empty_content":
                            counters["teacher_empty_content_attempts"] += 1
                        elif reason == "no_action_marker":
                            counters["teacher_no_action_marker_attempts"] += 1
                        elif reason == "empty_action":
                            counters["teacher_empty_action_attempts"] += 1
                        elif reason == "not_admissible_after_canonicalization":
                            counters["teacher_not_admissible_attempts"] += 1
 
                record = {
                    "protocol_version": "v5",
                    "episode_index": ep_idx,
                    **meta,
                    "turn": asdict(turn),
                    "selection": args.selection,
                    "teacher_model": args.teacher_model,
                    "teacher_temperature": args.teacher_temperature,
                    "teacher_samples_n": args.teacher_samples,
                    "teacher_max_retries": args.teacher_max_retries,
                    "target_mode": args.target_mode,
                    "teacher_samples": samples,
                }
                selected_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected_f.flush()
 
            print(
                json.dumps(
                    {
                        "episode": ep_idx + 1,
                        "games": len(games),
                        "success": meta["student_success"],
                        "steps": meta["student_steps"],
                        "invalid_student_actions": meta["student_invalid_actions"],
                        "repeated_state_rate": round(
                            float(meta["student_repeated_state_rate"]),
                            4,
                        ),
                        "selected_turns": [t.turn_index for t in selected],
                    },
                    ensure_ascii=False,
                )
            )
 
    planned = counters["teacher_planned_samples"]
    accepted = counters["teacher_accepted_samples"]
    attempts = counters["teacher_api_attempts"]
    valid_attempts = counters["teacher_valid_attempts"]
    nonempty_attempts = counters["teacher_nonempty_attempts"]
    parsed_attempts = counters["teacher_parsed_action_attempts"]
    student_turns = counters["student_turns"]
 
    for bucket, stats in depth_stats.items():
        p = stats["planned_samples"]
        stats["query_success_rate"] = (
            stats["accepted_samples"] / p if p else 0.0
        )
 
    summary = {
        "protocol_version": "v5",
        "config": {
            "seed": args.seed,
            "limit_games": args.limit_games,
            "max_steps": args.max_steps,
            "turns_per_episode": args.turns_per_episode,
            "selection": args.selection,
            "student_model": args.student_model,
            "student_temperature": args.student_temperature,
            "student_max_tokens": args.student_max_tokens,
            "tokenizer_path": args.tokenizer_path,
            "context_max_tokens": args.context_max_tokens,
            "context_reserve_tokens": args.context_reserve_tokens,
            "teacher_model": args.teacher_model,
            "teacher_temperature": args.teacher_temperature,
            "teacher_max_tokens": args.teacher_max_tokens,
            "teacher_samples": args.teacher_samples,
            "teacher_max_retries": args.teacher_max_retries,
            "target_mode": args.target_mode,
            "action_marker": args.action_marker,
        },
        "counts": counters,
        "rates": {
            "student_success_rate": (
                counters["student_successes"] / counters["episodes"]
                if counters["episodes"]
                else 0.0
            ),
            "student_invalid_rate": (
                counters["student_invalid_actions"] / student_turns
                if student_turns
                else 0.0
            ),
            "student_repeated_state_rate": (
                counters["student_repeated_state_turns"] / student_turns
                if student_turns
                else 0.0
            ),
            # Attempt-level diagnostics requested by A0-v2.
            "teacher_response_rate": (
                nonempty_attempts / attempts if attempts else 0.0
            ),
            "teacher_parse_rate_given_response": (
                parsed_attempts / nonempty_attempts if nonempty_attempts else 0.0
            ),
            "teacher_canonicalization_rate_given_parsed": (
                valid_attempts / parsed_attempts if parsed_attempts else 0.0
            ),
            # Go/No-Go metric: accepted query-level targets / planned targets.
            "teacher_query_success_rate": accepted / planned if planned else 0.0,
            # API stability metric: valid canonical actions / all API attempts.
            "teacher_attempt_valid_rate": (
                valid_attempts / attempts if attempts else 0.0
            ),
            # Cost metric: actual API attempts per accepted target.
            "teacher_attempts_per_accepted_target": (
                attempts / accepted if accepted else None
            ),
            "teacher_attempts_per_planned_target": (
                attempts / planned if planned else None
            ),
        },
        "teacher_depth_stats": depth_stats,
        "files": {
            "selected_turns": str(out_path),
            "trajectories": str(trajectory_path),
            "summary": str(summary_path),
        },
    }
 
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
