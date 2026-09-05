#!/usr/bin/env python3
"""sft_teacher_collect.py — 独立 SFT Teacher 轨迹采集器（v5.2 / QiangWHU 协议）
 
Design (based on qiangwhu/Qwen3-0.6B-ALFWorld-FullHistory-SFT protocol + our Route A):
- Teacher: DeepSeek-V4-Flash, thinking mode configurable (default enabled for expert demos),
  reasoning stays in reasoning_content; content is Action-only.
- Prompt: expert ALFWorld prompt with grounding constraints (target tracking / no invented objects).
- Records FULL metadata: game_id, task_type, teacher prompt, thinking, temperature, retry,
  parser, admissible, success, token cost.
- Only trajectories with won==True are kept for SFT (expert imitation).
- Step-level samples: (c_t, "Action: a_t") with factual history (no reasoning in history).
"""
from __future__ import annotations
 
import argparse
import json
import os
import random
import sys
from pathlib import Path
 
import yaml
from openai import OpenAI
from transformers import AutoTokenizer
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_harness.parser import parse_and_canonicalize  # noqa: E402
from agent_harness.context import TaskPreservingTruncator  # noqa: E402
 
# v6 P_T: expert Teacher prompt (Action-only content; light grounding constraints)
TEACHER_SYSTEM_PROMPT = """You are an expert ALFWorld household agent.
 
Complete the given household task by interacting with the environment one step at a time.
 
At each turn, use the task description, interaction history, current observation, and current admissible actions to determine the next action.
 
Base your decision only on objects, receptacles, locations, and state information supported by the task and the actual interaction history.
 
Keep the task's target object, its current location, carried objects, and relevant state changes such as cleaned, heated, or cooled consistent with the observed environment.
 
Do not invent object identities, object numbers, locations, carried objects, or state changes.
 
Choose exactly one action from the current admissible actions.
 
Return exactly:
Action: <command>
 
Use the action exactly as written in the admissible action list.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
 
def admissible_block(admissible) -> str:
    """v6 M25: admissible serialized as one '- action' per line."""
    return "\n".join(f"- {a}" for a in admissible)
 
 
def first_user_message(desc: str, observation: str, admissible) -> str:
    return (f"Task:\n{desc}\n\nObservation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def turn_user_message(observation: str, admissible) -> str:
    return (f"Observation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT Teacher trajectory collector (v5.2)")
    p.add_argument("--env-config", required=True)
    p.add_argument("--output", required=True, help="Output directory: trajectories.jsonl + summary.json")
    p.add_argument("--limit-games", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-api-key", default=None)
    p.add_argument("--teacher-key-file", default="/root/.deepseek_key")
    p.add_argument("--teacher-model", default="deepseek-v4-flash")
    p.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled",
                   help="v5.2: SFT/A1 expert demos use thinking enabled; temperature ignored in thinking mode")
    p.add_argument("--reasoning-effort", default="high",
                   help="Thinking-mode effort (low/medium/high), only meaningful when thinking=enabled")
    p.add_argument("--teacher-temperature", type=float, default=0.0,
                   help="Only effective when thinking=disabled (DeepSeek docs)")
    p.add_argument("--teacher-max-tokens", type=int, default=3000)
    p.add_argument("--teacher-max-retries", type=int, default=2,
                   help="Retries for empty content / no Action marker (technical failures only)")
    p.add_argument("--no-admissible", action="store_true",
                   help="v6 main protocol INJECTS admissible actions; this flag disables (diagnostic only)")
    p.add_argument("--tokenizer-path", default="/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft")
    p.add_argument("--context-max-tokens", type=int, default=4096)
    p.add_argument("--context-reserve-tokens", type=int, default=256)
    return p.parse_args()
 
 
def read_secret(cli_value, key_file, env_name):
    if cli_value and cli_value != "EMPTY":
        return cli_value.strip()
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()
    if key_file and Path(key_file).exists():
        raw = Path(key_file).read_text(encoding="utf-8").strip()
        if "=" in raw and not raw.startswith("sk-"):
            raw = raw.split("=", 1)[1]
        return raw.strip()
    raise RuntimeError(f"Missing API key: {env_name} or key file {key_file}")
 
 
def task_description(initial_observation: str) -> str:
    parts = str(initial_observation).split("\n\n")
    return "\n".join(parts[1:]).strip() if len(parts) > 1 else str(initial_observation).strip()
 
 
def list_train_games(env_config):
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    controller = AlfredTWEnv(env_config, train_eval="train")
    return list(controller.game_files)
 
 
def build_one_game_env(env_config, game: str):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    method = env_config["general"]["training_method"]
    if method == "dagger":
        max_episode_steps = env_config["dagger"]["training"]["max_nb_steps_per_episode"]
    elif method == "dqn":
        max_episode_steps = env_config["rl"]["training"]["max_nb_steps_per_episode"]
    else:
        raise ValueError(f"Unsupported training_method={method!r}")
    env_id = textworld.gym.register_games(
        [game], request_infos, batch_size=1, auto_reset=False, asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)
 
 
def unwrap_info(info, index=0):
    out = {}
    for key, value in info.items():
        try:
            out[key] = value[index]
        except Exception:
            out[key] = value
    return out
 
 
def classify_failure(raw: str, parsed) -> str | None:
    if not str(raw or "").strip():
        return "empty_content"
    if not parsed.had_action_marker:
        return "no_action_marker"
    if not parsed.valid:
        return "not_admissible_after_canonicalization"
    return None
 
 
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    env_cfg = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
 
    key = read_secret(args.teacher_api_key, args.teacher_key_file, "DEEPSEEK_API_KEY")
    teacher = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    truncator = TaskPreservingTruncator(tokenizer, max_context_tokens=args.context_max_tokens,
                                        reserve_tokens=args.context_reserve_tokens,
                                        enable_thinking=False)
 
    extra_body = {}
    if args.thinking == "disabled":
        extra_body["thinking"] = {"type": "disabled"}
    elif args.reasoning_effort:
        extra_body["thinking"] = {"type": "enabled", "effort": args.reasoning_effort}
 
    games = list_train_games(env_cfg)
    rng.shuffle(games)
    games = games[: max(0, args.limit_games)]
 
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    summary_path = out_dir / "summary.json"
 
    stats = {"episodes": 0, "successes": 0, "total_turns": 0, "invalid_actions": 0,
             "api_attempts": 0, "total_tokens": 0, "retry_count": 0}
 
    with traj_path.open("w", encoding="utf-8") as f:
        for ep_idx, game in enumerate(games):
            env = build_one_game_env(env_cfg, game)
            try:
                observations, info = env.reset()
                info0 = unwrap_info(info)
                desc = task_description(str(observations[0]))
                gf = str(info0.get("extra.gamefile") or game)
                task_type = gf.split("/")[-3] if "/" in gf else "?"
                init_admissible = list(info0.get("admissible_commands") or [])
                full_history = [
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": first_user_message(desc, desc, init_admissible if not args.no_admissible else [])},
                ]
                cur_obs = desc
                won = False
                done = False
                turns = []
                for step in range(args.max_steps):
                    admissible = list(info0.get("admissible_commands") or [])
                    obs_content = turn_user_message(cur_obs, admissible) if not args.no_admissible else f"Observation: {cur_obs.strip()}"
                    query, _tr = truncator.truncate(full_history)
                    if step > 0:
                        query.append({"role": "user", "content": obs_content})
 
                    # Teacher call with retry (technical failures only)
                    raw = ""
                    parsed = None
                    attempts = 0
                    for attempt in range(args.teacher_max_retries + 1):
                        attempts += 1
                        stats["api_attempts"] += 1
                        resp = teacher.chat.completions.create(
                            model=args.teacher_model, messages=query,
                            temperature=args.teacher_temperature,
                            max_tokens=args.teacher_max_tokens,
                            extra_body=extra_body,
                        )
                        c = resp.choices[0]
                        raw = (c.message.content or "") if c else ""
                        stats["total_tokens"] += getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else 0
                        parsed = parse_and_canonicalize(raw, admissible, marker_strategy="first")
                        reason = classify_failure(raw, parsed)
                        if reason in (None, "not_admissible_after_canonicalization"):
                            break
                        stats["retry_count"] += 1
                    action = parsed.canonical_action if (parsed and parsed.valid) else "look"
                    turns.append({
                        "turn_index": step,
                        "observation": cur_obs,
                        "admissible_actions": admissible,
                        "teacher_raw_response": raw,
                        "executed_action": action,
                        "valid": bool(parsed and parsed.valid),
                        "attempts": attempts,
                    })
                    stats["invalid_actions"] += int(not (parsed and parsed.valid))
                    stats["total_turns"] += 1
 
                    full_history.append({"role": "assistant", "content": f"Action: {action}"})
                    observations, rewards, dones, infos = env.step([action])
                    cur_obs = str(observations[0])
                    info0 = unwrap_info(infos)
                    done = bool(dones[0])
                    won = bool(info0.get("won", False))
                    if done:
                        break
                    full_history.append({"role": "user", "content": obs_content})
 
                stats["episodes"] += 1
                stats["successes"] += int(won)
                record = {
                    "game_id": gf,
                    "task_type": task_type,
                    "task": desc,
                    "won": bool(won),
                    "steps": len(turns),
                    "turns": turns,
                    "config": {
                        "thinking": args.thinking,
                        "reasoning_effort": args.reasoning_effort if args.thinking == "enabled" else None,
                        "temperature": args.teacher_temperature,
                        "max_tokens": args.teacher_max_tokens,
                        "max_retries": args.teacher_max_retries,
                        "admissible_injected": not args.no_admissible,
                        "system_prompt": TEACHER_SYSTEM_PROMPT,
                    },
                }
                # Keep ALL trajectories in the file (filter success at extraction time),
                # but mark won explicitly.
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({"episode": ep_idx + 1, "games": len(games), "won": bool(won),
                                  "steps": len(turns), "task_type": task_type}, ensure_ascii=False), flush=True)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
 
    n = stats["episodes"]
    summary = {
        "protocol": "v5.2-sft-collect",
        "config": {
            "teacher_model": args.teacher_model,
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort if args.thinking == "enabled" else None,
            "temperature": args.teacher_temperature,
            "max_tokens": args.teacher_max_tokens,
            "max_retries": args.teacher_max_retries,
            "admissible_injected": not args.no_admissible,
            "system_prompt": TEACHER_SYSTEM_PROMPT,
        },
        "stats": {
            "games": n,
            "success_rate": stats["successes"] / n if n else 0.0,
            "successes": stats["successes"],
            "total_turns": stats["total_turns"],
            "invalid_action_rate": stats["invalid_actions"] / stats["total_turns"] if stats["total_turns"] else 0.0,
            "api_attempts": stats["api_attempts"],
            "retry_count": stats["retry_count"],
            "total_tokens": stats["total_tokens"],
        },
        "files": {"trajectories": str(traj_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
