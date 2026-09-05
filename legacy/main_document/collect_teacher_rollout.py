#!/usr/bin/env python3
"""Teacher 轨迹重采（v5.1 协议）：DeepSeek 在训练集游戏上 rollout，输出 v5 trajectories。
 
SFT baseline 数据源。Teacher: deepseek-v4-flash, temp=0.5 (M21 与 OPD 统一),
max_tokens=3000, v5 SYSTEM_PROMPT, 纯 Action history (M22)。
输出: <output>/trajectories.jsonl (每游戏完整轨迹) + <output>/summary.json
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
 
SYSTEM_PROMPT = """You are an ALFWorld household agent. Solve the current task one environment turn at a time.
Reply in this format:
Thought: <brief reasoning>
Action: <one executable ALFWorld command>
Use the command grammar shown by the environment (for example go to, take, move, open, close, use, clean, heat, cool).
To place objects use "move X to Y" (NOT "put X in/on Y"). Object and receptacle names include their numbers, e.g. "fridge 1", "drawer 2".
Do not simulate future observations or future turns."""
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher trajectory collection (v5.1 SFT baseline data)")
    p.add_argument("--env-config", required=True)
    p.add_argument("--output", required=True, help="Directory for trajectories.jsonl + summary.json")
    p.add_argument("--limit-games", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-api-key", default=None)
    p.add_argument("--teacher-key-file", default="/root/.deepseek_key")
    p.add_argument("--teacher-model", default="deepseek-v4-flash")
    p.add_argument("--teacher-temperature", type=float, default=0.0,
                   help="M21': SFT demonstration collection uses deterministic decoding (temp=0)")
    p.add_argument("--teacher-max-tokens", type=int, default=3000)
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
 
 
def count_repeated(observations, min_run=3):
    cnt = 0
    run = 1
    for i in range(1, len(observations)):
        if observations[i] == observations[i - 1]:
            run += 1
        else:
            if run >= min_run:
                cnt += run - (min_run - 1)
            run = 1
    if run >= min_run:
        cnt += run - (min_run - 1)
    return cnt
 
 
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
 
    games = list_train_games(env_cfg)
    rng.shuffle(games)
    games = games[: max(0, args.limit_games)]
 
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    summary_path = out_dir / "summary.json"
 
    total_samples = 0
    successes = 0
    invalid_actions = 0
    total_steps = 0
 
    with traj_path.open("w", encoding="utf-8") as f:
        for ep_idx, game in enumerate(games):
            env = build_one_game_env(env_cfg, game)
            try:
                observations, info = env.reset()
                info0 = unwrap_info(info)
                desc = task_description(str(observations[0]))
                full_history = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Now solve this new task.\n{desc}"},
                ]
                cur_obs = desc
                won = False
                done = False
                turns = []
                obs_trace = []
                for step in range(args.max_steps):
                    admissible = list(info0.get("admissible_commands") or [])
                    obs_content = f"Observation: {cur_obs.strip()}"
                    query, _tr = truncator.truncate(full_history)
                    if step > 0:
                        query.append({"role": "user", "content": obs_content})
                    resp = teacher.chat.completions.create(
                        model=args.teacher_model, messages=query,
                        temperature=args.teacher_temperature, max_tokens=args.teacher_max_tokens,
                    )
                    content = (resp.choices[0].message.content or "") if resp.choices else ""
                    parsed = parse_and_canonicalize(content, admissible, marker_strategy="first")
                    action = parsed.canonical_action if parsed.valid else "look"
                    turns.append({
                        "turn_index": step,
                        "query_messages": query,
                        "observation": cur_obs,
                        "admissible_commands": admissible,
                        "teacher_raw": content,
                        "teacher_action": action,
                        "valid": bool(parsed.valid),
                    })
                    invalid_actions += int(not parsed.valid)
                    total_samples += 1
                    obs_trace.append(cur_obs)
 
                    full_history.append({"role": "assistant", "content": f"Action: {action}"})
                    observations, rewards, dones, infos = env.step([action])
                    cur_obs = str(observations[0])
                    info0 = unwrap_info(infos)
                    done = bool(dones[0])
                    won = bool(info0.get("won", False))
                    if done:
                        break
                    full_history.append({"role": "user", "content": obs_content})
 
                gf = str(info0.get("extra.gamefile") or game)
                total_steps += len(turns)
                successes += int(won)
                record = {
                    "episode_index": ep_idx,
                    "gamefile": gf,
                    "task_description": desc,
                    "success": bool(won),
                    "steps": len(turns),
                    "turns": turns,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({"episode": ep_idx + 1, "games": len(games), "success": bool(won),
                                  "steps": len(turns), "samples": total_samples}, ensure_ascii=False), flush=True)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
 
    n_games = len(games)
    summary = {
        "protocol": "v5.1",
        "teacher_model": args.teacher_model,
        "teacher_temperature": args.teacher_temperature,
        "teacher_max_tokens": args.teacher_max_tokens,
        "games": n_games,
        "success_rate": successes / n_games if n_games else 0.0,
        "successes": successes,
        "total_samples": total_samples,
        "invalid_action_rate": invalid_actions / total_steps if total_steps else 0.0,
        "avg_steps": total_steps / n_games if n_games else 0.0,
        "files": {"trajectories": str(traj_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
