#!/usr/bin/env python3
"""B0 zero-shot eval harness (RoMeRL-derived interaction shape, no memory, no few-shot).
 
Evaluates any OpenAI-compatible model (local vLLM student or DeepSeek teacher) on
ALFWorld eval_out_of_distribution games. Reuses the frozen parser from
agent_harness.parser and the same SYSTEM_PROMPT as the v5 collector so training
and evaluation behave identically.
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
    p = argparse.ArgumentParser(description="B0 zero-shot eval on ALFWorld OOD")
    p.add_argument("--env-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["eval_in_distribution", "eval_out_of_distribution"],
                   default="eval_out_of_distribution")
    p.add_argument("--limit-games", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
 
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--key-file", default="/root/.deepseek_key",
                   help="Used when --api-key is not given (DeepSeek teacher)")
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--provider", choices=["openai_compat", "deepseek"], default="openai_compat")
    p.add_argument("--with-admissible", action="store_true",
                   help="Append 'Admissible actions: [...]' to observations (training-like protocol)")
    p.add_argument("--tokenizer-path",
                   default="/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft")
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
 
 
def list_eval_games(env_config, split: str):
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    controller = AlfredTWEnv(env_config, train_eval=split)
    return list(controller.game_files)
 
 
def build_one_game_env(env_config, game: str, split: str):
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
 
    if args.provider == "deepseek":
        key = read_secret(args.api_key, args.key_file, "DEEPSEEK_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    else:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
 
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.context_max_tokens,
        reserve_tokens=args.context_reserve_tokens,
        enable_thinking=False,
    )
 
    games = list_eval_games(env_cfg, args.split)
    rng.shuffle(games)
    games = games[: max(0, args.limit_games)]
 
    results = []
    for ep_idx, game in enumerate(games):
        env = build_one_game_env(env_cfg, game, args.split)
        try:
            observations, info = env.reset()
            obs = str(observations[0])
            info0 = unwrap_info(info)
            desc = task_description(obs)
            full_history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Now solve this new task.\n{desc}"},
            ]
            current_observation = desc
            won = False
            done = False
            invalid = 0
            no_marker = 0
            obs_trace = []
            for step in range(args.max_steps):
                admissible = list(info0.get("admissible_commands") or [])
                obs_content = f"Observation: {current_observation.strip()}"
                if args.with_admissible:
                    obs_content += "\nAdmissible actions: " + str(admissible)
                query, _trunc_info = truncator.truncate(full_history)
                if step > 0:
                    query.append({"role": "user", "content": obs_content})
                resp = client.chat.completions.create(
                    model=args.model, messages=query,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
                content = (resp.choices[0].message.content or "") if resp.choices else ""
                parsed = parse_and_canonicalize(content, admissible, marker_strategy="first")
                action = parsed.canonical_action if parsed.valid else "look"
                if not parsed.valid:
                    invalid += 1
                    if not parsed.had_action_marker:
                        no_marker += 1
                obs_trace.append(current_observation)
 
                full_history.append({"role": "assistant", "content": f"Action: {action}"})
                observations, rewards, dones, infos = env.step([action])
                current_observation = str(observations[0])
                info0 = unwrap_info(infos)
                done = bool(dones[0])
                won = bool(info0.get("won", False))
                if done:
                    break
                full_history.append({"role": "user", "content": obs_content})
 
            gf = str(info0.get("extra.gamefile") or game)
            parts = gf.split("/")
            task_type = parts[-3] if len(parts) >= 3 else "?"
            results.append({
                "episode": ep_idx, "gamefile": gf, "task_type": task_type,
                "success": bool(won), "steps": step + 1,
                "invalid_actions": invalid, "no_marker_actions": no_marker,
                "repeated_state": count_repeated(obs_trace),
            })
            print(json.dumps({"episode": ep_idx + 1, "games": len(games),
                              "success": bool(won), "steps": step + 1,
                              "task_type": task_type}, ensure_ascii=False), flush=True)
        finally:
            try:
                env.close()
            except Exception:
                pass
 
    n = len(results)
    succ = sum(r["success"] for r in results)
    invalid = sum(r["invalid_actions"] for r in results)
    steps = sum(r["steps"] for r in results)
    reps = sum(r["repeated_state"] for r in results)
    by_type = {}
    for r in results:
        by_type.setdefault(r["task_type"], [0, 0])
        by_type[r["task_type"]][1] += 1
        by_type[r["task_type"]][0] += int(r["success"])
 
    summary = {
        "model": args.model, "split": args.split, "games": n,
        "success_rate": succ / n if n else 0.0,
        "successes": succ, "total_games": n,
        "invalid_rate": invalid / steps if steps else 0.0,
        "repeated_state_rate": reps / steps if steps else 0.0,
        "avg_steps": steps / n if n else 0.0,
        "by_task_type": {k: {"success": v[0], "total": v[1],
                             "rate": v[0] / v[1] if v[1] else 0.0} for k, v in by_type.items()},
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success_rate": summary["success_rate"], "by_task_type": by_type},
                     ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
