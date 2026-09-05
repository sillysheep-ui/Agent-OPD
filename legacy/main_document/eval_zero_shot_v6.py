#!/usr/bin/env python3
"""B0 eval harness (v6 protocol): P_S for Base/SFT student, P_T for Teacher,
with-admissible, Task/Observation/Admissible message structure."""
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
 
# v6 frozen prompts
STUDENT_SYSTEM_PROMPT = """You are an ALFWorld household agent.
 
Complete the given household task by interacting with the environment one step at a time.
 
At each turn, use the task description, interaction history, current observation, and current admissible actions to choose the next action.
 
Return exactly one action from the current admissible actions in this format:
Action: <command>
 
Use the action exactly as written in the admissible action list. Do not change object names, object numbers, or command syntax.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
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
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B0 v6 zero-shot eval")
    p.add_argument("--env-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="eval_out_of_distribution")
    p.add_argument("--limit-games", type=int, default=60)
    p.add_argument("--skip-first", type=int, default=0,
                   help="skip first N shuffled games (shard parallel eval)")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--provider", choices=["openai_compat", "deepseek"], default="openai_compat")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--key-file", default="/root/.deepseek_key")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled",
                   help="Teacher-only: v6 uses thinking enabled for expert demos; eval temperature needs disabled")
    p.add_argument("--no-admissible", action="store_true", help="diagnostic only")
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
 
 
def list_eval_games(env_config, split):
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
    env_cfg = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
 
    if args.provider == "deepseek":
        key = read_secret(args.api_key, args.key_file, "DEEPSEEK_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        system_prompt = TEACHER_SYSTEM_PROMPT
        extra_body = {"thinking": {"type": "disabled"}} if args.thinking == "disabled" else None
    else:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        system_prompt = STUDENT_SYSTEM_PROMPT
        extra_body = None
 
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    truncator = TaskPreservingTruncator(tokenizer, max_context_tokens=args.context_max_tokens,
                                        reserve_tokens=args.context_reserve_tokens,
                                        enable_thinking=False)
 
    games = list_eval_games(env_cfg, args.split)
    rng.shuffle(games)
    games = games[args.skip_first: args.skip_first + max(0, args.limit_games)]
 
    results = {"successes": 0, "games": 0, "invalid": 0, "steps_total": 0,
               "per_game": {}, "by_task_type": {}}
    print(json.dumps({"split": args.split, "model": args.model, "n_games": len(games)}, ensure_ascii=False), flush=True)
 
    for g in games:
        env = build_one_game_env(env_cfg, g, args.split)
        try:
            obs, info = env.reset()
            info0 = unwrap_info(info)
            desc = task_description(str(obs[0]))
            gf = str(info0.get("extra.gamefile") or g)
            task_type = gf.split("/")[-3] if "/" in gf else "?"
            full_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": first_user_message(desc, desc,
                                                               list(info0.get("admissible_commands") or [])
                                                               if not args.no_admissible else [])},
            ]
            cur_obs = desc
            won = False
            done = False
            invalid = 0
            for step in range(args.max_steps):
                admissible = list(info0.get("admissible_commands") or [])
                obs_content = turn_user_message(cur_obs, admissible) if not args.no_admissible else f"Observation: {cur_obs.strip()}"
                query, _ = truncator.truncate(full_history)
                if step > 0:
                    query.append({"role": "user", "content": obs_content})
                kwargs = dict(model=args.model, messages=query,
                              temperature=args.temperature, max_tokens=args.max_tokens)
                if extra_body:
                    kwargs["extra_body"] = extra_body
                resp = client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "") if resp.choices else ""
                parsed = parse_and_canonicalize(content, admissible, marker_strategy="first")
                action = parsed.canonical_action if parsed.valid else "look"
                invalid += int(not parsed.valid)
                full_history.append({"role": "assistant", "content": f"Action: {action}"})
                obs, rewards, dones, infos = env.step([action])
                cur_obs = str(obs[0])
                info0 = unwrap_info(infos)
                done = bool(dones[0])
                won = bool(info0.get("won", False))
                if done:
                    break
                full_history.append({"role": "user", "content": obs_content})
 
            results["games"] += 1
            results["successes"] += int(won)
            results["invalid"] += invalid
            results["steps_total"] += step + 1
            results["per_game"][gf] = int(won)
            results["by_task_type"].setdefault(task_type, {"success": 0, "total": 0})
            results["by_task_type"][task_type]["total"] += 1
            results["by_task_type"][task_type]["success"] += int(won)
            print(json.dumps({"game": task_type, "won": bool(won), "steps": step + 1, "invalid": invalid},
                             ensure_ascii=False), flush=True)
        finally:
            try:
                env.close()
            except Exception:
                pass
 
    results["success_rate"] = results["successes"] / max(results["games"], 1)
    results["invalid_rate"] = results["invalid"] / max(results["steps_total"], 1)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"final": results["successes"], "games": results["games"],
                      "success_rate": round(results["success_rate"], 4)}, ensure_ascii=False))
 
 
if __name__ == "__main__":
    main()
