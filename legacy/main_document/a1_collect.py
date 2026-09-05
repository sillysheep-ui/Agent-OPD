#!/usr/bin/env python3
"""a1_collect.py — A1 N=1 on-policy Teacher correction collection (v6 protocol).
 
Student (θ_SFT_v6, P_S) rolls out greedily on train games NOT overlapping SFT collection;
M=3 turns per episode are selected with a pre-fixed seed; Teacher (P_T, thinking enabled)
gives N=1 action correction on the exact same Student-induced state + admissible set.
Teacher prompt never enters training data; samples are rebuilt as (P_S, z_t^S) -> Action: a_t^T.
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
    p = argparse.ArgumentParser(description="A1 N=1 Teacher correction collection")
    p.add_argument("--env-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit-games", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m-turns", type=int, default=3)
    p.add_argument("--exclude-games-jsonl", action="append", default=[],
                   help="SFT/A1 collection trajectories/corrections jsonl(s); their game_ids are excluded (repeatable)")
    p.add_argument("--student-model", required=True, help="vLLM model id (sft-v6 LoRA)")
    p.add_argument("--student-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--student-temperature", type=float, default=0.0)
    p.add_argument("--student-max-tokens", type=int, default=256)
    p.add_argument("--teacher-api-key", default=None)
    p.add_argument("--teacher-key-file", default="/root/.deepseek_key")
    p.add_argument("--teacher-model", default="deepseek-v4-flash")
    p.add_argument("--teacher-max-tokens", type=int, default=3000)
    p.add_argument("--teacher-max-retries", type=int, default=2)
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
    return list(AlfredTWEnv(env_config, train_eval="train").game_files)
 
 
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
 
 
def admissible_block(admissible) -> str:
    return "\n".join(f"- {a}" for a in admissible)
 
 
def first_user_message(desc: str, observation: str, admissible) -> str:
    return (f"Task:\n{desc}\n\nObservation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def turn_user_message(observation: str, admissible) -> str:
    return (f"Observation:\n{observation.strip()}\n\nAdmissible actions:\n"
            + admissible_block(admissible))
 
 
def classify_failure(raw, parsed):
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
 
    # Exclude SFT collection games + previous A1 rounds
    excluded = set()
    for path in args.exclude_games_jsonl:
        for line in Path(path).open(encoding="utf-8"):
            if line.strip():
                excluded.add(json.loads(line).get("game_id"))
    games = [g for g in list_train_games(env_cfg) if g not in excluded]
    rng.shuffle(games)
    games = games[: max(0, args.limit_games)]
    print(json.dumps({"games_selected": len(games), "excluded_sft_games": len(excluded)},
                     ensure_ascii=False), flush=True)
 
    student = OpenAI(api_key="EMPTY", base_url=args.student_base_url)
    key = read_secret(args.teacher_api_key, args.teacher_key_file, "DEEPSEEK_API_KEY")
    teacher = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    truncator = TaskPreservingTruncator(tokenizer, max_context_tokens=args.context_max_tokens,
                                        reserve_tokens=args.context_reserve_tokens,
                                        enable_thinking=False)
    teacher_extra = {"thinking": {"type": "enabled", "effort": "high"}}
 
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corrections.jsonl"
    summary_path = out_dir / "summary.json"
 
    stats = {"episodes": 0, "student_wins": 0, "turns_total": 0,
             "corrections_total": 0, "corrections_valid": 0,
             "disagreement": 0, "teacher_retries": 0}
    corrections_all = []
 
    with out_path.open("w", encoding="utf-8") as f:
        for ep_idx, game in enumerate(games):
            env = build_one_game_env(env_cfg, game)
            try:
                observations, info = env.reset()
                info0 = unwrap_info(info)
                desc = task_description(str(observations[0]))
                gf = str(info0.get("extra.gamefile") or game)
                task_type = gf.split("/")[-3] if "/" in gf else "?"
                full_history = [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                    {"role": "user", "content": first_user_message(desc, desc,
                                                                   list(info0.get("admissible_commands") or []))},
                ]
                cur_obs = desc
                won = False
                done = False
                student_turns = []
                for step in range(args.max_steps):
                    admissible = list(info0.get("admissible_commands") or [])
                    obs_content = turn_user_message(cur_obs, admissible)
                    query, _ = truncator.truncate(full_history)
                    if step > 0:
                        query.append({"role": "user", "content": obs_content})
                    resp = student.chat.completions.create(
                        model=args.student_model, messages=query,
                        temperature=args.student_temperature, max_tokens=args.student_max_tokens)
                    content = (resp.choices[0].message.content or "") if resp.choices else ""
                    parsed = parse_and_canonicalize(content, admissible, marker_strategy="first")
                    action = parsed.canonical_action if parsed.valid else "look"
                    student_turns.append({
                        "turn_index": step,
                        "query_messages": query,
                        "observation": cur_obs,
                        "admissible_actions": admissible,
                        "student_raw": content,
                        "student_action": action,
                        "student_valid": bool(parsed.valid),
                    })
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
                stats["student_wins"] += int(won)
                stats["turns_total"] += len(student_turns)
 
                # M random turns (pre-fixed rng state per episode via ep seed)
                ep_rng = random.Random(args.seed * 1000 + ep_idx)
                n_sel = min(args.m_turns, len(student_turns))
                selected = ep_rng.sample(range(len(student_turns)), n_sel)
                for turn_idx in selected:
                    t = student_turns[turn_idx]
                    # Teacher correction on identical student state
                    raw = ""
                    parsed = None
                    for attempt in range(args.teacher_max_retries + 1):
                        resp = teacher.chat.completions.create(
                            model=args.teacher_model, messages=t["query_messages"],
                            temperature=0.0, max_tokens=args.teacher_max_tokens,
                            extra_body=teacher_extra)
                        c = resp.choices[0]
                        raw = (c.message.content or "") if c else ""
                        parsed = parse_and_canonicalize(raw, t["admissible_actions"], marker_strategy="first")
                        reason = classify_failure(raw, parsed)
                        if reason in (None, "not_admissible_after_canonicalization"):
                            break
                        stats["teacher_retries"] += 1
                    t_action = parsed.canonical_action if (parsed and parsed.valid) else "look"
                    valid = bool(parsed and parsed.valid)
                    disagree = valid and (t_action != t["student_action"])
                    stats["corrections_total"] += 1
                    stats["corrections_valid"] += int(valid)
                    stats["disagreement"] += int(disagree)
                    rec = {
                        "episode_index": ep_idx,
                        "gamefile": gf,
                        "task_type": task_type,
                        "student_won": bool(won),
                        "turn_index": turn_idx,
                        "turn_depth": turn_idx,
                        "query_messages": t["query_messages"],
                        "admissible_actions": t["admissible_actions"],
                        "student_action": t["student_action"],
                        "student_valid": t["student_valid"],
                        "teacher_raw": raw,
                        "teacher_action": t_action,
                        "teacher_valid": valid,
                        "disagreement": disagree,
                    }
                    corrections_all.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                print(json.dumps({"episode": ep_idx + 1, "games": len(games), "won": bool(won),
                                  "steps": len(student_turns), "task_type": task_type,
                                  "sel_turns": n_sel}, ensure_ascii=False), flush=True)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
 
    summary = {
        "protocol": "A1-v6",
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "m_turns": args.m_turns,
        "excluded_sft_games": len(excluded),
        "stats": stats,
        "disagreement_rate": stats["disagreement"] / max(stats["corrections_total"], 1),
        "correction_valid_rate": stats["corrections_valid"] / max(stats["corrections_total"], 1),
        "student_success_rate": stats["student_wins"] / max(stats["episodes"], 1),
        "files": {"corrections": str(out_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
