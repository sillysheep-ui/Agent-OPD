#!/usr/bin/env python3
"""CTRL multi-seed: Teacher-state OPD 采集（matched offline top-up）
Teacher 在 A1 同 50 games 上 rollout（temp 0.5——4 seed 不同轨迹）
→ 每 game 抽 3 turns（rng ξ——同 A1R M=3）→ correction（thinking enabled + temp 0——同 A1R）
唯一变量: 状态来源（Teacher-state vs Student-state）
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
 
import yaml
from openai import OpenAI
from transformers import AutoTokenizer
 
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
from agent_harness.parser import parse_and_canonicalize as pac
from agent_harness.context import TaskPreservingTruncator
 
AGENT_SYSTEM_PROMPT = """You are an ALFWorld household agent. Solve the current task one environment turn at a time.
Reply in this format:
Thought: <brief reasoning>
Action: <one executable ALFWorld command>
Use the command grammar shown by the environment (for example go to, take, move, open, close, use, clean, heat, cool).
To place objects use "move X to Y" (NOT "put X in/on Y"). Object and receptacle names include their numbers, e.g. "fridge 1", "drawer 2".
Do not simulate future observations or future turns."""
 
CORR_SYSTEM_PROMPT = """You are an expert ALFWorld household agent and teacher.
 
Given the task, interaction history, and current state, provide the correct next action.
 
Base your decision only on objects, receptacles, locations, and state information supported by the task and the actual interaction history.
 
Choose exactly one action from the current admissible actions.
 
Return exactly:
Action: <command>
 
Use the action exactly as written in the admissible action list.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
ENV_CONFIG = '/root/data/alfworld/configs/alfworld.yaml'
EPISODES = '/root/data/alfworld/opd/a1/a3/run1/episodes.jsonl'
MAX_STEPS = 50
CONTEXT_MAX = 4096
CONTEXT_RESERVE = 256
TOKENIZER_PATH = '/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft'
 
 
def build_one_game_env(env_cfg, game):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=['gamefile'])
    method = env_cfg['general']['training_method']
    if method == 'dagger':
        max_episode_steps = env_cfg['dagger']['training']['max_nb_steps_per_episode']
    elif method == 'dqn':
        max_episode_steps = env_cfg['rl']['training']['max_nb_steps_per_episode']
    else:
        raise ValueError(f'Unsupported training_method={method!r}')
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
 
 
def task_description(initial_observation):
    parts = str(initial_observation).split('\n\n')
    return '\n'.join(parts[1:]).strip() if len(parts) > 1 else str(initial_observation).strip()
 
 
def main():
    seed = int(sys.argv[1])
    OUT = f'/root/data/alfworld/opd/a1/a3/run1/ctrl_seed{seed}_corrections.jsonl'
    rng = random.Random(seed)
    env_cfg = yaml.safe_load(Path(ENV_CONFIG).read_text(encoding='utf-8'))
 
    key = Path('/root/.deepseek_key').read_text().strip()
    if '=' in key and not key.startswith('sk-'):
        key = key.split('=', 1)[1]
    teacher = OpenAI(api_key=key, base_url='https://api.deepseek.com')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    truncator = TaskPreservingTruncator(tokenizer, max_context_tokens=CONTEXT_MAX,
                                        reserve_tokens=CONTEXT_RESERVE, enable_thinking=False)
 
    games = [(ep['gamefile'], ep['task_type']) for ep in [json.loads(l) for l in open(EPISODES)]]
    print(f'{seed}: {len(games)} games（全量单进程）', flush=True)
 
    # 1) Teacher rollout（temp 0.5——探索）
    trajectories = []   # 每 game: {gamefile, task_type, turns: [...]}
    for gi, (gamefile, task_type) in enumerate(games):
        env = build_one_game_env(env_cfg, gamefile)
        info0 = None
        try:
            obs, info = env.reset()
            info0 = unwrap_info(info)
            desc = task_description(str(obs[0]))
            full_history = [
                {'role': 'system', 'content': AGENT_SYSTEM_PROMPT},
                {'role': 'user', 'content': f'Now solve this new task.\n{desc}'},
            ]
            cur_obs = desc
            done = False
            turns = []
            for step in range(MAX_STEPS):
                admissible = list(info0.get('admissible_commands') or [])
                obs_content = f'Observation: {cur_obs.strip()}'
                query, _ = truncator.truncate(full_history)
                if step > 0:
                    query.append({'role': 'user', 'content': obs_content})
                resp = teacher.chat.completions.create(
                    model='deepseek-v4-flash', messages=query,
                    temperature=0.5, max_tokens=3000)
                content = (resp.choices[0].message.content or '') if resp.choices else ''
                parsed = pac(content, admissible, marker_strategy='first')
                action = parsed.canonical_action if parsed.valid else 'look'
                turns.append({
                    'query_messages': query,      # 截断后 + 当前 obs（与 Student episodes 同构）
                    'admissible_actions': admissible,
                    'agent_action': action,
                })
                full_history.append({'role': 'assistant', 'content': f'Action: {action}'})
                obs, rewards, dones, infos = env.step([action])
                cur_obs = str(obs[0])
                info0 = unwrap_info(infos)
                done = bool(dones[0])
                if done:
                    break
                full_history.append({'role': 'user', 'content': obs_content})
        except Exception as e:
            print(f'  {gamefile}: rollout FAIL {e}', flush=True)
            turns = []
        finally:
            try:
                env.close()
            except Exception:
                pass
        trajectories.append({'gamefile': gamefile, 'task_type': task_type, 'turns': turns,
                             'won': bool(info0 and info0.get('won', False))})
        if (gi + 1) % 5 == 0:
            print(f'  [{gi+1}/{len(games)}] rollouts', flush=True)
 
    # 2) 抽状态 + correction
    n_sel = n_valid = 0
    with open(OUT, 'w') as f:
        for ep in trajectories:
            turns = ep['turns']
            if not turns:
                continue
            sel = rng.sample(range(len(turns)), min(3, len(turns)))
            for turn_idx in sel:
                t = turns[turn_idx]
                n_sel += 1
                # correction（thinking enabled + temp 0——与 A1R 相同）
                raw = ''
                valid = False
                action = 'look'
                for attempt in range(3):
                    resp = teacher.chat.completions.create(
                        model='deepseek-v4-flash', messages=t['query_messages'],
                        temperature=0.0, max_tokens=3000)
                    c = resp.choices[0]
                    raw = (c.message.content or '') if c else ''
                    parsed = pac(raw, t['admissible_actions'], marker_strategy='first')
                    valid = bool(parsed and parsed.valid)
                    action = parsed.canonical_action if valid else 'look'
                    if valid or raw.strip():
                        break
                n_valid += int(valid)
                rec = {
                    'gamefile': ep['gamefile'], 'task_type': ep['task_type'],
                    'turn_index': turn_idx,
                    'query_messages': t['query_messages'],
                    'admissible_actions': t['admissible_actions'],
                    'student_action': t['agent_action'],   # Teacher 轨迹动作（占位）
                    'student_valid': True,
                    'teacher_action': action, 'teacher_raw': raw,
                    'teacher_valid': valid,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'{seed}: selected={n_sel}, valid={n_valid}', flush=True)
 
 
if __name__ == '__main__':
    main()
