#!/usr/bin/env python3
"""Position Study State Fidelity Gate v2: 完整 prefix 从 episodes.jsonl 提取
engineering validation: 10 个 state——只验证 replay 链路，不看 rescue rate"""
import json
import sys
from pathlib import Path
import yaml
 
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
BASE = '/root/data/alfworld/opd/a1/a3/run1/'
ENV_CONFIG = '/root/data/alfworld/configs/alfworld.yaml'
 
def build_one_game_env(env_cfg, game):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=['gamefile'])
    method = env_cfg['general']['training_method']
    if method == 'dagger':
        max_episode_steps = env_cfg['dagger']['training']['max_nb_steps_per_episode']
    else:
        max_episode_steps = env_cfg['rl']['training']['max_nb_steps_per_episode']
    env_id = textworld.gym.register_games(
        [game], request_infos, batch_size=1, auto_reset=False, asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return textworld.gym.make(env_id)
 
def unwrap_info(info, index=0):
    out = {}
    for key, value in info.items():
        try:
            out[key] = value[index]
        except Exception:
            out[key] = value
    return out
 
# episodes（完整轨迹）
eps_by_game = {}
for l in open(BASE + 'episodes.jsonl'):
    e = json.loads(l)
    eps_by_game[e['gamefile']] = e
 
states = {}
for l in open(BASE + 'sage_to_judge.jsonl'):
    r = json.loads(l)
    states[(r['gamefile'], r['turn_index'])] = r
labels = {}
for l in open(BASE + 'sage_labels.jsonl'):
    r = json.loads(l)
    labels[(r['state_id'][0], r['state_id'][1])] = r['label']
 
d1 = [sid for sid in states if states[sid].get('D') == 1 and labels.get(sid) in ('Skip', 'Weak', 'Strong')]
print(f'D=1 有标签 states: {len(d1)}')
 
env_cfg = yaml.safe_load(Path(ENV_CONFIG).read_text(encoding='utf-8'))
 
import random
rng = random.Random(0)
sample = rng.sample(d1, min(10, len(d1)))
 
n_pass = n_fail = 0
for sid in sample:
    st = states[sid]
    gamefile = st['gamefile']
    turn = st['turn_index']
    snapshot_adm = set(st['admissible_actions'])
    # 完整 prefix 从 episodes 提取（turns[:turn] 的 student_action）
    ep = eps_by_game.get(gamefile)
    if ep is None:
        print(f'{gamefile}/t{turn}: game 不在 episodes ❌')
        n_fail += 1
        continue
    turns = ep['turns']
    if turn > len(turns):
        print(f'{gamefile}/t{turn}: turn 超出轨迹长度({len(turns)}) ❌')
        n_fail += 1
        continue
    prefix = [t['student_action'] for t in turns[:turn]]
    env = build_one_game_env(env_cfg, gamefile)
    try:
        obs, info = env.reset()
        info0 = unwrap_info(info)
        for a in prefix:
            obs, r, d, infos = env.step([a])
            info0 = unwrap_info(infos)
            if d[0]:
                break
        replay_adm = set(info0.get('admissible_commands') or [])
        adm_match = replay_adm == snapshot_adm
        status = '✅PASS' if adm_match else '❌FAIL'
        if adm_match:
            n_pass += 1
        else:
            n_fail += 1
        print(f'{gamefile.split("game_")[-1].split("/")[0]}/t{turn}: prefix={len(prefix)} '
              f'snap={len(snapshot_adm)} replay={len(replay_adm)} {status}')
        if not adm_match:
            print(f'  snap独有: {list(snapshot_adm - replay_adm)[:3]} | replay独有: {list(replay_adm - snapshot_adm)[:3]}')
    except Exception as e:
        print(f'{gamefile}/t{turn}: 异常 {str(e)[:80]}')
        n_fail += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
 
print(f'\nGate 结果: {n_pass}/{n_pass + n_fail} PASS')
