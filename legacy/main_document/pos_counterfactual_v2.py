#!/usr/bin/env python3
"""Position Study counterfactual rollout v2 —— 严格复用原 a1_collect 代码路径
核心原则: 唯一改变的变量 = 第 t 步执行 a_S 还是 a_T（其余全部冻结）
engineering validation 版（3 个 state——不看 rescue rate）"""
import json
import sys
import copy
from pathlib import Path
import yaml
from openai import OpenAI
from transformers import AutoTokenizer
 
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
# 复用原 collection 的代码路径（不是"看起来一样"——是同一函数）
from a1_collect import (STUDENT_SYSTEM_PROMPT, build_one_game_env, unwrap_info,
                        task_description, turn_user_message)
from agent_harness.parser import parse_and_canonicalize
from agent_harness.context import TaskPreservingTruncator
 
BASE = '/root/data/alfworld/opd/a1/a3/run1/'
ENV_CONFIG = '/root/data/alfworld/configs/alfworld.yaml'
TOKENIZER_PATH = '/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft'
VLLM_URL = 'http://127.0.0.1:8001/v1'
MODEL = 'sft'
# 冻结的 generation 参数（与 a1_collect parse_args 默认值一致）
STUDENT_TEMPERATURE = 0.0
STUDENT_MAX_TOKENS = 256
MAX_STEPS = 50
CONTEXT_MAX = 4096
CONTEXT_RESERVE = 256
 
def norm_text(x):
    return ' '.join(str(x).split())
 
eps_by_game = {json.loads(l)['gamefile']: json.loads(l) for l in open(BASE + 'episodes.jsonl')}
states = {}
for l in open(BASE + 'sage_to_judge.jsonl'):
    r = json.loads(l)
    states[(r['gamefile'], r['turn_index'])] = r
labels = {}
for l in open(BASE + 'sage_labels.jsonl'):
    r = json.loads(l)
    labels[(r['state_id'][0], r['state_id'][1])] = r['label']
 
d1 = [sid for sid in states if states[sid].get('D') == 1 and labels.get(sid) in ('Skip', 'Weak', 'Strong')]
 
env_cfg = yaml.safe_load(Path(ENV_CONFIG).read_text(encoding='utf-8'))
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
truncator = TaskPreservingTruncator(tok, max_context_tokens=CONTEXT_MAX,
                                    reserve_tokens=CONTEXT_RESERVE, enable_thinking=False)
client = OpenAI(api_key='EMPTY', base_url=VLLM_URL)
 
def extract_obs_from_messages(msgs):
    """从 query_messages 最后 user 消息提取 Observation 文本（规范化）"""
    for m in reversed(msgs):
        if m['role'] == 'user' and 'Observation:' in m['content']:
            c = m['content']
            obs = c.split('Observation:', 1)[1].split('Admissible actions:')[0]
            return norm_text(obs)
    return None
 
import random
rng = random.Random(0)
sample = rng.sample(d1, min(3, len(d1)))
 
for sid in sample:
    st = states[sid]
    gamefile = st['gamefile']
    turn = st['turn_index']
    snapshot_adm = list(st['admissible_actions'])   # 顺序敏感
    aS = st['student_action']
    # a_T 冻结规则: 第一个 valid（在 admissible）且 != aS 的 teacher_action（与原 D 判定一致）
    aT = next((str(t) for t in st.get('teacher_actions', [])
               if str(t) in snapshot_adm and str(t) != aS), None)
    ep = eps_by_game[gamefile]
    # prefix = 已 executed actions（student_action 就是 executed——含 fallback look）
    prefix = [t['student_action'] for t in ep['turns'][:turn]]
 
    env = build_one_game_env(env_cfg, gamefile)
    try:
        obs, info = env.reset()
        info0 = unwrap_info(info)
        # ---- State Fidelity Gate: 完整 replay + obs/adm 双匹配 ----
        for a in prefix:
            obs, r, d, infos = env.step([a])
            info0 = unwrap_info(infos)
            if d[0]:
                break
        replay_obs = norm_text(str(obs[0]))
        replay_adm = list(info0.get('admissible_commands') or [])
        snapshot_obs = extract_obs_from_messages(st['query_messages'])
        obs_match = (replay_obs == snapshot_obs)
        adm_match = (replay_adm == snapshot_adm)
        gate_ok = obs_match and adm_match
        if not gate_ok:
            print(f'{gamefile.split("game_")[-1].split("/")[0]}/t{turn}: gate FAIL '
                  f'(obs_match={obs_match} adm_match={adm_match})——跳过')
            continue
        assert len(prefix) == turn, f'prefix 长度 {len(prefix)} != turn {turn}（index 错位）'
 
        # ---- Counterfactual: 干预前 context = st['query_messages']（deepcopy，不重建）----
        hist = copy.deepcopy(st['query_messages'])
        assert hist[-1]['role'] == 'user', '干预前 context 最后一条应为 user（当前 obs）'
        assert aT in snapshot_adm and aT != aS, 'a_T 冻结规则违反'
        hist.append({'role': 'assistant', 'content': f'Action: {aT}'})
        obs, r, d, infos = env.step([aT])
        info0 = unwrap_info(infos)
        cur_obs = str(obs[0])
        won = bool(info0.get('won', False))
        done = bool(d[0])
        done_after_intervention = done
 
        # ---- 交还 SFT Student（严格复现 harness 语义）----
        remaining = MAX_STEPS - len(prefix) - 1
        steps_after = 0
        while not done and steps_after < remaining:
            admissible = list(info0.get('admissible_commands') or [])
            obs_content = turn_user_message(cur_obs, admissible)   # 复用原 formatter（含 admissible）
            hist.append({'role': 'user', 'content': obs_content})
            q, _ = truncator.truncate(hist)
            resp = client.chat.completions.create(
                model=MODEL, messages=q,
                temperature=STUDENT_TEMPERATURE, max_tokens=STUDENT_MAX_TOKENS)
            content = (resp.choices[0].message.content or '') if resp.choices else ''
            parsed = parse_and_canonicalize(content, admissible, marker_strategy='first')
            action = parsed.canonical_action if parsed.valid else 'look'
            hist.append({'role': 'assistant', 'content': f'Action: {action}'})
            obs, r, d, infos = env.step([action])
            info0 = unwrap_info(infos)
            cur_obs = str(obs[0])
            won = bool(info0.get('won', False))
            done = bool(d[0])
            steps_after += 1
            if done:
                break
 
        assert len(prefix) + 1 + steps_after <= MAX_STEPS, 'horizon 超限'
        YT = 1 if won else 0
        YS = 1 if ep['won'] else 0
        C = YT - YS
        # 完整记录落盘（含 C——validation 阶段不汇总不看）
        rec = {
            'state_id': f'{gamefile.split("game_")[-1].split("/")[0]}/t{turn}',
            'gate_pass': True,
            'aS': aS, 'aT': aT,
            'Y_S': YS, 'Y_T': YT, 'C': C,
            'steps_after': steps_after,
            'remaining': remaining,
        }
        with open('/root/data/alfworld/opd/final_test/pos_validation.jsonl', 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # console 不打印 C_t（预注册: 不看 rescue rate）
        print(json.dumps({
            'game_turn': rec['state_id'],
            'state_gate': {'obs_match': obs_match, 'adm_match': adm_match},
            'prefix_len': len(prefix),
            'remaining_before_intervention': remaining,
            'intervention': {'aS': aS, 'aT': aT, 'done_immediately': done_after_intervention},
            'continuation': {'steps_after': steps_after,
                             'budget_left': remaining - steps_after,
                             'final_won': won},
        }, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({'game_turn': gamefile.split('game_')[-1].split('/')[0] + f'/t{turn}',
                          'error': str(e)[:100]}, ensure_ascii=False))
    finally:
        try:
            env.close()
        except Exception:
            pass
