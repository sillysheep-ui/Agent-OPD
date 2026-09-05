#!/usr/bin/env python3
"""CTRL invalid correction 重调: user 消息附加 admissible（与 A1R 上下文对齐）"""
import json
import sys
from pathlib import Path
from openai import OpenAI
 
seed = int(sys.argv[1])
path = f'/root/data/alfworld/opd/a1/a3/run1/ctrl_seed{seed}_corrections.jsonl'
rows = [json.loads(l) for l in open(path)]
 
key = Path('/root/.deepseek_key').read_text().strip()
if '=' in key and not key.startswith('sk-'):
    key = key.split('=', 1)[1]
client = OpenAI(api_key=key, base_url='https://api.deepseek.com')
 
CORR_SYSTEM_PROMPT = """You are an expert ALFWorld household agent and teacher.
 
Given the task, interaction history, and current state, provide the correct next action.
 
Base your decision only on objects, receptacles, locations, and state information supported by the task and the actual interaction history.
 
Choose exactly one action from the current admissible actions.
 
Return exactly:
Action: <command>
 
Use the action exactly as written in the admissible action list.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
from agent_harness.parser import parse_and_canonicalize as pac
 
n_fixed = 0
for r in rows:
    if r.get('teacher_valid'):
        continue
    msgs = [dict(m) for m in r['query_messages']]
    # 附加 admissible 到最后 user 消息
    adm_text = '\n'.join(f'- {a}' for a in r['admissible_actions'])
    last_user_idx = max(i for i, m in enumerate(msgs) if m['role'] == 'user')
    msgs[last_user_idx] = dict(msgs[last_user_idx])
    msgs[last_user_idx]['content'] += f'\n\nAdmissible actions:\n{adm_text}'
    raw = ''
    valid = False
    action = 'look'
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model='deepseek-v4-flash',
                messages=[{'role': 'system', 'content': CORR_SYSTEM_PROMPT}] + msgs,
                temperature=0.0, max_tokens=3000)
            c = resp.choices[0]
            raw = (c.message.content or '') if c else ''
        except Exception as e:
            print(f'ERR: {e}')
            continue
        parsed = pac(raw, r['admissible_actions'], marker_strategy='first')
        valid = bool(parsed and parsed.valid)
        action = parsed.canonical_action if valid else 'look'
        if valid:
            break
    r['teacher_action'] = action
    r['teacher_raw'] = raw
    r['teacher_valid'] = valid
    if valid:
        n_fixed += 1
 
with open(path, 'w') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
n_valid = sum(1 for r in rows if r['teacher_valid'])
print(f'seed{seed}: fixed {n_fixed}, total valid {n_valid}/{len(rows)}')
