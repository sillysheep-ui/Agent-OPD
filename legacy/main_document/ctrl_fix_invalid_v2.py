#!/usr/bin/env python3
"""CTRL invalid correction 重调 v2: 附加 admissible + timeout + 即时落盘 + 日志"""
import json
import sys
import time
from pathlib import Path
from openai import OpenAI
 
seed = int(sys.argv[1])
path = f'/root/data/alfworld/opd/a1/a3/run1/ctrl_seed{seed}_corrections.jsonl'
rows = [json.loads(l) for l in open(path)]
 
key = Path('/root/.deepseek_key').read_text().strip()
if '=' in key and not key.startswith('sk-'):
    key = key.split('=', 1)[1]
client = OpenAI(api_key=key, base_url='https://api.deepseek.com', timeout=60)
 
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
 
def fix_one(r):
    msgs = [dict(m) for m in r['query_messages']]
    adm_text = '\n'.join(f'- {a}' for a in r['admissible_actions'])
    last_user_idx = max(i for i, m in enumerate(msgs) if m['role'] == 'user')
    msgs[last_user_idx] = dict(msgs[last_user_idx])
    msgs[last_user_idx]['content'] += f'\n\nAdmissible actions:\n{adm_text}'
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model='deepseek-v4-flash',
                messages=[{'role': 'system', 'content': CORR_SYSTEM_PROMPT}] + msgs,
                temperature=0.0, max_tokens=3000)
            c = resp.choices[0]
            raw = (c.message.content or '') if c else ''
        except Exception as e:
            print(f'  ERR: {str(e)[:80]}', flush=True)
            time.sleep(3 * (attempt + 1))
            continue
        parsed = pac(raw, r['admissible_actions'], marker_strategy='first')
        if parsed and parsed.valid:
            r['teacher_action'] = parsed.canonical_action
            r['teacher_raw'] = raw
            r['teacher_valid'] = True
            return True
    r['teacher_action'] = 'look'
    r['teacher_valid'] = False
    return False
 
n_fixed = 0
for i, r in enumerate(rows):
    if r.get('teacher_valid'):
        continue
    ok = fix_one(r)
    if ok:
        n_fixed += 1
    # 即时落盘（每条）
    with open(path, 'w') as f:
        for rr in rows:
            f.write(json.dumps(rr, ensure_ascii=False) + '\n')
    if (i + 1) % 10 == 0:
        print(f'  [{i+1}/{len(rows)}] fixed so far: {n_fixed}', flush=True)
 
n_valid = sum(1 for r in rows if r['teacher_valid'])
print(f'seed{seed}: fixed {n_fixed}, total valid {n_valid}/{len(rows)}', flush=True)
