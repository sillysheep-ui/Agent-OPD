#!/usr/bin/env python3
"""M1 probe set 标注: rollout 中排除 A1/A3/A4/A5 已选 turns 后, seed 777 抽 100 states
Teacher 配置: thinking enabled + temp 0（与 A1/A3/A4/A5 采集一致）"""
import json
import random
import sys
from pathlib import Path
from openai import OpenAI
 
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
from agent_harness.parser import parse_and_canonicalize as pac
 
EPISODES = '/root/data/alfworld/opd/a1/a3/run1/episodes.jsonl'
OUT = '/root/data/alfworld/opd/a1/a3/run1/m1_probe_100.jsonl'
 
def load_valid_turns(paths):
    import glob
    keys = set()
    for p in paths:
        for f in glob.glob(p) if any(ch in p for ch in '*?') else [p]:
            for l in open(f):
                try:
                    r = json.loads(l)
                    if any(a['teacher_valid'] for a in r.get('teacher_actions', [r])):
                        keys.add((r['gamefile'], r['turn_index']))
                except Exception:
                    pass
    return keys
 
def main():
    episodes = [json.loads(l) for l in open(EPISODES)]
    # 已选 turns（A1/A3/A4/A5 训练样本）
    used = load_valid_turns([
        '/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl',
        '/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl',
        '/root/data/alfworld/opd/a1/a3/run1/a4_corrections.jsonl',
        '/root/data/alfworld/opd/a1/a3/run1/a5_corrections.jsonl',
    ])
    # 候选: 未被选过的 turns
    cands = [(ep, t) for ep in episodes for t in ep['turns']
             if (ep['gamefile'], t['turn_index']) not in used]
    print(f'candidates: {len(cands)} (excluded {len(used)} used)')
    rng = random.Random(777)
    chosen = rng.sample(cands, 100)
 
    key = Path('/root/.deepseek_key').read_text().strip()
    if '=' in key and not key.startswith('sk-'):
        key = key.split('=', 1)[1]
    teacher = OpenAI(api_key=key, base_url='https://api.deepseek.com')
    extra = {'thinking': {'type': 'enabled', 'effort': 'high'}}
 
    stats = {'valid': 0, 'invalid': 0, 'retries': 0}
    with open(OUT, 'w') as f:
        for ep, t in chosen:
            raw, parsed = '', None
            for attempt in range(3):
                resp = teacher.chat.completions.create(
                    model='deepseek-v4-flash', messages=t['query_messages'],
                    temperature=0.0, max_tokens=3000, extra_body=extra)
                c = resp.choices[0]
                raw = (c.message.content or '') if c else ''
                parsed = pac(raw, t['admissible_actions'], marker_strategy='first')
                if parsed and (parsed.had_action_marker or parsed.valid):
                    break
                stats['retries'] += 1
            valid = bool(parsed and parsed.valid)
            t_action = parsed.canonical_action if valid else 'look'
            stats['valid'] += int(valid)
            stats['invalid'] += int(not valid)
            rec = {
                'gamefile': ep['gamefile'], 'task_type': ep['task_type'],
                'turn_index': t['turn_index'],
                'query_messages': t['query_messages'],
                'admissible_actions': t['admissible_actions'],
                'student_action': t['student_action'],
                'teacher_action': t_action, 'teacher_valid': valid,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
    print(json.dumps({'stats': stats, 'n': 100, 'valid_rate': stats['valid'] / 100}))
 
if __name__ == '__main__':
    main()
