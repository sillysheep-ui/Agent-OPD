#!/usr/bin/env python3
"""M2: State Representativeness —— selection 是否造成 state-distribution bias
对比: 完整 Student rollout (1410 turns) vs A1/A3/A4/A5 corrections 的特征分布
"""
import json
import math
from collections import Counter
 
EPISODES = '/root/data/alfworld/opd/a1/a3/run1/episodes.jsonl'
GROUPS = {
    'rollout(1410)': None,
    'A1-random': '/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl',
    'A3-top': '/root/data/alfworld/opd/a1/a3/run1/a3_corrections_merged.jsonl',
    'A4-bounded': '/root/data/alfworld/opd/a1/a3/run1/a4_corrections.jsonl',
    'A5-mixed': '/root/data/alfworld/opd/a1/a3/run1/a5_corrections.jsonl',
}
 
def extract_obs(messages):
    """从 query_messages 提取当前 Observation 文本"""
    for m in reversed(messages):
        if m['role'] == 'user':
            c = m['content']
            if 'Observation:' in c:
                part = c.split('Observation:', 1)[1]
                return part.split('Admissible actions:')[0].strip()
    return ''
 
def features_for_turn(turn, ep_won, prev_obs=None, max_steps=50):
    act = (turn.get('student_action') or 'look')
    act_type = act.split()[0] if act.split() else 'none'
    obs = extract_obs(turn['query_messages'])
    return {
        'turn_depth': min(turn['turn_index'] / max_steps, 1.0),   # 0-1 归一
        'action_type': act_type,
        'n_admissible': min(len(turn['admissible_actions']), 30),  # 封顶 30
        'technical': 1 if (not turn.get('student_valid', True)) else 0,
        'empty_response': 1 if (not str(turn.get('student_raw') or '').strip()) else 0,
        'repeat_obs': 1 if (prev_obs is not None and obs == prev_obs and obs) else 0,
        'lost_episode': 1 if (not ep_won) else 0,
    }
 
def js_div(p, q):
    """JS divergence between two categorical distributions (dicts over same keys)"""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0) + q.get(k, 0)) for k in keys}
    def kl(a, b):
        s = 0.0
        for k in keys:
            pa, qa = a.get(k, 0), b.get(k, 0)
            if pa > 0:
                s += pa * math.log(pa / max(qa, 1e-12))
        return s
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
 
def dist_of(rows, key):
    c = Counter(r[key] for r in rows)
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}
 
# 1) 全量 rollout 特征
rollout_rows = []
eps = [json.loads(l) for l in open(EPISODES)]
for ep in eps:
    prev_obs = None
    for t in ep['turns']:
        rollout_rows.append(features_for_turn(t, ep['won'], prev_obs))
        prev_obs = extract_obs(t['query_messages'])
print(f'rollout: {len(rollout_rows)} turns')
 
# 2) 各组 corrections 特征
group_rows = {}
for name, path in GROUPS.items():
    if path is None:
        continue
    if 'a3_corrections_merged' in path:
        import glob
        files = sorted(glob.glob('/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl'))
        recs = []
        for f in files:
            for l in open(f):
                try: recs.append(json.loads(l))
                except Exception: pass
    else:
        recs = []
        for l in open(path):
            try: recs.append(json.loads(l))
            except Exception: pass
    # 用 correction 的 turn 定位 rollout 的对应 turn（同 gamefile + turn_index）
    rows = []
    # 预计算每 turn 的 prev_obs
    prev_map = {}
    for ep in eps:
        prev = None
        for t in ep['turns']:
            prev_map[(ep['gamefile'], t['turn_index'])] = prev
            prev = extract_obs(t['query_messages'])
    ep_map = {(ep['gamefile'], t['turn_index']): (ep, t) for ep in eps for t in ep['turns']}
    for r in recs:
        hit = ep_map.get((r['gamefile'], r['turn_index']))
        if hit:
            ep, t = hit
            rows.append(features_for_turn(t, ep['won'], prev_map.get((r['gamefile'], r['turn_index']))))
    group_rows[name] = rows
    print(f'{name}: {len(rows)} turns')
 
# 3) 每特征 JS divergence（各组 vs rollout）
FEATURES = ['turn_depth', 'action_type', 'n_admissible', 'technical', 'empty_response', 'repeat_obs', 'lost_episode']
print(f'\n=== JS divergence vs rollout（越大 = 分布偏离越严重）===')
print(f'{"feature":<16}' + ''.join(f'{g:>12}' for g in group_rows))
for f in FEATURES:
    p = dist_of(rollout_rows, f)
    line = f'{f:<16}'
    for name, rows in group_rows.items():
        q = dist_of(rows, f)
        line += f'{js_div(p, q):>12.4f}'
    print(line)
 
# 4) 关键特征直方图（technical / empty / repeat_obs / lost）
print(f'\n=== 关键比例对比 ===')
for f in ['technical', 'empty_response', 'repeat_obs', 'lost_episode']:
    p = dist_of(rollout_rows, f)
    line = f'{f:<16}'
    for name, rows in group_rows.items():
        q = dist_of(rows, f)
        line += f'{(q.get(1, 0)*100):>11.1f}%'
    print(f'{line}   (rollout: {p.get(1,0)*100:.1f}%)')
 
print(f'\n=== turn_depth 均值 ===')
p = sum(r['turn_depth'] for r in rollout_rows) / len(rollout_rows)
line = f'{"turn_depth":<16}'
for name, rows in group_rows.items():
    m = sum(r['turn_depth'] for r in rows) / len(rows)
    line += f'{m:>12.3f}'
print(f'{line}   (rollout: {p:.3f})')
