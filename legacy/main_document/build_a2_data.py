#!/usr/bin/env python3
"""A2 训练数据 v2: 加 state_weight 列（1/K_s）——per-state normalization"""
import json
import pandas as pd
 
def build(n, seed, state_intersection=None):
    recs = []
    skipped = 0
    for l in open(f'/root/data/alfworld/opd/a1/a3/run1/a2_n{n}_seed{seed}_corrections.jsonl'):
        try:
            recs.append(json.loads(l))
        except Exception:
            skipped += 1
    rows = []
    n_states = 0
    for r in recs:
        key = (r['gamefile'], r['turn_index'])
        if state_intersection is not None and key not in state_intersection:
            continue                      # coverage 对齐: 只保留交集 state
        msgs = list(r['query_messages'])
        valid_actions = [a for a in r['teacher_actions'] if a['teacher_valid']]
        if not valid_actions:
            continue
        n_states += 1
        K_s = len(valid_actions)          # state 级 valid Teacher samples（交集内）
        for a in valid_actions:
            rows.append({
                'messages': msgs + [{'role': 'assistant', 'content': f"Action: {a['teacher_action']}"}],
                'enable_thinking': False,
                'gamefile': r['gamefile'],
                'task_type': r['task_type'],
                'turn_index': r['turn_index'],
                'teacher_action': a['teacher_action'],
                'student_action': r['student_action'],
                'disagreement': a['disagreement'],
                'state_weight': 1.0 / K_s,   # per-state normalization: 1/K_s
            })
    out = f'/root/data/alfworld/opd/a1/a3/run1/a2_v2_n{n}_seed{seed}_train.parquet'
    pd.DataFrame(rows).to_parquet(out, index=False)
    w = pd.DataFrame(rows)['state_weight']
    print(f'n{n}-seed{seed}: {n_states} states, {len(rows)} samples, '
          f'weight: min={w.min():.4f} max={w.max():.4f} mean={w.mean():.4f}')
 
# 交集对齐: 先算 N1/N3 的 valid state 集合
for seed in [42, 7, 123, 2024]:
    s1, s3 = set(), set()
    for l in open(f'/root/data/alfworld/opd/a1/a3/run1/a2_n1_seed{seed}_corrections.jsonl'):
        r = json.loads(l)
        if any(a['teacher_valid'] for a in r['teacher_actions']):
            s1.add((r['gamefile'], r['turn_index']))
    for l in open(f'/root/data/alfworld/opd/a1/a3/run1/a2_n3_seed{seed}_corrections.jsonl'):
        r = json.loads(l)
        if any(a['teacher_valid'] for a in r['teacher_actions']):
            s3.add((r['gamefile'], r['turn_index']))
    inter = s1 & s3
    print(f'seed{seed}: 交集 = {len(inter)} states (N1-only={len(s1-inter)}, N3-only={len(s3-inter)})')
    build(1, seed, inter)
    build(3, seed, inter)
 
# 验证: N1/N3 state 集合完全一致
print('\n=== coverage 验证 ===')
for seed in [42, 7, 123, 2024]:
    d1 = pd.read_parquet(f'/root/data/alfworld/opd/a1/a3/run1/a2_v2_n1_seed{seed}_train.parquet')
    d3 = pd.read_parquet(f'/root/data/alfworld/opd/a1/a3/run1/a2_v2_n3_seed{seed}_train.parquet')
    k1 = set(zip(d1['gamefile'], d1['turn_index']))
    k3 = set(zip(d3['gamefile'], d3['turn_index']))
    print(f'seed{seed}: N1={len(k1)} states, N3={len(k3)} states, 一致={k1==k3}')
    # state 级权重和 = 1.0
    g = d3.groupby(['gamefile', 'turn_index'])['state_weight'].sum()
    print(f'  N3 Σweight/state: min={g.min():.6f} max={g.max():.6f} (应=1.0)')
