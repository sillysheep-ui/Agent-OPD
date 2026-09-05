#!/usr/bin/env python3
"""构建 3 个 seed 的训练数据"""
import json
import pandas as pd
 
for seed in [7, 123, 2024]:
    recs = []
    skipped = 0
    for l in open(f'/root/data/alfworld/opd/a1/a3/run1/a1_seed{seed}_corrections.jsonl'):
        try:
            recs.append(json.loads(l))
        except Exception:
            skipped += 1
    print(f'seed{seed}: {len(recs)} 条解析成功, 跳过 {skipped} 坏行')
    valid = [r for r in recs if r['teacher_valid']]
    rows = []
    for r in valid:
        msgs = list(r['query_messages'])
        rows.append({
            'messages': msgs + [{'role': 'assistant', 'content': f"Action: {r['teacher_action']}"}],
            'enable_thinking': False,
            'gamefile': r['gamefile'],
            'task_type': r['task_type'],
            'turn_index': r['turn_index'],
            'teacher_action': r['teacher_action'],
            'student_action': r['student_action'],
            'disagreement': r['disagreement'],
        })
    out = f'/root/data/alfworld/opd/a1/a3/run1/a1_seed{seed}_train.parquet'
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f'seed{seed}: {len(valid)} 条 -> {out}')
