#!/usr/bin/env python3
"""构建 A1-disagree-only: 30 条 student_valid & disagreement（纯决策分歧）"""
import json
import pandas as pd
 
corr = '/root/data/alfworld/opd/a1/run1/corrections.jsonl'
out = '/root/data/alfworld/opd/a1/run1/a1_disagree_train.parquet'
 
recs = [json.loads(l) for l in open(corr)]
valid = [r for r in recs if r['teacher_valid']]
dis = [r for r in valid if r['disagreement'] and r['student_valid']]
print(f'纯分歧样本（student valid + disagreement）: {len(dis)}')
 
rows = []
for r in dis:
    msgs = list(r['query_messages'])
    rows.append({
        'messages': msgs + [{'role': 'assistant', 'content': f"Action: {r['teacher_action']}"}],
        'enable_thinking': False,
        'gamefile': r['gamefile'],
        'task_type': r['task_type'],
        'turn_index': r['turn_index'],
        'teacher_action': r['teacher_action'],
        'student_action': r['student_action'],
        'disagreement': True,
    })
pd.DataFrame(rows).to_parquet(out, index=False)
print(f'已写: {out} ({len(rows)} 条)')
from collections import Counter
print('类型分布:', dict(Counter(r['task_type'] for r in dis)))
