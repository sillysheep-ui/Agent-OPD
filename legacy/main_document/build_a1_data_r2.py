#!/usr/bin/env python3
"""build_a1_data_r2.py — 第二轮 corrections -> veRL parquet"""
import json
import pandas as pd
 
corr_path = '/root/data/alfworld/opd/a1/run2/corrections.jsonl'
out = '/root/data/alfworld/opd/a1/run2/a1_ce_r2_train.parquet'
 
recs = [json.loads(l) for l in open(corr_path)]
valid = [r for r in recs if r['teacher_valid']]
print(f'corrections: {len(recs)}, valid: {len(valid)}')
 
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
pd.DataFrame(rows).to_parquet(out, index=False)
print(f'已写: {out} ({len(rows)} 条)')
