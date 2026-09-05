#!/usr/bin/env python3
"""build_a3_data.py — A3 corrections -> veRL parquet（有效 138 条）"""
import json
import pandas as pd
 
recs = [json.loads(l) for l in open('/root/data/alfworld/opd/a1/a3/run1/corrections.jsonl')]
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
        'entropy': r.get('entropy', 0.0),
        'teacher_action': r['teacher_action'],
        'student_action': r['student_action'],
        'disagreement': r['disagreement'],
    })
df = pd.DataFrame(rows)
out = '/root/data/alfworld/opd/a1/a3/run1/a3_ce_train.parquet'
df.to_parquet(out, index=False)
print(f'已写: {out} ({len(rows)} 条)')
