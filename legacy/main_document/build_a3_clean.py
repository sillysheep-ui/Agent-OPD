#!/usr/bin/env python3
"""A3-clean: A3 138 条剔除 technical(空响应) -> 94 条纯决策样本训练集"""
import json
import pandas as pd
 
recs = [json.loads(l) for l in open('/root/data/alfworld/opd/a1/a3/run1/corrections.jsonl')]
valid = [r for r in recs if r['teacher_valid']]
tech = [r for r in valid if not r.get('student_valid', True)]
clean = [r for r in valid if r.get('student_valid', True)]
print(f'valid: {len(valid)}, technical: {len(tech)}, clean: {len(clean)}')
 
rows = []
for r in clean:
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
out = '/root/data/alfworld/opd/a1/a3/run1/a3_clean_train.parquet'
pd.DataFrame(rows).to_parquet(out, index=False)
print(f'已写: {out} ({len(rows)} 条)')
# 看 clean 集里 disagreement 占比
dis = sum(1 for r in clean if r['disagreement'])
print(f'clean 集 disagreement: {dis}/{len(clean)} = {dis/len(clean):.1%}')
