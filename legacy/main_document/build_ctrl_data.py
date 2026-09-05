#!/usr/bin/env python3
"""CTRL 训练数据构建（与 A1 格式同构: messages/enable_thinking/gamefile/...）"""
import json
import pandas as pd
 
for seed in [42, 7, 123, 2024]:
    rows = []
    for l in open(f'/root/data/alfworld/opd/a1/a3/run1/ctrl_seed{seed}_corrections.jsonl'):
        r = json.loads(l)
        if not r.get('teacher_valid'):
            continue
        # target 内嵌（与 A1 格式同构: messages 最后一条 = assistant "Action: X"）
        msgs = list(r['query_messages'])
        msgs.append({'role': 'assistant', 'content': f"Action: {r['teacher_action']}"})
        rows.append({
            'messages': msgs,
            'enable_thinking': False,
            'gamefile': r['gamefile'],
            'task_type': r['task_type'],
            'turn_index': r['turn_index'],
            'teacher_action': r['teacher_action'],
            'student_action': r.get('student_action'),
        })
    df = pd.DataFrame(rows)
    out = f'/root/data/alfworld/opd/a1/a3/run1/ctrl_seed{seed}_train.parquet'
    df.to_parquet(out, index=False)
    print(f'seed{seed}: {len(df)} samples -> {out}')
