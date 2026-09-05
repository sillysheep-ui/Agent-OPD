#!/usr/bin/env python3
"""build_a1_data.py — A1 corrections -> veRL parquet + matched offline control data.
 
A1: (P_S, z_t^S) -> Action: a_t^T   (valid corrections only, from a1/run1)
Control: same count of (P_S, z_t^T) -> Action: a_t^T sampled from SFT v6 train data
"""
import json
import random
from pathlib import Path
 
import pandas as pd
 
STUDENT_SYSTEM_PROMPT = """You are an ALFWorld household agent.
 
Complete the given household task by interacting with the environment one step at a time.
 
At each turn, use the task description, interaction history, current observation, and current admissible actions to choose the next action.
 
Return exactly one action from the current admissible actions in this format:
Action: <command>
 
Use the action exactly as written in the admissible action list. Do not change object names, object numbers, or command syntax.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
corr_path = '/root/data/alfworld/opd/a1/run1/corrections.jsonl'
sft_train = '/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_train.parquet'
out_prefix = '/root/data/alfworld/opd/a1/run1/a1_ce'
 
recs = [json.loads(l) for l in open(corr_path)]
valid = [r for r in recs if r['teacher_valid']]
print(f'corrections: {len(recs)}, valid: {len(valid)}')
 
# A1 samples: query_messages already contain (P_S, z_t^S); append teacher target
a1_rows = []
for r in valid:
    msgs = list(r['query_messages'])
    # ensure system is P_S (query_messages came from student context which used P_S)
    a1_rows.append({
        'messages': msgs + [{'role': 'assistant', 'content': f"Action: {r['teacher_action']}"}],
        'enable_thinking': False,
        'gamefile': r['gamefile'],
        'task_type': r['task_type'],
        'turn_index': r['turn_index'],
        'teacher_action': r['teacher_action'],
        'student_action': r['student_action'],
        'disagreement': r['disagreement'],
    })
 
# Control: matched count sampled from SFT v6 train (Teacher-state data)
df = pd.read_parquet(sft_train)
rng = random.Random(42)
n = len(a1_rows)
ctrl_idx = rng.sample(range(len(df)), n)
ctrl_rows = []
for i in ctrl_idx:
    row = df.iloc[i]
    ctrl_rows.append({
        'messages': row['messages'],
        'enable_thinking': False,
        'gamefile': row['gamefile'],
        'task_type': row['task_type'],
        'turn_index': row['turn_index'],
        'teacher_action': row['teacher_action'],
        'student_action': None,
        'disagreement': None,
    })
 
print(f'A1 rows: {len(a1_rows)}, Control rows: {len(ctrl_rows)}')
 
train_path = Path(out_prefix + '_train.parquet')
ctrl_path = Path(out_prefix + '_control_train.parquet')
pd.DataFrame(a1_rows).to_parquet(train_path, index=False)
pd.DataFrame(ctrl_rows).to_parquet(ctrl_path, index=False)
 
manifest = {
    'protocol': 'A1-v6',
    'a1_samples': len(a1_rows),
    'a1_valid_rate': len(valid) / len(recs),
    'disagreement_rate': sum(1 for r in valid if r['disagreement']) / len(valid),
    'control_samples': len(ctrl_rows),
    'control_source': sft_train,
    'control_sampling_seed': 42,
    'training_target': 'Action:<cmd>',
    'files': {'a1_train': str(train_path), 'control_train': str(ctrl_path)},
}
Path(out_prefix + '_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
