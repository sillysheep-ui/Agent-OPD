#!/usr/bin/env python3
"""A3 合并 shard + disagreement density vs random"""
import json, glob
from collections import Counter
 
recs = []
for f in glob.glob('/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl'):
    for line in open(f):
        if line.strip():
            recs.append(json.loads(line))
print(f'total corrections: {len(recs)}')
 
t_valid = [r for r in recs if r['teacher_valid']]
agree = [r for r in t_valid if r['student_valid'] and not r['disagreement']]
dec_dis = [r for r in t_valid if r['student_valid'] and r['disagreement']]
tech = [r for r in t_valid if not r['student_valid']]
inv = [r for r in recs if not r['teacher_valid']]
print(f'valid: {len(t_valid)}')
print(f'agreement: {len(agree)} ({len(agree)/len(t_valid):.1%})')
print(f'decision-disagreement: {len(dec_dis)} ({len(dec_dis)/len(t_valid):.1%})')
print(f'technical: {len(tech)} ({len(tech)/len(t_valid):.1%})')
print(f'teacher-invalid: {len(inv)}')
 
# 对比 random A1 第一轮
print('\n=== 对比 random A1 第一轮 ===')
print(f'random:  agreement 63.4% | decision-disagreement 20.7% | technical 15.9%')
print(f'selective: agreement {len(agree)/len(t_valid):.1%} | decision-disagreement {len(dec_dis)/len(t_valid):.1%} | technical {len(tech)/len(t_valid):.1%}')
 
# disagreement density 定义对比（用户口径：disagreement 占比）
print(f'\nrandom disagreement density (含technical): {53/150:.1%}')
print(f'selective disagreement density (含technical): {(len(dec_dis)+len(tech))/len(recs):.1%}')
 
# entropy 分布
ents = [r['entropy'] for r in recs]
print(f'\nselected entropy: min {min(ents):.3f} | median {sorted(ents)[len(ents)//2]:.3f} | max {max(ents):.3f}')
 
# 类别分布
types = Counter(r['task_type'] for r in recs)
print(f'\n类别分布: {dict(types)}')
 
# 保存合并文件
with open('/root/data/alfworld/opd/a1/a3/run1/corrections.jsonl', 'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('\nmerged -> run1/corrections.jsonl')
