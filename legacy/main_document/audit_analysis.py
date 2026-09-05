#!/usr/bin/env python3
"""SAGE 人工 audit 分析: 人工 vs SAGE 一致性 + 关键条件概率"""
import json
from collections import Counter, defaultdict
 
BASE = '/root/data/alfworld/opd/a1/a3/run1/'
 
# SAGE 数据
states = {}
for l in open(BASE + 'sage_to_judge.jsonl'):
    r = json.loads(l)
    states[(r['gamefile'].split('game_')[-1].split('/')[0], r['turn_index'])] = r
labels = {}
for l in open(BASE + 'sage_labels.jsonl'):
    r = json.loads(l)
    labels[(str(r['state_id'][0]).split('game_')[-1].split('/')[0], r['state_id'][1])] = r['label']
 
# 人工结果
human = [json.loads(l) for l in open(BASE + 'sage_audit_human_results.jsonl')]
 
# join
joined = []
for h in human:
    g, t = h['game_turn'].split('/')
    t = int(t)
    sid = (g, t)
    sage = labels.get(sid, 'MISSING')
    st = states.get(sid)
    D = st.get('D') if st else None
    joined.append({**h, 'sage_label': sage, 'D': D})
 
print('=== 人工 vs SAGE intervention（confusion）===')
cm = Counter((h['sage_label'], h['human_intervention']) for h in joined)
hdr = 'SAGE\\人工'  # noqa
print(f'{hdr:<10}{"Skip":>8}{"Weak":>8}{"Strong":>8}{"总":>6}')
for sage in ['Skip', 'Weak', 'Strong', 'MISSING']:
    row = [cm.get((sage, hh), 0) for hh in ['Skip', 'Weak', 'Strong']]
    print(f'{sage:<10}{row[0]:>8}{row[1]:>8}{row[2]:>8}{sum(row):>6}')
 
# 一致性率（精确匹配）
agree = sum(1 for h in joined if h['sage_label'] == h['human_intervention'])
print(f'\n精确一致率: {agree}/{len(joined)} = {agree/len(joined)*100:.1f}%')
 
# 关键: 人工对 SAGE 判定的确认（降级/升级分析）
print('\n=== SAGE Skip 的人工判定 ===')
sub = [h for h in joined if h['sage_label'] == 'Skip']
print(f'SAGE Skip (n={len(sub)}): 人工 Skip {sum(1 for h in sub if h["human_intervention"]=="Skip")} / '
      f'Weak {sum(1 for h in sub if h["human_intervention"]=="Weak")} / '
      f'Strong {sum(1 for h in sub if h["human_intervention"]=="Strong")}')
 
print('\n=== SAGE Strong 的人工判定 ===')
sub = [h for h in joined if h['sage_label'] == 'Strong']
print(f'SAGE Strong (n={len(sub)}): 人工 Strong {sum(1 for h in sub if h["human_intervention"]=="Strong")} / '
      f'Skip/Weak {sum(1 for h in sub if h["human_intervention"]!="Strong")}')
 
# 核心: P(Student acceptable | D=1, SAGE=Skip)
print('\n=== 核心验证 ===')
sub = [h for h in joined if h['D'] == 1 and h['sage_label'] == 'Skip']
if sub:
    acc = sum(1 for h in sub if h['human_acceptable'] == 'Yes')
    print(f'P(Student acceptable | D=1, SAGE Skip) = {acc}/{len(sub)} = {acc/len(sub)*100:.1f}%')
 
# P(Teacher better | D=1)
sub = [h for h in joined if h['D'] == 1]
if sub:
    better = sum(1 for h in sub if h['human_teacher_vs'] == 'Better')
    worse = sum(1 for h in sub if h['human_teacher_vs'] == 'Worse')
    equiv = sum(1 for h in sub if h['human_teacher_vs'] == 'Equivalent')
    print(f'P(Teacher better | D=1) = {better}/{len(sub)} = {better/len(sub)*100:.1f}%')
    print(f'P(Teacher worse | D=1) = {worse}/{len(sub)} = {worse/len(sub)*100:.1f}%')
    print(f'P(Equivalent | D=1) = {equiv}/{len(sub)} = {equiv/len(sub)*100:.1f}%')
 
# P(acceptable | D=1) 总体
sub = [h for h in joined if h['D'] == 1]
acc_all = sum(1 for h in sub if h['human_acceptable'] == 'Yes')
print(f'\nP(Student acceptable | D=1) 总体 = {acc_all}/{len(sub)} = {acc_all/len(sub)*100:.1f}%')
 
# 人工 Strong 里 SAGE 的分布
print('\n=== 人工 Strong (n=24) 的 SAGE 标签 ===')
sub = [h for h in joined if h['human_intervention'] == 'Strong']
print(Counter(h['sage_label'] for h in sub))
