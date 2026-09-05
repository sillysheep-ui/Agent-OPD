#!/usr/bin/env python3
"""SAGE Audit 分析: 四组主表 + P(I|D) + P(V_T|I) + U_g + game-level cluster bootstrap"""
import json
import random
from collections import defaultdict
 
BASE = '/root/data/alfworld/opd/a1/a3/run1/'
GROUPS = ['A1', 'A3', 'A4', 'A5']
 
# 1. 加载 390 unique states（含 D, V_T, groups）
states = {}
for l in open(BASE + 'sage_to_judge.jsonl'):
    r = json.loads(l)
    states[(r['gamefile'], r['turn_index'])] = r
 
# 2. 加载 labels（judge 结果）+ deterministic 分支
labels = {}
for l in open(BASE + 'sage_labels.jsonl'):
    r = json.loads(l)
    labels[(r['state_id'][0], r['state_id'][1])] = r['label']
 
def get_label(sid):
    st = states[sid]
    if not st['student_valid']:
        return 'Strong'   # deterministic
    return labels.get(sid, 'MISSING')
 
# 3. 四组主表（membership 计数——重叠 state 在所属组都算）
print('=== 四组主表（selected turns 口径——含 teacher-invalid） ===')
print(f'{"Group":<6}{"N":>5}{"Disagree":>10}{"Skip":>8}{"Weak":>8}{"SemStrong":>11}{"DetStrong":>11}{"StrongAll":>10}{"meanI":>7}')
for g in GROUPS:
    sids = [sid for sid, st in states.items() if g in st['groups']]
    n = len(sids)
    skip = weak = sem_strong = det_strong = 0
    for sid in sids:
        lab = get_label(sid)
        if lab == 'Skip': skip += 1
        elif lab == 'Weak': weak += 1
        elif lab == 'Strong':
            if states[sid]['student_valid']: sem_strong += 1
            else: det_strong += 1
    d = sum(1 for sid in sids if states[sid].get('D') == 1)
    mean_i = (0 * skip + 0.5 * weak + 1 * (sem_strong + det_strong)) / n
    print(f'{g:<6}{n:>5}{d/n*100:>9.1f}%{skip:>8}{weak:>8}{sem_strong:>11}{det_strong:>11}{sem_strong+det_strong:>10}{mean_i:>7.3f}')
 
# 4. P(I|D) 与 P(D|I)（unique state 口径）
print('\n=== P(I | D)（可执行 turns——student_valid=True） ===')
exec_sids = [sid for sid, st in states.items() if st['student_valid']]
for dval in [1, 0]:
    sub = [sid for sid in exec_sids if states[sid].get('D') == dval]
    n = len(sub)
    if n == 0: continue
    c = defaultdict(int)
    for sid in sub:
        c[get_label(sid)] += 1
    print(f'D={dval} (n={n}): ' + ' '.join(f'{k}={v/n*100:.1f}%' for k, v in sorted(c.items())))
 
print('\n=== P(D | I) ===')
for lab in ['Skip', 'Weak', 'Strong']:
    sub = [sid for sid in exec_sids if get_label(sid) == lab]
    n = len(sub)
    if n == 0: continue
    d1 = sum(1 for sid in sub if states[sid].get('D') == 1)
    print(f'I={lab} (n={n}): P(D=1)={d1/n*100:.1f}%')
 
# 5. P(V_T | I)（Teacher reliability 连接）
print('\n=== P(V_T=1 | I)（unique state 口径） ===')
for lab in ['Skip', 'Weak', 'Strong']:
    sub = [sid for sid in states if get_label(sid) == lab]
    n = len(sub)
    if n == 0: continue
    vt = sum(1 for sid in sub if states[sid].get('teacher_valid'))
    print(f'I={lab} (n={n}): P(V_T=1)={vt/n*100:.1f}%')
 
# 6. U_g = P(D=1, I=Skip | g)（unnecessary-disagreement rate）
print('\n=== U_g = P(D=1, I=Skip)（可执行 turns） ===')
for g in GROUPS:
    sids = [sid for sid, st in states.items() if g in st['groups'] and st['student_valid']]
    n = len(sids)
    if n == 0: continue
    u = sum(1 for sid in sids if states[sid].get('D') == 1 and get_label(sid) == 'Skip')
    print(f'{g}: U={u/n*100:.1f}% (n={n})')
 
# 7. game-level cluster bootstrap: ΔStrong (A3-A1)
print('\n=== game-level cluster bootstrap: ΔStrong (A3-A1, 可执行 turns) ===')
def group_by_game(sids):
    byg = defaultdict(list)
    for sid in sids:
        byg[sid[0]].append(sid)
    return byg
s1 = [sid for sid, st in states.items() if 'A1' in st['groups'] and st['student_valid']]
s3 = [sid for sid, st in states.items() if 'A3' in st['groups'] and st['student_valid']]
g1, g3 = group_by_game(s1), group_by_game(s3)
common = sorted(set(g1) & set(g3))
def strong_rate(byg, games):
    sids = [sid for g in games for sid in byg[g]]
    if not sids: return 0.0
    return sum(1 for sid in sids if get_label(sid) == 'Strong') / len(sids)
r1, r3 = strong_rate(g1, common), strong_rate(g3, common)
print(f'common games: {len(common)}, Strong: A1={r1:.3f}, A3={r3:.3f}, Δ={r3-r1:+.3f}')
rng = random.Random(42)
diffs = []
for _ in range(5000):
    gs = [rng.choice(common) for _ in common]
    diffs.append(strong_rate(g3, gs) - strong_rate(g1, gs))
diffs.sort()
lo, hi = diffs[125], diffs[4975]
print(f'95% CI of ΔStrong: [{lo:+.3f}, {hi:+.3f}]', '✅ 显著' if hi < 0 or lo > 0 else '❌ 不显著')
