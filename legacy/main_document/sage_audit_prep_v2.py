#!/usr/bin/env python3
"""Sanity #3 audit 包 v2: 完整 History + blind 两阶段 + 定向抽样
Pass A 字段在前（Task/History/Obs/Admissible/Student——无 teacher/label/group）
Pass B 在后（Teacher action——先填 A 再看 B）
分层: D=1&Skip（A1 15 + A3 20）+ D=0&Skip 10 + Weak 全部 + Semantic Strong 全部——unique state
"""
import json
import random
from collections import defaultdict
 
BASE = '/root/data/alfworld/opd/a1/a3/run1/'
OUT_MD = BASE + 'sage_audit_human_v2.md'
 
states = {}
for l in open(BASE + 'sage_to_judge.jsonl'):
    r = json.loads(l)
    states[(r['gamefile'], r['turn_index'])] = r
 
labels = {}
for l in open(BASE + 'sage_labels.jsonl'):
    r = json.loads(l)
    labels[(r['state_id'][0], r['state_id'][1])] = r['label']
 
def render_history(st):
    """完整历史渲染（factual History——与 SAGE judge 信息等价）"""
    parts = []
    for m in st['query_messages']:
        role = m['role']
        c = m['content']
        if role == 'system':
            continue   # system prompt 对判断无信息量
        if role == 'user':
            # 提取 obs（含 admissible——user 消息主体）
            if 'Your task is to' in c:
                parts.append(f'[TASK] {c.strip()[:2000]}')
            elif 'Observation:' in c:
                parts.append(f'[OBS] {c.strip()}')
            else:
                parts.append(f'[USER] {c.strip()[:800]}')
        else:
            parts.append(f'[AGENT] {c.strip()[:300]}')
    return '\n'.join(parts)
 
def teacher_actions_of(st):
    return [str(t) for t in st.get('teacher_actions', [])]
 
def first_teacher_of(st):
    acts = teacher_actions_of(st)
    return acts[0] if acts else 'N/A'
 
# ---- 分层抽样 ----
rng = random.Random(2026)
def sample_pool(pred, n):
    pool = [sid for sid, st in states.items()
            if st['student_valid'] and labels.get(sid) in ('Skip', 'Weak', 'Strong') and pred(sid, st)]
    return rng.sample(pool, min(n, len(pool)))
 
sel = []   # (sid, [groups])
# 1) D=1 & Skip（核心——A1/A3 定向）
for g, n in [('A1', 15), ('A3', 20)]:
    for sid in sample_pool(lambda sid, st: st.get('D') == 1 and labels.get(sid) == 'Skip' and g in st['groups'], n):
        sel.append((sid, [g]))
# 2) D=0 & Skip（negative control——10）
for sid in sample_pool(lambda sid, st: st.get('D') == 0 and labels.get(sid) == 'Skip', 10):
    sel.append((sid, list(states[sid]['groups'])))
# 3) Weak 全部
for sid in sample_pool(lambda sid, st: labels.get(sid) == 'Weak', 100):
    sel.append((sid, list(states[sid]['groups'])))
# 4) Semantic Strong 全部
for sid in sample_pool(lambda sid, st: labels.get(sid) == 'Strong', 100):
    sel.append((sid, list(states[sid]['groups'])))
 
# unique state（去重——保留首现的 groups）
seen = {}
for sid, gs in sel:
    if sid not in seen:
        seen[sid] = gs
    else:
        seen[sid] = sorted(set(seen[sid]) | set(gs))
items = [(sid, gs) for sid, gs in seen.items()]
 
print(f'unique states: {len(items)}')
from collections import Counter
print('构成:', Counter(labels[sid] for sid, _ in items))
 
# ---- 输出 markdown（blind 两阶段）----
with open(OUT_MD, 'w') as f:
    f.write('# SAGE 人工 Audit v2（blind 两阶段）\n\n')
    f.write('**规则**：\n')
    f.write('1. 每条先看 **Pass A**（Task/History/Obs/Admissible/Student action）——填 `human_intervention` 和 `human_student_acceptable`\n')
    f.write('2. **再**往下看 Pass B（Teacher action）——填 `human_teacher_vs_student`\n')
    f.write('3. 不要回改 Pass A（保持 blind）\n\n')
    f.write('| human_intervention | human_student_acceptable | human_teacher_vs_student |\n')
    f.write('|---|---|---|\n')
    f.write('| Skip / Weak / Strong / Uncertain | Yes / No / Uncertain | Better / Equivalent / Worse / Unclear / N/A |\n\n')
    f.write('---\n\n')
    for i, (sid, gs) in enumerate(items):
        st = states[sid]
        f.write(f'## 样本 {i+1}（game={sid[0].split("game_")[-1].split("/")[0]}, turn={sid[1]}）\n\n')
        f.write('### Pass A（先填——不要看 Pass B）\n\n')
        f.write(f'**History / Observation**（完整——SAGE judge 同信息集）：\n\n')
        f.write('```\n' + render_history(st)[:6000] + '\n```\n\n')
        f.write(f'**Admissible actions**（完整）：\n\n```\n' + '\n'.join(st['admissible_actions']) + '\n```\n\n')
        f.write(f'**Student action**: `{st["student_action"]}`\n\n')
        f.write(f'- human_intervention: \n- human_student_acceptable: \n- human_note: \n\n')
        f.write('### Pass B（填完 Pass A 再看）\n\n')
        f.write(f'**Teacher action**: `{first_teacher_of(st)}`\n\n')
        f.write(f'- human_teacher_vs_student: \n\n')
        f.write('---\n\n')
print(f'written {OUT_MD}')
