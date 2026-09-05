#!/usr/bin/env python3
"""Position Study 前置数据检查: 114 D=1 states 字段完整性"""
import json
from collections import Counter, defaultdict
 
BASE = '/root/data/alfworld/opd/a1/a3/run1/'
states = {}
for l in open(BASE + 'sage_to_judge.jsonl'):
    r = json.loads(l)
    states[(r['gamefile'], r['turn_index'])] = r
labels = {}
for l in open(BASE + 'sage_labels.jsonl'):
    r = json.loads(l)
    labels[(r['state_id'][0], r['state_id'][1])] = r['label']
 
# 114 D=1 states
d1 = [sid for sid in states if states[sid].get('D') == 1]
print(f'D=1 states: {len(d1)}')
 
# 字段检查
def check(sid):
    st = states[sid]
    # teacher_action（valid 的第一个）
    tacts = [str(t) for t in st.get('teacher_actions', [])]
    aS = st.get('student_action')
    adm = st.get('admissible_actions', [])
    msgs = st.get('query_messages', [])
    # executed prefix（assistant 消息的 Action）
    prefix = [m['content'] for m in msgs if m['role'] == 'assistant']
    return {
        'aS_valid': bool(aS) and aS in adm,
        'aT_valid': bool(tacts) and tacts[0] in adm and tacts[0] != aS,
        'has_adm': len(adm) > 0,
        'n_msgs': len(msgs),
        'n_prefix': len(prefix),
        'label': labels.get(sid),
        'groups': st.get('groups', []),
        'turn': sid[1],
    }
 
ok = 0
issues = Counter()
for sid in d1:
    c = check(sid)
    if c['aS_valid'] and c['aT_valid'] and c['has_adm'] and c['n_prefix'] >= 1:
        ok += 1
    else:
        for k, v in [('aS_valid', c['aS_valid']), ('aT_valid', c['aT_valid']),
                     ('has_adm', c['has_adm']), ('n_prefix', c['n_prefix'])]:
            if not v:
                issues[k] += 1
print(f'完整通过字段检查: {ok}/{len(d1)}')
print(f'问题分布: {dict(issues)}')
 
# 标签分布（H-P2 需要）
print('\n=== I_T 分布（D=1）===')
print(Counter(labels.get(sid) for sid in d1))
 
# membership（H-P4 需要）
print('\n=== A1/A3 membership（D=1）===')
a1_m = sum(1 for sid in d1 if 'A1' in states[sid].get('groups', []))
a3_m = sum(1 for sid in d1 if 'A3' in states[sid].get('groups', []))
both = sum(1 for sid in d1 if 'A1' in states[sid].get('groups', []) and 'A3' in states[sid].get('groups', []))
print(f'A1: {a1_m}, A3: {a3_m}, overlap: {both}')
 
# turn depth 分布（H-P3）
turns = sorted(sid[1] for sid in d1)
print('\n=== turn depth 分布 ===')
print(f'min={turns[0]}, max={turns[-1]}, 前10: {turns[:10]}')
