#!/usr/bin/env python3
"""SAGE Intervention Audit v2（4 修正版）
1. 原始 selected turns（含 teacher-invalid）——不条件在 V_T=1
2. unique state (gamefile, turn_index) 只 judge 一次——保存 membership
3. deterministic Strong（student_valid=False）不调 Teacher——intervention_source 区分
4. 主分析: P(I|D), P(V_T|I), U_g = P(D=1, I=Skip)
"""
import json
import sys
import re
from collections import defaultdict
 
OUT = '/root/data/alfworld/opd/a1/a3/run1/sage_audit.jsonl'
 
# 各组原始 selected turns（含 invalid）
GROUPS = {
    'A1': ['/root/data/alfworld/opd/a1/run1/corrections.jsonl'],
    'A3': ['/root/data/alfworld/opd/a1/a3/run1/shard{}/corrections.jsonl'.format(i) for i in range(4)],
    'A4': ['/root/data/alfworld/opd/a1/a3/run1/a4_shard{}/corrections.jsonl'.format(i) for i in range(4)],
    'A5': ['/root/data/alfworld/opd/a1/a3/run1/a5_shard{}/corrections.jsonl'.format(i) for i in range(4)],
}
 
def parse_action(action_str):
    """解析 student_action 字段（可能是字符串或 dict）"""
    if isinstance(action_str, dict):
        return action_str.get('action') or action_str.get('canonical_action') or str(action_str)
    return str(action_str)
 
def main():
    # 1. 合并原始 selected turns
    states = {}   # state_id -> {gamefile, turn_index, query_messages, admissible, student_action, student_valid, groups: set, teacher_actions: [...]}
    for g, paths in GROUPS.items():
        for p in paths:
            for l in open(p):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                sid = (r['gamefile'], r['turn_index'])
                st = states.setdefault(sid, {
                    'gamefile': r['gamefile'], 'turn_index': r['turn_index'],
                    'query_messages': r['query_messages'],
                    'admissible_actions': r.get('admissible_actions', []),
                    'student_action': parse_action(r.get('student_action', '')),
                    'student_valid': r.get('student_valid', True),
                    'groups': set(), 'teacher_actions': [],
                })
                st['groups'].add(g)
                # teacher actions（旧格式直接字段 / 新格式数组）
                if 'teacher_actions' in r:
                    st['teacher_actions'].extend(
                        [a for a in r['teacher_actions'] if a.get('teacher_valid')])
                elif r.get('teacher_action') is not None and r.get('teacher_valid'):
                    st['teacher_actions'].append(r['teacher_action'])
    print(f'unique states: {len(states)}')
    for g in GROUPS:
        print(f'  {g}: {sum(1 for s in states.values() if g in s["groups"])} memberships')
 
    # 2. 每条 state 计算 D（disagreement）和 V_T
    for st in states.values():
        st['teacher_valid'] = len(st['teacher_actions']) > 0
        # D = 任一 valid teacher action 与 student 不同
        st['D'] = int(any(str(t) != st['student_action'] for t in st['teacher_actions'])) if st['teacher_valid'] else None
        # 标签初始化
        st['intervention'] = None
        st['intervention_source'] = None
 
    # 3. 统计 deterministic 分支
    det = sum(1 for s in states.values() if not s['student_valid'])
    print(f'deterministic Strong (student_valid=False): {det}')
 
    # 4. 待 judge 的 state（student_valid=True——需 Teacher 判定）
    to_judge = [s for s in states.values() if s['student_valid']]
    print(f'to judge: {len(to_judge)}')
 
    # 5. 输出待标注文件（供 judge 脚本消费）
    with open('/root/data/alfworld/opd/a1/a3/run1/sage_to_judge.jsonl', 'w') as f:
        for s in to_judge:
            s_out = dict(s)
            s_out['groups'] = sorted(s['groups'])
            f.write(json.dumps(s_out, ensure_ascii=False) + '\n')
    print(f'written sage_to_judge.jsonl ({len(to_judge)} states)')
 
if __name__ == '__main__':
    main()
