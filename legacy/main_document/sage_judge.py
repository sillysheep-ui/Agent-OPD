#!/usr/bin/env python3
"""SAGE Intervention Judge: 对 310 个 unique state 调 Teacher 判定 Skip/Weak/Strong
- blind: 只输入 (Task, History, Observation, A_t, a_S)——不出现 entropy/disagreement/Teacher action/valid/success
- Teacher 配置: thinking enabled + temp 0（与采集一致）
- 并发 8（API 限速安全）
"""
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
 
def read_key():
    raw = open('/root/.deepseek_key').read().strip()
    # 文件是 export 格式（export DEEPSEEK_API_KEY=sk-...）——提取 = 后的部分
    if '=' in raw and not raw.startswith('sk-'):
        raw = raw.split('=', 1)[1]
    return raw.strip()
 
API_KEY = read_key()
BASE_URL = 'https://api.deepseek.com'
MODEL = 'deepseek-v4-flash'
 
JUDGE_SYSTEM = (
    'You are an expert ALFWorld task supervisor. Your job: for each student turn, decide '
    'whether teacher intervention (a corrective action) is necessary.\n'
    '- Skip: the student action is reasonable/fine for this state; no correction needed.\n'
    '- Weak: the action is questionable or suboptimal; a correction may help but is not critical.\n'
    '- Strong: the action is clearly wrong or harmful; a correction is necessary.\n'
    'Output exactly one word: Skip, Weak, or Strong.'
)
 
def call_teacher(messages):
    payload = {
        'model': MODEL,
        'messages': messages,
        'max_tokens': 2000,    # thinking 会占 token——留足空间让标签出现
        'temperature': 0,
        'stream': False,
    }
    req = urllib.request.Request(
        BASE_URL + '/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == 2:
                return f'__ERROR__: {e}'
            time.sleep(3 * (attempt + 1))
 
def build_user_prompt(st):
    """blind prompt: (Task, History, Observation_t, A_t, a_S)"""
    # 从 query_messages 提取 Task / History / Observation / Admissible
    msgs = st['query_messages']
    task = ''
    obs = ''
    admissible = ''
    history = []
    for m in msgs:
        c = m.get('content', '')
        if m['role'] == 'user':
            if 'Your task is to' in c:
                task = c.split('Your task is to')[1].split('.')[0].strip().lstrip(':').strip()
            if 'Observation:' in c:
                obs_part = c.split('Observation:', 1)[1]
                obs = obs_part.split('Admissible actions:')[0].strip()
                adm = obs_part.split('Admissible actions:', 1)[1] if 'Admissible actions:' in obs_part else ''
                admissible = adm.strip()
            history.append(c[:200])   # 历史截断（防止超长）
    hist_text = '\n'.join(history[-6:]) if history else ''   # 最近 6 条
    return (
        f'Task: {task}\n\n'
        f'Recent history:\n{hist_text}\n\n'
        f'Current observation: {obs}\n'
        f'Admissible actions: {admissible}\n'
        f'Student action: {st["student_action"]}\n\n'
        f'Intervention label (Skip/Weak/Strong):'
    )
 
def parse_label(text):
    t = text.strip().upper()
    # thinking 模式下取最后一次出现的标签（final answer 在尾部）
    best, pos = None, -1
    for lab in ['STRONG', 'WEAK', 'SKIP']:
        p = t.rfind(lab)
        if p > pos:
            pos, best = p, lab
    if best:
        return best.title()
    return f'PARSE_FAIL:{t[-30:]}'
 
def judge_one(st):
    user = build_user_prompt(st)
    resp = call_teacher([
        {'role': 'system', 'content': JUDGE_SYSTEM},
        {'role': 'user', 'content': user},
    ])
    if resp.startswith('__ERROR__'):
        return {'state_id': (st['gamefile'], st['turn_index']), 'label': None, 'raw': resp, 'error': True}
    return {
        'state_id': (st['gamefile'], st['turn_index']),
        'label': parse_label(resp), 'raw': resp, 'error': False,
    }
 
def main():
    rows = [json.loads(l) for l in open('/root/data/alfworld/opd/a1/a3/run1/sage_to_judge.jsonl')]
    out_path = '/root/data/alfworld/opd/a1/a3/run1/sage_labels.jsonl'
    # 断点续跑: 已标注的 state_id 跳过（PARSE_FAIL 的除外——需重试）
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                if r.get('label') and not str(r['label']).startswith('PARSE_FAIL'):
                    done.add((r['state_id'][0], r['state_id'][1]))
            except Exception:
                pass
    todo = [r for r in rows if (r['gamefile'], r['turn_index']) not in done]
    print(f'total {len(rows)}, already done {len(done)}, to judge {len(todo)}', flush=True)
 
    fout = open(out_path, 'a')
    lock = __import__('threading').Lock()
    n_err = 0
    def work(st):
        nonlocal n_err
        res = judge_one(st)
        if res['error']:
            with lock:
                n_err += 1
        with lock:
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
            fout.flush()
        return res
 
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, res in enumerate(ex.map(work, todo)):
            if (i + 1) % 50 == 0:
                print(f'  [{i+1}/{len(todo)}]', flush=True)
    fout.close()
    print(f'done: {len(todo)} labels, errors: {n_err}', flush=True)
    from collections import Counter
    labels = [json.loads(l)['label'] for l in open(out_path)]
    print('label distribution:', dict(Counter(labels)), flush=True)
 
if __name__ == '__main__':
    main()
