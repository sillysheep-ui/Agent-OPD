#!/usr/bin/env python3
"""A1R 多 seed 验证: 同一 rollout（A3 episodes），不同 seed 随机选点 + Teacher"""
import json
import random
import sys
from pathlib import Path
from openai import OpenAI
 
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
from agent_harness.parser import parse_and_canonicalize as pac
 
SEED = int(sys.argv[1])
M = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1
EPISODES = '/root/data/alfworld/opd/a1/a3/run1/episodes.jsonl'
OUT = f'/root/data/alfworld/opd/a1/a3/run1/a2_n{N}_seed{SEED}_corrections.jsonl'
TEACHER_SYSTEM_PROMPT = """You are an expert ALFWorld household agent.
 
Complete the given household task by interacting with the environment one step at a time.
 
At each turn, use the task description, interaction history, current observation, and current admissible actions to determine the next action.
 
Base your decision only on objects, receptacles, locations, and state information supported by the task and the actual interaction history.
 
Keep the task's target object, its current location, carried objects, and relevant state changes such as cleaned, heated, or cooled consistent with the observed environment.
 
Do not invent object identities, object numbers, locations, carried objects, or state changes.
 
Choose exactly one action from the current admissible actions.
 
Return exactly:
Action: <command>
 
Use the action exactly as written in the admissible action list.
 
Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""
 
def classify_failure(raw, parsed):
    if not str(raw or '').strip():
        return 'empty_content'
    if not parsed.had_action_marker:
        return 'no_action_marker'
    if not parsed.valid:
        return 'not_admissible_after_canonicalization'
    return None
 
def main():
    episodes = [json.loads(l) for l in open(EPISODES)]
    key = Path('/root/.deepseek_key').read_text().strip()
    if '=' in key and not key.startswith('sk-'):
        key = key.split('=', 1)[1]
    teacher = OpenAI(api_key=key, base_url='https://api.deepseek.com')
    # A2: N>1 Teacher sampling requires real temperature -> thinking DISABLED
    # (thinking mode ignores temperature; A1/M6 used thinking enabled + temp 0)
    extra = {'thinking': {'type': 'disabled'}}
    teacher_temp = 0.5
 
    rng = random.Random(SEED)
    stats = {'episodes': 0, 'student_wins': 0, 'corrections_total': 0,
             'corrections_valid': 0, 'disagreement': 0, 'teacher_retries': 0}
    with open(OUT, 'w') as f:
        for ep_idx, ep in enumerate(episodes):
            stats['episodes'] += 1
            stats['student_wins'] += int(ep['won'])
            turns = ep['turns']
            selected = rng.sample(range(len(turns)), min(M, len(turns)))
            for turn_idx in selected:
                t = turns[turn_idx]
                # sample N Teacher actions per state (iid given s)
                n_valid = 0
                n_disagree = 0
                t_actions = []
                for j in range(N):
                    raw = ''
                    parsed = None
                    for attempt in range(3):
                        resp = teacher.chat.completions.create(
                            model='deepseek-v4-flash', messages=t['query_messages'],
                            temperature=teacher_temp, max_tokens=3000, extra_body=extra)
                        c = resp.choices[0]
                        raw = (c.message.content or '') if c else ''
                        parsed = pac(raw, t['admissible_actions'], marker_strategy='first')
                        reason = classify_failure(raw, parsed)
                        if reason in (None, 'not_admissible_after_canonicalization'):
                            break
                        stats['teacher_retries'] += 1
                    t_action = parsed.canonical_action if (parsed and parsed.valid) else 'look'
                    valid = bool(parsed and parsed.valid)
                    disagree = valid and (t_action != t['student_action'])
                    n_valid += int(valid)
                    n_disagree += int(disagree)
                    t_actions.append({'teacher_raw': raw, 'teacher_action': t_action,
                                      'teacher_valid': valid, 'disagreement': disagree})
                stats['corrections_total'] += 1
                stats['corrections_valid'] += int(n_valid > 0)
                stats['disagreement'] += int(n_disagree > 0)
                rec = {
                    'episode_index': ep_idx, 'gamefile': ep['gamefile'], 'task_type': ep['task_type'],
                    'student_won': ep['won'], 'turn_index': turn_idx, 'turn_depth': turn_idx,
                    'query_messages': t['query_messages'], 'admissible_actions': t['admissible_actions'],
                    'student_action': t['student_action'], 'student_valid': t['student_valid'],
                    'n': N, 'teacher_actions': t_actions,
                    'disagreement': n_disagree > 0,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
    print(json.dumps({'seed': SEED, 'n': N, 'stats': stats,
                      'disagreement_rate': stats['disagreement'] / max(stats['corrections_total'], 1),
                      'valid_rate': stats['corrections_valid'] / max(stats['corrections_total'], 1)}))
 
if __name__ == '__main__':
    main()
