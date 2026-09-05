#!/usr/bin/env python3
"""M3 v3 FINAL: Correction Transfer Retention
S_i = source absorption (token-normalized Δlogp = loss reduction)
T_i = neighbor transfer retention
N(s_i) = top-K by sim (0.5 BOW cosine + 0.5 Jaccard(A^-_i, A^-_j))
         subject to: task 同 / a_i^T∈A(s_j) / game 异 / 未训练
covariates: task_type, action_type, technical, repeat_obs, sim_top10
"""
import json
import random
import sys
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
BASE = '/root/data/alfworld/checkpoints/sft_v6_merged_bf16'
EPISODES = '/root/data/alfworld/opd/a1/a3/run1/episodes.jsonl'
A1_CKPT = '/root/data/alfworld/checkpoints/d3_a1_strong/global_step_108'
A3_CKPT = '/root/data/alfworld/checkpoints/a3_selective/global_step_102'
OUT = '/root/data/alfworld/opd/a1/a3/run1/m3v3_result.jsonl'
TARGET_PREFIX = 'Action: '
K_NEIGHBORS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SIM_MODE = sys.argv[3] if len(sys.argv) > 3 else 'both'   # both|bow|jaccard
 
SELECTED_PATHS = [
    '/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl',
    '/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl',
    '/root/data/alfworld/opd/a1/a3/run1/a4_corrections.jsonl',
    '/root/data/alfworld/opd/a1/a3/run1/a5_corrections.jsonl',
]
 
STOPWORDS = {'the', 'a', 'an', 'you', 'are', 'is', 'in', 'of', 'to', 'and', 'on', 'with', 'at', 'from'}
 
def load_model(lora_path=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(torch.bfloat16).cuda().eval()
    return model, tok
 
def mean_token_logp(model, tok, messages, action):
    """token-normalized logp（loss reduction 的正向量）"""
    with torch.no_grad():
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = tok(text, return_tensors='pt')['input_ids'].to('cuda')
        target = tok(TARGET_PREFIX + action, return_tensors='pt')['input_ids'].to('cuda')
        full = torch.cat([prompt, target], dim=1)
        out = model(input_ids=full, use_cache=False)
        logits = out.logits[0, prompt.shape[1]-1:-1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        target_ids = full[0, prompt.shape[1]:]
        return logp.gather(1, target_ids.unsqueeze(1)).squeeze(1).mean().item()
 
def extract_obs(messages):
    for m in reversed(messages):
        if m['role'] == 'user':
            c = m['content']
            if 'Observation:' in c:
                part = c.split('Observation:', 1)[1]
                return part.split('Admissible actions:')[0].strip()
    return ''
 
def bow_vec(text):
    toks = [w for w in text.lower().split() if w not in STOPWORDS and w.isalnum()]
    return Counter(toks)
 
def cosine_bow(a, b):
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    na = sum(a.values()) ** 0.5
    nb = sum(b.values()) ** 0.5
    return inter / (na * nb + 1e-9)
 
def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
 
def main():
    group = sys.argv[1]
    ckpt = A1_CKPT if group == 'A1' else A3_CKPT
    corr_path = ('/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl'
                 if group == 'A1' else '/root/data/alfworld/opd/a1/a3/run1/corrections.jsonl')
    import glob as G
    if group == 'A3':
        corr_path = G.glob('/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl')
 
    episodes = [json.loads(l) for l in open(EPISODES)]
 
    # 全部 selected（防泄漏）
    selected = set()
    for p in SELECTED_PATHS:
        for f in G.glob(p) if any(ch in p for ch in '*?') else [p]:
            for l in open(f):
                try:
                    r = json.loads(l)
                    selected.add((r['gamefile'], r['turn_index']))
                except Exception:
                    pass
 
    # transfer pool: 非 selected + student_valid + 特征（obs BOW + admissible）
    pool = {}
    prev_obs_map = {}
    for ep in episodes:
        prev = None
        for t in ep['turns']:
            prev_obs_map[(ep['gamefile'], t['turn_index'])] = prev
            prev = extract_obs(t['query_messages'])
    for ep in episodes:
        for t in ep['turns']:
            if (ep['gamefile'], t['turn_index']) in selected:
                continue
            if not t.get('student_valid', False):
                continue
            obs = extract_obs(t['query_messages'])
            pool.setdefault(ep['task_type'], []).append({
                'gamefile': ep['gamefile'], 'turn': t,
                'bow': bow_vec(obs),
                'admissible': set(t['admissible_actions']),
            })
    print(f'{group}: pool {sum(len(v) for v in pool.values())} states', flush=True)
 
    # corrections
    recs = []
    for f in corr_path if isinstance(corr_path, list) else [corr_path]:
        for l in open(f):
            try:
                r = json.loads(l)
                if r.get('teacher_valid'):
                    recs.append(r)
            except Exception:
                pass
    print(f'{group}: {len(recs)} corrections', flush=True)
 
    model_sft, tok = load_model()
    model_after, _ = load_model(ckpt)
 
    results = []
    n_no_neighbor = 0
    for i, r in enumerate(recs):
        aT = r['teacher_action']
        s_i_msgs = r['query_messages']
        # self absorption
        d_self = (mean_token_logp(model_after, tok, s_i_msgs, aT)
                  - mean_token_logp(model_sft, tok, s_i_msgs, aT))
        # neighbor selection: sim top-K（candidates 先按约束过滤）
        A_i = set(r['admissible_actions'])
        A_i_minus = A_i - {aT}
        bow_i = bow_vec(extract_obs(s_i_msgs))
        cands = []
        for s in pool.get(r['task_type'], []):
            if s['gamefile'] == r['gamefile']:
                continue
            if aT not in s['admissible']:
                continue
            if SIM_MODE == 'bow':
                sim = cosine_bow(bow_i, s['bow'])
            elif SIM_MODE == 'jaccard':
                sim = jaccard(A_i_minus, s['admissible'] - {aT})
            else:
                sim = 0.5 * cosine_bow(bow_i, s['bow']) + 0.5 * jaccard(A_i_minus, s['admissible'] - {aT})
            cands.append((sim, s))
        cands.sort(key=lambda x: -x[0])
        neigh = cands[:K_NEIGHBORS]
        if len(neigh) < 5:
            n_no_neighbor += 1
        if len(neigh) == 0:
            continue    # 无合法邻居——transfer 未定义
        sim_mean = sum(sim for sim, _ in neigh) / len(neigh)
        d_trans = []
        for sim, s in neigh:
            d_trans.append(mean_token_logp(model_after, tok, s['turn']['query_messages'], aT)
                           - mean_token_logp(model_sft, tok, s['turn']['query_messages'], aT))
        # covariates
        st_act = r.get('student_action') or 'look'
        st_type = st_act.split()[0] if st_act.split() else 'none'
        prev_obs = prev_obs_map.get((r['gamefile'], r['turn_index']))
        cur_obs = extract_obs(s_i_msgs)
        repeat_obs = 1 if (prev_obs is not None and cur_obs == prev_obs and cur_obs) else 0
        results.append({
            'group': group, 'gamefile': r['gamefile'], 'task_type': r['task_type'],
            'turn_index': r['turn_index'], 'action': aT,
            'S_i': d_self,
            'T_i': sum(d_trans) / len(d_trans),
            'sim_topK': sim_mean,
            'n_neighbors': len(neigh),
            'action_type': st_type,
            'technical': 0 if r.get('student_valid', True) else 1,
            'repeat_obs': repeat_obs,
            'disagreement': r.get('disagreement', None),
        })
        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(recs)}]', flush=True)
 
    suffix = f'_k{K_NEIGHBORS}_{SIM_MODE}'
    out_path = OUT.replace('.jsonl', f'_{group}{suffix}.jsonl')   # 按 group 分文件——并发写同一文件会互相截断
    with open(out_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    import statistics
    print(json.dumps({
        'group': group, 'n': len(results), 'n_no_neighbor_lt5': n_no_neighbor,
        'S_mean': statistics.mean(x['S_i'] for x in results),
        'T_mean': statistics.mean(x['T_i'] for x in results),
        'sim_topK_mean': statistics.mean(x['sim_topK'] for x in results),
        'mean_n_neighbors': statistics.mean(x['n_neighbors'] for x in results),
    }))
 
if __name__ == '__main__':
    main()
