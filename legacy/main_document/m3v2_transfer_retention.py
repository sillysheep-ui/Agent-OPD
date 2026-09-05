#!/usr/bin/env python3
"""M3 v2: Correction Update Transfer Retention（修正版）
- transfer pool = rollout ∖ (A1∪A3∪A4∪A5 全部 selected states)（防泄漏）
- 同 game 排除 + student_valid + a_i^T ∈ admissible(s_j)（语义适用性）
- token-normalized Δlogp（mean token logp——与训练 loss 一致）为主
- K=10 邻居
"""
import json
import random
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
BASE = '/root/data/alfworld/checkpoints/sft_v6_merged_bf16'
EPISODES = '/root/data/alfworld/opd/a1/a3/run1/episodes.jsonl'
A1_CKPT = '/root/data/alfworld/checkpoints/d3_a1_strong/global_step_108'
A3_CKPT = '/root/data/alfworld/checkpoints/a3_selective/global_step_102'
OUT = '/root/data/alfworld/opd/a1/a3/run1/m3v2_result.jsonl'
TARGET_PREFIX = 'Action: '
K_NEIGHBORS = 10
 
# 全部 selection 方法的 selected turns（防泄漏）
SELECTED_PATHS = [
    '/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl',          # A1
    '/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl',             # A3
    '/root/data/alfworld/opd/a1/a3/run1/a4_corrections.jsonl',                 # A4
    '/root/data/alfworld/opd/a1/a3/run1/a5_corrections.jsonl',                 # A5
]
 
def load_model(lora_path=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(torch.bfloat16).cuda().eval()
    return model, tok
 
def mean_token_logp_of(model, tok, messages, action):
    """action 的 mean token log-prob（token-normalized——主指标）"""
    with torch.no_grad():
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = tok(text, return_tensors='pt')['input_ids'].to('cuda')
        target = tok(TARGET_PREFIX + action, return_tensors='pt')['input_ids'].to('cuda')
        full = torch.cat([prompt, target], dim=1)
        out = model(input_ids=full, use_cache=False)
        logits = out.logits[0, prompt.shape[1]-1:-1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        target_ids = full[0, prompt.shape[1]:]
        per_token = logp.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        return per_token.mean().item()   # token-normalized
 
def main():
    group = sys.argv[1]
    ckpt = A1_CKPT if group == 'A1' else A3_CKPT
    corr_path = ('/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl'
                 if group == 'A1' else '/root/data/alfworld/opd/a1/a3/run1/corrections.jsonl')
    import glob
    if group == 'A3':
        corr_path = glob.glob('/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl')
 
    episodes = [json.loads(l) for l in open(EPISODES)]
 
    # selected turns（全部方法）——防泄漏
    import glob as G
    selected = set()
    for p in SELECTED_PATHS:
        for f in G.glob(p) if any(ch in p for ch in '*?') else [p]:
            for l in open(f):
                try:
                    r = json.loads(l)
                    selected.add((r['gamefile'], r['turn_index']))
                except Exception:
                    pass
    print(f'selected turns (excluded): {len(selected)}', flush=True)
 
    # 正常状态池: 非 selected + student_valid
    pool = {}
    for ep in episodes:
        for t in ep['turns']:
            if (ep['gamefile'], t['turn_index']) in selected:
                continue
            if not t.get('student_valid', False):
                continue
            pool.setdefault(ep['task_type'], []).append((ep['gamefile'], t))
    print(f'{group}: pool sizes { {k: len(v) for k, v in pool.items()} }', flush=True)
 
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
    rng = random.Random(2026)
 
    results = []
    for i, r in enumerate(recs):
        aT = r['teacher_action']
        s_i_msgs = r['query_messages']
        # self（原状态——admissible 必然成立）
        d_self = (mean_token_logp_of(model_after, tok, s_i_msgs, aT)
                  - mean_token_logp_of(model_sft, tok, s_i_msgs, aT))
        # transfer: 同 task_type、非同 game、admissible 过滤
        cands = [x for x in pool.get(r['task_type'], [])
                 if x[0] != r['gamefile'] and aT in x[1]['admissible_actions']]
        if len(cands) > K_NEIGHBORS:
            cands = rng.sample(cands, K_NEIGHBORS)
        d_trans = []
        for gf, t in cands:
            d_trans.append(mean_token_logp_of(model_after, tok, t['query_messages'], aT)
                           - mean_token_logp_of(model_sft, tok, t['query_messages'], aT))
        if len(d_trans) == 0:
            continue    # 无合法 transfer state——剔除（transfer 未定义）
        results.append({
            'group': group, 'gamefile': r['gamefile'], 'task_type': r['task_type'],
            'turn_index': r['turn_index'], 'action': aT,
            'dlogp_self': d_self,
            'dlogp_transfer_mean': sum(d_trans) / max(len(d_trans), 1),
            'n_transfer': len(d_trans),
            'disagreement': r.get('disagreement', None),
        })
        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(recs)}]', flush=True)
 
    with open(OUT, 'a') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    import statistics
    ds = [x['dlogp_self'] for x in results]
    dt = [x['dlogp_transfer_mean'] for x in results]
    n_t = [x['n_transfer'] for x in results]
    print(json.dumps({
        'group': group, 'n': len(results),
        'dlogp_self_mean': statistics.mean(ds),
        'dlogp_transfer_mean': statistics.mean(dt),
        'transfer_ratio': statistics.mean(dt) / max(statistics.mean(ds), 1e-9),
        'mean_n_transfer': statistics.mean(n_t),
    }))
 
if __name__ == '__main__':
    main()
