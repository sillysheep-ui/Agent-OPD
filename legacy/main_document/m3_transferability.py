#!/usr/bin/env python3
"""M3: Transferability —— correction (s_i, a_i^T) 在相似状态的迁移
T(s_i, s_j) = logp_{θ_after}(a_i^T | s_j) - logp_{θ_SFT}(a_i^T | s_j)
- 原状态 s_i 的 Δlogp（验证学进去了）
- K 个同 task_type 的 rollout 状态 s_j（student_valid, 排除 s_i, seed 固定）
- 对比: A1 (用 A1-strong 测) vs A3 (用 A3-selective 测)
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
OUT = '/root/data/alfworld/opd/a1/a3/run1/m3_result.jsonl'
TARGET_PREFIX = 'Action: '
K_NEIGHBORS = 5
 
def load_model(lora_path=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(torch.bfloat16).cuda().eval()
    return model, tok
 
def logp_of(model, tok, messages, action):
    """action 的序列 logp 之和（sum of token logprobs——冻结定义）"""
    with torch.no_grad():
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = tok(text, return_tensors='pt')['input_ids'].to('cuda')
        target = tok(TARGET_PREFIX + action, return_tensors='pt')['input_ids'].to('cuda')
        full = torch.cat([prompt, target], dim=1)
        out = model(input_ids=full, use_cache=False)
        logits = out.logits[0, prompt.shape[1]-1:-1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        target_ids = full[0, prompt.shape[1]:]
        return logp.gather(1, target_ids.unsqueeze(1)).squeeze(1).sum().item()
 
def main():
    group = sys.argv[1]   # A1 or A3
    ckpt = A1_CKPT if group == 'A1' else A3_CKPT
    corr_path = ('/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl'
                 if group == 'A1' else '/root/data/alfworld/opd/a1/a3/run1/corrections.jsonl')
    import glob
    if group == 'A3':
        corr_path = glob.glob('/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl')
 
    episodes = [json.loads(l) for l in open(EPISODES)]
    # 每个 task_type 的正常状态池（student_valid=True）
    pool = {}
    for ep in episodes:
        for t in ep['turns']:
            if t.get('student_valid', False):
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
        s_i_msgs = r['query_messages']
        aT = r['teacher_action']
        # 原状态
        lp_sft_i = logp_of(model_sft, tok, s_i_msgs, aT)
        lp_after_i = logp_of(model_after, tok, s_i_msgs, aT)
        d_self = lp_after_i - lp_sft_i
        # 迁移状态: 同 task_type 抽 K 个（排除自身 game+turn）
        cands = [x for x in pool.get(r['task_type'], [])
                 if x[0] != r['gamefile']]
        if len(cands) > K_NEIGHBORS:
            cands = rng.sample(cands, K_NEIGHBORS)
        d_trans = []
        for gf, t in cands:
            lp_sft_j = logp_of(model_sft, tok, t['query_messages'], aT)
            lp_after_j = logp_of(model_after, tok, t['query_messages'], aT)
            d_trans.append(lp_after_j - lp_sft_j)
        res = {
            'group': group, 'gamefile': r['gamefile'], 'task_type': r['task_type'],
            'turn_index': r['turn_index'], 'action': aT,
            'dlogp_self': d_self,
            'dlogp_transfer_mean': sum(d_trans) / max(len(d_trans), 1),
            'n_transfer': len(d_trans),
            'disagreement': r.get('disagreement', None),
        }
        results.append(res)
        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(recs)}]', flush=True)
 
    with open(OUT, 'a') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    import statistics
    ds = [x['dlogp_self'] for x in results]
    dt = [x['dlogp_transfer_mean'] for x in results]
    print(json.dumps({
        'group': group, 'n': len(results),
        'dlogp_self_mean': statistics.mean(ds),
        'dlogp_transfer_mean': statistics.mean(dt),
        'transfer_ratio': statistics.mean(dt) / max(statistics.mean(ds), 1e-9),
    }))
 
if __name__ == '__main__':
    main()
