#!/usr/bin/env python3
"""给 A1 run1 的 145 条补算 normalized entropy（同一评分模型/方法）"""
import sys
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# device selection via CUDA_VISIBLE_DEVICES (set by launcher); do NOT call
# set_device with the physical index — it goes out of range under the mask.
 
BASE = '/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507'
LORA = '/root/data/alfworld/checkpoints/sft_v6/global_step_542'
CORR = '/root/data/alfworld/opd/a1/run1/corrections.jsonl'
OUT = '/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl'
 
recs = [json.loads(l) for l in open(CORR)]
valid = [r for r in recs if r['teacher_valid']]
print(f'valid: {len(valid)}')
 
torch.cuda.set_device(0)  # already set above via argv
scorer = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2").cuda().eval()
scorer = PeftModel.from_pretrained(scorer, LORA).eval()
tok2 = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok2.padding_side = 'left'
lm_head = scorer.get_output_embeddings()
 
def turn_entropy(msgs, acts, bs=1):
    texts = [tok2.apply_chat_template(msgs + [{'role': 'assistant', 'content': f'Action: {a}'}],
                                      tokenize=False, add_generation_prompt=False) for a in acts]
    enc_all = tok2(texts)
    ids_list = [torch.tensor(row) for row in enc_all['input_ids']]
    with torch.no_grad():
        logps = []
        for s in range(0, len(ids_list), bs):
            chunk_ids = ids_list[s:s+bs]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [x.flip(0) for x in chunk_ids], batch_first=True, padding_value=tok2.pad_token_id).flip(1)
            attn = (input_ids != tok2.pad_token_id).long()
            out = scorer(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :-1].to(torch.bfloat16)
            targets = input_ids[:, 1:].cuda()
            pad_id = tok2.pad_token_id or 0
            valid_mask = (targets != pad_id).float()
            B, T, H = hidden.shape
            seq_logp = torch.zeros(B, device='cuda')
            CHUNK = 256
            for c in range(0, T, CHUNK):
                h = hidden[:, c:c+CHUNK]
                lg = lm_head(h).to(torch.bfloat16)
                lsm = torch.nn.functional.log_softmax(lg, dim=-1)
                tgt = targets[:, c:c+CHUNK]
                v = valid_mask[:, c:c+CHUNK]
                per = lsm.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
                seq_logp += (per * v).sum(dim=1)
            logps.extend(seq_logp.tolist())
    vals = torch.tensor(logps, dtype=torch.float32)
    vals = vals - vals.max()
    probs = torch.softmax(vals, dim=0)
    ent = -(probs * torch.log(probs + 1e-12)).sum().item()
    return ent / math.log(max(len(acts), 2))
 
import math
for i, r in enumerate(valid):
    if i % 4 != GPU:  # 4-way shard by index
        continue
    r['entropy'] = round(turn_entropy(r['query_messages'], r['admissible_actions']), 4)
    if i % 30 == 0:
        print(f'  gpu{GPU} [{i}/{len(valid)}] ent={r["entropy"]}', flush=True)
 
out = OUT.replace('.jsonl', f'_gpu{GPU}.jsonl')
with open(out, 'w') as f:
    for r in valid:
        if 'entropy' in r:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'gpu{GPU} saved {out}')
