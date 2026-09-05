#!/usr/bin/env python3
"""M1: Gradient Alignment —— LoRA 参数空间的 correction 梯度 vs G_ref
 
phase=ref: 计算 probe set 的平均梯度 G_ref（保存）
phase=group: 对每组 corrections 算 g_i → 三指标（Ā / P(A<0) / cos(G_D, G_ref)）
"""
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
 
BASE = '/root/data/alfworld/checkpoints/sft_v6_merged_bf16'
LORA_CFG = dict(r=16, lora_alpha=32, target_modules=[
    'v_proj', 'o_proj', 'down_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj'],
    lora_dropout=0.0, bias='none')
PROBE = '/root/data/alfworld/opd/a1/a3/run1/m1_probe_100.jsonl'
GREF_PATH = '/root/data/alfworld/opd/a1/a3/run1/m1_gref.pt'
GROUPS = {
    'A1': '/root/data/alfworld/opd/a1/run1/corrections_with_entropy.jsonl',
    'A3': '/root/data/alfworld/opd/a1/a3/run1/shard*/corrections.jsonl',
    'A4': '/root/data/alfworld/opd/a1/a3/run1/a4_corrections.jsonl',
    'A5': '/root/data/alfworld/opd/a1/a3/run1/a5_corrections.jsonl',
}
TARGET_PREFIX = 'Action: '
 
def load_model():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    # 关键顺序: 先在 base 上 enable checkpointing（PEFT 之后 enable 会断计算图）
    model.gradient_checkpointing_enable()
    from peft import get_peft_model
    lora = LoraConfig(**LORA_CFG)
    # 关键: 固定 LoRA A 的随机初始化——所有进程同基（否则 cosine 跨进程不可比）
    torch.manual_seed(20260831)
    model = get_peft_model(model, lora)
    model = model.to(torch.bfloat16).cuda().train()   # train 模式（dropout=0 无随机性）
    # 冻结 base，只留 LoRA 可训练
    for n, p in model.named_parameters():
        p.requires_grad = 'lora' in n
    return model, tok
 
def grads_of(model, messages, action):
    """返回 state-level mean loss 对 LoRA 参数的梯度（flatten 1D）"""
    model.zero_grad()
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = tok(text, return_tensors='pt').to('cuda')
    target_text = TARGET_PREFIX + action
    target = tok(target_text, return_tensors='pt')['input_ids'].to('cuda')
    # 拼接 prompt + target（同训练: response + eos）
    full_ids = torch.cat([prompt['input_ids'], target, torch.tensor([[tok.eos_token_id]]).to('cuda')], dim=1)
    # 注意: 不能包 no_grad——backward 需要计算图（LoRA 参数 requires_grad=True）
    out = model(input_ids=full_ids, use_cache=False)
    logits = out.logits[0, prompt['input_ids'].shape[1]-1:-1, :]  # 对齐 target 位置
    target_ids = full_ids[0, prompt['input_ids'].shape[1]:]
    loss = F.cross_entropy(logits.float(), target_ids, reduction='mean')  # state-level mean
    if not loss.requires_grad:
        print(f'[WARN] loss no grad: logits.requires_grad={logits.requires_grad}, '
              f'P={prompt["input_ids"].shape[1]}, full_len={full_ids.shape[1]}', flush=True)
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().cpu()
    return g
 
def load_recs(path):
    import glob
    recs = []
    for f in glob.glob(path) if any(ch in path for ch in '*?') else [path]:
        for l in open(f):
            try:
                r = json.loads(l)
                if 'teacher_actions' in r:
                    for a in r['teacher_actions']:
                        if a['teacher_valid']:
                            recs.append((r['gamefile'], r['query_messages'], a['teacher_action']))
                elif r.get('teacher_valid'):
                    recs.append((r['gamefile'], r['query_messages'], r['teacher_action']))
            except Exception:
                pass
    return recs
 
if __name__ == '__main__':
    phase = sys.argv[1]
    model, tok = load_model()
    if phase == 'ref':
        recs = load_recs(PROBE)
        print(f'probe: {len(recs)}', flush=True)
        gs = []
        for i, (gf, msgs, act) in enumerate(recs):
            gs.append(grads_of(model, msgs, act))
            if (i + 1) % 10 == 0:
                print(f'  [{i+1}/{len(recs)}]', flush=True)
        G = torch.stack(gs).mean(0)
        torch.save(G, GREF_PATH)
        print(f'G_ref saved: {G.shape}, norm={G.norm().item():.4f}')
    elif phase == 'group':
        group = sys.argv[2]
        G = torch.load(GREF_PATH, weights_only=True)
        recs = load_recs(GROUPS[group])
        print(f'{group}: {len(recs)} corrections', flush=True)
        cos_all, neg = [], 0
        gs = []
        indiv = []
        for i, (gf, msgs, act) in enumerate(recs):
            g = grads_of(model, msgs, act)
            c = F.cosine_similarity(g.unsqueeze(0), G.unsqueeze(0)).item()
            cos_all.append(c)
            neg += int(c < 0)
            gs.append(g)
            indiv.append({'gamefile': gf, 'A_i': c})
            if (i + 1) % 20 == 0:
                print(f'  [{i+1}/{len(recs)}] meanA={sum(cos_all)/len(cos_all):.4f}', flush=True)
        GD = torch.stack(gs).mean(0)
        A_D = F.cosine_similarity(GD.unsqueeze(0), G.unsqueeze(0)).item()
        with open(f'/root/data/alfworld/opd/a1/a3/run1/m1_indiv_{group}.jsonl', 'w') as f:
            for d in indiv:
                f.write(json.dumps(d) + '\n')
        print(json.dumps({
            'group': group, 'n': len(recs),
            'mean_alignment': sum(cos_all) / len(cos_all),
            'neg_rate': neg / len(cos_all),
            'dataset_alignment': A_D,
            'G_D_norm': GD.norm().item(),
        }))
