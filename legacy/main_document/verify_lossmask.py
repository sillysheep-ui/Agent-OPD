#!/usr/bin/env python3
"""loss mask 端到端验证：FinalTurnOnlySFTDataset 在新格式数据上"""
import sys
sys.path.insert(0, '/cfs/data/private/yangchunyu/ld/agent_opd_route_a')
sys.path.insert(0, '/root/code/verl')
 
from transformers import AutoTokenizer
from final_turn_dataset import FinalTurnOnlySFTDataset
from torch.utils.data import DataLoader
 
tok = AutoTokenizer.from_pretrained('/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507', trust_remote_code=True)
 
ds = FinalTurnOnlySFTDataset(
    parquet_files='/root/data/alfworld/sft_v6/dryrun_train.parquet',
    tokenizer=tok,
    config={
        'truncation': 'left',
        'max_length': 4096,
        'multiturn': {'messages_key': 'messages',
                      'tools_key': 'tools',
                      'enable_thinking_key': 'enable_thinking'},
    },
)
print(f'dataset 大小: {len(ds)}')
 
checked = 0
for i in range(min(4, len(ds))):
    item = ds[i]
    loss_mask = item['loss_mask']
    tokens = item['input_ids']
    n = len(loss_mask)
    nz = [j for j in range(n) if loss_mask[j] == 1]
    print(f'\n样本 {i}: 总 token {n}, loss=1 区域: {len(nz)} token')
    if nz:
        # 解码 loss=1 区域
        seg = tokens[nz[0]:nz[-1] + 1]
        text = tok.decode(seg)
        print(f'  loss=1 解码: {text[:80]!r}')
        print(f'  区域: [{nz[0]}..{nz[-1]}] (前 5: {nz[:5]}, 后 5: {nz[-5:]})')
        # 检查前面是否有 loss
        if nz[0] > 0 and any(loss_mask[j] == 1 for j in range(0, nz[0])):
            print('  ⚠️ target 之前也有 loss!')
        else:
            print('  ✅ target 之前的 token 全部 loss=0')
    checked += 1
print(f'\n验证完成: {checked} 条')
