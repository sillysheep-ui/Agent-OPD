#!/usr/bin/env python3
"""合并 SFT_v6 LoRA 到 base，输出完整模型"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
BASE = '/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507'
LORA = '/root/data/alfworld/checkpoints/sft_v6/global_step_542'
OUT = '/root/data/alfworld/checkpoints/sft_v6_merged'
 
print('loading base...')
model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.float16)
print('loading lora...')
model = PeftModel.from_pretrained(model, LORA)
print('merging...')
model = model.merge_and_unload()
print('saving...')
model.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.save_pretrained(OUT)
print(f'merged model saved to {OUT}')
