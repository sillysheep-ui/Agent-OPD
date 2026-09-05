#!/usr/bin/env python3
"""合并 a1s(step108) LoRA 到 merged_bf16 -> merged_a1s_bf16（第二轮起点）"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
BASE = '/root/data/alfworld/checkpoints/sft_v6_merged_bf16'
LORA = '/root/data/alfworld/checkpoints/d3_a1_strong/global_step_108'
OUT = '/root/data/alfworld/checkpoints/sft_v6_merged_a1s_bf16'
 
print('loading merged_bf16...')
model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.bfloat16)
print('loading a1s lora...')
model = PeftModel.from_pretrained(model, LORA)
model = model.merge_and_unload().to(torch.bfloat16)
print('saving...')
model.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.save_pretrained(OUT)
print(f'saved to {OUT}')
