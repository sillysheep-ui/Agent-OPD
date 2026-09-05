#!/bin/bash
# vLLM: Qwen3-4B-Instruct-2507 base + sft_v6 LoRA (global_step_542)
export VLLM_USE_V1=0
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --lora-modules sft-v6=/root/data/alfworld/checkpoints/sft_v6/global_step_542 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  > /tmp/vllm_sft_v6.log 2>&1 &
echo "vllm started pid=$!"
