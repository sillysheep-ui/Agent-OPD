#!/bin/bash
# vLLM: base + A1 LoRA + Control LoRA（一次服务两个模型）
export VLLM_USE_V1=0
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --lora-modules a1-opd=/root/data/alfworld/checkpoints/a1_opd/global_step_36 control=/root/data/alfworld/checkpoints/a1_control/global_step_36 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 \
  > /tmp/vllm_a1.log 2>&1 &
echo "vllm_a1 started pid=$!"
