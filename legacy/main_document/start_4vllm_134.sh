#!/bin/bash
# 4 vLLM + 4 模型 134 OOD repeated eval
export VLLM_USE_V1=0
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora --lora-modules sft-v6=/root/data/alfworld/checkpoints/sft_v6/global_step_542 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 > /tmp/vllm_sft6.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
  --enable-lora --lora-modules a1s=/root/data/alfworld/checkpoints/d3_a1_strong/global_step_108 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8001 > /tmp/vllm_a1s.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
  --enable-lora --lora-modules ctrls=/root/data/alfworld/checkpoints/d3_ctrl_strong/global_step_108 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8002 > /tmp/vllm_ctrls.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
  --enable-lora --lora-modules dis=/root/data/alfworld/checkpoints/d3_a1_disagree/global_step_21 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8003 > /tmp/vllm_dis.log 2>&1 &
echo "4 vllm started (8000-8003)"
