#!/bin/bash
# 3 seed vLLM（显式 checkpoint 映射）
export VLLM_USE_V1=0
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
  --enable-lora --lora-modules s7=/root/data/alfworld/checkpoints/a1_seed7/global_step_108 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 > /tmp/vllm_seed7.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
  --enable-lora --lora-modules s123=/root/data/alfworld/checkpoints/a1_seed123/global_step_108 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8001 > /tmp/vllm_seed123.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
  --enable-lora --lora-modules s2024=/root/data/alfworld/checkpoints/a1_seed2024/global_step_111 \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8002 > /tmp/vllm_seed2024.log 2>&1 &
echo "3 seed vllm started"
