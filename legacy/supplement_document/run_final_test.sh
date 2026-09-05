#!/bin/bash
# Untouched held-out ID final test（方案 B——串行 4 模型 × 140 games_eval_id）
# 冻结规则: 一旦开始跑，不允许任何模型/协议修改；纯技术 bug 才重评（4 个全重）
cd /root/data/alfworld/scripts
export VLLM_USE_V1=0
mkdir -p /root/data/alfworld/opd/final_test
 
declare -A CKPT=(
  [sft]=/root/data/alfworld/checkpoints/sft_v6_merged_bf16
  [a1r42]=/root/data/alfworld/checkpoints/d3_a1_strong/global_step_108
  [ctrl42]=/root/data/alfworld/checkpoints/ctrl_seed42/global_step_93
  [a1mc42]=/root/data/alfworld/checkpoints/a1_mc_seed42/global_step_93
)
 
# SFT 是全量模型（无 LoRA）——单独处理
for name in sft a1r42 ctrl42 a1mc42; do
  rm -rf /tmp/tmp* 2>/dev/null
  echo "=== $(date '+%H:%M') start $name disk:$(df -h / | tail -1 | awk '{print $4}') ==="
  if [ "$name" = "sft" ]; then
    CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
      --model ${CKPT[$name]} --served-model-name sft \
      --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 \
      > /tmp/vllm_${name}.log 2>&1 &
    vpid=$!
    model_arg=sft
  else
    CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
      --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
      --enable-lora --lora-modules ${name}=${CKPT[$name]} \
      --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 \
      > /tmp/vllm_${name}.log 2>&1 &
    vpid=$!
    model_arg=$name
  fi
  for t in $(seq 1 40); do
    curl -s --max-time 3 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
    sleep 5
  done
  python3 -u /cfs/data/private/yangchunyu/ld/agent_opd_route_a/eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --split eval_in_distribution \
    --output /root/data/alfworld/opd/final_test/${name}_id140.json \
    --limit-games 140 --seed 42 \
    --model ${model_arg} --base-url http://127.0.0.1:8000/v1 \
    > /root/data/alfworld/opd/final_test/${name}_id140.log 2>&1
  ec=$?
  kill $vpid 2>/dev/null
  wait $vpid 2>/dev/null
  pkill -9 -f '[v]llm' 2>/dev/null   # 清理孤儿 engine 子进程（防 CUDA 泄漏）
  sleep 3
  rm -rf /tmp/tmp* /tmp/vllm_${name}.log 2>/dev/null
  echo "=== $(date '+%H:%M') done $name exit=$ec ==="
  ls -la /root/data/alfworld/opd/final_test/${name}_id140.json 2>/dev/null || echo "JSON MISSING $name"
done
echo "FINAL_TEST_DONE"
