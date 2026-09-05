#!/bin/bash
# A2 串行评测: 8 模型 × 134（每轮 1 vLLM + 1 评测 + 停——磁盘绝对安全）
cd /root/data/alfworld/scripts
export VLLM_USE_V1=0
mkdir -p /root/data/alfworld/opd/a1/a3/eval/134/a2
eval_one() {
  local name=$1 model=$2 ckpt=$3
  rm -rf /tmp/tmp* /tmp/vllm_*.log 2>/dev/null
  echo "=== $(date '+%H:%M') start $name disk:$(df -h / | tail -1 | awk '{print $4}') ==="
  CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
    --enable-lora --lora-modules ${model}=${ckpt} \
    --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 \
    > /tmp/vllm_${name}.log 2>&1 &
  local vpid=$!
  for t in $(seq 1 30); do
    curl -s --max-time 3 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
    sleep 5
  done
  python3 -u /cfs/data/private/yangchunyu/ld/agent_opd_route_a/eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/eval/134/a2/${name}_134.json \
    --limit-games 134 --seed 42 \
    --model ${model} --base-url http://127.0.0.1:8000/v1 \
    > /root/data/alfworld/opd/a1/a3/eval/134/a2/${name}_134.log 2>&1
  local ec=$?
  kill $vpid 2>/dev/null
  wait $vpid 2>/dev/null
  sleep 3
  echo "=== $(date '+%H:%M') done $name exit=$ec disk:$(df -h / | tail -1 | awk '{print $4}') ==="
  ls -la /root/data/alfworld/opd/a1/a3/eval/134/a2/${name}_134.json 2>/dev/null || echo "JSON MISSING $name"
}
 
# 8 模型（N=1 × 4 + N=3 × 4）——checkpoint 步数: N=1 用 final(102), N=3 用 final(306) 或 step_200
eval_one a2_n1_seed42 n142 $(ls -d /root/data/alfworld/checkpoints/a2_n1_seed42/global_step_* | sort -t_ -k3 -n | tail -1)
eval_one a2_n1_seed7 n17 $(ls -d /root/data/alfworld/checkpoints/a2_n1_seed7/global_step_* | sort -t_ -k3 -n | tail -1)
eval_one a2_n1_seed123 n1123 $(ls -d /root/data/alfworld/checkpoints/a2_n1_seed123/global_step_* | sort -t_ -k3 -n | tail -1)
eval_one a2_n1_seed2024 n12024 $(ls -d /root/data/alfworld/checkpoints/a2_n1_seed2024/global_step_* | sort -t_ -k3 -n | tail -1)
eval_one a2_n3_seed42 n342 $(ls -d /root/data/alfworld/checkpoints/a2_n3_seed42/global_step_* | sort -t_ -k3 -n | tail -1)
eval_one a2_n3_seed7 n37 $(ls -d /root/data/alfworld/checkpoints/a2_n3_seed7/global_step_* | sort -t_ -k3 -n | tail -1)
eval_one a2_n3_seed123 n3123 $(ls -d /root/data/alfworld/checkpoints/a2_n3_seed123/global_step_* | sort -t_ -k3 -n | tail -1)
eval_one a2_n3_seed2024 n32024 $(ls -d /root/data/alfworld/checkpoints/a2_n3_seed2024/global_step_* | sort -t_ -k3 -n | tail -1)
echo "A2_SERIAL_EVAL_DONE"
