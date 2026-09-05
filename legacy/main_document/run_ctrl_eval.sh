#!/bin/bash
# CTRL 134 评测（串行 4 个——磁盘安全）
cd /root/data/alfworld/scripts
export VLLM_USE_V1=0
mkdir -p /root/data/alfworld/opd/a1/a3/eval/134/ctrl
eval_one() {
  local name=$1 ckpt=$2
  rm -rf /tmp/tmp* /tmp/vllm_*.log 2>/dev/null
  echo "=== $(date '+%H:%M') start $name disk:$(df -h / | tail -1 | awk '{print $4}') ==="
  CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
    --enable-lora --lora-modules ${name}=${ckpt} \
    --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 \
    > /tmp/vllm_${name}.log 2>&1 &
  local vpid=$!
  for t in $(seq 1 30); do
    curl -s --max-time 3 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
    sleep 5
  done
  python3 -u /cfs/data/private/yangchunyu/ld/agent_opd_route_a/eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/eval/134/ctrl/${name}_134.json \
    --limit-games 134 --seed 42 \
    --model ${name} --base-url http://127.0.0.1:8000/v1 \
    > /root/data/alfworld/opd/a1/a3/eval/134/ctrl/${name}_134.log 2>&1
  local ec=$?
  kill $vpid 2>/dev/null
  wait $vpid 2>/dev/null
  sleep 3
  echo "=== $(date '+%H:%M') done $name exit=$ec ==="
  ls -la /root/data/alfworld/opd/a1/a3/eval/134/ctrl/${name}_134.json 2>/dev/null || echo "JSON MISSING $name"
}
 
eval_one ctrl_seed42 /root/data/alfworld/checkpoints/ctrl_seed42/global_step_93
eval_one ctrl_seed7 /root/data/alfworld/checkpoints/ctrl_seed7/global_step_96
eval_one ctrl_seed123 /root/data/alfworld/checkpoints/ctrl_seed123/global_step_93
eval_one ctrl_seed2024 /root/data/alfworld/checkpoints/ctrl_seed2024/global_step_96
echo "CTRL_EVAL_DONE"
