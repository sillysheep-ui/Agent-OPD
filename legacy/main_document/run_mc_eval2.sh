#!/bin/bash
# A1 matched-count 134 评测（2 并发——每轮后清磁盘）
cd /root/data/alfworld/scripts
export VLLM_USE_V1=0
mkdir -p /root/data/alfworld/opd/a1/a3/eval/134/a1_mc
eval_one() {
  local name=$1 port=$2 gpu=$3
  local ckpt=$(ls -d /root/data/alfworld/checkpoints/${name}/global_step_* | sort -t_ -k3 -n | tail -1)
  rm -rf /tmp/tmp* 2>/dev/null
  echo "=== $(date '+%H:%M') start $name disk:$(df -h / | tail -1 | awk '{print $4}') ==="
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m vllm.entrypoints.openai.api_server \
    --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
    --enable-lora --lora-modules ${name}=${ckpt} \
    --max-model-len 8192 --gpu-memory-utilization 0.8 --port $port \
    > /tmp/vllm_${name}.log 2>&1 &
  local vpid=$!
  for t in $(seq 1 40); do
    curl -s --max-time 3 http://127.0.0.1:$port/v1/models > /dev/null 2>&1 && break
    sleep 5
  done
  python3 -u /cfs/data/private/yangchunyu/ld/agent_opd_route_a/eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/eval/134/a1_mc/${name}_134.json \
    --limit-games 134 --seed 42 \
    --model ${name} --base-url http://127.0.0.1:$port/v1 \
    > /root/data/alfworld/opd/a1/a3/eval/134/a1_mc/${name}_134.log 2>&1
  local ec=$?
  kill $vpid 2>/dev/null
  wait $vpid 2>/dev/null
  sleep 3
  rm -rf /tmp/tmp* /tmp/vllm_${name}.log 2>/dev/null
  echo "=== $(date '+%H:%M') done $name exit=$ec disk:$(df -h / | tail -1 | awk '{print $4}') ==="
  ls -la /root/data/alfworld/opd/a1/a3/eval/134/a1_mc/${name}_134.json 2>/dev/null || echo "JSON MISSING $name"
}
 
# 第一批（42, 7——GPU 0/1）
eval_one a1_mc_seed42 8000 0 &
eval_one a1_mc_seed7 8001 1 &
wait
# 第二批（123, 2024——GPU 0/1）
eval_one a1_mc_seed123 8000 0 &
eval_one a1_mc_seed2024 8001 1 &
wait
echo "MC_EVAL_DONE"
