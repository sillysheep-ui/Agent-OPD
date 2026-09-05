#!/bin/bash
# seed42 补跑（Errno28 崩溃后——GPU 2 单发）
cd /root/data/alfworld/scripts
export VLLM_USE_V1=0
name=a1_mc_seed42
ckpt=/root/data/alfworld/checkpoints/a1_mc_seed42/global_step_93
rm -rf /tmp/tmp* 2>/dev/null
echo "=== $(date '+%H:%M') start $name (rerun) disk:$(df -h / | tail -1 | awk '{print $4}') ==="
CUDA_VISIBLE_DEVICES=2 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /root/data/alfworld/checkpoints/sft_v6_merged_bf16 \
  --enable-lora --lora-modules ${name}=${ckpt} \
  --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8002 \
  > /tmp/vllm_${name}.log 2>&1 &
vpid=$!
for t in $(seq 1 40); do
  curl -s --max-time 3 http://127.0.0.1:8002/v1/models > /dev/null 2>&1 && break
  sleep 5
done
python3 -u /cfs/data/private/yangchunyu/ld/agent_opd_route_a/eval_zero_shot_v6.py \
  --env-config /root/data/alfworld/configs/alfworld.yaml \
  --output /root/data/alfworld/opd/a1/a3/eval/134/a1_mc/a1_mc_seed42_134.json \
  --limit-games 134 --seed 42 \
  --model ${name} --base-url http://127.0.0.1:8002/v1 \
  > /root/data/alfworld/opd/a1/a3/eval/134/a1_mc/a1_mc_seed42_134.log 2>&1
ec=$?
kill $vpid 2>/dev/null
wait $vpid 2>/dev/null
sleep 3
rm -rf /tmp/tmp* /tmp/vllm_${name}.log 2>/dev/null
echo "=== $(date '+%H:%M') done $name exit=$ec ==="
ls -la /root/data/alfworld/opd/a1/a3/eval/134/a1_mc/${name}_134.json 2>/dev/null || echo "JSON MISSING $name"
echo "MC_RERUN42_DONE"
