#!/bin/bash
# 3 seed × 134 OOD repeated（3 卡并行，vLLM 已在 8000-8002）
cd /root/data/alfworld/scripts
mkdir -p /root/data/alfworld/opd/a1/a3/eval/134
run_eval() {
  local name=$1 port=$2 model=$3
  python3 -u /cfs/data/private/yangchunyu/ld/agent_opd_route_a/eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/eval/134/${name}_134.json \
    --limit-games 134 --seed 42 \
    --model ${model} --base-url http://127.0.0.1:${port}/v1 \
    > /root/data/alfworld/opd/a1/a3/eval/134/${name}_134.log 2>&1
}
run_eval a1_seed7 8000 s7 &
P1=$!
run_eval a1_seed123 8001 s123 &
P2=$!
run_eval a1_seed2024 8002 s2024 &
P3=$!
wait $P1 $P2 $P3
echo "SEED134_DONE"
