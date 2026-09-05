#!/bin/bash
# 3 seed 60 dev 评测（并行）
cd /root/data/alfworld/scripts
mkdir -p /root/data/alfworld/opd/a1/a3/eval
run_eval() {
  local name=$1 port=$2 model=$3
  python3 -u /cfs/data/private/yangchunyu/ld/agent_opd_route_a/eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/eval/${name}_60.json \
    --limit-games 60 --seed 42 \
    --model ${model} --base-url http://127.0.0.1:${port}/v1 \
    > /root/data/alfworld/opd/a1/a3/eval/${name}_60.log 2>&1
}
run_eval a1_seed7 8000 s7 &
P1=$!
run_eval a1_seed123 8001 s123 &
P2=$!
run_eval a1_seed2024 8002 s2024 &
P3=$!
wait $P1 $P2 $P3
echo "SEED_EVAL_DONE"
