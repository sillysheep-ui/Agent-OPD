#!/bin/bash
# 4 路 60 dev 评测（每个模型 60 games）
cd /cfs/data/private/yangchunyu/ld/agent_opd_route_a
mkdir -p /root/data/alfworld/opd/a1/d3_eval
run_eval() {
  local name=$1 port=$2 model=$3
  python3 -u eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/d3_eval/${name}_60.json \
    --limit-games 60 --seed 42 \
    --model ${model} --base-url http://127.0.0.1:${port}/v1 \
    > /root/data/alfworld/opd/a1/d3_eval/${name}_60.log 2>&1
}
run_eval sft_v6 8000 sft-v6 &
P1=$!
run_eval a1_strong 8001 a1s &
P2=$!
run_eval ctrl_strong 8002 ctrls &
P3=$!
run_eval a1_disagree 8003 dis &
P4=$!
wait $P1 $P2 $P3 $P4
echo "D3_EVAL_DONE"
