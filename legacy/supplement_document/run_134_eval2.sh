#!/bin/bash
# 134 OOD paired eval: 三个模型（双 vLLM 并行）
cd /cfs/data/private/yangchunyu/ld/agent_opd_route_a
mkdir -p /root/data/alfworld/opd/a1/eval_134
# sft-v6 on GPU0 (base)
python3 -u eval_zero_shot_v6.py \
  --env-config /root/data/alfworld/configs/alfworld.yaml \
  --output /root/data/alfworld/opd/a1/eval_134/sft_v6_134.json \
  --limit-games 134 --seed 42 \
  --model sft-v6 --base-url http://127.0.0.1:8000/v1 \
  > /root/data/alfworld/opd/a1/eval_134/sft_v6_134.log 2>&1 &
# a1-opd + control on GPU1 (merged)
python3 -u eval_zero_shot_v6.py \
  --env-config /root/data/alfworld/configs/alfworld.yaml \
  --output /root/data/alfworld/opd/a1/eval_134/a1_opd_134.json \
  --limit-games 134 --seed 42 \
  --model a1-opd --base-url http://127.0.0.1:8001/v1 \
  > /root/data/alfworld/opd/a1/eval_134/a1_opd_134.log 2>&1 &
python3 -u eval_zero_shot_v6.py \
  --env-config /root/data/alfworld/configs/alfworld.yaml \
  --output /root/data/alfworld/opd/a1/eval_134/control_134.json \
  --limit-games 134 --seed 42 \
  --model control --base-url http://127.0.0.1:8001/v1 \
  > /root/data/alfworld/opd/a1/eval_134/control_134.log 2>&1 &
wait
echo "ALL_134_DONE"
