#!/bin/bash
# 60 OOD dev: A1-OPD vs Control（并行）
cd /cfs/data/private/yangchunyu/ld/agent_opd_route_a
python3 -u eval_zero_shot_v6.py \
  --env-config /root/data/alfworld/configs/alfworld.yaml \
  --output /root/data/alfworld/opd/a1/eval_dev/a1_opd_60.json \
  --limit-games 60 --seed 42 \
  --model a1-opd --base-url http://127.0.0.1:8000/v1 \
  > /root/data/alfworld/opd/a1/eval_dev/a1_opd_60.log 2>&1 &
python3 -u eval_zero_shot_v6.py \
  --env-config /root/data/alfworld/configs/alfworld.yaml \
  --output /root/data/alfworld/opd/a1/eval_dev/control_60.json \
  --limit-games 60 --seed 42 \
  --model control --base-url http://127.0.0.1:8000/v1 \
  > /root/data/alfworld/opd/a1/eval_dev/control_60.log 2>&1 &
wait
echo "BOTH_EVAL_DONE"
