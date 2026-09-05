#!/bin/bash
# 134 OOD paired formal eval: SFT_v6 / A1-OPD / Control 同一任务集
cd /cfs/data/private/yangchunyu/ld/agent_opd_route_a
for m in sft-v6 a1-opd control; do
  python3 -u eval_zero_shot_v6.py \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/eval_134/${m}_134.json \
    --limit-games 134 --seed 42 \
    --model ${m} --base-url http://127.0.0.1:8000/v1 \
    > /root/data/alfworld/opd/a1/eval_134/${m}_134.log 2>&1
  echo "${m}_134_DONE exit=$?"
done
echo "ALL_134_DONE"
