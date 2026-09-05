#!/bin/bash
# A4 采集: 复用 A3 episodes + 重算全量 entropy（持久化）+ bounded [0.6,0.9] 选点 + Teacher
cd /root/data/alfworld/scripts
for i in 0 1 2 3; do
  python3 -u a3_collect.py \
    --phase score \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/run1/a4_shard${i} \
    --episodes-input /root/data/alfworld/opd/a1/a3/run1/episodes_${i}.jsonl \
    --only-games-jsonl /root/data/alfworld/opd/a1/run1/corrections.jsonl \
    --selection bounded \
    --bounded-lo 0.6 --bounded-hi 0.9 \
    --gpu ${i} \
    > /root/data/alfworld/opd/a1/a3/run1/a4_shard${i}.log 2>&1 &
done
wait
echo "A4_ALL_DONE"
