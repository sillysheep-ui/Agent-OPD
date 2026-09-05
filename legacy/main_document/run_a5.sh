#!/bin/bash
# A5 采集: mixed 选点（1 random + 2 bounded[0.6,0.8]），复用 A3 episodes
cd /root/data/alfworld/scripts
for i in 0 1 2 3; do
  (export CUDA_VISIBLE_DEVICES=${i}; python3 -u a3_collect.py \
    --phase score \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/run1/a5_shard${i} \
    --episodes-input /root/data/alfworld/opd/a1/a3/run1/episodes_${i}.jsonl \
    --only-games-jsonl /root/data/alfworld/opd/a1/run1/corrections.jsonl \
    --selection mixed \
    --bounded-lo 0.6 --bounded-hi 0.8 \
    --gpu 0 \
    > /root/data/alfworld/opd/a1/a3/run1/a5_shard${i}.log 2>&1) &
done
wait
echo "A5_ALL_DONE"
