#!/bin/bash
# A3 正式 Phase 2: 4 进程并行（每 GPU 跑 1/4 games 的评分 + Teacher）
cd /root/data/alfworld/scripts
# 拆 episodes 为 4 份
python3 - << 'EOF'
import json
lines = open('/root/data/alfworld/opd/a1/a3/run1/episodes.jsonl').readlines()
n = len(lines)
print(f'episodes: {n}')
for i in range(4):
    part = lines[i*n//4:(i+1)*n//4] if i < 3 else lines[3*n//4:]
    with open(f'/root/data/alfworld/opd/a1/a3/run1/episodes_{i}.jsonl', 'w') as f:
        f.writelines(part)
    print(f'  part{i}: {len(part)}')
EOF
# 4 并行评分+Teacher（每进程输出自己 shard 的 corrections）
for i in 0 1 2 3; do
  python3 -u a3_collect.py \
    --phase score \
    --env-config /root/data/alfworld/configs/alfworld.yaml \
    --output /root/data/alfworld/opd/a1/a3/run1/shard${i} \
    --episodes-input /root/data/alfworld/opd/a1/a3/run1/episodes_${i}.jsonl \
    --only-games-jsonl /root/data/alfworld/opd/a1/run1/corrections.jsonl \
    --gpu ${i} \
    > /root/data/alfworld/opd/a1/a3/run1/phase2_shard${i}.log 2>&1 &
done
wait
echo "A3_PHASE2_ALL_DONE"
