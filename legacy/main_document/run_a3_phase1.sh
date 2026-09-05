#!/bin/bash
# A3 正式 Phase 1: rollout 50 games（vLLM 8000 sft-v6）
cd /root/data/alfworld/scripts
python3 -u a3_collect.py \
  --phase rollout \
  --env-config /root/data/alfworld/configs/alfworld.yaml \
  --output /root/data/alfworld/opd/a1/a3/run1 \
  --only-games-jsonl /root/data/alfworld/opd/a1/run1/corrections.jsonl \
  --student-model sft-v6 \
  --student-base-url http://127.0.0.1:8000/v1 \
  > /root/data/alfworld/opd/a1/a3/run1_phase1.log 2>&1
echo "A3_PHASE1_DONE exit=$?"
