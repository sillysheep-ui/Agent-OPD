#!/bin/bash
# M3 v2: A1/A3 真并行（2 卡）
rm -f /root/data/alfworld/opd/a1/a3/run1/m3v2_result.jsonl
(export CUDA_VISIBLE_DEVICES=0; python3 -u /root/data/alfworld/scripts/m3v2_transfer_retention.py A1 > /tmp/m3v2_a1.log 2>&1) &
P1=$!
(export CUDA_VISIBLE_DEVICES=1; python3 -u /root/data/alfworld/scripts/m3v2_transfer_retention.py A3 > /tmp/m3v2_a3.log 2>&1) &
P2=$!
wait $P1 $P2
echo "M3V2_DONE"
