#!/bin/bash
# M3 v3 主配置: K=10 both（A1/A3 并行）
(export CUDA_VISIBLE_DEVICES=0; python3 -u /root/data/alfworld/scripts/m3v3_transfer_retention.py A1 10 both > /tmp/m3v3_a1.log 2>&1) &
P1=$!
(export CUDA_VISIBLE_DEVICES=1; python3 -u /root/data/alfworld/scripts/m3v3_transfer_retention.py A3 10 both > /tmp/m3v3_a3.log 2>&1) &
P2=$!
wait $P1 $P2
echo "M3V3_DONE"
