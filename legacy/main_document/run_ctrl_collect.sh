#!/bin/bash
# CTRL 4-seed 采集（每 seed 1 进程——串行 50 games）
cd /root/data/alfworld/scripts
for s in 42 7 123 2024; do
  (python3 -u ctr_multiseed.py $s > /tmp/ctrl_${s}.log 2>&1) &
done
wait
echo "CTRL_COLLECT_DONE"
