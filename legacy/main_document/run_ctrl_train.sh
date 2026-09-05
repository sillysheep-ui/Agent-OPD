#!/bin/bash
# CTRL 4-seed 训练（与 A1 同配置: 3ep/lr 2e-5/LoRA r16/batch 4/merged_bf16）
cd /root/data/alfworld/scripts
export MODEL_PATH=/root/data/alfworld/checkpoints/sft_v6_merged_bf16
export VAL_FILES=/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_val.parquet
export EPOCHS=3
export LR=2e-5
export SAVE_FREQ=30
export HF_ENDPOINT=https://hf-mirror.com
export NUM_GPUS=1
export MODEL_DTYPE=bf16
for i in 0 1 2 3; do
  seeds=(42 7 123 2024)
  seed=${seeds[$i]}
  (export CUDA_VISIBLE_DEVICES=$i TRAIN_FILES=/root/data/alfworld/opd/a1/a3/run1/ctrl_seed${seed}_train.parquet OUTPUT_DIR=/root/data/alfworld/checkpoints/ctrl_seed${seed}; \
   bash /cfs/data/private/yangchunyu/ld/agent_opd_route_a/run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/a3/run1/ctrl_seed${seed}_train.log 2>&1) &
done
wait
echo "CTRL_TRAIN_DONE"
