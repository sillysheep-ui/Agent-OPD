#!/bin/bash
# A4 训练: 起点 merged_bf16 + A4 128 条 × 3 epochs（同配置）
cd /root/data/alfworld/scripts
export MODEL_PATH=/root/data/alfworld/checkpoints/sft_v6_merged_bf16
export TRAIN_FILES=/root/data/alfworld/opd/a1/a3/run1/a4_ce_train.parquet
export VAL_FILES=/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_val.parquet
export OUTPUT_DIR=/root/data/alfworld/checkpoints/a4_bounded
export EPOCHS=3
export LR=2e-5
export SAVE_FREQ=30
export HF_ENDPOINT=https://hf-mirror.com
export NUM_GPUS=1
export MODEL_DTYPE=bf16
CUDA_VISIBLE_DEVICES=0 bash /cfs/data/private/yangchunyu/ld/agent_opd_route_a/run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/a3/run1/a4_train.log 2>&1
echo "A4_TRAIN_DONE=$?"
