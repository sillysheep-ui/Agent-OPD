#!/bin/bash
# D3 并行: 3 训练 × 1 GPU（bf16）
cd /cfs/data/private/yangchunyu/ld/agent_opd_route_a
export MODEL_PATH=/root/data/alfworld/checkpoints/sft_v6_merged
export VAL_FILES=/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_val.parquet
export EPOCHS=3
export LR=2e-5
export SAVE_FREQ=30
export HF_ENDPOINT=https://hf-mirror.com
export NUM_GPUS=1
export MODEL_DTYPE=bf16
 
CUDA_VISIBLE_DEVICES=0 TRAIN_FILES=/root/data/alfworld/opd/a1/run1/a1_ce_train.parquet \
  OUTPUT_DIR=/root/data/alfworld/checkpoints/d3_a1_strong \
  bash run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/run1/d3_a1_strong.log 2>&1 &
P1=$!
 
CUDA_VISIBLE_DEVICES=1 TRAIN_FILES=/root/data/alfworld/opd/a1/run1/a1_ce_control_train.parquet \
  OUTPUT_DIR=/root/data/alfworld/checkpoints/d3_ctrl_strong \
  bash run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/run1/d3_ctrl_strong.log 2>&1 &
P2=$!
 
CUDA_VISIBLE_DEVICES=2 TRAIN_FILES=/root/data/alfworld/opd/a1/run1/a1_disagree_train.parquet \
  OUTPUT_DIR=/root/data/alfworld/checkpoints/d3_a1_disagree \
  bash run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/run1/d3_a1_disagree.log 2>&1 &
P3=$!
 
echo "parallel training started: $P1 $P2 $P3"
wait $P1 $P2 $P3
echo "ALL_D3_PARALLEL_DONE"
