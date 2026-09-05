#!/bin/bash
# D3: 3 个训练串行（同配置 3 epochs, lr 2e-5, 从 merged 出发）
cd /cfs/data/private/yangchunyu/ld/agent_opd_route_a
export MODEL_PATH=/root/data/alfworld/checkpoints/sft_v6_merged
export VAL_FILES=/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_val.parquet
export EPOCHS=3
export LR=2e-5
export SAVE_FREQ=30
export HF_ENDPOINT=https://hf-mirror.com
export NUM_GPUS=4
 
echo "=== 1/3 A1-full-strong ==="
export TRAIN_FILES=/root/data/alfworld/opd/a1/run1/a1_ce_train.parquet
export OUTPUT_DIR=/root/data/alfworld/checkpoints/d3_a1_strong
bash run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/run1/d3_a1_strong.log 2>&1
echo "A1_STRONG_DONE=$?"
 
echo "=== 2/3 CTRL-full-strong ==="
export TRAIN_FILES=/root/data/alfworld/opd/a1/run1/a1_ce_control_train.parquet
export OUTPUT_DIR=/root/data/alfworld/checkpoints/d3_ctrl_strong
bash run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/run1/d3_ctrl_strong.log 2>&1
echo "CTRL_STRONG_DONE=$?"
 
echo "=== 3/3 A1-disagree-only ==="
export TRAIN_FILES=/root/data/alfworld/opd/a1/run1/a1_disagree_train.parquet
export OUTPUT_DIR=/root/data/alfworld/checkpoints/d3_a1_disagree
bash run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/run1/d3_a1_disagree.log 2>&1
echo "DISAGREE_DONE=$?"
 
echo "ALL_D3_DONE"
