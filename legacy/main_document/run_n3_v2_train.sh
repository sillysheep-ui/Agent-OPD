#!/bin/bash
# N=3 修复版重训（GPU1-3 并行 4 个——评测占 GPU0 不受影响）
cd /root/data/alfworld/scripts
export MODEL_PATH=/root/data/alfworld/checkpoints/sft_v6_merged_bf16
export VAL_FILES=/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_val.parquet
export EPOCHS=3
export LR=2e-5
export SAVE_FREQ=200
export HF_ENDPOINT=https://hf-mirror.com
export NUM_GPUS=1
export MODEL_DTYPE=bf16
seeds=(42 7 123 2024)
# 3 卡并行 4 个（第 4 个排队）
for i in 0 1 2; do
  seed=${seeds[$i]}
  (export CUDA_VISIBLE_DEVICES=$((i+1)) TRAIN_FILES=/root/data/alfworld/opd/a1/a3/run1/a2_v2_n3_seed${seed}_train.parquet OUTPUT_DIR=/root/data/alfworld/checkpoints/a2_n3_seed${seed}_v2; \
   bash /cfs/data/private/yangchunyu/ld/agent_opd_route_a/run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/a3/run1/a2_v2_n3_seed${seed}_train.log 2>&1) &
done
wait
# 第 4 个
(export CUDA_VISIBLE_DEVICES=1 TRAIN_FILES=/root/data/alfworld/opd/a1/a3/run1/a2_v2_n3_seed2024_train.parquet OUTPUT_DIR=/root/data/alfworld/checkpoints/a2_n3_seed2024_v2; \
 bash /cfs/data/private/yangchunyu/ld/agent_opd_route_a/run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/a3/run1/a2_v2_n3_seed2024_train.log 2>&1)
echo "N3_V2_TRAIN_DONE"
