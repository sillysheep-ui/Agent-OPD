#!/bin/bash
# A2 训练: 8 个（N=1/N=3 × 4 seed），4 卡 2 轮
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
 
# 轮 1: N=1（4 个）
for i in 0 1 2 3; do
  seed=${seeds[$i]}
  (export CUDA_VISIBLE_DEVICES=$i TRAIN_FILES=/root/data/alfworld/opd/a1/a3/run1/a2_n1_seed${seed}_train.parquet OUTPUT_DIR=/root/data/alfworld/checkpoints/a2_n1_seed${seed}; \
   bash /cfs/data/private/yangchunyu/ld/agent_opd_route_a/run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/a3/run1/a2_n1_seed${seed}_train.log 2>&1) &
done
wait
echo "A2_N1_TRAIN_DONE"
 
# 轮 2: N=3（4 个）
for i in 0 1 2 3; do
  seed=${seeds[$i]}
  (export CUDA_VISIBLE_DEVICES=$i TRAIN_FILES=/root/data/alfworld/opd/a1/a3/run1/a2_n3_seed${seed}_train.parquet OUTPUT_DIR=/root/data/alfworld/checkpoints/a2_n3_seed${seed}; \
   bash /cfs/data/private/yangchunyu/ld/agent_opd_route_a/run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/a3/run1/a2_n3_seed${seed}_train.log 2>&1) &
done
wait
echo "A2_N3_TRAIN_DONE"
