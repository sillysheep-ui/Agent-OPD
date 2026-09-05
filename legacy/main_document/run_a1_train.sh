#!/bin/bash
# A1 OPD update: FinalTurnOnly CE on 145 student-state corrections
cd /cfs/data/private/yangchunyu/ld/agent_opd_route_a
export MODEL_PATH=/root/data/alfworld/checkpoints/sft_v6_merged
export TRAIN_FILES=/root/data/alfworld/opd/a1/run1/a1_ce_train.parquet
export VAL_FILES=/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_val.parquet
export OUTPUT_DIR=/root/data/alfworld/checkpoints/a1_opd
export EPOCHS=1
export LR=2e-5
export SAVE_FREQ=10
export HF_ENDPOINT=https://hf-mirror.com
export NUM_GPUS=4
bash run_verl_sft_v6.sh > /root/data/alfworld/opd/a1/run1/a1_train.log 2>&1
echo "A1_TRAIN_DONE exit=$?"
