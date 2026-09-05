#!/usr/bin/env bash
set -euo pipefail
 
# Route-A SFT v6: FinalTurnOnlySFTDataset + multiturn config (veRL 0.4.1)
# data.multiturn.* keys (NOT flat data.messages_key)
 
VERL_ROOT=${VERL_ROOT:-/root/code/verl}
MODEL_PATH=${MODEL_PATH:-/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507}
TRAIN_FILES=${TRAIN_FILES:-/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_train.parquet}
VAL_FILES=${VAL_FILES:-/root/data/alfworld/sft_v6/alfworld_teacher_sft_v6_val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-/root/data/alfworld/checkpoints/sft_v6}
NUM_GPUS=${NUM_GPUS:-4}
LR=${LR:-2e-4}
EPOCHS=${EPOCHS:-2}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TRAIN_BSZ=${TRAIN_BSZ:-4}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
SAVE_FREQ=${SAVE_FREQ:-50}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
 
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export TOKENIZERS_PARALLELISM=true
 
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
 
cd "${VERL_ROOT}"
 
ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.fsdp_sft_trainer"}
MODEL_KEY=${MODEL_KEY:-"model.partial_pretrain"}
CUSTOM_DATASET=${CUSTOM_DATASET:-${SCRIPT_DIR}/final_turn_dataset.py}
 
mkdir -p "${OUTPUT_DIR}"
 
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" ${ENTRYPOINT} \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.multiturn.tools_key=tools \
  data.multiturn.enable_thinking_key=enable_thinking \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=left \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
  data.custom_cls.path="${CUSTOM_DATASET}" \
  data.custom_cls.name=FinalTurnOnlySFTDataset \
  "${MODEL_KEY}"="${MODEL_PATH}" \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.target_modules=all-linear \
  model.fsdp_config.model_dtype="${MODEL_DTYPE:-fp32}" \
  optim.lr="${LR}" \
  optim.warmup_steps_ratio="${WARMUP_RATIO}" \
  trainer.total_epochs="${EPOCHS}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name=agent-omniopd \
  trainer.experiment_name=sft-v6 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.logger='["console"]' \
  "$@"
