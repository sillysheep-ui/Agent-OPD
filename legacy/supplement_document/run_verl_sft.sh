#!/usr/bin/env bash
set -euo pipefail
 
# Route-A OPD update using veRL SFT engine and FinalTurnOnlySFTDataset.
# Defaults are deliberately conservative for the first reality check.
 
VERL_ROOT=${VERL_ROOT:-/root/code/verl}
MODEL_PATH=${MODEL_PATH:-/cfs/data/private/yangchunyu/ld/models/qwen3-4b-sft}
TRAIN_FILES=${TRAIN_FILES:-/root/data/alfworld/opd/route_a_round0/train.parquet}
VAL_FILES=${VAL_FILES:-/root/data/alfworld/opd/route_a_round0/val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-/root/data/alfworld/checkpoints/opd_route_a_round0}
NUM_GPUS=${NUM_GPUS:-4}
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-1}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TRAIN_BSZ=${TRAIN_BSZ:-4}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
SAVE_FREQ=${SAVE_FREQ:-10}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
 
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export TOKENIZERS_PARALLELISM=true
# If you need the HF mirror, export HF_ENDPOINT before invoking this script.
 
cd "${VERL_ROOT}"
 
# veRL 0.4.1 on this server only exposes fsdp_sft_trainer, which reads
# model.partial_pretrain (NOT model.path). Newer upstream uses
# verl.trainer.sft_trainer + model.path — override ENTRYPOINT/MODEL_KEY for that.
ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.fsdp_sft_trainer"}
MODEL_KEY=${MODEL_KEY:-"model.partial_pretrain"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUSTOM_DATASET=${CUSTOM_DATASET:-${SCRIPT_DIR}/final_turn_dataset.py}
 
mkdir -p "${OUTPUT_DIR}"
 
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" ${ENTRYPOINT} \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.messages_key=messages \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
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
  optim.lr="${LR}" \
  optim.warmup_steps_ratio="${WARMUP_RATIO}" \
  trainer.total_epochs="${EPOCHS}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name=agent-omniopd \
  trainer.experiment_name=route-a-round0 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.logger='["console"]' \
  "$@"
