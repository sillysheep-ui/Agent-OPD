#!/bin/bash
# Host-side launcher for the standard (traditional) per-token OPD arm.
#
# The Student and the frozen Teacher run in one container and are split across
# disjoint GPU sets, because veRL allocates a separate resource pool per role:
#   GPUS 0..NUM_GPUS-1              -> Student pool (rollout + FSDP actor)
#   GPUS NUM_GPUS..NUM_GPUS+TEACHER_GPUS-1 -> Teacher pool (TP=TEACHER_TP)
# The objective does not depend on this split; only wall-clock does. Global
# batch, micro batch per GPU, learning rate, and optimizer steps are fixed here
# so two GPU layouts stay comparable.
#
# Everything is overridable; the defaults reproduce the 2026-09-23 run.
set -euo pipefail

RUN=${RUN:?run name required}
ROOT=${ROOT:-/data/yangchunyu/ld/omniopd_runs}
OUT=${OUT:-$ROOT/$RUN}
LOG=${LOG:-$ROOT/$RUN.host.log}
IMG=${IMG:-omniopd-verl080:vllm010-td010}
NAME=${NAME:-omniopd-opd-standard}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-4}
TEACHER_GPUS=${TEACHER_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-2}
TRAIN_BSZ=${TRAIN_BSZ:-64}
MICRO_BSZ=${MICRO_BSZ:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
LR=${LR:-1e-6}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-86}
SAVE_FREQ=${SAVE_FREQ:-43}
VERL_ROOT=${VERL_ROOT:-/data/yangchunyu/ld/verl-v0.8.0}
AGENT_ROOT=${AGENT_ROOT:-/data/yangchunyu/ld/agent_omniopd_v080}
STUDENT_MODEL=${STUDENT_MODEL:-/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507}
TEACHER_MODEL=${TEACHER_MODEL:-/cfs/data/private/zhangsl/Model/Qwen/Qwen3-14B}
OPD_DATA=${OPD_DATA:-$ROOT/opd_train_20260922_01/rows.jsonl}
OPD_DATA_MANIFEST=${OPD_DATA_MANIFEST:-$ROOT/opd_train_20260922_01/rows.manifest.json}

if [ -e "$OUT" ]; then
  echo "refusing to reuse an existing output directory: $OUT" >&2
  exit 2
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" --network host --runtime=nvidia --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e NUM_GPUS="$NUM_GPUS" -e TEACHER_GPUS="$TEACHER_GPUS" -e TEACHER_TP="$TEACHER_TP" \
  -e TRAIN_BSZ="$TRAIN_BSZ" -e MICRO_BSZ="$MICRO_BSZ" \
  -e MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" -e MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
  -e LR="$LR" -e TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
  -e SAVE_FREQ="$SAVE_FREQ" \
  -e VERL_ROOT=/opt/verl \
  -e OPD_DATA="$OPD_DATA" -e OPD_DATA_MANIFEST="$OPD_DATA_MANIFEST" \
  -e OPD_STUDENT_MODEL=/models/student -e OPD_TEACHER_MODEL=/models/teacher \
  -e OPD_OUTPUT="/runs/$RUN" \
  -v "$VERL_ROOT":/opt/verl:ro \
  -v "$AGENT_ROOT":/opt/agent:ro \
  -v "$STUDENT_MODEL":/models/student:ro \
  -v "$TEACHER_MODEL":/models/teacher:ro \
  -v "$ROOT":/runs \
  -w /opt/agent --entrypoint bash "$IMG" \
  -c 'bash scripts/run_verl_opd_train.sh' > /dev/null

echo "training container started: $RUN -> $OUT (GPUs $GPUS)" > "$LOG"
docker logs -f "$NAME" >> "$LOG" 2>&1
