#!/bin/bash
# Exercise the whole post-training chain on a two-step run before any long run.
#
# The 2026-09-23 losses were both end-of-run surprises: a launcher that saved
# nothing, and a converter that could not read a DTensor shard. Each cost a full
# training budget because it was only discovered after the run. This script
# spends a few minutes to prove, in order, that a short run writes a checkpoint
# and a completion manifest, that the adapter converts with base-shape checks,
# and that the adapter merges into a servable model without changing it.
#
# Exit code 0 means the long run's post-processing path is verified.
set -uo pipefail

ROOT=${ROOT:-/data/yangchunyu/ld/omniopd_runs}
RUN=${RUN:-opd_preflight_$(date -u +%Y%m%d_%H%M%S)}
AGENT_ROOT=${AGENT_ROOT:-/data/yangchunyu/ld/agent_omniopd_v080}
TRAIN_IMG=${TRAIN_IMG:-omniopd-verl080:vllm010-td010}
EVAL_IMG=${EVAL_IMG:-omniopd-verl080:alfworld}
BASE=${BASE:-/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507}
GPUS=${GPUS:-0,1,2,3}
STEPS=${STEPS:-2}
LOG=$ROOT/$RUN.preflight.log
OUT=$ROOT/$RUN
ADAPTER=$ROOT/opd_adapter_$RUN
MERGED=$ROOT/opd_merged_$RUN

echo "preflight started $(date -u) for $RUN" > "$LOG"

env RUN="$RUN" LOG="$ROOT/$RUN.host.log" GPUS="$GPUS" NUM_GPUS=2 TEACHER_GPUS=2 \
  TEACHER_TP=2 TOTAL_TRAINING_STEPS="$STEPS" SAVE_FREQ="$STEPS" \
  bash "$AGENT_ROOT/scripts/host/run_opd_standard.sh" >> "$LOG" 2>&1 &

for _ in $(seq 1 240); do
  if [ -f "$OUT/completion_manifest.json" ]; then
    echo "1/3 short run wrote its completion manifest $(date -u)" >> "$LOG"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^omniopd-opd-standard$'; then
    echo "FAIL: the short run exited without a completion manifest $(date -u)" >> "$LOG"
    exit 1
  fi
  sleep 15
done
if [ ! -f "$OUT/completion_manifest.json" ]; then
  echo "FAIL: timed out waiting for the short run $(date -u)" >> "$LOG"
  exit 1
fi
for _ in $(seq 1 60); do
  docker ps --format '{{.Names}}' | grep -q '^omniopd-opd-standard$' || break
  sleep 10
done
if [ ! -d "$OUT/checkpoints/global_step_$STEPS" ]; then
  echo "FAIL: no checkpoint at $OUT/checkpoints/global_step_$STEPS" >> "$LOG"
  exit 1
fi

docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$AGENT_ROOT":/opt/agent:ro -v "$BASE":/models/base:ro -v "$ROOT":/runs \
  --entrypoint python "$TRAIN_IMG" /opt/agent/scripts/convert_opd_checkpoint.py \
  --checkpoint "/runs/$RUN/checkpoints/global_step_$STEPS" \
  --output "/runs/opd_adapter_$RUN" --base-model /models/base >> "$LOG" 2>&1
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "FAIL: adapter conversion did not produce adapter_config.json" >> "$LOG"
  exit 1
fi
echo "2/3 adapter converted and matched against the base layers $(date -u)" >> "$LOG"

docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -v "$AGENT_ROOT":/opt/agent:ro -v "$BASE":/models/base:ro \
  -v "$ADAPTER":/adapters/opd:ro -v "$MERGED":/merged \
  --entrypoint python "$EVAL_IMG" /opt/agent/scripts/merge_adapter_for_serving.py \
  --base-model /models/base --adapter /adapters/opd --output-dir /merged >> "$LOG" 2>&1
if [ ! -f "$MERGED/merge_manifest.json" ]; then
  echo "FAIL: the adapter could not be merged into a servable model" >> "$LOG"
  exit 1
fi
echo "3/3 merge verified $(date -u)" >> "$LOG"
echo "preflight PASS: the long run's post-processing path is verified" >> "$LOG"
tail -n 40 "$LOG"
