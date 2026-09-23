#!/bin/bash
# Host-side chain that turns a finished standard-OPD run into a paired
# ALFWorld measurement:
#   1. wait for the run's completion manifest and for the container to exit
#   2. convert the veRL FSDP LoRA shard into a PEFT adapter
#   3. serve the base Student with the adapter as model `opd` (LoRA path), and
#      fall back to merging the adapter into a full model if vLLM refuses it
#   4. evaluate `opd` and the frozen `base4b` Student on the same game list
#   5. print both win rates (the base number is the no-update control)
#
# The evaluation prompt file is referenced by path, not copied, because the
# upstream repository is not licensed for redistribution in this repository.
set -u

ROOT=${ROOT:-/data/yangchunyu/ld/omniopd_runs}
RUN=${RUN:?run name required}
STEPS=${STEPS:-86}
NAME=${NAME:-omniopd-opd-standard}
OUT=$ROOT/$RUN
CKPT=$OUT/checkpoints/global_step_$STEPS
ADAPTER=$ROOT/opd_adapter_$RUN
MERGED=$ROOT/opd_merged_$RUN
EVALDIR=$ROOT/opd_eval_$RUN
LOG=$ROOT/opd_eval_chain_$RUN.log
TRAIN_IMG=${TRAIN_IMG:-omniopd-verl080:vllm010-td010}
EVAL_IMG=${EVAL_IMG:-omniopd-verl080:alfworld}
BASE=${BASE:-/cfs/data/private/zhangsl/Model/Qwen/Qwen3-4B-Instruct-2507}
VERL_ROOT=${VERL_ROOT:-/data/yangchunyu/ld/verl-v0.8.0}
AGENT_ROOT=${AGENT_ROOT:-/data/yangchunyu/ld/agent_omniopd_v080}
REFERENCE_ROOT=${REFERENCE_ROOT:-/data/yangchunyu/ld/reference_protocols}
ALFWORLD_DATA=${ALFWORLD_DATA:-/cfs/data/private/yangchunyu/liud/ld/alfworld_data_0.4.2}
PORT=${PORT:-18101}
GPU=${GPU:-4}
PROMPT_JSON=${PROMPT_JSON:-/reference/AgentBoard/agentboard/prompts/VanillaAgent/alfworld_base.json}
VAL_DATA=${VAL_DATA:-/runs/valid_seen_probe/val_all.jsonl}
MAX_STEPS=${MAX_STEPS:-30}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-32768}
RESERVE_TOKENS=${RESERVE_TOKENS:-4096}
MAX_TOKENS=${MAX_TOKENS:-4096}

mkdir -p "$EVALDIR"
echo "chain started $(date -u)" > "$LOG"

wait_for_manifest() {
  for _ in $(seq 1 900); do
    if [ -f "$OUT/completion_manifest.json" ]; then
      echo "training completion manifest ready $(date -u)" >> "$LOG"; return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^$NAME$"; then
      echo "training container gone without a completion manifest; abort $(date -u)" >> "$LOG"; return 1
    fi
    sleep 30
  done
  echo "timed out waiting for the completion manifest $(date -u)" >> "$LOG"; return 1
}

wait_for_gpus() {
  for _ in $(seq 1 90); do
    docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo "training container exited $(date -u)" >> "$LOG"; return 0; }
    sleep 10
  done
  echo "training container still present after the manifest; abort" >> "$LOG"; return 1
}

start_lora_server() {
  docker rm -f omniopd-vllm-opd-eval >/dev/null 2>&1
  docker run -d --name omniopd-vllm-opd-eval --network host --runtime=nvidia --shm-size=16g \
    -e NVIDIA_VISIBLE_DEVICES="$GPU" -e VLLM_USE_V1=1 \
    -v "$BASE":/models/base:ro -v "$ADAPTER":/adapters/opd:ro \
    --entrypoint vllm "$EVAL_IMG" serve /models/base --served-model-name base4b \
    --enable-lora --lora-modules opd=/adapters/opd \
    --max-model-len "$MAX_CONTEXT_TOKENS" --port "$PORT" --host 127.0.0.1 \
    --gpu-memory-utilization 0.5 >> "$LOG" 2>&1
  for _ in $(seq 1 40); do
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"opd"' && { echo "LoRA server ready $(date -u)" >> "$LOG"; return 0; }
    sleep 15
  done
  echo "LoRA server did not expose the adapter $(date -u)" >> "$LOG"; return 1
}

start_plain_server() {
  docker rm -f omniopd-vllm-opd-eval >/dev/null 2>&1
  docker run -d --name omniopd-vllm-opd-eval --network host --runtime=nvidia --shm-size=16g \
    -e NVIDIA_VISIBLE_DEVICES="$GPU" -e VLLM_USE_V1=1 -v "$1":/models/served:ro \
    --entrypoint vllm "$EVAL_IMG" serve /models/served --served-model-name "$2" \
    --max-model-len "$MAX_CONTEXT_TOKENS" --port "$PORT" --host 127.0.0.1 \
    --gpu-memory-utilization 0.5 >> "$LOG" 2>&1
  for _ in $(seq 1 40); do
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$2\"" && { echo "$2 server ready $(date -u)" >> "$LOG"; return 0; }
    sleep 15
  done
  echo "$2 server did not come up $(date -u)" >> "$LOG"; return 1
}

evaluate() {
  echo "=== evaluating $1 -> $2 $(date -u) ===" >> "$LOG"
  _evaluate_with_val_data "$1" "$2" "$VAL_DATA"
}

# A three-game pass over the served model, so a broken adapter or a server that
# answers nonsense is caught in minutes instead of after the full game list.
smoke_evaluate() {
  if [ ! -f "$ROOT/valid_seen_probe/val_3.jsonl" ]; then
    echo "smoke game list missing; skipping the smoke pass $(date -u)" >> "$LOG"
    return 0
  fi
  echo "=== smoke: 3 games on $1 $(date -u) ===" >> "$LOG"
  _evaluate_with_val_data "$1" "smoke_$2" "/runs/valid_seen_probe/val_3.jsonl"
}

_evaluate_with_val_data() {
  docker run --rm --network host -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/opt/agent/src:/opt/verl -e ALFWORLD_DATA=/alfworld \
    -v "$VERL_ROOT":/opt/verl:ro -v "$AGENT_ROOT":/opt/agent:ro \
    -v "$REFERENCE_ROOT":/reference:ro -v "$ALFWORLD_DATA":/alfworld:ro \
    -v "$BASE":/models/student:ro -v "$ROOT":/runs -w /opt/agent --entrypoint python "$EVAL_IMG" \
    scripts/evaluate_coldstart_sft.py --env-config configs/alfworld_textworld.yaml \
    --tokenizer /models/student --val-data "$3" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$1" --prompt-json "$PROMPT_JSON" \
    --max-steps "$MAX_STEPS" --max-context-tokens "$MAX_CONTEXT_TOKENS" \
    --reserve-tokens "$RESERVE_TOKENS" --max-tokens "$MAX_TOKENS" \
    --output "$EVALDIR/$2" >> "$LOG" 2>&1
  echo "evaluation of $1 finished with exit=$? $(date -u)" >> "$LOG"
}

summarize() {
  python3 - "$EVALDIR" >> "$LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for name, label in (("closed_loop_opd.json", "OPD Student"), ("closed_loop_base.json", "frozen base Student")):
    path = root / name
    if not path.is_file():
        print(f"{label}: missing {path}")
        continue
    summary = json.loads(path.read_text())["summary"]
    wins, games = summary["wins"], summary["games"]
    print("%s: %d/%d = %.2f%%" % (label, wins, games, 100 * wins / games))
    for task, value in sorted(summary["by_task_type"].items()):
        print("   %-32s %d/%d" % (task, value["wins"], value["games"]))
PY
}

wait_for_manifest || exit 1
wait_for_gpus || exit 1

if [ ! -d "$CKPT" ]; then echo "no checkpoint at $CKPT; abort" >> "$LOG"; exit 1; fi
echo "=== checkpoint tree ===" >> "$LOG"; find "$CKPT" -maxdepth 3 >> "$LOG" 2>&1
du -sh "$CKPT" >> "$LOG" 2>&1

echo "=== converting the FSDP shard to a PEFT adapter $(date -u) ===" >> "$LOG"
docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$AGENT_ROOT":/opt/agent:ro -v "$ROOT":/runs -v "$BASE":/models/base:ro \
  --entrypoint python "$TRAIN_IMG" /opt/agent/scripts/convert_opd_checkpoint.py \
  --checkpoint "/runs/$RUN/checkpoints/global_step_$STEPS" --output "/runs/opd_adapter_$RUN" \
  --base-model "/models/base" >> "$LOG" 2>&1
if [ ! -f "$ADAPTER/adapter_config.json" ]; then echo "conversion failed; abort $(date -u)" >> "$LOG"; exit 1; fi
ls -l "$ADAPTER" >> "$LOG" 2>&1

if start_lora_server; then
  smoke_evaluate opd closed_loop_opd.json
  evaluate opd closed_loop_opd.json
  evaluate base4b closed_loop_base.json
  summarize
  docker rm -f omniopd-vllm-opd-eval >/dev/null 2>&1
  echo "chain done $(date -u)" >> "$LOG"
  exit 0
fi

echo "=== falling back to a merged model $(date -u) ===" >> "$LOG"
docker rm -f omniopd-vllm-opd-eval >/dev/null 2>&1
# The merge script verifies that the adapter actually changes the logits and
# that the exported checkpoint stays close to the adapter-applied model.
docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -v "$AGENT_ROOT":/opt/agent:ro \
  -v "$BASE":/models/base:ro -v "$ADAPTER":/adapters/opd:ro -v "$ROOT":/runs \
  --entrypoint python "$EVAL_IMG" /opt/agent/scripts/merge_adapter_for_serving.py \
  --base-model /models/base --adapter /adapters/opd \
  --output-dir "/runs/opd_merged_$RUN" >> "$LOG" 2>&1
if [ ! -f "$MERGED/config.json" ]; then echo "merge failed; abort $(date -u)" >> "$LOG"; exit 1; fi

if start_plain_server "$MERGED" opd; then
  smoke_evaluate opd closed_loop_opd.json
  evaluate opd closed_loop_opd.json
else
  echo "could not serve the adapted Student $(date -u)" >> "$LOG"
fi
if start_plain_server "$BASE" base4b; then
  evaluate base4b closed_loop_base.json
else
  echo "could not serve the frozen base Student $(date -u)" >> "$LOG"
fi
summarize
docker rm -f omniopd-vllm-opd-eval >/dev/null 2>&1
echo "chain done (fallback path) $(date -u)" >> "$LOG"
