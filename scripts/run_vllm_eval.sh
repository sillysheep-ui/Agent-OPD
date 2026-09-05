#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_DEVICE:?set CUDA_DEVICE}"
: "${BASE_MODEL:?set BASE_MODEL}"
: "${BASE_SERVED_MODEL:?set BASE_SERVED_MODEL to a stable base-model alias}"
: "${LORA_PATH:?set LORA_PATH to the final global_step checkpoint}"
: "${TRAINING_MANIFEST:?set TRAINING_MANIFEST to its launch_manifest.json}"
: "${SERVED_MODEL:?set SERVED_MODEL}"
: "${PORT:?set PORT}"
: "${EVAL_OUTPUT:?set EVAL_OUTPUT}"
: "${ENV_CONFIG:?set ENV_CONFIG}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"
: "${EVAL_SPLIT:?set EVAL_SPLIT}"
: "${LIMIT_GAMES:?set LIMIT_GAMES}"
: "${GAME_LIST:?set GAME_LIST to a frozen JSON game list}"
: "${GAME_LIST_MANIFEST:?set GAME_LIST_MANIFEST to its manifest}"
: "${TRAINING_SEED:?set TRAINING_SEED to the checkpoint training seed}"
: "${ROLLOUT_SEED:?set ROLLOUT_SEED identically for every compared checkpoint}"
: "${MAX_STEPS:?set MAX_STEPS}"
: "${MAX_CONTEXT_TOKENS:?set MAX_CONTEXT_TOKENS}"
: "${RESERVE_TOKENS:?set RESERVE_TOKENS}"
: "${MAX_TOKENS:?set MAX_TOKENS}"
: "${VLLM_MAX_MODEL_LEN:?set VLLM_MAX_MODEL_LEN}"

for integer_name in TRAINING_SEED ROLLOUT_SEED MAX_STEPS MAX_CONTEXT_TOKENS RESERVE_TOKENS MAX_TOKENS VLLM_MAX_MODEL_LEN; do
  integer_value=${!integer_name}
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_name} must be a non-negative integer" >&2
    exit 2
  fi
done
if (( MAX_STEPS <= 0 || MAX_CONTEXT_TOKENS <= 0 || RESERVE_TOKENS < MAX_TOKENS || MAX_TOKENS <= 0 )); then
  echo "steps/tokens must be positive and RESERVE_TOKENS must cover MAX_TOKENS" >&2
  exit 2
fi
if (( MAX_CONTEXT_TOKENS > VLLM_MAX_MODEL_LEN )); then
  echo "MAX_CONTEXT_TOKENS exceeds the served VLLM_MAX_MODEL_LEN" >&2
  exit 2
fi
if [[ "${BASE_SERVED_MODEL}" == "${SERVED_MODEL}" ]]; then
  echo "BASE_SERVED_MODEL and the LoRA SERVED_MODEL alias must be distinct" >&2
  exit 2
fi

task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/omniopd-eval.XXXXXX")
server_log="${task_tmp}/vllm.log"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -rf "${task_tmp}"
}
trap cleanup EXIT INT TERM

vllm_version=$(python3 -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')
if [[ -z "${vllm_version}" ]]; then
  echo "could not resolve the installed vLLM version" >&2
  exit 2
fi

command=(python3 -m vllm.entrypoints.openai.api_server
  --model "${BASE_MODEL}"
  --served-model-name "${BASE_SERVED_MODEL}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization 0.8
  --port "${PORT}")
command+=(--enable-lora --lora-modules "${SERVED_MODEL}=${LORA_PATH}")
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${command[@]}" >"${server_log}" 2>&1 &
server_pid=$!

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 "${repo_root}/scripts/wait_for_model.py" \
  --url "http://127.0.0.1:${PORT}/v1" \
  --expected-model "${SERVED_MODEL}" \
  --timeout 200

eval_command=(python3 "${repo_root}/scripts/evaluate.py"
  --env-config "${ENV_CONFIG}"
  --output "${EVAL_OUTPUT}"
  --model "${SERVED_MODEL}"
  --base-model-artifact "${BASE_MODEL}"
  --checkpoint-artifact "${LORA_PATH}"
  --training-manifest "${TRAINING_MANIFEST}"
  --base-url "http://127.0.0.1:${PORT}/v1"
  --tokenizer "${TOKENIZER_PATH}"
  --inference-runtime "vllm:${vllm_version}"
  --split "${EVAL_SPLIT}"
  --limit-games "${LIMIT_GAMES}"
  --training-seed "${TRAINING_SEED}"
  --rollout-seed "${ROLLOUT_SEED}"
  --max-steps "${MAX_STEPS}"
  --max-context-tokens "${MAX_CONTEXT_TOKENS}"
  --reserve-tokens "${RESERVE_TOKENS}"
  --max-tokens "${MAX_TOKENS}")
eval_command+=(
  --game-list "${GAME_LIST}"
  --game-list-manifest "${GAME_LIST_MANIFEST}"
)
"${eval_command[@]}"
