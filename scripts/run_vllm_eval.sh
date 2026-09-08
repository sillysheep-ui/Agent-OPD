#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_DEVICE:?set CUDA_DEVICE}"
: "${BASE_MODEL:?set BASE_MODEL}"
: "${BASE_SERVED_MODEL:?set BASE_SERVED_MODEL to a stable base-model alias}"
: "${LORA_PATH:?set LORA_PATH to the final global_step checkpoint}"
: "${TRAINING_MANIFEST:?set TRAINING_MANIFEST to its launch_manifest.json}"
: "${TRAINING_COMPLETION_MANIFEST:?set TRAINING_COMPLETION_MANIFEST to its completion_manifest.json}"
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
: "${ENVIRONMENT_SEED:?set ENVIRONMENT_SEED to the frozen game-list environment seed}"
: "${MAX_STEPS:?set MAX_STEPS}"
: "${MAX_CONTEXT_TOKENS:?set MAX_CONTEXT_TOKENS}"
: "${RESERVE_TOKENS:?set RESERVE_TOKENS}"
: "${MAX_TOKENS:?set MAX_TOKENS}"
: "${VLLM_MAX_MODEL_LEN:?set VLLM_MAX_MODEL_LEN}"

for integer_name in TRAINING_SEED ROLLOUT_SEED ENVIRONMENT_SEED MAX_STEPS MAX_CONTEXT_TOKENS RESERVE_TOKENS MAX_TOKENS VLLM_MAX_MODEL_LEN; do
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

python3 - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise SystemExit(f"refusing to start: loopback port {port} is already in use: {error}")
PY

task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/omniopd-eval.XXXXXX")
server_log="${task_tmp}/vllm.log"
service_manifest="${EVAL_OUTPUT}.service_manifest.json"
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
if [[ -e "${service_manifest}" ]]; then
  echo "Refusing to overwrite service attestation: ${service_manifest}" >&2
  exit 2
fi

command=(python3 -m vllm.entrypoints.openai.api_server
  --model "${BASE_MODEL}"
  --tokenizer "${TOKENIZER_PATH}"
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

# Record the exact argv and content identities only after the locally launched
# child reports ready.  evaluate.py additionally verifies that it is a direct
# child of this wrapper and that this server PID is still alive.
python3 - \
  "${service_manifest}" \
  "${server_pid}" \
  "$$" \
  "${PORT}" \
  "${BASE_SERVED_MODEL}" \
  "${SERVED_MODEL}" \
  "${BASE_MODEL}" \
  "${TOKENIZER_PATH}" \
  "${LORA_PATH}" \
  "${TRAINING_MANIFEST}" \
  "${TRAINING_COMPLETION_MANIFEST}" \
  "${vllm_version}" \
  "${CUDA_DEVICE}" \
  "${repo_root}/scripts/run_vllm_eval.sh" \
  -- "${command[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(sys.argv[14]).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from omniopd.evaluation import validate_training_completion_manifest
from omniopd.provenance import fingerprint_path, sha256_file, sha256_json
from omniopd.service_attestation import live_service_process_snapshot
from omniopd.validation import validate_lora_checkpoint_directory

(
    destination_raw,
    server_pid,
    wrapper_pid,
    port,
    base_alias,
    adapter_alias,
    base_model_raw,
    tokenizer_raw,
    checkpoint_raw,
    launch_raw,
    completion_raw,
    vllm_version,
    cuda_visible_devices,
    wrapper_raw,
) = sys.argv[1:15]
if sys.argv[15] != "--":
    raise SystemExit("internal service-attestation argv delimiter is missing")
command = sys.argv[16:]
destination = Path(destination_raw)
launch_path = Path(launch_raw).resolve()
completion_path = Path(completion_raw).resolve()
checkpoint_path = Path(checkpoint_raw).resolve()
base_model_path = Path(base_model_raw).resolve()
tokenizer_path = Path(tokenizer_raw).resolve()
launch = json.loads(launch_path.read_text(encoding="utf-8"))
completion = json.loads(completion_path.read_text(encoding="utf-8"))
checkpoint = fingerprint_path(checkpoint_path)
base_model = fingerprint_path(base_model_path)
tokenizer = fingerprint_path(tokenizer_path)
resolved_config_path = launch_path.parent / "resolved_config.yaml"
validate_training_completion_manifest(
    completion,
    launch,
    launch_manifest_sha256=sha256_file(launch_path),
    completion_manifest_sha256=sha256_file(completion_path),
    checkpoint_fingerprint=checkpoint,
    current_checkpoint_format=validate_lora_checkpoint_directory(
        checkpoint_path,
        expected_rank=int(launch["hyperparameters"]["lora_rank"]),
        expected_alpha=float(launch["hyperparameters"]["lora_alpha"]),
        expected_target_modules_policy=str(launch["hyperparameters"]["target_modules"]),
    ),
    resolved_config_fingerprint=fingerprint_path(resolved_config_path),
)
payload = {
    "protocol_version": "omniopd-v1",
    "artifact": "omniopd_local_vllm_service_attestation",
    "schema_version": 2,
    "attestation_scope": "local_wrapper_child_process",
    "service_role": "checkpoint_evaluation",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "wrapper_pid": int(wrapper_pid),
    "server_pid": int(server_pid),
    "base_url": f"http://127.0.0.1:{int(port)}/v1",
    "port": int(port),
    "cuda_visible_devices": cuda_visible_devices,
    "vllm_version": vllm_version,
    "inference_runtime": f"vllm:{vllm_version}",
    "served_aliases": {"base": base_alias, "adapter": adapter_alias},
    "launch_command": command,
    "launch_command_sha256": sha256_json(command),
    "wrapper": fingerprint_path(wrapper_raw),
    "base_model_artifact": base_model,
    "tokenizer_artifact": tokenizer,
    "checkpoint_artifact": checkpoint,
    "training_launch_manifest": fingerprint_path(launch_path),
    "training_completion_manifest": fingerprint_path(completion_path),
    "training_launch_manifest_sha256": sha256_file(launch_path),
    "training_completion_manifest_sha256": sha256_file(completion_path),
    "process_observation": live_service_process_snapshot(
        server_pid=int(server_pid),
        wrapper_pid=int(wrapper_pid),
        port=int(port),
        launch_command=command,
    ),
    "trust_boundary": {
        "local": "argv_parent_port_and_artifact_content_verified_on_same_trusted_host",
        "limitation": "not_cryptographic_remote_attestation",
    },
}
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
PY

eval_command=(python3 "${repo_root}/scripts/evaluate.py"
  --env-config "${ENV_CONFIG}"
  --output "${EVAL_OUTPUT}"
  --model "${SERVED_MODEL}"
  --base-model-artifact "${BASE_MODEL}"
  --checkpoint-artifact "${LORA_PATH}"
  --training-manifest "${TRAINING_MANIFEST}"
  --training-completion-manifest "${TRAINING_COMPLETION_MANIFEST}"
  --service-manifest "${service_manifest}"
  --base-url "http://127.0.0.1:${PORT}/v1"
  --tokenizer "${TOKENIZER_PATH}"
  --inference-runtime "vllm:${vllm_version}"
  --split "${EVAL_SPLIT}"
  --limit-games "${LIMIT_GAMES}"
  --training-seed "${TRAINING_SEED}"
  --rollout-seed "${ROLLOUT_SEED}"
  --environment-seed "${ENVIRONMENT_SEED}"
  --max-steps "${MAX_STEPS}"
  --max-context-tokens "${MAX_CONTEXT_TOKENS}"
  --reserve-tokens "${RESERVE_TOKENS}"
  --max-tokens "${MAX_TOKENS}")
eval_command+=(
  --game-list "${GAME_LIST}"
  --game-list-manifest "${GAME_LIST_MANIFEST}"
)
"${eval_command[@]}"
