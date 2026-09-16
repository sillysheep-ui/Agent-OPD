#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_DEVICE:?set CUDA_DEVICE}"
: "${MODEL_PATH:?set MODEL_PATH to the complete Student model}"
: "${SERVED_MODEL:?set SERVED_MODEL to a stable Student alias}"
: "${PORT:?set PORT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${ENV_CONFIG:?set ENV_CONFIG}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"
: "${LIMIT_GAMES:?set LIMIT_GAMES}"
: "${GAME_ORDER_SEED:?set GAME_ORDER_SEED}"
: "${ENVIRONMENT_SEED:?set ENVIRONMENT_SEED}"
: "${MAX_STEPS:?set MAX_STEPS}"
: "${MAX_CONTEXT_TOKENS:?set MAX_CONTEXT_TOKENS}"
: "${RESERVE_TOKENS:?set RESERVE_TOKENS}"
: "${MAX_TOKENS:?set MAX_TOKENS}"
: "${VLLM_MAX_MODEL_LEN:?set VLLM_MAX_MODEL_LEN}"

for integer_name in PORT LIMIT_GAMES GAME_ORDER_SEED ENVIRONMENT_SEED MAX_STEPS MAX_CONTEXT_TOKENS RESERVE_TOKENS MAX_TOKENS VLLM_MAX_MODEL_LEN; do
  integer_value=${!integer_name}
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_name} must be a non-negative integer" >&2
    exit 2
  fi
done
if (( PORT <= 0 || PORT > 65535 || LIMIT_GAMES <= 0 || MAX_STEPS <= 0 || MAX_TOKENS <= 0 )); then
  echo "port, games, steps, and generated tokens must be positive" >&2
  exit 2
fi
if (( RESERVE_TOKENS < MAX_TOKENS || MAX_CONTEXT_TOKENS <= RESERVE_TOKENS || MAX_CONTEXT_TOKENS > VLLM_MAX_MODEL_LEN )); then
  echo "token limits are inconsistent with truncation or the served context" >&2
  exit 2
fi
if [[ ! -e "${MODEL_PATH}" || ! -e "${TOKENIZER_PATH}" || ! -f "${ENV_CONFIG}" ]]; then
  echo "MODEL_PATH, TOKENIZER_PATH, and ENV_CONFIG must be local fingerprintable artifacts" >&2
  exit 2
fi

exclude_args=()
while (( $# > 0 )); do
  if [[ "$1" != "--exclude-jsonl" || $# -lt 2 || ! -f "$2" ]]; then
    echo "only repeatable --exclude-jsonl EXISTING_PATH arguments are accepted" >&2
    exit 2
  fi
  exclude_args+=("$1" "$2")
  shift 2
done

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

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
service_manifest="${OUTPUT_DIR}.service_manifest.json"
if [[ -e "${OUTPUT_DIR}" || -e "${service_manifest}" ]]; then
  echo "refusing to reuse the state-pool output or service attestation" >&2
  exit 2
fi
task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/omniopd-state-pool.XXXXXX")
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
command=(python3 -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --tokenizer "${TOKENIZER_PATH}"
  --served-model-name "${SERVED_MODEL}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization 0.8
  --host 127.0.0.1
  --port "${PORT}")
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${command[@]}" >"${server_log}" 2>&1 &
server_pid=$!

python3 "${repo_root}/scripts/wait_for_model.py" \
  --url "http://127.0.0.1:${PORT}/v1" \
  --expected-model "${SERVED_MODEL}" \
  --timeout 200

python3 - \
  "${service_manifest}" "${server_pid}" "$$" "${PORT}" \
  "${SERVED_MODEL}" "${MODEL_PATH}" "${TOKENIZER_PATH}" "${vllm_version}" \
  "${CUDA_DEVICE}" "${repo_root}/scripts/run_vllm_state_pool.sh" \
  -- "${command[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(sys.argv[10]).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from omniopd.provenance import fingerprint_path, sha256_json
from omniopd.service_attestation import live_service_process_snapshot

(
    destination_raw,
    server_pid,
    wrapper_pid,
    port,
    served_model,
    model_raw,
    tokenizer_raw,
    vllm_version,
    cuda_visible_devices,
    wrapper_raw,
) = sys.argv[1:11]
if sys.argv[11] != "--":
    raise SystemExit("internal service-attestation argv delimiter is missing")
command = sys.argv[12:]
payload = {
    "protocol_version": "omniopd-v1",
    "artifact": "omniopd_local_vllm_service_attestation",
    "schema_version": 2,
    "attestation_scope": "local_wrapper_child_process",
    "service_role": "student_state_pool",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "wrapper_pid": int(wrapper_pid),
    "server_pid": int(server_pid),
    "base_url": f"http://127.0.0.1:{int(port)}/v1",
    "port": int(port),
    "cuda_visible_devices": cuda_visible_devices,
    "vllm_version": vllm_version,
    "inference_runtime": f"vllm:{vllm_version}",
    "served_model": served_model,
    "launch_command": command,
    "launch_command_sha256": sha256_json(command),
    "wrapper": fingerprint_path(wrapper_raw),
    "model_artifact": fingerprint_path(model_raw),
    "tokenizer_artifact": fingerprint_path(tokenizer_raw),
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
destination = Path(destination_raw)
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
PY

python3 "${repo_root}/scripts/collect_state_pool.py" \
  --env-config "${ENV_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --behavior-model "${SERVED_MODEL}" \
  --behavior-artifact "${MODEL_PATH}" \
  --behavior-service-manifest "${service_manifest}" \
  --behavior-url "http://127.0.0.1:${PORT}/v1" \
  --inference-runtime "vllm:${vllm_version}" \
  --behavior-thinking-mode disabled \
  --behavior-temperature 0 \
  --state-source student \
  --tokenizer "${TOKENIZER_PATH}" \
  --split train \
  --limit-games "${LIMIT_GAMES}" \
  --max-steps "${MAX_STEPS}" \
  --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
  --reserve-tokens "${RESERVE_TOKENS}" \
  --behavior-max-tokens "${MAX_TOKENS}" \
  --seed "${GAME_ORDER_SEED}" \
  --environment-seed "${ENVIRONMENT_SEED}" \
  "${exclude_args[@]}"
