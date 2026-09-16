#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_DEVICE:?set CUDA_DEVICE}"
: "${MODEL_PATH:?set MODEL_PATH to the complete frozen behavior Student}"
: "${SERVED_MODEL:?set SERVED_MODEL to the exact behavior-model alias}"
: "${PORT:?set PORT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${CORRECTIONS:?set CORRECTIONS}"
: "${CORRECTION_MANIFEST:?set CORRECTION_MANIFEST}"
: "${STATE_POOL:?set STATE_POOL}"
: "${STATE_POOL_MANIFEST:?set STATE_POOL_MANIFEST}"
: "${ENV_CONFIG:?set ENV_CONFIG}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"
: "${MAX_STEPS:?set MAX_STEPS}"
: "${MAX_CONTEXT_TOKENS:?set MAX_CONTEXT_TOKENS}"
: "${RESERVE_TOKENS:?set RESERVE_TOKENS}"
: "${MAX_TOKENS:?set MAX_TOKENS}"
: "${ROLLOUT_SEED:?set ROLLOUT_SEED}"
: "${BOOTSTRAP_SEED:?set BOOTSTRAP_SEED}"
: "${BOOTSTRAP_REPLICATES:?set BOOTSTRAP_REPLICATES}"
: "${CONFIDENCE:?set CONFIDENCE}"
: "${VLLM_MAX_MODEL_LEN:?set VLLM_MAX_MODEL_LEN}"

for integer_name in PORT MAX_STEPS MAX_CONTEXT_TOKENS RESERVE_TOKENS MAX_TOKENS ROLLOUT_SEED BOOTSTRAP_SEED BOOTSTRAP_REPLICATES VLLM_MAX_MODEL_LEN; do
  integer_value=${!integer_name}
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_name} must be a non-negative integer" >&2
    exit 2
  fi
done
if (( PORT <= 0 || PORT > 65535 || MAX_STEPS <= 0 || MAX_TOKENS <= 0 || BOOTSTRAP_REPLICATES <= 0 )); then
  echo "port, steps, generated tokens, and bootstrap replicates must be positive" >&2
  exit 2
fi
if (( RESERVE_TOKENS < MAX_TOKENS || MAX_CONTEXT_TOKENS <= RESERVE_TOKENS || MAX_CONTEXT_TOKENS > VLLM_MAX_MODEL_LEN )); then
  echo "token limits are inconsistent with truncation or the served context" >&2
  exit 2
fi
if ! python3 - "${CONFIDENCE}" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 < value < 1.0:
    raise SystemExit(1)
PY
then
  echo "CONFIDENCE must be finite and strictly between zero and one" >&2
  exit 2
fi

for required_path in "${MODEL_PATH}" "${TOKENIZER_PATH}" "${CORRECTIONS}" "${CORRECTION_MANIFEST}" "${STATE_POOL}" "${STATE_POOL_MANIFEST}" "${ENV_CONFIG}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "required Position input does not exist: ${required_path}" >&2
    exit 2
  fi
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
  echo "refusing to reuse the Position output or service attestation" >&2
  exit 2
fi
task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/omniopd-position.XXXXXX")
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
  --timeout 600

python3 - \
  "${service_manifest}" "${server_pid}" "$$" "${PORT}" \
  "${SERVED_MODEL}" "${MODEL_PATH}" "${TOKENIZER_PATH}" "${vllm_version}" \
  "${CUDA_DEVICE}" "${repo_root}/scripts/run_vllm_position.sh" \
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
    "service_role": "position_counterfactual",
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

python3 "${repo_root}/scripts/run_position_counterfactual.py" \
  --corrections "${CORRECTIONS}" \
  --correction-manifest "${CORRECTION_MANIFEST}" \
  --state-pool "${STATE_POOL}" \
  --state-pool-manifest "${STATE_POOL_MANIFEST}" \
  --env-config "${ENV_CONFIG}" \
  --student-model "${SERVED_MODEL}" \
  --student-url "http://127.0.0.1:${PORT}/v1" \
  --student-service-manifest "${service_manifest}" \
  --inference-runtime "vllm:${vllm_version}" \
  --student-artifact "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
  --reserve-tokens "${RESERVE_TOKENS}" \
  --max-tokens "${MAX_TOKENS}" \
  --rollout-seed "${ROLLOUT_SEED}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --confidence "${CONFIDENCE}"
