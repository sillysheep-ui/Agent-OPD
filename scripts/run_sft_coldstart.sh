#!/usr/bin/env bash
# Cold-start SFT launcher for the shared Student initialization.
#
# This is the dedicated launcher for the expert cold-start corpus.  It is NOT
# run_verl_train.sh: that one is bound to the fixed-budget breadth/depth arms and
# refuses to start without an annotation-pair manifest.  Cold start has no pair,
# so this launcher binds the data audit instead and keeps every other gate
# (clean trees, frozen veRL tag/revision, explicit optimizer steps).
set -euo pipefail

: "${SFT_MODEL:?Student model directory required}"
: "${SFT_TRAIN_FILES:?training rows required}"
: "${SFT_VAL_FILES:?validation rows required}"
: "${SFT_DATA_AUDIT:?expert SFT data audit required}"
: "${SFT_OUTPUT_DIR:?new output directory required}"
: "${VERL_ROOT:?veRL checkout required}"
: "${NUM_GPUS:?GPU count required}"
: "${TOTAL_TRAINING_STEPS:?explicit optimizer step budget required}"

LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-1}
TRAIN_BSZ=${TRAIN_BSZ:-4}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
TRAINING_DTYPE=${TRAINING_DTYPE:-bf16}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
SEED=${SEED:-42}
SAVE_FREQ=${SAVE_FREQ:-0}
TEST_FREQ=${TEST_FREQ:-0}
PROJECT_NAME=${PROJECT_NAME:-agent_omniopd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft_coldstart}

require_positive_integer() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 2
  fi
}

require_positive_integer NUM_GPUS "${NUM_GPUS}"
require_positive_integer TOTAL_TRAINING_STEPS "${TOTAL_TRAINING_STEPS}"
require_positive_integer EPOCHS "${EPOCHS}"
require_positive_integer TRAIN_BSZ "${TRAIN_BSZ}"
require_positive_integer MICRO_BSZ "${MICRO_BSZ}"
require_positive_integer MAX_LENGTH "${MAX_LENGTH}"
require_positive_integer LORA_ALPHA "${LORA_ALPHA}"
require_nonnegative_integer LORA_RANK "${LORA_RANK}"
require_nonnegative_integer SEED "${SEED}"
require_nonnegative_integer SAVE_FREQ "${SAVE_FREQ}"
require_nonnegative_integer TEST_FREQ "${TEST_FREQ}"

if (( TRAIN_BSZ % NUM_GPUS != 0 )); then
  echo "TRAIN_BSZ=${TRAIN_BSZ} must be divisible by NUM_GPUS=${NUM_GPUS}" >&2
  exit 2
fi
LOCAL_TRAIN_BSZ=$((TRAIN_BSZ / NUM_GPUS))
if (( LOCAL_TRAIN_BSZ % MICRO_BSZ != 0 )); then
  echo "local train batch ${LOCAL_TRAIN_BSZ} must be divisible by MICRO_BSZ=${MICRO_BSZ}" >&2
  exit 2
fi
if [[ "${MODEL_DTYPE}" != "fp32" ]]; then
  echo "MODEL_DTYPE must be fp32 so optimizer master parameters stay full precision" >&2
  exit 2
fi
if [[ "${TRAINING_DTYPE}" != "bf16" ]]; then
  echo "TRAINING_DTYPE must be bf16: this trainer uses bf16 forward with fp32 CE" >&2
  exit 2
fi
if [[ "${ALLOW_RESUME:-0}" == "1" ]]; then
  echo "Checkpoint resume is unsupported for this launcher" >&2
  exit 2
fi

python_candidate=${PYTHON_BIN:-python3}
if ! command -v -- "${python_candidate}" >/dev/null 2>&1; then
  echo "PYTHON_BIN is not executable: ${python_candidate}" >&2
  exit 2
fi
python_bin=$("${python_candidate}" -c 'from pathlib import Path; import sys; print(Path(sys.executable).resolve())')
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
resolve_path() {
  "${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}
VERL_ROOT=$(resolve_path "${VERL_ROOT}")
SFT_MODEL=$(resolve_path "${SFT_MODEL}")
SFT_TRAIN_FILES=$(resolve_path "${SFT_TRAIN_FILES}")
SFT_VAL_FILES=$(resolve_path "${SFT_VAL_FILES}")
SFT_DATA_AUDIT=$(resolve_path "${SFT_DATA_AUDIT}")
SFT_OUTPUT_DIR=$(resolve_path "${SFT_OUTPUT_DIR}")

export VERL_ROOT SFT_MODEL SFT_TRAIN_FILES SFT_VAL_FILES SFT_DATA_AUDIT
export SFT_OUTPUT_DIR NUM_GPUS TRAIN_BSZ LOCAL_TRAIN_BSZ MICRO_BSZ EPOCHS
export TOTAL_TRAINING_STEPS LR MAX_LENGTH MODEL_DTYPE TRAINING_DTYPE
export LORA_RANK LORA_ALPHA SEED SAVE_FREQ TEST_FREQ PROJECT_NAME EXPERIMENT_NAME
export OMNIOPD_REPO_ROOT="${repo_root}"
export PYTHONPATH="${repo_root}/src:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["OMNIOPD_REPO_ROOT"]).resolve()
sys.path.insert(0, str(root / "src"))
from omniopd.provenance import fingerprint_path, sha256_file


def fail(message: str) -> None:
    raise SystemExit(message)


def git_status(path: Path, role: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        fail(f"cannot read {role} Git worktree: {completed.stderr.strip()}")
    if completed.stdout.strip():
        fail(f"{role} worktree is dirty; commit or remove changes before launching")


def git_rev(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if completed.returncode != 0:
        fail(f"cannot read {path} revision")
    return completed.stdout.strip()


verl_root = Path(os.environ["VERL_ROOT"]).resolve()
model = Path(os.environ["SFT_MODEL"]).resolve()
train_path = Path(os.environ["SFT_TRAIN_FILES"]).resolve()
val_path = Path(os.environ["SFT_VAL_FILES"]).resolve()
audit_path = Path(os.environ["SFT_DATA_AUDIT"]).resolve()
output_dir = Path(os.environ["SFT_OUTPUT_DIR"]).resolve()

if output_dir.exists():
    fail(f"refusing to reuse existing output directory: {output_dir}")
for label, path in (
    ("Student model", model / "config.json"),
    ("training rows", train_path),
    ("validation rows", val_path),
    ("data audit", audit_path),
):
    if not path.is_file():
        fail(f"{label} is missing: {path}")

git_status(root, "OmniOPD")
git_status(verl_root, "veRL")
try:
    tag = subprocess.run(
        ["git", "-C", str(verl_root), "describe", "--tags", "--exact-match", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError) as error:
    fail(f"veRL checkout must be exactly on a release tag: {error}")
if tag != "v0.8.0":
    fail(f"cold-start SFT requires veRL v0.8.0, got {tag!r}")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("artifact") != "expert_sft_data_audit" or not audit.get("training_ready"):
    fail("data audit is not a training-ready expert SFT audit")
if audit.get("weighting_mode") != "game_state_mean":
    fail("data audit does not carry the game_state_mean weighting contract")
outputs = audit.get("outputs") or {}
if outputs.get("train", {}).get("sha256") != sha256_file(train_path):
    fail("training rows do not match the data audit hash")
if outputs.get("validation", {}).get("sha256") != sha256_file(val_path):
    fail("validation rows do not match the data audit hash")

payload = {
    "artifact": "sft_coldstart_launch_manifest",
    "supervision_source": audit.get("supervision_source"),
    "code_revision": git_rev(root),
    "verl": {"path": str(verl_root), "revision": git_rev(verl_root), "tag": tag},
    "inputs": {
        "model": fingerprint_path(model),
        "training_rows": fingerprint_path(train_path),
        "validation_rows": fingerprint_path(val_path),
        "data_audit": fingerprint_path(audit_path),
    },
    "training_contract": {
        "num_gpus": int(os.environ["NUM_GPUS"]),
        "global_train_batch_size": int(os.environ["TRAIN_BSZ"]),
        "local_train_batch_size": int(os.environ["LOCAL_TRAIN_BSZ"]),
        "micro_batch_size_per_gpu": int(os.environ["MICRO_BSZ"]),
        "total_optimizer_steps": int(os.environ["TOTAL_TRAINING_STEPS"]),
        "configured_total_epochs_compatibility_only": int(os.environ["EPOCHS"]),
        "seed": int(os.environ["SEED"]),
        "learning_rate": float(os.environ["LR"]),
        "max_length": int(os.environ["MAX_LENGTH"]),
        "model_load_dtype": os.environ["MODEL_DTYPE"],
        "forward_and_fsdp_param_dtype": os.environ["TRAINING_DTYPE"],
        "lora_rank": int(os.environ["LORA_RANK"]),
        "lora_alpha": int(os.environ["LORA_ALPHA"]),
        "weighting": audit.get("weighting_mode"),
        "target_format": audit.get("target_format"),
        "encoded_sequence_overflow_policy": "error",
        "confirmatory_use_allowed": bool(audit.get("confirmatory_use_allowed")),
    },
}
output_dir.mkdir(parents=True, exist_ok=False)
(output_dir / "launch_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

cd "${repo_root}"
"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  "${repo_root}/integrations/verl/fsdp_sft_trainer.py" \
  --config-dir "${repo_root}/configs" \
  --config-name verl_v080_omniopd_sft \
  "$@" \
  data.train_files="${SFT_TRAIN_FILES}" \
  data.val_files="${SFT_VAL_FILES}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=error \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
  model.partial_pretrain="${SFT_MODEL}" \
  model.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.target_modules=all-linear \
  model.strategy=fsdp \
  optim.lr="${LR}" \
  trainer.total_epochs="${EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${SFT_OUTPUT_DIR}/checkpoints" \
  trainer.seed="${SEED}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='["console"]' \
  hydra.run.dir="${SFT_OUTPUT_DIR}/hydra" \
  ulysses_sequence_parallel_size=1 \
  use_remove_padding=false \
  2>&1 | tee "${SFT_OUTPUT_DIR}/train.log"

"${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

root = Path(os.environ["OMNIOPD_REPO_ROOT"]).resolve()
sys.path.insert(0, str(root / "src"))
from omniopd.provenance import fingerprint_path, sha256_file

output_dir = Path(os.environ["SFT_OUTPUT_DIR"]).resolve()
checkpoint = output_dir / "checkpoints" / f"global_step_{int(os.environ['TOTAL_TRAINING_STEPS'])}"
if not (checkpoint / "actor").is_dir():
    raise SystemExit(f"training finished without the expected checkpoint: {checkpoint}")
payload = {
    "artifact": "sft_coldstart_completion_manifest",
    "total_optimizer_steps": int(os.environ["TOTAL_TRAINING_STEPS"]),
    "checkpoint": fingerprint_path(checkpoint),
    "training_log": {
        "path": str(output_dir / "train.log"),
        "sha256": sha256_file(output_dir / "train.log"),
    },
    "launch_manifest_sha256": sha256_file(output_dir / "launch_manifest.json"),
}
(output_dir / "completion_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"completion manifest written for {checkpoint}")
PY
