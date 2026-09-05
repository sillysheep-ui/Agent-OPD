#!/usr/bin/env bash
set -euo pipefail

: "${VERL_ROOT:?set VERL_ROOT to the veRL checkout}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${TRAIN_FILES:?set TRAIN_FILES}"
: "${VAL_FILES:?set VAL_FILES}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${NUM_GPUS:?set NUM_GPUS}"
: "${TOTAL_TRAINING_STEPS:?set TOTAL_TRAINING_STEPS identically for every compared arm}"
: "${EXPERIMENT_CONFIG:?set EXPERIMENT_CONFIG to the resolved arm YAML}"
: "${TRAIN_AUDIT:?set TRAIN_AUDIT to the build-data audit JSON}"
: "${TRAINING_SEED:?set TRAINING_SEED explicitly to one configured replicate seed}"

LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-1}
TRAIN_BSZ=${TRAIN_BSZ:-4}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
TRAINING_DTYPE=${TRAINING_DTYPE:-bf16}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
SAVE_FREQ=${SAVE_FREQ:-0}
TEST_FREQ=${TEST_FREQ:-0}
PROJECT_NAME=${PROJECT_NAME:-agent_omniopd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-audited_sft}

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name=$1
  local value=$2
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
require_nonnegative_integer TRAINING_SEED "${TRAINING_SEED}"
require_nonnegative_integer LORA_RANK "${LORA_RANK}"
require_positive_integer LORA_ALPHA "${LORA_ALPHA}"
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
MICROBATCHES_PER_STEP=$((LOCAL_TRAIN_BSZ / MICRO_BSZ))

if [[ "${MODEL_DTYPE}" != "fp32" ]]; then
  echo "MODEL_DTYPE must be fp32 so optimizer master parameters remain full precision" >&2
  exit 2
fi
if [[ "${TRAINING_DTYPE}" != "bf16" ]]; then
  echo "TRAINING_DTYPE must be bf16: the audited trainer uses bf16 forward/parameters and fp32 CE/reduction" >&2
  exit 2
fi

if [[ "${ALLOW_RESUME:-0}" == "1" ]]; then
  echo "Checkpoint resume is unsupported: checkpoints omit optimizer/scheduler state" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse existing OUTPUT_DIR=${OUTPUT_DIR}" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
trainer_path="${repo_root}/integrations/verl/fsdp_sft_trainer.py"
verl_config_dir="${VERL_ROOT}/verl/trainer/config"
verl_config_file="${verl_config_dir}/sft_trainer.yaml"
python_bin=${PYTHON_BIN:-python3}

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must be a local fingerprintable model file or directory" >&2
  exit 2
fi

for required_path in \
  "${trainer_path}" \
  "${repo_root}/src/omniopd/loss.py" \
  "${repo_root}/src/omniopd/sampler.py" \
  "${repo_root}/src/omniopd/torch_dataset.py" \
  "${EXPERIMENT_CONFIG}" \
  "${TRAIN_AUDIT}" \
  "${TRAIN_FILES}" \
  "${VAL_FILES}" \
  "${verl_config_file}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required file is missing: ${required_path}" >&2
    exit 2
  fi
done

export PYTHONPATH="${repo_root}/src:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LR EPOCHS TRAIN_BSZ LOCAL_TRAIN_BSZ MICRO_BSZ MICROBATCHES_PER_STEP
export MAX_LENGTH TRAINING_SEED MODEL_DTYPE TRAINING_DTYPE LORA_RANK LORA_ALPHA
export SAVE_FREQ TEST_FREQ PROJECT_NAME EXPERIMENT_NAME TOTAL_TRAINING_STEPS
export VERL_ROOT MODEL_PATH TRAIN_FILES VAL_FILES OUTPUT_DIR NUM_GPUS
export EXPERIMENT_CONFIG TRAIN_AUDIT
export OMNIOPD_REPO_ROOT="${repo_root}"
export OMNIOPD_TRAINER_PATH="${trainer_path}"
export OMNIOPD_LAUNCHER_PATH="${BASH_SOURCE[0]}"

"${python_bin}" - "$@" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; package/__version__ checks still apply.
    tomllib = None

import yaml

from omniopd.provenance import fingerprint_code_tree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_tree(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for child in sorted((item for item in path.rglob("*") if item.is_file())):
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(sha256(child).encode("ascii") + b"\n")
        count += 1
        total_bytes += size
    return {"tree_sha256": digest.hexdigest(), "files": count, "bytes": total_bytes}


def describe_input(raw: str) -> dict[str, object]:
    description: dict[str, object] = {"value": raw}
    path = Path(raw).expanduser()
    if path.is_file():
        description.update(
            {"resolved_path": str(path.resolve()), "sha256": sha256(path)}
        )
    elif path.is_dir():
        description.update({"resolved_path": str(path.resolve()), **describe_tree(path)})
    return description


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_verl_versions(path: Path) -> list[str]:
    """Collect independent version declarations for the selected veRL tree."""

    versions = []
    try:
        value = importlib.metadata.version("verl")
        if value:
            versions.append(value)
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import verl

        value = getattr(verl, "__version__", None)
        if value:
            versions.append(str(value))
    except (ImportError, AttributeError):
        pass
    pyproject = path / "pyproject.toml"
    if pyproject.is_file() and tomllib is not None:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
        value = project.get("version")
        if value:
            versions.append(str(value))
    return sorted(set(versions))


root = Path(os.environ["OMNIOPD_REPO_ROOT"]).resolve()
verl_root = Path(os.environ["VERL_ROOT"]).resolve()
required_verl_version = "0.4.1"
resolved_verl_versions = resolve_verl_versions(verl_root)
if resolved_verl_versions != [required_verl_version]:
    raise SystemExit(
        "the audited integration is locked to veRL 0.4.1; "
        f"resolved declarations={resolved_verl_versions or ['missing']}"
    )
root_revision = git_revision(root)
verl_revision = git_revision(verl_root)
if not root_revision or not verl_revision:
    raise SystemExit("both OmniOPD and veRL must be immutable Git revisions")
experiment_config_path = Path(os.environ["EXPERIMENT_CONFIG"]).resolve()
train_audit_path = Path(os.environ["TRAIN_AUDIT"]).resolve()
train_path = Path(os.environ["TRAIN_FILES"]).resolve()
val_path = Path(os.environ["VAL_FILES"]).resolve()
experiment = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
train_audit = json.loads(train_audit_path.read_text(encoding="utf-8"))
expected = {
    "experiment": experiment.get("experiment"),
    "experiment_config_sha256": sha256(experiment_config_path),
    "training_output_sha256": sha256(train_path),
    "validation_output_sha256": sha256(val_path),
}
actual = {
    "experiment": train_audit.get("experiment"),
    "experiment_config_sha256": train_audit.get("experiment_config", {}).get("sha256"),
    "training_output_sha256": train_audit.get("training_output_sha256"),
    "validation_output_sha256": train_audit.get("validation_output_sha256"),
}
if actual != expected:
    raise SystemExit(
        f"training data/audit/config binding mismatch: actual={actual}, expected={expected}"
    )
configured_training = experiment["training"]
resolved_contract = {
    "global_batch_size": int(os.environ["TRAIN_BSZ"]),
    "micro_batch_size_per_gpu": int(os.environ["MICRO_BSZ"]),
    "max_length": int(os.environ["MAX_LENGTH"]),
    "model_load_dtype": os.environ["MODEL_DTYPE"],
    "compute_dtype": os.environ["TRAINING_DTYPE"],
    "reduction_dtype": "fp32",
    "learning_rate": float(os.environ["LR"]),
    "lora_rank": int(os.environ["LORA_RANK"]),
    "lora_alpha": int(os.environ["LORA_ALPHA"]),
    "seed": int(os.environ["TRAINING_SEED"]),
    "total_optimizer_steps": int(os.environ["TOTAL_TRAINING_STEPS"]),
    "weighting": train_audit.get("weighting_mode"),
    "sampling_unit": train_audit.get("training_contract", {}).get("sampling_unit"),
}
configured_projection = {
    key: configured_training[key]
    for key in resolved_contract
    if key != "seed"
}
resolved_projection = {
    key: value for key, value in resolved_contract.items() if key != "seed"
}
try:
    configured_seeds = [int(value) for value in configured_training["seeds"]]
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"training.seeds is missing or invalid: {error}") from error
if (
    configured_projection != resolved_projection
    or resolved_contract["seed"] not in configured_seeds
    or len(configured_seeds) != len(set(configured_seeds))
):
    raise SystemExit(
        "launcher settings disagree with the preregistered training config: "
        f"resolved={resolved_contract}, configured={configured_projection}, "
        f"allowed_seeds={configured_seeds}"
    )
files = {
    "trainer": Path(os.environ["OMNIOPD_TRAINER_PATH"]).resolve(),
    "launcher": Path(os.environ["OMNIOPD_LAUNCHER_PATH"]).resolve(),
    "loss": root / "src/omniopd/loss.py",
    "sampler": root / "src/omniopd/sampler.py",
    "dataset": root / "src/omniopd/torch_dataset.py",
    "verl_config": Path(os.environ["VERL_ROOT"]).resolve()
    / "verl/trainer/config/sft_trainer.yaml",
}
payload = {
    "protocol_version": "omniopd-v1",
    "artifact": "omniopd_training_launch",
    "schema_version": 1,
    "experiment": experiment.get("experiment"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "user_hydra_overrides": sys.argv[1:],
    "verl_root": str(verl_root),
    "verl_version": required_verl_version,
    "verl_version_declarations": resolved_verl_versions,
    "verl_git_revision": verl_revision,
    "code": fingerprint_code_tree(root),
    "code_revision": root_revision,
    "implementation": {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in files.items()
    },
    "inputs": {
        "train_files": describe_input(os.environ["TRAIN_FILES"]),
        "val_files": describe_input(os.environ["VAL_FILES"]),
        "model_path": describe_input(os.environ["MODEL_PATH"]),
        "experiment_config": describe_input(os.environ["EXPERIMENT_CONFIG"]),
        "train_audit": describe_input(os.environ["TRAIN_AUDIT"]),
    },
    "training_contract": {
        "num_gpus": int(os.environ["NUM_GPUS"]),
        "global_train_batch_size": int(os.environ["TRAIN_BSZ"]),
        "local_train_batch_size": int(os.environ["LOCAL_TRAIN_BSZ"]),
        "micro_batch_size_per_gpu": int(os.environ["MICRO_BSZ"]),
        "microbatches_per_optimizer_step": int(os.environ["MICROBATCHES_PER_STEP"]),
        "total_optimizer_steps": int(os.environ["TOTAL_TRAINING_STEPS"]),
        "configured_total_epochs_compatibility_only": int(os.environ["EPOCHS"]),
        "seed": int(os.environ["TRAINING_SEED"]),
        "model_load_dtype": os.environ["MODEL_DTYPE"],
        "forward_and_fsdp_param_dtype": os.environ["TRAINING_DTYPE"],
        "cross_entropy_dtype": "fp32",
        "gradient_reduce_dtype": "fp32",
        "attention_implementation": "sdpa",
        "distributed_strategy": "fsdp1_use_orig_params_true",
        "state_weight_normalization": "uniform_sample_expected_mass",
        "sampler_padding_weight": 0.0,
        "encoded_sequence_overflow_policy": "error",
        "validation_role": "arm_internal_sanity_only_no_early_stopping",
        "dataset_contract": resolved_contract,
    },
    "hyperparameters": {
        "learning_rate": os.environ["LR"],
        "max_length": int(os.environ["MAX_LENGTH"]),
        "lora_rank": int(os.environ["LORA_RANK"]),
        "lora_alpha": int(os.environ["LORA_ALPHA"]),
        "save_frequency": int(os.environ["SAVE_FREQ"]),
        "validation_frequency": int(os.environ["TEST_FREQ"]),
        "project_name": os.environ["PROJECT_NAME"],
        "experiment_name": os.environ["EXPERIMENT_NAME"],
    },
}
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=False)
manifest = output_dir / "launch_manifest.json"
manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

cd "${repo_root}"
"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  "${trainer_path}" \
  --config-dir "${verl_config_dir}" \
  --config-name sft_trainer \
  "$@" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=error \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
  model.partial_pretrain="${MODEL_PATH}" \
  model.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.target_modules=all-linear \
  optim.lr="${LR}" \
  trainer.total_epochs="${EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.seed="${TRAINING_SEED}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='["console"]' \
  ulysses_sequence_parallel_size=1 \
  use_remove_padding=false \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
