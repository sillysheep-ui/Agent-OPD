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
: "${ANNOTATION_PAIR_MANIFEST:?set ANNOTATION_PAIR_MANIFEST to the validated annotation-pair JSON}"
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

# Resolve every caller-supplied path before any existence check, fingerprint,
# directory creation, export, or cwd change.  This keeps the manifest and the
# trainer on the same bytes even when the wrapper is launched outside the repo.
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
MODEL_PATH=$(resolve_path "${MODEL_PATH}")
TRAIN_FILES=$(resolve_path "${TRAIN_FILES}")
VAL_FILES=$(resolve_path "${VAL_FILES}")
OUTPUT_DIR=$(resolve_path "${OUTPUT_DIR}")
EXPERIMENT_CONFIG=$(resolve_path "${EXPERIMENT_CONFIG}")
TRAIN_AUDIT=$(resolve_path "${TRAIN_AUDIT}")
ANNOTATION_PAIR_MANIFEST=$(resolve_path "${ANNOTATION_PAIR_MANIFEST}")

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse existing OUTPUT_DIR=${OUTPUT_DIR}" >&2
  exit 2
fi

trainer_path="${repo_root}/integrations/verl/fsdp_sft_trainer.py"
verl_config_dir="${VERL_ROOT}/verl/trainer/config"
verl_config_file="${verl_config_dir}/sft_trainer.yaml"

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
  "${ANNOTATION_PAIR_MANIFEST}" \
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
export EXPERIMENT_CONFIG TRAIN_AUDIT ANNOTATION_PAIR_MANIFEST
export OMNIOPD_REPO_ROOT="${repo_root}"
export OMNIOPD_TRAINER_PATH="${trainer_path}"
export OMNIOPD_LAUNCHER_PATH="${repo_root}/scripts/run_verl_train.sh"

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

from omniopd.evaluation import validate_annotation_pair_manifest
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
    return {
        "kind": "directory",
        "tree_sha256": digest.hexdigest(),
        "files": count,
        "bytes": total_bytes,
    }


def describe_input(raw: str) -> dict[str, object]:
    description: dict[str, object] = {"value": raw}
    path = Path(raw).expanduser()
    if path.is_file():
        description.update(
            {
                "resolved_path": str(path.resolve()),
                "kind": "file",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    elif path.is_dir():
        description.update({"resolved_path": str(path.resolve()), **describe_tree(path)})
    return description


def content_identity(value):
    if not isinstance(value, dict):
        return value
    return {
        key: content_identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


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


def require_clean_git_tree(path: Path, role: str) -> None:
    try:
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot audit {role} Git worktree: {error}") from error
    if status:
        raise SystemExit(
            f"{role} Git worktree has tracked or untracked files; "
            "commit or remove them before a confirmatory launch"
        )


def distribution_version(name: str) -> str:
    try:
        value = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit(f"required training distribution is missing: {name}") from error
    if not value:
        raise SystemExit(f"required training distribution has no version: {name}")
    return value


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
require_clean_git_tree(root, "OmniOPD")
require_clean_git_tree(verl_root, "veRL")
try:
    import torch
except ImportError as error:
    raise SystemExit("PyTorch is required for the audited training runtime") from error
if not torch.cuda.is_available() or torch.cuda.device_count() < int(os.environ["NUM_GPUS"]):
    raise SystemExit("the audited training launch requires the declared CUDA GPUs")
if not torch.version.cuda:
    raise SystemExit("the audited training launch requires a CUDA-enabled PyTorch build")
try:
    driver_versions = sorted(
        set(
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
        )
    )
except (OSError, subprocess.CalledProcessError) as error:
    raise SystemExit("nvidia-smi driver identity is required") from error
if not driver_versions:
    raise SystemExit("nvidia-smi returned no driver identity")
training_runtime = {
    "python": platform.python_version(),
    "python_executable": describe_input(sys.executable),
    "platform": platform.platform(),
    "distributions": {
        name: distribution_version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "numpy",
            "hydra-core",
        )
    },
    "torch_cuda": torch.version.cuda,
    "cudnn": None if torch.backends.cudnn.version() is None else str(torch.backends.cudnn.version()),
    "gpu_count": torch.cuda.device_count(),
    "declared_gpu_count": int(os.environ["NUM_GPUS"]),
    "gpus": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ],
    "nvidia_driver_versions": driver_versions,
}
experiment_config_path = Path(os.environ["EXPERIMENT_CONFIG"]).resolve()
train_audit_path = Path(os.environ["TRAIN_AUDIT"]).resolve()
annotation_pair_path = Path(os.environ["ANNOTATION_PAIR_MANIFEST"]).resolve()
train_path = Path(os.environ["TRAIN_FILES"]).resolve()
val_path = Path(os.environ["VAL_FILES"]).resolve()
experiment = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
train_audit = json.loads(train_audit_path.read_text(encoding="utf-8"))
annotation_pair = json.loads(annotation_pair_path.read_text(encoding="utf-8"))
try:
    validated_annotation_pair = validate_annotation_pair_manifest(
        annotation_pair,
        manifest_sha256=sha256(annotation_pair_path),
    )
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"invalid annotation-pair manifest: {error}") from error
if (
    not isinstance(experiment, dict)
    or experiment.get("protocol_version") != "omniopd-v1"
    or not isinstance(experiment.get("experiment"), str)
    or not experiment["experiment"]
):
    raise SystemExit("experiment config is missing its canonical protocol/arm identity")
if train_path == val_path or train_audit_path in {train_path, val_path}:
    raise SystemExit("training, validation, and build-audit inputs must be distinct files")
if annotation_pair_path in {train_path, val_path, train_audit_path, experiment_config_path}:
    raise SystemExit("annotation-pair manifest must be a distinct input file")
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
audit_contract = train_audit.get("training_contract", {})
if (
    actual != expected
    or train_audit.get("artifact") != "action_only_training_data_audit"
    or train_audit.get("protocol_version") != "omniopd-v1"
    or train_audit.get("code") != fingerprint_code_tree(root)
    or train_audit.get("code_revision") != root_revision
    or not isinstance(train_audit.get("correction_manifest"), dict)
    or not train_audit["correction_manifest"].get("sha256")
    or not isinstance(train_audit.get("annotation_contract"), dict)
    or not isinstance(train_audit.get("split"), dict)
    or int(train_audit.get("training_rows", 0)) <= 0
    or int(train_audit.get("validation_rows", 0)) <= 0
):
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
if (
    int(audit_contract.get("global_batch_size", -1)) != resolved_contract["global_batch_size"]
    or int(audit_contract.get("total_optimizer_steps", -1))
    != resolved_contract["total_optimizer_steps"]
    or audit_contract.get("weighting") != resolved_contract["weighting"]
    or int(audit_contract.get("split_seed", -1))
    != int(configured_training["data_split_seed"])
):
    raise SystemExit("build audit and resolved training contract disagree")
try:
    arm_contract = {
        "comparison_family": "fixed_teacher_budget",
        "experiment": experiment["experiment"],
        "teacher_budget_definition": str(experiment["teacher_budget_definition"]),
        "games": int(experiment["games"]),
        "states_per_game": int(experiment["states_per_game"]),
        "distinct_states_M": int(experiment["distinct_states_M"]),
        "teacher_samples_per_state_N": int(
            experiment["teacher_samples_per_state_N"]
        ),
        "teacher_budget_B": int(experiment["teacher_budget_B"]),
        "teacher_sampling_profile": str(experiment["teacher_sampling_profile"]),
        "teacher_max_tokens": int(experiment["teacher_max_tokens"]),
        "invalid_policy": str(experiment["invalid_policy"]),
        "selection": str(experiment["selection"]),
        "selection_seed": int(experiment["selection_seed"]),
        "state_pool_games_G": int(experiment["state_pool"]["games_G"]),
        "state_pool_seed": int(experiment["state_pool"]["seed"]),
        "state_pool_contract": experiment["state_pool"],
        "weighting": str(configured_training["weighting"]),
        "sampling_unit": str(configured_training["sampling_unit"]),
        "data_split_seed": int(configured_training["data_split_seed"]),
    }
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"fixed-budget arm config is incomplete: {error}") from error
if (
    arm_contract["teacher_budget_definition"] != "annotation_api_attempts"
    or arm_contract["invalid_policy"] != "retain_as_missing_no_free_retry"
    or arm_contract["teacher_max_tokens"] <= 0
    or arm_contract["games"] <= 0
    or arm_contract["states_per_game"] <= 0
    or arm_contract["distinct_states_M"]
    != arm_contract["games"] * arm_contract["states_per_game"]
    or arm_contract["state_pool_games_G"] != arm_contract["games"]
    or not isinstance(arm_contract["state_pool_contract"], dict)
    or int(arm_contract["state_pool_contract"].get("environment_seed", -1)) < 0
    or arm_contract["state_pool_contract"].get("state_source") != "student"
    or arm_contract["distinct_states_M"]
    * arm_contract["teacher_samples_per_state_N"]
    != arm_contract["teacher_budget_B"]
):
    raise SystemExit("experiment does not encode a valid fixed-B arm")
pair_contract = annotation_pair.get("pair_contract")
if (
    annotation_pair.get("artifact") != "omniopd_annotation_pair"
    or annotation_pair.get("protocol_version") != "omniopd-v1"
    or int(annotation_pair.get("schema_version", -1)) != 2
    or annotation_pair.get("pair_kind") != "fixed_budget_annotation_runs"
    or annotation_pair.get("comparison_contrast") != "breadth_depth"
    or not isinstance(pair_contract, dict)
    or pair_contract.get("comparison_contrast") != "breadth_depth"
    or annotation_pair.get("pair_contract_sha256")
    != hashlib.sha256(
        json.dumps(
            pair_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
):
    raise SystemExit("annotation-pair manifest has an invalid schema or contract digest")
members = pair_contract.get("members")
member = members.get(experiment["experiment"]) if isinstance(members, dict) else None
manifest_entries = annotation_pair.get("annotation_manifests")
shared_pair_contract = pair_contract.get("shared_contract")
audit_annotation_contract = train_audit.get("annotation_contract", {})
matching_entries = (
    [entry for entry in manifest_entries if entry.get("experiment") == experiment["experiment"]]
    if isinstance(manifest_entries, list)
    and all(isinstance(entry, dict) for entry in manifest_entries)
    else []
)
if (
    not isinstance(members, dict)
    or len(members) != 2
    or not isinstance(shared_pair_contract, dict)
    or shared_pair_contract.get("code") != fingerprint_code_tree(root)
    or shared_pair_contract.get("code_revision") != root_revision
    or shared_pair_contract.get("teacher_budget_definition")
    != "annotation_api_attempts"
    or shared_pair_contract.get("invalid_policy")
    != "retain_as_missing_no_free_retry"
    or shared_pair_contract != audit_annotation_contract.get("shared_contract")
    or not isinstance(manifest_entries, list)
    or len(manifest_entries) != 2
    or {entry.get("experiment") for entry in manifest_entries}
    != set(members)
    or any(
        entry.get("kind") != "file"
        or int(entry.get("bytes", -1)) <= 0
        or entry.get("sha256")
        != members[entry["experiment"]].get("annotation_manifest_sha256")
        for entry in manifest_entries
    )
):
    raise SystemExit("annotation-pair shared contract or member inventory is invalid")
behavior_student = shared_pair_contract.get("behavior_student")
if (
    not isinstance(behavior_student, dict)
    or content_identity(describe_input(os.environ["MODEL_PATH"]))
    != behavior_student.get("behavior_artifact")
    or behavior_student.get("tokenizer") != behavior_student.get("behavior_artifact")
):
    raise SystemExit(
        "MODEL_PATH must be the single complete frozen behavior Student, including tokenizer"
    )
if not isinstance(member, dict) or member.get("experiment") != experiment["experiment"]:
    raise SystemExit("current experiment is not a member of the annotation pair")
if len(matching_entries) != 1:
    raise SystemExit("annotation-pair manifest does not uniquely identify this arm manifest")
expected_pair_member = audit_annotation_contract.get("member")
if not isinstance(expected_pair_member, dict):
    raise SystemExit("training audit has no canonical annotation member contract")
pair_mismatches = {
    key: (member.get(key), expected_value)
    for key, expected_value in expected_pair_member.items()
    if member.get(key) != expected_value
}
if matching_entries[0].get("sha256") != member.get("annotation_manifest_sha256"):
    pair_mismatches["annotation_manifests.sha256"] = (
        matching_entries[0].get("sha256"),
        member.get("annotation_manifest_sha256"),
    )
if pair_mismatches:
    raise SystemExit(f"annotation pair does not bind the realized training inputs: {pair_mismatches}")
annotation_pair_binding = {
    "pair_schema_version": validated_annotation_pair["schema_version"],
    "annotation_pair_manifest_sha256": validated_annotation_pair[
        "annotation_pair_manifest_sha256"
    ],
    "pair_contract_sha256": annotation_pair["pair_contract_sha256"],
    "pair_contract": pair_contract,
    "comparison_contrast": "breadth_depth",
    "effect_direction": validated_annotation_pair["effect_direction"],
    "arm_roles": validated_annotation_pair["arm_roles"],
    "experiment": experiment["experiment"],
    "member": member,
}
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
    "schema_version": 2,
    "experiment": experiment.get("experiment"),
    "arm_contract": arm_contract,
    "annotation_pair_binding": annotation_pair_binding,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "user_hydra_overrides": sys.argv[1:],
    "verl_root": str(verl_root),
    "verl_version": required_verl_version,
    "verl_version_declarations": resolved_verl_versions,
    "verl_git_revision": verl_revision,
    "git_worktrees_clean": {"omniopd": True, "verl": True},
    "training_runtime": training_runtime,
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
        "annotation_pair": describe_input(os.environ["ANNOTATION_PAIR_MANIFEST"]),
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
        "target_modules": "all-linear",
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

# This block is reached only when both torchrun and tee completed successfully
# (set -o pipefail).  Consequently a failed/interrupted run cannot obtain a
# canonical completion manifest merely because a checkpoint directory exists.
"${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from omniopd.evaluation import validate_training_launch_manifest
from omniopd.provenance import fingerprint_path, sha256_json
from omniopd.validation import validate_lora_checkpoint_directory


def content_identity(value):
    if not isinstance(value, dict):
        return value
    return {
        key: content_identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
launch_path = output_dir / "launch_manifest.json"
completion_path = output_dir / "completion_manifest.json"
resolved_config_path = output_dir / "resolved_config.yaml"
train_log_path = output_dir / "train.log"
launch = json.loads(launch_path.read_text(encoding="utf-8"))
identity = validate_training_launch_manifest(launch)
final_step = int(identity["total_optimizer_steps"])
checkpoint_path = output_dir / f"global_step_{final_step}"
if completion_path.exists():
    raise SystemExit("refusing to overwrite a training completion manifest")
for role, path in {
    "final checkpoint": checkpoint_path,
    "resolved trainer config": resolved_config_path,
    "training log": train_log_path,
}.items():
    if not path.exists():
        raise SystemExit(f"successful trainer exit did not produce {role}: {path}")
checkpoint = fingerprint_path(checkpoint_path)
if checkpoint.get("kind") != "directory" or int(checkpoint.get("files", 0)) <= 0:
    raise SystemExit("final checkpoint is not a non-empty content-addressed directory")
checkpoint_format = validate_lora_checkpoint_directory(
    checkpoint_path,
    expected_rank=int(launch["hyperparameters"]["lora_rank"]),
    expected_alpha=float(launch["hyperparameters"]["lora_alpha"]),
    expected_target_modules_policy=str(launch["hyperparameters"]["target_modules"]),
)
payload = {
    "protocol_version": "omniopd-v1",
    "artifact": "omniopd_training_completion",
    "schema_version": 2,
    "completion_status": "completed",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": identity["experiment"],
    "training_seed": identity["training_seed"],
    "final_global_step": final_step,
    "code": launch["code"],
    "code_revision": launch["code_revision"],
    "launch_manifest": fingerprint_path(launch_path),
    "final_checkpoint": checkpoint,
    "checkpoint_format": checkpoint_format,
    "resolved_trainer_config": fingerprint_path(resolved_config_path),
    "train_log": fingerprint_path(train_log_path),
    "training_inputs_sha256": sha256_json(content_identity(launch["inputs"])),
    "training_contract_sha256": sha256_json(launch["training_contract"]),
    "hyperparameters_sha256": sha256_json(launch["hyperparameters"]),
    "annotation_pair_binding": identity["annotation_pair_binding"],
}
with completion_path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
PY
