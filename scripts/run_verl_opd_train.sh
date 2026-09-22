#!/usr/bin/env bash
# Standard token-level OPD training on a fixed Student state pool.
#
# The states come from a collected Student pool; the response is generated
# on-policy at training time and scored by the frozen Teacher on the identical
# prefix.  This is the "traditional per-token OPD" comparator arm.
set -euo pipefail

: "${OPD_DATA:?token-OPD rows required}"
: "${OPD_DATA_MANIFEST:?data manifest required}"
: "${OPD_STUDENT_MODEL:?Student model required}"
: "${OPD_TEACHER_MODEL:?frozen Teacher model required}"
: "${OPD_OUTPUT:?new output directory required}"
: "${NUM_GPUS:?student GPU count required}"
: "${TOTAL_TRAINING_STEPS:?fixed optimizer-step budget required}"

LR=${LR:-1e-6}
TRAIN_BSZ=${TRAIN_BSZ:-8}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
TEACHER_GPU_MEMORY=${TEACHER_GPU_MEMORY:-0.60}
STUDENT_GPU_MEMORY=${STUDENT_GPU_MEMORY:-0.30}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-standard_token_opd}
PROJECT_NAME=${PROJECT_NAME:-agent_omniopd}

python_candidate=${PYTHON_BIN:-python3}
command -v -- "${python_candidate}" >/dev/null || { echo "PYTHON_BIN missing" >&2; exit 2; }
python_bin=$("${python_candidate}" -c 'from pathlib import Path; import sys; print(Path(sys.executable).resolve())')
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
resolve() { "${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"; }
VERL_ROOT=$(resolve "${VERL_ROOT:?set VERL_ROOT}")
OPD_DATA=$(resolve "${OPD_DATA}")
OPD_DATA_MANIFEST=$(resolve "${OPD_DATA_MANIFEST}")
OPD_STUDENT_MODEL=$(resolve "${OPD_STUDENT_MODEL}")
OPD_TEACHER_MODEL=$(resolve "${OPD_TEACHER_MODEL}")
OPD_OUTPUT=$(resolve "${OPD_OUTPUT}")

export OPD_DATA OPD_DATA_MANIFEST OPD_STUDENT_MODEL OPD_TEACHER_MODEL OPD_OUTPUT
export OMNIOPD_REPO_ROOT="${repo_root}" VERL_ROOT
export PYTHONPATH="${repo_root}/src:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

root = Path(os.environ["OMNIOPD_REPO_ROOT"]).resolve()
sys.path.insert(0, str(root / "src"))
from omniopd.provenance import fingerprint_path, sha256_file

data = Path(os.environ["OPD_DATA"]).resolve()
manifest_path = Path(os.environ["OPD_DATA_MANIFEST"]).resolve()
student = Path(os.environ["OPD_STUDENT_MODEL"]).resolve()
teacher = Path(os.environ["OPD_TEACHER_MODEL"]).resolve()
output = Path(os.environ["OPD_OUTPUT"]).resolve()

if output.exists():
    raise SystemExit(f"refusing to reuse existing output directory: {output}")
for label, path in (
    ("OPD rows", data),
    ("data manifest", manifest_path),
    ("Student config", student / "config.json"),
    ("Teacher config", teacher / "config.json"),
):
    if not path.is_file():
        raise SystemExit(f"{label} is missing: {path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("artifact") != "verl_token_opd_training_data":
    raise SystemExit("data manifest is not a token-OPD training manifest")
if manifest.get("teacher_context") != "identical_to_student_prefix":
    raise SystemExit("this launcher implements standard OPD and needs the shared prefix")
if manifest.get("rows", {}).get("sha256") != sha256_file(data):
    raise SystemExit("OPD rows do not match the data manifest")

output.mkdir(parents=True, exist_ok=False)
print(json.dumps({
    "artifact": "standard_token_opd_launch",
    "rows": fingerprint_path(data),
    "data_manifest": fingerprint_path(manifest_path),
    "student_model": fingerprint_path(student / "config.json"),
    "teacher_model": fingerprint_path(teacher / "config.json"),
}, ensure_ascii=False, indent=2))
PY

cd "${repo_root}"
"${python_bin}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="${OPD_DATA}" \
  data.val_files="${OPD_DATA}" \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.val_batch_size="${TRAIN_BSZ}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  data.shuffle=true \
  data.dataloader_num_workers=0 \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path="${OPD_STUDENT_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BSZ}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BSZ}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${STUDENT_GPU_MEMORY}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 2)) \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${repo_root}/configs/verl_v080_opd_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=omniopd_action_token_opd \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=omniopd.verl_opd.OmniOPDAgentLoopManager \
  trainer.balance_batch=false \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.val_before_train=false \
  trainer.save_freq=0 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.resume_mode=disable \
  trainer.default_local_dir="${OPD_OUTPUT}/checkpoints" \
  trainer.rollout_data_dir="${OPD_OUTPUT}/rollouts" \
  hydra.run.dir="${OPD_OUTPUT}/hydra" \
  distillation.enabled=true \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="${OPD_TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY}" \
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 2)) \
  distillation.teacher_models.teacher_model.inference.temperature=1.0 \
  distillation.distillation_loss.loss_mode=k2 \
  distillation.distillation_loss.use_task_rewards=false \
  distillation.distillation_loss.use_policy_gradient=false \
  distillation.distillation_loss.loss_max_clamp=null \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  "$@" \
  2>&1 | tee "${OPD_OUTPUT}/train.log"

"${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

root = Path(os.environ["OMNIOPD_REPO_ROOT"]).resolve()
sys.path.insert(0, str(root / "src"))
from omniopd.provenance import fingerprint_path, sha256_file

output = Path(os.environ["OPD_OUTPUT"]).resolve()
checkpoint = output / "checkpoints" / f"global_step_{int(os.environ['TOTAL_TRAINING_STEPS'])}"
if not checkpoint.is_dir():
    raise SystemExit(f"training finished without the expected checkpoint: {checkpoint}")
payload = {
    "artifact": "standard_token_opd_completion",
    "total_optimizer_steps": int(os.environ["TOTAL_TRAINING_STEPS"]),
    "checkpoint": fingerprint_path(checkpoint),
    "train_log": {"path": str(output / "train.log"), "sha256": sha256_file(output / "train.log")},
}
(output / "completion_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"completion manifest written for {checkpoint}")
PY
