#!/usr/bin/env bash
# veRL's native on-policy distillation plus one hook: the Teacher prompt is
# rendered with the Teacher's own chat template (see src/omniopd/opd_native.py).
#
# This launcher deliberately mirrors examples/on_policy_distillation_trainer/
# run_qwen3_0.6b_opd_veomni.sh flag for flag, with four documented deviations:
#   * fsdp instead of veomni (our server runs the FSDP engine),
#   * our ALFWorld fixed-state rows instead of gsm8k parquet,
#   * LoRA rank 16 instead of full fine-tuning, with LR raised accordingly
#     (the official 1e-5 assumes every weight moves; a rank-16 update needs a
#     larger step to reach a comparable ||dW||/||W||),
#   * max_response_length 512 because an ALFWorld action line is short.
set -euo pipefail

: "${OPD_DATA:?training rows required}"
: "${OPD_DATA_MANIFEST:?data manifest required}"
: "${OPD_STUDENT_MODEL:?Student model required}"
: "${OPD_TEACHER_MODEL:?frozen Teacher model required}"
: "${OPD_OUTPUT:?new output directory required}"
: "${NUM_GPUS:?student GPU count required}"
: "${TOTAL_TRAINING_STEPS:?fixed optimizer-step budget required}"

# Official OPD defaults, overridable.
LOSS_MODE=${LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
TRAIN_BSZ=${TRAIN_BSZ:-64}
MICRO_BSZ=${MICRO_BSZ:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
LR=${LR:-1e-4}
SAVE_FREQ=${SAVE_FREQ:-43}
TEACHER_GPU_MEMORY=${TEACHER_GPU_MEMORY:-0.60}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_GPUS=${TEACHER_GPUS:-2}
# veRL asserts that (num_replicas * per_replica_world_size) equals the size of
# the distillation resource pool, so derive the replica count instead of hoping.
TEACHER_REPLICAS=${TEACHER_REPLICAS:-$((TEACHER_GPUS / TEACHER_TP))}
STUDENT_GPU_MEMORY=${STUDENT_GPU_MEMORY:-0.30}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TEMPERATURE=${TEMPERATURE:-1.0}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-native_opd}
PROJECT_NAME=${PROJECT_NAME:-agent_omniopd}
AGENT_LOOP_MANAGER=${AGENT_LOOP_MANAGER:-omniopd.opd_native.TeacherTemplateAgentLoopManager}

case "${SAVE_FREQ}" in
  ''|*[!0-9]*) echo "SAVE_FREQ must be a positive integer, got '${SAVE_FREQ}'" >&2; exit 2 ;;
esac
if [ "${SAVE_FREQ}" -le 0 ]; then
  echo "SAVE_FREQ must be positive: veRL skips every checkpoint when it is 0" >&2
  exit 2
fi
if [ $((TEACHER_REPLICAS * TEACHER_TP)) -ne "${TEACHER_GPUS}" ]; then
  echo "TEACHER_GPUS=${TEACHER_GPUS} must be a multiple of TEACHER_TP=${TEACHER_TP}" >&2
  exit 2
fi
echo "teacher pool: ${TEACHER_GPUS} GPUs = ${TEACHER_REPLICAS} replica(s) x TP ${TEACHER_TP}"
echo "checkpoint schedule: every ${SAVE_FREQ} steps plus the final step ${TOTAL_TRAINING_STEPS}"

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

export OPD_DATA OPD_DATA_MANIFEST OPD_STUDENT_MODEL OPD_TEACHER_MODEL OPD_OUTPUT SAVE_FREQ
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
if manifest.get("rows", {}).get("sha256") != sha256_file(data):
    raise SystemExit("OPD rows do not match the data manifest")

output.mkdir(parents=True, exist_ok=False)
print(json.dumps({
    "artifact": "native_opd_launch",
    "rows": fingerprint_path(data),
    "data_manifest": fingerprint_path(manifest_path),
    "student_model": fingerprint_path(student / "config.json"),
    "teacher_model": fingerprint_path(teacher / "config.json"),
}, ensure_ascii=False, indent=2))
PY

data_source=$("${python_bin}" -c '
import json,sys
print(json.loads(open(sys.argv[1]).readline()).get("data_source",""))
' "${OPD_DATA}")
if [ -z "${data_source}" ]; then
  echo "training rows carry no data_source to route the Teacher" >&2
  exit 2
fi

cd "${repo_root}"
"${python_bin}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
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
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${repo_root}/configs/verl_v080_native_opd_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=native_opd_zero_reward \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class="${AGENT_LOOP_MANAGER}" \
  trainer.balance_batch=false \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.val_before_train=false \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.resume_mode=disable \
  trainer.default_local_dir="${OPD_OUTPUT}/checkpoints" \
  trainer.rollout_data_dir="${OPD_OUTPUT}/rollouts" \
  hydra.run.dir="${OPD_OUTPUT}/hydra" \
  distillation.enabled=True \
  distillation.n_gpus_per_node="${TEACHER_GPUS}" \
  distillation.nnodes=1 \
  +distillation.teacher_models.teacher.key="${data_source}" \
  +distillation.teacher_models.teacher.model_path="${OPD_TEACHER_MODEL}" \
  +distillation.teacher_models.teacher.num_replicas="${TEACHER_REPLICAS}" \
  +distillation.teacher_models.teacher.inference.name=vllm \
  +distillation.teacher_models.teacher.inference.tensor_model_parallel_size="${TEACHER_TP}" \
  +distillation.teacher_models.teacher.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY}" \
  +distillation.teacher_models.teacher.inference.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1)) \
  distillation.distillation_loss.loss_mode="${LOSS_MODE}" \
  distillation.distillation_loss.topk="${DISTILLATION_TOPK}" \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT}" \
  distillation.distillation_loss.loss_max_clamp=10.0 \
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
    raise SystemExit(
        f"training finished without the expected checkpoint: {checkpoint}. "
        f"save_freq={os.environ['SAVE_FREQ']}; veRL only enters its checkpoint "
        "branch for a positive save_freq and always saves on the last step."
    )
payload = {
    "artifact": "native_opd_completion",
    "total_optimizer_steps": int(os.environ["TOTAL_TRAINING_STEPS"]),
    "checkpoint": fingerprint_path(checkpoint),
    "train_log": {"path": str(output / "train.log"), "sha256": sha256_file(output / "train.log")},
}
(output / "completion_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"completion manifest written for {checkpoint}")
PY
