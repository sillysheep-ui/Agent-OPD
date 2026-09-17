#!/usr/bin/env bash
# Nonconfirmatory one-step veRL v0.8.0 Student/Teacher token-OPD smoke.
set -euo pipefail

: "${OPD_SMOKE_DATA:?one-row ALFWorld JSONL required}"
: "${OPD_STUDENT_MODEL:?SFT Student directory required}"
: "${OPD_TEACHER_MODEL:?frozen Teacher directory required}"
: "${OPD_SMOKE_OUTPUT:?new output directory required}"

if [[ ! -f "$OPD_SMOKE_DATA" || ! -f "$OPD_STUDENT_MODEL/config.json" || ! -f "$OPD_TEACHER_MODEL/config.json" ]]; then
  echo "OPD smoke input or model config is missing" >&2
  exit 2
fi
if [[ -e "$OPD_SMOKE_OUTPUT" ]]; then
  echo "OPD smoke output already exists: $OPD_SMOKE_OUTPUT" >&2
  exit 2
fi
mkdir -p "$OPD_SMOKE_OUTPUT"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$OPD_SMOKE_DATA" \
  data.val_files="$OPD_SMOKE_DATA" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=64 \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  data.shuffle=false \
  data.dataloader_num_workers=0 \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path="$OPD_STUDENT_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=0 \
  actor_rollout_ref.rollout.max_model_len=1152 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/opt/agent/configs/verl_v080_opd_agent_loops.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=omniopd_action_token_opd \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=omniopd.verl_opd.OmniOPDAgentLoopManager \
  trainer.balance_batch=false \
  trainer.logger='["console"]' \
  trainer.project_name=omniopd_technical_smoke \
  trainer.experiment_name=one_state_one_step \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=false \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$OPD_SMOKE_OUTPUT/checkpoints" \
  trainer.rollout_data_dir="$OPD_SMOKE_OUTPUT/rollouts" \
  distillation.enabled=true \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$OPD_TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.45 \
  distillation.teacher_models.teacher_model.inference.max_model_len=1152 \
  distillation.distillation_loss.loss_mode=k3 \
  distillation.distillation_loss.use_task_rewards=false \
  distillation.distillation_loss.use_policy_gradient=false \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  "$@"
