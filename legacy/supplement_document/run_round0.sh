#!/usr/bin/env bash
set -euo pipefail
 
# Assumptions:
#   1) Student SFT model is already served by vLLM at http://127.0.0.1:8000/v1
#   2) /root/.deepseek_key contains the Teacher key (or DEEPSEEK_API_KEY is exported)
#   3) ENV_CONFIG points to your rebuilt ALFWorld YAML with train games and training_method: dqn
 
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_CONFIG=${ENV_CONFIG:-/root/data/alfworld/configs/alfworld.yaml}
STUDENT_MODEL=${STUDENT_MODEL:-qwen3-4b-sft}
WORKDIR=${WORKDIR:-/root/data/alfworld/opd/route_a_round0}
LIMIT_GAMES=${LIMIT_GAMES:-20}
M=${M:-3}
N=${N:-1}
TEACHER_TEMP=${TEACHER_TEMP:-0.5}
 
mkdir -p "${WORKDIR}"
 
python "${SCRIPT_DIR}/opd_collect.py" \
  --env-config "${ENV_CONFIG}" \
  --output "${WORKDIR}/selected_turns.jsonl" \
  --limit-games "${LIMIT_GAMES}" \
  --turns-per-episode "${M}" \
  --selection random \
  --student-model "${STUDENT_MODEL}" \
  --student-temperature 0.0 \
  --teacher-model deepseek-v4-flash \
  --teacher-temperature "${TEACHER_TEMP}" \
  --teacher-samples "${N}" \
  --target-mode full \
  --action-marker first
 
python "${SCRIPT_DIR}/opd_prepare.py" \
  --input "${WORKDIR}/selected_turns.jsonl" \
  --train-output "${WORKDIR}/train.parquet" \
  --val-output "${WORKDIR}/val.parquet" \
  --val-fraction 0.1
 
echo "Collection/preparation complete."
echo "Inspect ${WORKDIR}/selected_turns.jsonl before training."
echo "Then run: TRAIN_FILES=${WORKDIR}/train.parquet VAL_FILES=${WORKDIR}/val.parquet bash "${SCRIPT_DIR}/run_verl_sft.sh""
