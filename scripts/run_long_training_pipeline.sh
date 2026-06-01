#!/usr/bin/env bash
set -euo pipefail

NUM_THREADS="${NUM_THREADS:-50}"
USE_CUDA="${USE_CUDA:-False}"
REPORT_DIR="${REPORT_DIR:-reports/long_training}"
LOG_DIR="${REPORT_DIR}/logs"
MANIFEST="${REPORT_DIR}/manifest.env"

mkdir -p "${LOG_DIR}" "${REPORT_DIR}/eval" "${REPORT_DIR}/curves"
: > "${MANIFEST}"

latest_run_dir() {
  local env_name="$1"
  ls -td "tmp/${env_name}"/* | head -n 1
}

record_var() {
  local name="$1"
  local value="$2"
  printf '%s=%q\n' "${name}" "${value}" | tee -a "${MANIFEST}"
}

run_train() {
  local label="$1"
  shift
  echo "[$(date '+%F %T')] START ${label}" | tee -a "${LOG_DIR}/pipeline.log"
  conda run -n EAI python train.py "$@" 2>&1 | tee "${LOG_DIR}/${label}.log"
  echo "[$(date '+%F %T')] DONE ${label}" | tee -a "${LOG_DIR}/pipeline.log"
}

run_eval() {
  local label="$1"
  shift
  echo "[$(date '+%F %T')] EVAL ${label}" | tee -a "${LOG_DIR}/pipeline.log"
  conda run -n EAI python scripts/evaluate_checkpoint.py "$@" 2>&1 | tee "${LOG_DIR}/${label}.log"
}

run_train fixed_run_to_goal --cfg config/run-to-goal-ants-v0.yaml --num_threads "${NUM_THREADS}" --use_cuda "${USE_CUDA}"
FIXED_RUN_DIR="$(latest_run_dir run-to-goal-ants-v0)"
record_var FIXED_RUN_DIR "${FIXED_RUN_DIR}"

run_train devants_warmup --cfg config/repro/devants-compatible-warmup-long.yaml --num_threads "${NUM_THREADS}" --use_cuda "${USE_CUDA}"
WARMUP_RUN_DIR="$(latest_run_dir robo-sumo-devants-v0)"
record_var WARMUP_RUN_DIR "${WARMUP_RUN_DIR}"

run_train devants_confrontation --cfg config/repro/devants-confrontation-long.yaml --ckpt_dir "${WARMUP_RUN_DIR}/models" --ckpt 100 --num_threads "${NUM_THREADS}" --use_cuda "${USE_CUDA}"
CONFRONTATION_RUN_DIR="$(latest_run_dir robo-sumo-devants-v0)"
record_var CONFRONTATION_RUN_DIR "${CONFRONTATION_RUN_DIR}"

run_train reproduction_robo_sumo_devants --cfg config/robo-sumo-devants-v0.yaml --num_threads "${NUM_THREADS}" --use_cuda "${USE_CUDA}"
REPRO_RUN_DIR="$(latest_run_dir robo-sumo-devants-v0)"
record_var REPRO_RUN_DIR "${REPRO_RUN_DIR}"

run_eval fixed_run_to_goal_eval --cfg config/run-to-goal-ants-v0.yaml --ckpt_dir "${FIXED_RUN_DIR}/models" --ckpt 1000 --episodes 100 --seed 300 --out "${REPORT_DIR}/eval/fixed_run_to_goal_epoch1000.json"
run_eval warmup_eval --cfg config/repro/devants-compatible-warmup-long.yaml --ckpt_dir "${WARMUP_RUN_DIR}/models" --ckpt 100 --episodes 100 --seed 301 --out "${REPORT_DIR}/eval/devants_warmup_epoch0100.json"
run_eval confrontation_eval --cfg runs/robo-sumo-devants-v0/config.yml --ckpt_dir "${CONFRONTATION_RUN_DIR}/models" --ckpt 1000 --episodes 100 --seed 302 --out "${REPORT_DIR}/eval/devants_confrontation_epoch1000.json"
run_eval original_runs_eval --cfg runs/robo-sumo-devants-v0/config.yml --ckpt_dir runs/robo-sumo-devants-v0/models --ckpt best --episodes 100 --seed 303 --out "${REPORT_DIR}/eval/original_runs_best.json"
run_eval reproduction_eval --cfg runs/robo-sumo-devants-v0/config.yml --ckpt_dir "${REPRO_RUN_DIR}/models" --ckpt 1000 --episodes 100 --seed 304 --out "${REPORT_DIR}/eval/reproduction_epoch1000.json"

conda run -n EAI python scripts/plot_training_curves.py \
  --run_dir "${FIXED_RUN_DIR}" \
  --run_dir "${WARMUP_RUN_DIR}" \
  --run_dir "${CONFRONTATION_RUN_DIR}" \
  --run_dir "${REPRO_RUN_DIR}" \
  --out_dir "${REPORT_DIR}/curves" 2>&1 | tee "${LOG_DIR}/plot_curves.log"

echo "[$(date '+%F %T')] LONG TRAINING PIPELINE COMPLETE" | tee -a "${LOG_DIR}/pipeline.log"
