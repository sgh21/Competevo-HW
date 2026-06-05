#!/usr/bin/env bash
set -euo pipefail

CFG="${CFG:-config/repro/unified-devant-training.yaml}"
WARMUP_CKPT_DIR="${WARMUP_CKPT_DIR:-tmp/robo-sumo-devants-v0/base-rtg-fixed-epoch100-20260603_183245/models}"
WARMUP_CKPT="${WARMUP_CKPT:-best}"
NUM_THREADS="${NUM_THREADS:-12}"
USE_CUDA="${USE_CUDA:-True}"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
CHECK_SECONDS="${CHECK_SECONDS:-300}"
REPORT_DIR="${REPORT_DIR:-reports/unified_modes}"
RUN_SET="${RUN_SET:-parallel-$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${REPORT_DIR}/logs/${RUN_SET}"
PID_DIR="${REPORT_DIR}/pids/${RUN_SET}"
MANIFEST="${REPORT_DIR}/formal_parallel_manifest.tsv"
PROGRESS_LOG="${LOG_DIR}/formal_parallel_progress.log"

mkdir -p "${LOG_DIR}" "${PID_DIR}" "${REPORT_DIR}/eval" "${REPORT_DIR}/curves"

printf 'label\tmode\tmorph_optim_agents\trun_dir\tstdout_log\tpid\tstarted_at\tfinished_at\tstatus\n' > "${MANIFEST}"
: > "${PROGRESS_LOG}"

LABELS=(
  "${RUN_SET}-sp-fixed"
  "${RUN_SET}-tp-fixed"
  "${RUN_SET}-sp-morph"
  "${RUN_SET}-tp-morph"
  "${RUN_SET}-tp-mixed-a0morph"
)
MODES=(
  "selfplay"
  "two_player"
  "selfplay"
  "two_player"
  "two_player"
)
MORPHS=(
  "none"
  "none"
  "all"
  "all"
  "0"
)

latest_run_dir_for_label() {
  local label="$1"
  ls -td "tmp/robo-sumo-devants-v0/${label}-"* 2>/dev/null | head -n 1 || true
}

latest_train_log() {
  local run_dir="$1"
  find "${run_dir}/log" -maxdepth 1 -type f -name '*.log' 2>/dev/null | sort | tail -n 1 || true
}

latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}/models" -maxdepth 2 -type f -name 'epoch_*.p' -printf '%f\n' 2>/dev/null | sort | tail -n 1 || true
}

last_field() {
  local pattern="$1"
  local sed_expr="$2"
  local log="$3"
  grep -E "${pattern}" "${log}" | tail -n 1 | sed -E "${sed_expr}" || true
}

progress_line() {
  local label="$1"
  local run_dir="$2"
  local train_log
  train_log="$(latest_train_log "${run_dir}")"
  if [[ -z "${train_log}" ]]; then
    printf '[%s] %s run_dir=%s waiting_for_log latest_ckpt=%s\n' \
      "$(date '+%F %T')" "${label}" "${run_dir}" "$(latest_checkpoint "${run_dir}")"
    return
  fi

  local iteration total reward0 reward1 win0 win1 reward_sp win_sp
  iteration="$(last_field '#-+ Iteration [0-9]+' 's/.*Iteration ([0-9]+).*/\1/' "${train_log}")"
  total="$(last_field 'Total time:' 's/.*Total time:[[:space:]]*([0-9.]+) min.*/\1/' "${train_log}")"
  reward0="$(last_field 'Agent_0 gets eval reward:' 's/.*reward: (-?[0-9.]+).*/\1/' "${train_log}")"
  reward1="$(last_field 'Agent_1 gets eval reward:' 's/.*reward: (-?[0-9.]+).*/\1/' "${train_log}")"
  win0="$(last_field 'Agent_0 gets win rate:' 's/.*rate: ([0-9.]+).*/\1/' "${train_log}")"
  win1="$(last_field 'Agent_1 gets win rate:' 's/.*rate: ([0-9.]+).*/\1/' "${train_log}")"
  reward_sp="$(last_field 'Agent gets eval reward:' 's/.*reward: (-?[0-9.]+).*/\1/' "${train_log}")"
  win_sp="$(last_field 'Agent gets win rate:' 's/.*rate: ([0-9.]+).*/\1/' "${train_log}")"

  if [[ -n "${reward_sp}" || -n "${win_sp}" ]]; then
    printf '[%s] %s epoch=%s total_min=%s reward=%s win=%s latest_ckpt=%s run_dir=%s\n' \
      "$(date '+%F %T')" "${label}" "${iteration:-?}" "${total:-?}" \
      "${reward_sp:-?}" "${win_sp:-?}" "$(latest_checkpoint "${run_dir}")" "${run_dir}"
  else
    printf '[%s] %s epoch=%s total_min=%s reward0=%s reward1=%s win0=%s win1=%s latest_ckpt=%s run_dir=%s\n' \
      "$(date '+%F %T')" "${label}" "${iteration:-?}" "${total:-?}" \
      "${reward0:-?}" "${reward1:-?}" "${win0:-?}" "${win1:-?}" \
      "$(latest_checkpoint "${run_dir}")" "${run_dir}"
  fi
}

update_manifest_status() {
  local label="$1"
  local finished_at="$2"
  local status="$3"
  awk -F '\t' -v OFS='\t' -v label="${label}" -v finished="${finished_at}" -v status="${status}" '
    $1 == label {$8 = finished; $9 = status}
    {print}
  ' "${MANIFEST}" > "${MANIFEST}.tmp"
  mv "${MANIFEST}.tmp" "${MANIFEST}"
}

declare -a PIDS RUN_DIRS STATUSES
for i in "${!LABELS[@]}"; do
  PIDS[$i]=""
  RUN_DIRS[$i]=""
  STATUSES[$i]="PENDING"
done

start_job() {
  local i="$1"
  local label="${LABELS[$i]}"
  local mode="${MODES[$i]}"
  local morph="${MORPHS[$i]}"
  local stdout_log="${LOG_DIR}/${label}.stdout.log"
  local started_at
  started_at="$(date '+%F %T')"

  printf '[%s] START %s mode=%s morph=%s threads=%s\n' \
    "${started_at}" "${label}" "${mode}" "${morph}" "${NUM_THREADS}" | tee -a "${PROGRESS_LOG}"

  PYTHONUNBUFFERED=1 conda run -n EAI python train.py \
    --cfg "${CFG}" \
    --ckpt_dir "${WARMUP_CKPT_DIR}" \
    --ckpt "${WARMUP_CKPT}" \
    --num_threads "${NUM_THREADS}" \
    --use_cuda "${USE_CUDA}" \
    --gpu_index "${GPU_INDEX}" \
    --game_mode "${mode}" \
    --morph_optim_agents "${morph}" \
    --run_label "${label}" \
    > "${stdout_log}" 2>&1 &

  local pid="$!"
  PIDS[$i]="${pid}"
  STATUSES[$i]="RUNNING"
  printf '%s\n' "${pid}" > "${PID_DIR}/${label}.pid"

  local run_dir=""
  for _ in $(seq 1 60); do
    run_dir="$(latest_run_dir_for_label "${label}")"
    [[ -n "${run_dir}" ]] && break
    sleep 1
  done
  if [[ -z "${run_dir}" ]]; then
    run_dir="tmp/robo-sumo-devants-v0/${label}-UNKNOWN"
  fi
  RUN_DIRS[$i]="${run_dir}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t\tRUNNING\n' \
    "${label}" "${mode}" "${morph}" "${run_dir}" "${stdout_log}" "${pid}" "${started_at}" >> "${MANIFEST}"
  progress_line "${label}" "${run_dir}" | tee -a "${PROGRESS_LOG}"
}

running_count() {
  local count=0
  for status in "${STATUSES[@]}"; do
    [[ "${status}" == "RUNNING" ]] && count=$((count + 1))
  done
  printf '%s\n' "${count}"
}

next_pending_index() {
  for i in "${!STATUSES[@]}"; do
    if [[ "${STATUSES[$i]}" == "PENDING" ]]; then
      printf '%s\n' "${i}"
      return 0
    fi
  done
  return 1
}

completed_count() {
  local count=0
  for status in "${STATUSES[@]}"; do
    [[ "${status}" == DONE || "${status}" == FAILED* ]] && count=$((count + 1))
  done
  printf '%s\n' "${count}"
}

fill_slots() {
  while [[ "$(running_count)" -lt "${MAX_PARALLEL}" ]]; do
    local next_i
    if ! next_i="$(next_pending_index)"; then
      return
    fi
    start_job "${next_i}"
  done
}

check_finished() {
  for i in "${!LABELS[@]}"; do
    [[ "${STATUSES[$i]}" == "RUNNING" ]] || continue
    local pid="${PIDS[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi

    local status=0
    wait "${pid}" || status="$?"
    local finished_at
    finished_at="$(date '+%F %T')"
    progress_line "${LABELS[$i]}" "${RUN_DIRS[$i]}" | tee -a "${PROGRESS_LOG}"
    if [[ "${status}" -eq 0 ]]; then
      STATUSES[$i]="DONE"
      update_manifest_status "${LABELS[$i]}" "${finished_at}" "DONE"
      printf '[%s] DONE %s\n' "${finished_at}" "${LABELS[$i]}" | tee -a "${PROGRESS_LOG}"
    else
      STATUSES[$i]="FAILED_${status}"
      update_manifest_status "${LABELS[$i]}" "${finished_at}" "FAILED_${status}"
      printf '[%s] FAILED %s status=%s\n' "${finished_at}" "${LABELS[$i]}" "${status}" | tee -a "${PROGRESS_LOG}"
    fi
  done
}

log_hourly_progress() {
  for i in "${!LABELS[@]}"; do
    [[ "${STATUSES[$i]}" == "RUNNING" ]] || continue
    progress_line "${LABELS[$i]}" "${RUN_DIRS[$i]}" | tee -a "${PROGRESS_LOG}"
  done
}

printf '[%s] PARALLEL FORMAL TRAINING START run_set=%s max_parallel=%s threads=%s\n' \
  "$(date '+%F %T')" "${RUN_SET}" "${MAX_PARALLEL}" "${NUM_THREADS}" | tee -a "${PROGRESS_LOG}"

fill_slots
last_progress_ts="$(date +%s)"

while [[ "$(completed_count)" -lt "${#LABELS[@]}" ]]; do
  sleep "${CHECK_SECONDS}"
  check_finished
  fill_slots

  now_ts="$(date +%s)"
  if (( now_ts - last_progress_ts >= POLL_SECONDS )); then
    log_hourly_progress
    last_progress_ts="${now_ts}"
  fi
done

printf '[%s] ALL PARALLEL FORMAL TRAINING COMPLETE\n' "$(date '+%F %T')" | tee -a "${PROGRESS_LOG}"
