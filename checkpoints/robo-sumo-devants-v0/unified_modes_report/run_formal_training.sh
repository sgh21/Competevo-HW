#!/usr/bin/env bash
set -euo pipefail

CFG="${CFG:-config/repro/unified-devant-training.yaml}"
WARMUP_CKPT_DIR="${WARMUP_CKPT_DIR:-tmp/robo-sumo-devants-v0/base-rtg-fixed-epoch100-20260603_183245/models}"
WARMUP_CKPT="${WARMUP_CKPT:-best}"
NUM_THREADS="${NUM_THREADS:-24}"
USE_CUDA="${USE_CUDA:-True}"
GPU_INDEX="${GPU_INDEX:-0}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
REPORT_DIR="${REPORT_DIR:-reports/unified_modes}"
LOG_DIR="${REPORT_DIR}/logs"
PID_DIR="${REPORT_DIR}/pids"
MANIFEST="${REPORT_DIR}/formal_manifest.tsv"
PROGRESS_LOG="${LOG_DIR}/formal_progress.log"

mkdir -p "${LOG_DIR}" "${PID_DIR}" "${REPORT_DIR}/eval" "${REPORT_DIR}/curves"

printf 'label\tmode\tmorph_optim_agents\trun_dir\tstdout_log\tpid\tstarted_at\tfinished_at\tstatus\n' > "${MANIFEST}"
: > "${PROGRESS_LOG}"

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

progress_line() {
  local label="$1"
  local run_dir="$2"
  local train_log
  train_log="$(latest_train_log "${run_dir}")"
  if [[ -z "${train_log}" ]]; then
    printf '[%s] %s run_dir=%s waiting for logger; latest_ckpt=%s\n' \
      "$(date '+%F %T')" "${label}" "${run_dir}" "$(latest_checkpoint "${run_dir}")"
    return
  fi

  local iteration total reward0 reward1 win0 win1 reward_sp win_sp
  iteration="$(grep -E '#-+ Iteration [0-9]+' "${train_log}" | tail -n 1 | sed -E 's/.*Iteration ([0-9]+).*/\1/' || true)"
  total="$(grep -E 'Total time:' "${train_log}" | tail -n 1 | sed -E 's/.*Total time:[[:space:]]*([0-9.]+) min.*/\1/' || true)"
  reward0="$(grep -E 'Agent_0 gets eval reward:' "${train_log}" | tail -n 1 | sed -E 's/.*reward: ([^.]*(\.[0-9]+)?).*/\1/' || true)"
  reward1="$(grep -E 'Agent_1 gets eval reward:' "${train_log}" | tail -n 1 | sed -E 's/.*reward: ([^.]*(\.[0-9]+)?).*/\1/' || true)"
  win0="$(grep -E 'Agent_0 gets win rate:' "${train_log}" | tail -n 1 | sed -E 's/.*rate: ([0-9.]+).*/\1/' || true)"
  win1="$(grep -E 'Agent_1 gets win rate:' "${train_log}" | tail -n 1 | sed -E 's/.*rate: ([0-9.]+).*/\1/' || true)"
  reward_sp="$(grep -E 'Agent gets eval reward:' "${train_log}" | tail -n 1 | sed -E 's/.*reward: ([^.]*(\.[0-9]+)?).*/\1/' || true)"
  win_sp="$(grep -E 'Agent gets win rate:' "${train_log}" | tail -n 1 | sed -E 's/.*rate: ([0-9.]+).*/\1/' || true)"

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

run_one() {
  local label="$1"
  local mode="$2"
  local morph="$3"
  local stdout_log="${LOG_DIR}/${label}.stdout.log"
  local pid_file="${PID_DIR}/${label}.pid"
  local started_at
  started_at="$(date '+%F %T')"

  printf '[%s] START %s mode=%s morph=%s\n' "${started_at}" "${label}" "${mode}" "${morph}" | tee -a "${PROGRESS_LOG}"
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
  printf '%s\n' "${pid}" > "${pid_file}"

  local run_dir=""
  for _ in $(seq 1 60); do
    run_dir="$(latest_run_dir_for_label "${label}")"
    [[ -n "${run_dir}" ]] && break
    sleep 1
  done
  if [[ -z "${run_dir}" ]]; then
    run_dir="tmp/robo-sumo-devants-v0/${label}-UNKNOWN"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t\tRUNNING\n' \
    "${label}" "${mode}" "${morph}" "${run_dir}" "${stdout_log}" "${pid}" "${started_at}" >> "${MANIFEST}"
  progress_line "${label}" "${run_dir}" | tee -a "${PROGRESS_LOG}"

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
    if kill -0 "${pid}" 2>/dev/null; then
      progress_line "${label}" "${run_dir}" | tee -a "${PROGRESS_LOG}"
    fi
  done

  local status=0
  wait "${pid}" || status="$?"
  local finished_at
  finished_at="$(date '+%F %T')"
  progress_line "${label}" "${run_dir}" | tee -a "${PROGRESS_LOG}"

  if [[ "${status}" -eq 0 ]]; then
    printf '[%s] DONE %s\n' "${finished_at}" "${label}" | tee -a "${PROGRESS_LOG}"
    sed -i -E "s#^(${label}\t.*\t)RUNNING\$#\\1DONE#" "${MANIFEST}"
    sed -i -E "s#^(${label}\t[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*)\t\tDONE\$#\\1\t${finished_at}\tDONE#" "${MANIFEST}"
  else
    printf '[%s] FAILED %s status=%s\n' "${finished_at}" "${label}" "${status}" | tee -a "${PROGRESS_LOG}"
    sed -i -E "s#^(${label}\t.*\t)RUNNING\$#\\1FAILED_${status}#" "${MANIFEST}"
    sed -i -E "s#^(${label}\t[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*)\t\tFAILED_${status}\$#\\1\t${finished_at}\tFAILED_${status}#" "${MANIFEST}"
    return "${status}"
  fi
}

run_one "formal-sp-fixed" "selfplay" "none"
run_one "formal-tp-fixed" "two_player" "none"
run_one "formal-sp-morph" "selfplay" "all"
run_one "formal-tp-morph" "two_player" "all"
run_one "formal-tp-mixed-a0morph" "two_player" "0"

printf '[%s] ALL FORMAL TRAINING COMPLETE\n' "$(date '+%F %T')" | tee -a "${PROGRESS_LOG}"
