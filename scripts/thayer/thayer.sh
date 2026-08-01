#!/usr/bin/env bash
# Thayer: download C4, then run the same benchmark suite as the NERSC launcher,
# sharded across local NVIDIA GPUs instead of submitted to Slurm.
#
# Example: scripts/thayer/thayer.sh --gpus 0,1,2,3
#
# GPUs are partitioned into worker groups of --gpus-per-job. All workers share one
# --claim-dir, so they dynamically pull from a single pool of pending (model,
# optimizer) runs: a group that finishes early grabs the next unclaimed run
# instead of exiting idle. Each worker pins its GPUs via CUDA_VISIBLE_DEVICES,
# gets its own torchrun master port, and scans the pool from a shard offset.
#
# Env: PYTHON, LOG_DIR, CLAIM_DIR (default: fresh mktemp, removed on exit),
#      KEEP_CLAIM_DIR=1, BASE_PORT, WANDB_API_KEY, WANDB_NAME_PREFIX, SEED,
#      SKIP_EXISTING=1, TRACE=1, DRY_RUN=1

set -euo pipefail
[[ "${TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_DIR="${PROJECT_ROOT}/llama_pretraining/scripts/benchmark_c4"
RUN_ONE="${BENCH_DIR}/run_one.sh"
DOWNLOAD_PY="${BENCH_DIR}/download_c4.py"

# shellcheck source=../../llama_pretraining/scripts/benchmark_c4/jobs.sh
source "${BENCH_DIR}/jobs.sh"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
BASE_PORT="${BASE_PORT:-29500}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-htmuonv2_c4}"

on_err() {
  echo "${0##*/}: exit $? at line ${BASH_LINENO[0]}" >&2
}
trap on_err ERR

usage() {
  cat <<USAGE
Usage: $0 --gpus <id[,id...]> [options]

Downloads C4 into the HuggingFace cache, then runs the C4 benchmark suite
(models x optimizers) across the given GPUs with a shared claim pool.

Options:
  --gpus <ids>         GPU ids to use, comma- or space-separated (required)
  --gpus-per-job <n>   GPUs per concurrent run (default 2)
  --models <csv>       models to benchmark (default 60m,135m)
  --opts <csv>         optimizer labels to run (default: full suite)
  --train-shards <n>   C4 train shards to prefetch (default 16; 0 = val+tokenizer)
  --hf-home <dir>      HF cache root (default: \$SCRATCH/hf_cache or ./.hf_cache)
  --skip-download      do not prefetch the dataset
  --download-only      only download, do not run benchmarks
  --dry-run            print the plan only

Env:
  PYTHON            interpreter (default: python)
  LOG_DIR           if set, each run logs to \$LOG_DIR/<model>_<label>.log
  CLAIM_DIR         shared claim-lock dir (default: fresh mktemp dir)
  KEEP_CLAIM_DIR=1  do not delete the claim dir on exit
  BASE_PORT         first torchrun master port (default 29500)
  SKIP_EXISTING=1   skip runs whose checkpoint dir already exists
  TRACE=1           set -x
  DRY_RUN=1         print commands only
USAGE
}

GPUS=""
GPUS_PER_JOB=2
MODELS="60m,135m"
OPTS=""
TRAIN_SHARDS="${TRAIN_SHARDS:-16}"
SKIP_DOWNLOAD=0
DOWNLOAD_ONLY=0

_default_hf_home() {
  if [[ -n "${HF_HOME:-}" ]]; then
    echo "${HF_HOME}"
  elif [[ -n "${SCRATCH:-}" ]]; then
    echo "${SCRATCH}/hf_cache"
  else
    echo "${PROJECT_ROOT}/.hf_cache"
  fi
}
HF_HOME_DIR="$(_default_hf_home)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --gpus)
      if [[ $# -lt 2 ]]; then
        echo >&2 "--gpus requires an argument"
        exit 2
      fi
      GPUS="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#--gpus=}"
      shift
      ;;
    --gpus-per-job) GPUS_PER_JOB="$2"; shift 2 ;;
    --gpus-per-job=*) GPUS_PER_JOB="${1#--gpus-per-job=}"; shift ;;
    --models) MODELS="$2"; shift 2 ;;
    --opts) OPTS="$2"; shift 2 ;;
    --train-shards) TRAIN_SHARDS="$2"; shift 2 ;;
    --hf-home) HF_HOME_DIR="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo >&2 "unknown arg: $1"
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${GPUS}" && "${DOWNLOAD_ONLY}" -eq 0 ]]; then
  usage >&2
  exit 2
fi

export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

cd "${PROJECT_ROOT}"

echo "[thayer] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[thayer] HF_HOME=${HF_HOME}"

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  echo "[thayer] downloading C4 / tokenizer (train_shards=${TRAIN_SHARDS})..."
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${PYTHON} ${DOWNLOAD_PY} --hf-home ${HF_HOME} --train-shards ${TRAIN_SHARDS} --smoke"
  else
    "${PYTHON}" "${DOWNLOAD_PY}" \
      --hf-home "${HF_HOME}" \
      --train-shards "${TRAIN_SHARDS}" \
      --smoke
  fi
else
  echo "[thayer] skip download"
fi

if [[ "${DOWNLOAD_ONLY}" -eq 1 ]]; then
  echo "[thayer] download-only; not running benchmarks"
  exit 0
fi

# Comma- or space-separated GPU ids.
IFS=', ' read -r -a GPU_ARR <<<"${GPUS}"

_clean=()
for g in "${GPU_ARR[@]}"; do
  [[ -n "$g" ]] && _clean+=("$g")
done
GPU_ARR=("${_clean[@]}")

if [[ "${#GPU_ARR[@]}" -eq 0 ]]; then
  echo >&2 "no GPU ids parsed from --gpus='${GPUS}'"
  exit 2
fi

if ! [[ "${GPUS_PER_JOB}" =~ ^[0-9]+$ ]] || [[ "${GPUS_PER_JOB}" -lt 1 ]]; then
  echo >&2 "--gpus-per-job must be a positive integer, got '${GPUS_PER_JOB}'"
  exit 2
fi

# Partition the GPU list into worker groups; a trailing partial group still runs.
# Not named GROUPS: bash reserves that for the caller's group ids.
GPU_GROUPS=()
for ((i = 0; i < ${#GPU_ARR[@]}; i += GPUS_PER_JOB)); do
  group="${GPU_ARR[$i]}"
  for ((j = i + 1; j < i + GPUS_PER_JOB && j < ${#GPU_ARR[@]}; j++)); do
    group+=",${GPU_ARR[$j]}"
  done
  GPU_GROUPS+=("${group}")
done
NUM_SHARDS="${#GPU_GROUPS[@]}"

# Job pool: every (model, optimizer) pair, same suite as the NERSC launcher.
SUITE=()
while IFS= read -r job; do
  [[ -n "${job}" ]] && SUITE+=("${job}")
done < <(benchmark_select_jobs "${OPTS}")

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"

ALL_JOBS=()
for model in "${MODEL_LIST[@]}"; do
  model="$(benchmark_trim "$(echo "${model}" | tr '[:upper:]' '[:lower:]')")"
  [[ -z "${model}" ]] && continue
  for job in "${SUITE[@]}"; do
    ALL_JOBS+=("${model}|${job}")
  done
done

if [[ "${#ALL_JOBS[@]}" -eq 0 ]]; then
  echo >&2 "no jobs selected (models='${MODELS}' opts='${OPTS}')"
  exit 2
fi

if [[ -n "${LOG_DIR}" ]]; then
  mkdir -p "${LOG_DIR}"
fi

# Shared claim dir lets every worker steal from one pending pool. A fresh dir per
# launch avoids inheriting stale locks from a crashed earlier run.
CLEAN_CLAIM_DIR=0
if [[ -z "${CLAIM_DIR:-}" ]]; then
  CLAIM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/c4_bench_claims.XXXXXX")"
  if [[ "${KEEP_CLAIM_DIR:-0}" != "1" ]]; then
    CLEAN_CLAIM_DIR=1
  fi
else
  mkdir -p "${CLAIM_DIR}"
fi

echo "[thayer] launching ${NUM_SHARDS} worker(s) over ${#ALL_JOBS[@]} run(s)"
echo "[thayer] gpu groups: ${GPU_GROUPS[*]}"
echo "[thayer] shared claim dir: ${CLAIM_DIR}"

# One worker per GPU group: scan the pool from a shard offset, claim what is free.
run_worker() {
  local shard="$1"
  local group="$2"

  local group_arr
  IFS=',' read -r -a group_arr <<< "${group}"
  local group_size="${#group_arr[@]}"

  local total="${#ALL_JOBS[@]}"
  local rc=0
  local k idx entry model rest label opt flags
  local nproc devices port save_dir wandb_name key

  for ((k = 0; k < total; k++)); do
    idx=$(( (k + shard) % total ))
    entry="${ALL_JOBS[$idx]}"
    model="${entry%%|*}"
    rest="${entry#*|}"
    label="${rest%%|*}"
    rest="${rest#*|}"
    opt="${rest%%|*}"
    flags="${rest#*|}"

    save_dir="${PROJECT_ROOT}/llama_pretraining/checkpoints/c4_${model}_${label}"
    wandb_name="${WANDB_NAME_PREFIX}_${model}_${label}"
    key="${model}__${label}"

    if [[ "${SKIP_EXISTING}" == "1" && -d "${save_dir}" ]]; then
      continue
    fi

    # mkdir is atomic, so exactly one worker wins each run.
    if ! mkdir "${CLAIM_DIR}/${key}" 2>/dev/null; then
      continue
    fi

    nproc="$(benchmark_nproc_for "${model}" "${group_size}")"
    devices="${group_arr[0]}"
    for ((j = 1; j < nproc; j++)); do
      devices+=",${group_arr[$j]}"
    done
    port=$(( BASE_PORT + shard * 100 + idx % 100 ))

    local cmd=(bash "${RUN_ONE}" --model "${model}" --optimizer "${opt}")
    # shellcheck disable=SC2206
    local flag_arr=(${flags})
    if [[ ${#flag_arr[@]} -gt 0 ]]; then
      cmd+=("${flag_arr[@]}")
    fi

    echo "[thayer] shard ${shard} gpus=${devices} nproc=${nproc} -> ${model}/${label}"

    if [[ -n "${LOG_DIR}" ]]; then
      env CUDA_VISIBLE_DEVICES="${devices}" NPROC="${nproc}" MASTER_PORT="${port}" \
        SAVE_DIR="${save_dir}" WANDB_NAME="${wandb_name}" \
        "${cmd[@]}" >"${LOG_DIR}/${model}_${label}.log" 2>&1 || rc=$?
    else
      env CUDA_VISIBLE_DEVICES="${devices}" NPROC="${nproc}" MASTER_PORT="${port}" \
        SAVE_DIR="${save_dir}" WANDB_NAME="${wandb_name}" \
        "${cmd[@]}" || rc=$?
    fi

    if [[ "${rc}" -ne 0 ]]; then
      echo "[thayer] shard ${shard}: ${model}/${label} exited ${rc}" >&2
    fi
  done

  return "${rc}"
}

maybe_clean_claims() {
  if [[ "${CLEAN_CLAIM_DIR}" == "1" && -n "${CLAIM_DIR:-}" ]]; then
    rm -rf "${CLAIM_DIR}" 2>/dev/null || true
  fi
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for i in "${!GPU_GROUPS[@]}"; do
    echo "[dry-run] shard ${i}/${NUM_SHARDS} on GPUs ${GPU_GROUPS[$i]}"
  done
  for entry in "${ALL_JOBS[@]}"; do
    model="${entry%%|*}"
    rest="${entry#*|}"
    label="${rest%%|*}"
    rest="${rest#*|}"
    opt="${rest%%|*}"
    flags="${rest#*|}"
    echo "[dry-run] pending ${model}/${label}: run_one.sh --model ${model} --optimizer ${opt} ${flags}"
  done
  maybe_clean_claims
  exit 0
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[thayer] warn: WANDB_API_KEY unset; runs will not log to WandB"
fi

PIDS=()
GPU_FOR_PID=()
SHARD_FOR_PID=()
for i in "${!GPU_GROUPS[@]}"; do
  run_worker "$i" "${GPU_GROUPS[$i]}" &
  PIDS+=("$!")
  GPU_FOR_PID+=("${GPU_GROUPS[$i]}")
  SHARD_FOR_PID+=("$i")
done

kill_tree() {
  local pid="$1" child
  for child in $(pgrep -P "${pid}" 2>/dev/null); do
    kill_tree "${child}"
  done
  kill "${pid}" 2>/dev/null || true
}

cleanup() {
  echo "[thayer] caught signal, killing workers" >&2
  for pid in "${PIDS[@]}"; do
    kill_tree "$pid"
  done
}
trap cleanup INT TERM

exit_code=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "$pid"; then
    echo "[thayer] shard ${SHARD_FOR_PID[$idx]} (GPUs ${GPU_FOR_PID[$idx]}) finished ok"
  else
    ec=$?
    echo "[thayer] shard ${SHARD_FOR_PID[$idx]} (GPUs ${GPU_FOR_PID[$idx]}) exited ${ec}" >&2
    if [[ "${exit_code}" -eq 0 ]]; then
      exit_code="${ec}"
    fi
  fi
done

maybe_clean_claims

exit "${exit_code}"
