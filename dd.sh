#!/usr/bin/env bash
# Thayer: shard dd_training sweep across local GPUs. --gpus is parsed here only,
# stripped before sweep.py (NERSC: Slurm allocation; no CUDA sharding CLI).
# Invokes plain python -m scripts.dd_training.sweep (no uv/sbatch/srun).
#
# Example: scripts/thayer/dd.sh --gpus 0,1 --setting resnet18k_cifar10 --out-dir ./exp
#
# All shards share one --claim-dir, so they dynamically pull from a single pool
# of pending (not-yet-trained) runs: a GPU that finishes early grabs the next
# unclaimed run instead of exiting idle. Each shard pins one GPU via
# CUDA_VISIBLE_DEVICES and gets --shard i --num-shards N (label / scan offset).
#
# Env: PYTHON, LOG_DIR, CLAIM_DIR (default: fresh mktemp, removed on exit),
#      KEEP_CLAIM_DIR=1, TRACE=1, DRY_RUN=1

set -euo pipefail
[[ "${TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"

on_err() {
  echo "${0##*/}: exit $? at line ${BASH_LINENO[0]}" >&2
}
trap on_err ERR

usage() {
  cat <<USAGE
Usage: $0 --gpus <id[,id...]> [sweep args...]

--gpus is Thayer-only (removed before sweep). Remaining args go to
python -m scripts.dd_training.sweep per GPU, with --shard i --num-shards N,
a shared --claim-dir, and CUDA_VISIBLE_DEVICES set.

Env:
  PYTHON         interpreter (default: python)
  LOG_DIR        if set, stderr/stdout -> \$LOG_DIR/shard\${i}.log
  CLAIM_DIR      shared claim-lock dir (default: fresh mktemp dir)
  KEEP_CLAIM_DIR=1  do not delete the claim dir on exit
  TRACE=1        set -x
  DRY_RUN=1      print commands only
USAGE
}

GPUS=""
FWD=()
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
    *)
      FWD+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${GPUS}" ]]; then
  usage >&2
  exit 2
fi

# Comma- or space-separated GPU ids.
IFS=', ' read -r -a GPU_ARR <<<"${GPUS}"

# Drop empty tokens after split.
_clean=()
for g in "${GPU_ARR[@]}"; do
  [[ -n "$g" ]] && _clean+=("$g")
done
GPU_ARR=("${_clean[@]}")
NUM_SHARDS="${#GPU_ARR[@]}"

if [[ "${NUM_SHARDS}" -eq 0 ]]; then
  echo >&2 "no GPU ids parsed from --gpus='${GPUS}'"
  exit 2
fi

cd "${PROJECT_ROOT}"

if [[ -n "${LOG_DIR}" ]]; then
  mkdir -p "${LOG_DIR}"
fi

# Shared claim dir lets every shard steal from one pending pool. A fresh dir per
# launch avoids inheriting stale locks from a crashed earlier run.
CLEAN_CLAIM_DIR=0
if [[ -z "${CLAIM_DIR:-}" ]]; then
  CLAIM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dd_sweep_claims.XXXXXX")"
  if [[ "${KEEP_CLAIM_DIR:-0}" != "1" ]]; then
    CLEAN_CLAIM_DIR=1
  fi
else
  mkdir -p "${CLAIM_DIR}"
fi

echo "[thayer] launching ${NUM_SHARDS} shard(s) across GPUs: ${GPU_ARR[*]}"
echo "[thayer] shared claim dir: ${CLAIM_DIR}"

PIDS=()
GPU_FOR_PID=()
SHARD_FOR_PID=()
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "${PYTHON}" -m scripts.dd_training.sweep
       "${FWD[@]}" --claim-dir "${CLAIM_DIR}"
       --shard "$i" --num-shards "${NUM_SHARDS}")

  echo "[thayer] shard ${i}/${NUM_SHARDS} on GPU ${gpu}: ${cmd[*]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  if [[ -n "${LOG_DIR}" ]]; then
    "${cmd[@]}" >"${LOG_DIR}/shard${i}.log" 2>&1 &
  else
    "${cmd[@]}" &
  fi
  PIDS+=("$!")
  GPU_FOR_PID+=("${gpu}")
  SHARD_FOR_PID+=("${i}")
done

maybe_clean_claims() {
  if [[ "${CLEAN_CLAIM_DIR}" == "1" && -n "${CLAIM_DIR:-}" ]]; then
    rm -rf "${CLAIM_DIR}" 2>/dev/null || true
  fi
}

if [[ "${DRY_RUN}" == "1" ]]; then
  maybe_clean_claims
  exit 0
fi

cleanup() {
  echo "[thayer] caught signal, killing shards" >&2
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

exit_code=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "$pid"; then
    echo "[thayer] shard ${SHARD_FOR_PID[$idx]} (GPU ${GPU_FOR_PID[$idx]}) finished ok"
  else
    ec=$?
    echo "[thayer] shard ${SHARD_FOR_PID[$idx]} (GPU ${GPU_FOR_PID[$idx]}) exited ${ec}" >&2
    if [[ "${exit_code}" -eq 0 ]]; then
      exit_code="${ec}"
    fi
  fi
done

maybe_clean_claims

exit "${exit_code}"
