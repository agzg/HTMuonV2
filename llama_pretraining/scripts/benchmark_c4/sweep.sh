#!/usr/bin/env bash
# Sweep spectral-family optimizers on C4 for LLaMA 60M / 135M.
# Compares HTMuon against Freon, DynMuon, SoftMuon, ContraMuon, Muon, and fixed-p.
#
# Usage:
#   ./sweep.sh                          # both models, default suite
#   ./sweep.sh --models 60m             # 60M only
#   ./sweep.sh --models 60m,135m --dry-run
#   ./sweep.sh --opts htmuon,freon,dynmuon
#
# On NERSC/Perlmutter (from repo root):
#   export ACCOUNT=m####_g
#   ./scripts/nersc/c4_batch.sh -- ./llama_pretraining/scripts/benchmark_c4/sweep.sh --models 60m
#
# Env: SEED, WANDB_NAME_PREFIX, SKIP_EXISTING=1, DRY_RUN=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_one.sh"

MODELS="60m,135m"
OPTS=""
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-htmuonv2_c4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --opts) OPTS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    *)
      echo >&2 "unknown arg: $1"
      exit 1
      ;;
  esac
done

# shellcheck source=jobs.sh
source "${SCRIPT_DIR}/jobs.sh"

JOBS=()
while IFS= read -r job; do
  [[ -n "${job}" ]] && JOBS+=("${job}")
done < <(benchmark_select_jobs "${OPTS}")

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"

run_job() {
  local model="$1"
  local label="$2"
  local opt="$3"
  local flags="$4"

  local save_dir="${LLAMA_DIR}/checkpoints/c4_${model}_${label}"
  local wandb_name="${WANDB_NAME_PREFIX}_${model}_${label}"

  if [[ "${SKIP_EXISTING}" == "1" && -d "${save_dir}" ]]; then
    echo "[skip] ${save_dir} exists"
    return 0
  fi

  # shellcheck disable=SC2206
  local flag_arr=(${flags})
  local cmd=(
    bash "${RUN_ONE}"
    --model "${model}"
    --optimizer "${opt}"
  )
  if [[ ${#flag_arr[@]} -gt 0 ]]; then
    cmd+=("${flag_arr[@]}")
  fi

  export SAVE_DIR="${save_dir}"
  export WANDB_NAME="${wandb_name}"
  export MASTER_PORT="${MASTER_PORT:-$((20119 + RANDOM % 1000))}"

  echo "============================================================"
  echo "[sweep] model=${model} label=${label} opt=${opt}"
  echo "[sweep] flags=${flags}"
  echo "[sweep] save=${save_dir}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] SAVE_DIR=${save_dir} WANDB_NAME=${wandb_name} ${cmd[*]}"
    return 0
  fi

  "${cmd[@]}"
}

for model in "${MODEL_LIST[@]}"; do
  model="$(benchmark_trim "$(echo "${model}" | tr '[:upper:]' '[:lower:]')")"
  [[ -z "${model}" ]] && continue
  for job in "${JOBS[@]}"; do
    label="${job%%|*}"
    rest="${job#*|}"
    opt="${rest%%|*}"
    flags="${rest#*|}"
    run_job "${model}" "${label}" "${opt}" "${flags}"
  done
done

echo "[sweep] done"
