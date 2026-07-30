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

# label|optimizer|extra flags for run_one
DEFAULT_JOBS=(
  "muon|muon|"
  "htmuon|htmuon|--power 0.125"
  "freon|freon|--freon_c 0.6666667"
  "freon_c075|freon|--freon_c 0.75"
  "dynmuon|dynmuon|--dynmuon_pmax 1.0 --dynmuon_pmin -0.25 --dynmuon_tau 0.04 --dynmuon_width 0.04"
  "softmuon|softmuon|--soft_alpha 0.5"
  "contramuon|contramuon|--contra_coeff 0.5"
  "spectral_p0|spectral_p|--power 0.0"
  "spectral_p_neg025|spectral_p|--power -0.25"
  "spectral_p0125|spectral_p|--power 0.125"
)

if [[ -n "${OPTS}" ]]; then
  JOBS=()
  IFS=',' read -r -a wanted <<< "${OPTS}"
  for w in "${wanted[@]}"; do
    w="${w#"${w%%[![:space:]]*}"}"
    w="${w%"${w##*[![:space:]]}"}"
    [[ -z "${w}" ]] && continue
    found=0
    for job in "${DEFAULT_JOBS[@]}"; do
      label="${job%%|*}"
      if [[ "${label}" == "${w}" ]]; then
        JOBS+=("${job}")
        found=1
        break
      fi
    done
    if [[ "${found}" -eq 0 ]]; then
      JOBS+=("${w}|${w}|")
    fi
  done
else
  JOBS=("${DEFAULT_JOBS[@]}")
fi

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
  model="$(echo "${model}" | tr '[:upper:]' '[:lower:]')"
  model="${model#"${model%%[![:space:]]*}"}"
  model="${model%"${model##*[![:space:]]}"}"
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
