#!/usr/bin/env bash
# Download C4 (into HF cache), then submit one Slurm batch per model (60M / 135M).
#
# From repo root on a login node:
#   export ACCOUNT=m####_g
#   export WANDB_API_KEY=...
#   ./scripts/nersc/launch_c4_benchmarks.sh
#
# Options:
#   --models 60m,135m          models to launch (default both)
#   --opts htmuon,freon,...    subset of optimizer labels (default: full suite)
#   --train-shards N           C4 train shards to prefetch (default 16; 0 = val+tok only)
#   --hf-home DIR              HF cache root (default: $SCRATCH/hf_cache or $PSCRATCH/hf_cache)
#   --skip-download            do not prefetch dataset
#   --dry-run                  print sbatch commands only
#   --download-only            only download, do not submit jobs
#
# Env: ACCOUNT, ALLOC_TIME, CONDA_ENV, NUM_GPUS_60M (default 2), NUM_GPUS_135M (default 4),
#      SBATCH_QOS, WANDB_API_KEY, WANDB_NAME_PREFIX, TRACE=1

set -euo pipefail
[[ "${TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCH="${SCRIPT_DIR}/c4_batch.sh"
DOWNLOAD_PY="${PROJECT_ROOT}/llama_pretraining/scripts/benchmark_c4/download_c4.py"
SWEEP="${PROJECT_ROOT}/llama_pretraining/scripts/benchmark_c4/sweep.sh"

MODELS="60m,135m"
OPTS=""
TRAIN_SHARDS="${TRAIN_SHARDS:-16}"
SKIP_DOWNLOAD=0
DRY_RUN=0
DOWNLOAD_ONLY=0

_default_hf_home() {
  if [[ -n "${HF_HOME:-}" ]]; then
    echo "${HF_HOME}"
  elif [[ -n "${PSCRATCH:-}" ]]; then
    echo "${PSCRATCH}/hf_cache"
  elif [[ -n "${SCRATCH:-}" ]]; then
    echo "${SCRATCH}/hf_cache"
  else
    echo "${PROJECT_ROOT}/.hf_cache"
  fi
}
HF_HOME_DIR="$(_default_hf_home)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --opts) OPTS="$2"; shift 2 ;;
    --train-shards) TRAIN_SHARDS="$2"; shift 2 ;;
    --hf-home) HF_HOME_DIR="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,24p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo >&2 "unknown arg: $1"
      exit 1
      ;;
  esac
done

export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

echo "[launch] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[launch] HF_HOME=${HF_HOME}"
echo "[launch] models=${MODELS}"

activate_conda() {
  if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
  fi
  local env="${CONDA_ENV:-htmuon}"
  if [[ -n "${env}" ]] && command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env}"
    echo "[launch] conda env=${env}"
  fi
}

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  activate_conda
  echo "[launch] downloading C4 / tokenizer (train_shards=${TRAIN_SHARDS})..."
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] python ${DOWNLOAD_PY} --hf-home ${HF_HOME} --train-shards ${TRAIN_SHARDS} --smoke"
  else
    python "${DOWNLOAD_PY}" \
      --hf-home "${HF_HOME}" \
      --train-shards "${TRAIN_SHARDS}" \
      --smoke
  fi
else
  echo "[launch] skip download"
fi

if [[ "${DOWNLOAD_ONLY}" -eq 1 ]]; then
  echo "[launch] download-only; not submitting jobs"
  exit 0
fi

if [[ -z "${ACCOUNT:-}" && "${DRY_RUN}" -eq 0 ]]; then
  echo >&2 "set ACCOUNT before launching batches (e.g. export ACCOUNT=m4790_g)"
  exit 1
fi

gpus_for_model() {
  case "$1" in
    60m|60) echo "${NUM_GPUS_60M:-2}" ;;
    135m|135) echo "${NUM_GPUS_135M:-4}" ;;
    *) echo "${NUM_GPUS:-4}" ;;
  esac
}

time_for_model() {
  case "$1" in
    60m|60) echo "${ALLOC_TIME_60M:-${ALLOC_TIME:-24:00:00}}" ;;
    135m|135) echo "${ALLOC_TIME_135M:-${ALLOC_TIME:-36:00:00}}" ;;
    *) echo "${ALLOC_TIME:-24:00:00}" ;;
  esac
}

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
SUBMITTED=()

for model in "${MODEL_LIST[@]}"; do
  model="$(echo "${model}" | tr '[:upper:]' '[:lower:]')"
  model="${model#"${model%%[![:space:]]*}"}"
  model="${model%"${model##*[![:space:]]}"}"
  [[ -z "${model}" ]] && continue

  n_gpus="$(gpus_for_model "${model}")"
  alloc="$(time_for_model "${model}")"
  job_name="htmuon_c4_${model}"

  sweep_args=(--models "${model}")
  if [[ -n "${OPTS}" ]]; then
    sweep_args+=(--opts "${OPTS}")
  fi

  echo "------------------------------------------------------------"
  echo "[launch] submit model=${model} gpus=${n_gpus} time=${alloc}"
  echo "[launch] sweep ${sweep_args[*]}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ACCOUNT=${ACCOUNT:-unset} NUM_GPUS=${n_gpus} ALLOC_TIME=${alloc} JOB_NAME=${job_name} \\"
    echo "  ${BATCH} -- ${SWEEP} ${sweep_args[*]}"
    SUBMITTED+=("${model}:dry-run")
    continue
  fi

  # c4_batch.sh self-sbatches when SLURM_JOB_ID is unset
  job_out="$(
    ACCOUNT="${ACCOUNT}" \
    NUM_GPUS="${n_gpus}" \
    ALLOC_TIME="${alloc}" \
    JOB_NAME="${job_name}" \
    CONDA_ENV="${CONDA_ENV:-htmuon}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
    bash "${BATCH}" -- bash "${SWEEP}" "${sweep_args[@]}"
  )"
  echo "${job_out}"
  SUBMITTED+=("${model}:${job_out}")
done

echo "============================================================"
echo "[launch] submitted ${#SUBMITTED[@]} batch(es):"
for s in "${SUBMITTED[@]}"; do
  echo "  ${s}"
done
echo "[launch] monitor with: squeue -u \$USER"
