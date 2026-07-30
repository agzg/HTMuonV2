#!/usr/bin/env bash
# NERSC / Slurm batch wrapper for C4 LLaMA spectral-optimizer benchmarks.
# Pattern adapted from the project's prior Perlmutter dd_batch setup.
#
# Submit from login (forwards remaining args to the inner command):
#   export ACCOUNT=m####_g
#   ./scripts/nersc/c4_batch.sh -- ./llama_pretraining/scripts/benchmark_c4/sweep.sh --models 60m
#
# Or a single run:
#   ./scripts/nersc/c4_batch.sh -- ./llama_pretraining/scripts/benchmark_c4/run_one.sh \
#       --model 60m --optimizer htmuon --power 0.125
#
# Env: ACCOUNT, ALLOC_TIME, NUM_GPUS, CONSTRAINT, SBATCH_QOS, CPUS_PER_TASK,
# JOB_NAME, SLURM_OUTPUT, SLURM_ERROR, CONDA_ENV (default htmuon), TRACE=1

set -euo pipefail
[[ "${TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

on_err() {
  echo "${0##*/}: exit $? at line ${BASH_LINENO[0]}" >&2
}
trap on_err ERR

ACCOUNT="${ACCOUNT:-}"
ALLOC_TIME="${ALLOC_TIME:-24:00:00}"
NUM_GPUS="${NUM_GPUS:-4}"
CONSTRAINT="${CONSTRAINT:-gpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
JOB_NAME="${JOB_NAME:-htmuon_c4}"
SLURM_OUTPUT="${SLURM_OUTPUT:-%x-%j.out}"
SLURM_ERROR="${SLURM_ERROR:-%x-%j.err}"
CONDA_ENV="${CONDA_ENV:-htmuon}"

_default_qos() {
  if [[ "${NUM_GPUS:-1}" -le 2 ]]; then
    echo shared
  else
    echo regular
  fi
}
SBATCH_QOS="${SBATCH_QOS:-$(_default_qos)}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${ACCOUNT}" ]]; then
    echo >&2 "set ACCOUNT before sbatch (e.g. export ACCOUNT=m4790_g)"
    exit 1
  fi
  exec sbatch \
    --account="${ACCOUNT}" \
    --constraint="${CONSTRAINT}" \
    --qos="${SBATCH_QOS}" \
    --time="${ALLOC_TIME}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus-per-task="${NUM_GPUS}" \
    --job-name="${JOB_NAME}" \
    --output="${SLURM_OUTPUT}" \
    --error="${SLURM_ERROR}" \
    --export=ALL \
    "${BASH_SOURCE[0]}" "$@"
fi

if [[ "${1:-}" != "--inner" ]]; then
  echo "[batch] job=${SLURM_JOB_ID:-?} host=$(hostname -s 2>/dev/null || hostname)"
  cd "${PROJECT_ROOT}" || exit 1
  if command -v module &>/dev/null; then
    module load conda 2>/dev/null || true
  fi
  if [[ -n "${CONDA_ENV}" ]] && command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
  export SLURM_CPU_BIND=cores
  exec srun -N 1 -n 1 -c "${CPUS_PER_TASK}" --gpus-per-task "${NUM_GPUS}" \
    bash "${BASH_SOURCE[0]}" --inner "$@"
fi

shift
cd "${PROJECT_ROOT}" || exit 1
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ "$#" -eq 0 ]]; then
  echo >&2 "usage: $0 -- <command> [args...]"
  exit 1
fi
exec "$@"
