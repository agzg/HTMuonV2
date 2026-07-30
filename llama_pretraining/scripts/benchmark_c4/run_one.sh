#!/usr/bin/env bash
# Launch one C4 pretraining run (LLaMA 60M or 135M) with a spectral-family optimizer.
#
# Examples:
#   ./run_one.sh --model 60m --optimizer htmuon --power 0.125
#   ./run_one.sh --model 135m --optimizer dynmuon
#   ./run_one.sh --model 60m --optimizer freon --freon_c 0.6667
#   ./run_one.sh --model 60m --optimizer softmuon --soft_alpha 0.5
#   ./run_one.sh --model 60m --optimizer spectral_p --power -0.25
#
# Env overrides: WANDB_API_KEY, WANDB_NAME, CUDA_VISIBLE_DEVICES, NPROC, MASTER_PORT,
# SEED, LR, LRMUON, BATCH_SIZE, TOTAL_BATCH_SIZE, STEPS, WARMUP, SAVE_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${LLAMA_DIR}"

MODEL="60m"
OPTIMIZER="htmuon"
POWER="0.125"
FREON_C="0.6666667"
SOFT_ALPHA="0.5"
CONTRA_COEFF="0.5"
DYN_PMAX="1.0"
DYN_PMIN="-0.25"
DYN_TAU="0.04"
DYN_WIDTH="0.04"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --power) POWER="$2"; shift 2 ;;
    --freon_c) FREON_C="$2"; shift 2 ;;
    --soft_alpha) SOFT_ALPHA="$2"; shift 2 ;;
    --contra_coeff) CONTRA_COEFF="$2"; shift 2 ;;
    --dynmuon_pmax) DYN_PMAX="$2"; shift 2 ;;
    --dynmuon_pmin) DYN_PMIN="$2"; shift 2 ;;
    --dynmuon_tau) DYN_TAU="$2"; shift 2 ;;
    --dynmuon_width) DYN_WIDTH="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

MODEL_LC="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
case "${MODEL_LC}" in
  60m|llama_60m|60)
    MODEL_CONFIG="configs/llama_60m.json"
    NPROC="${NPROC:-2}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-512}"
    STEPS="${STEPS:-10000}"
    WARMUP="${WARMUP:-1000}"
    DEFAULT_DEVICES="0,1"
    ;;
  135m|llama_135m|135)
    MODEL_CONFIG="configs/llama_135m.json"
    NPROC="${NPROC:-4}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-512}"
    STEPS="${STEPS:-20000}"
    WARMUP="${WARMUP:-2000}"
    DEFAULT_DEVICES="0,1,2,3"
    ;;
  *)
    echo >&2 "unknown --model ${MODEL} (use 60m or 135m)"
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_DEVICES}}"
MASTER_PORT="${MASTER_PORT:-20119}"
SEED="${SEED:-5}"
LR="${LR:-0.001}"
LRMUON="${LRMUON:-0.03}"
WANDB_NAME="${WANDB_NAME:-c4_${MODEL_LC}_${OPTIMIZER}}"
SAVE_DIR="${SAVE_DIR:-checkpoints/c4_${MODEL_LC}_${OPTIMIZER}}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[warn] WANDB_API_KEY unset; set it before long runs if you want WandB logging"
fi

OPT_ARGS=(
  --model_config "${MODEL_CONFIG}"
  --optimizer "${OPTIMIZER}"
  --seed "${SEED}"
  --lr "${LR}"
  --lrmuon "${LRMUON}"
  --power "${POWER}"
  --batch_size "${BATCH_SIZE}"
  --total_batch_size "${TOTAL_BATCH_SIZE}"
  --num_training_steps "${STEPS}"
  --warmup_steps "${WARMUP}"
  --weight_decay 0.1
  --dtype bfloat16
  --eval_every 1000
  --wandb_name "${WANDB_NAME}"
  --target_eval_tokens 10000000
  --save_every "${STEPS}"
  --save_dir "${SAVE_DIR}"
  --freon_c "${FREON_C}"
  --soft_alpha "${SOFT_ALPHA}"
  --contra_coeff "${CONTRA_COEFF}"
  --dynmuon_pmax "${DYN_PMAX}"
  --dynmuon_pmin "${DYN_PMIN}"
  --dynmuon_tau "${DYN_TAU}"
  --dynmuon_width "${DYN_WIDTH}"
)

echo "[run_one] model=${MODEL_LC} opt=${OPTIMIZER} nproc=${NPROC} steps=${STEPS} save=${SAVE_DIR}"
torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  --master_addr=localhost \
  torchrun_main_HTMuon.py \
  "${OPT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
