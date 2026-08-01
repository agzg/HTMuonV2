#!/usr/bin/env bash
# Shared C4 benchmark definitions, sourced by sweep.sh (Slurm path) and
# scripts/thayer/thayer.sh (local multi-GPU path) so both launch the same suite.

# Entry format: label|optimizer|extra flags for run_one.sh
BENCHMARK_JOBS=(
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

benchmark_trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

# benchmark_select_jobs [csv of labels]
# Prints one "label|optimizer|flags" entry per line. Empty input selects everything.
# An unknown label is passed through as a bare optimizer name with default flags.
benchmark_select_jobs() {
  local wanted_csv="${1:-}"
  if [[ -z "${wanted_csv}" ]]; then
    printf '%s\n' "${BENCHMARK_JOBS[@]}"
    return 0
  fi

  local wanted=() w job label found
  IFS=',' read -r -a wanted <<< "${wanted_csv}"
  for w in "${wanted[@]}"; do
    w="$(benchmark_trim "${w}")"
    [[ -z "${w}" ]] && continue
    found=0
    for job in "${BENCHMARK_JOBS[@]}"; do
      label="${job%%|*}"
      if [[ "${label}" == "${w}" ]]; then
        printf '%s\n' "${job}"
        found=1
        break
      fi
    done
    if [[ "${found}" -eq 0 ]]; then
      printf '%s|%s|\n' "${w}" "${w}"
    fi
  done
}

# benchmark_batch_ratio <model>
# total_batch_size / batch_size for the model, i.e. the largest usable world size.
# torchrun_main_HTMuon.py asserts gradient_accumulation * batch_size * world_size
# == total_batch_size, so nproc must be a divisor of this. Keep in sync with the
# per-model defaults in run_one.sh.
benchmark_batch_ratio() {
  case "$1" in
    60m|llama_60m|60) echo 2 ;;
    135m|llama_135m|135) echo 4 ;;
    *) echo 1 ;;
  esac
}

# benchmark_nproc_for <model> <available gpus>
# Largest divisor of the model's batch ratio that fits in the available GPUs.
benchmark_nproc_for() {
  local model="$1" avail="$2"
  local ratio n
  ratio="$(benchmark_batch_ratio "${model}")"
  for ((n = avail > ratio ? ratio : avail; n > 1; n--)); do
    if (( ratio % n == 0 )); then
      echo "${n}"
      return 0
    fi
  done
  echo 1
}
