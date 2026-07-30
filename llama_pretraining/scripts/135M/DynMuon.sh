#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../benchmark_c4" && pwd)"
exec bash "${ROOT}/run_one.sh" --model 135m --optimizer dynmuon "$@"
