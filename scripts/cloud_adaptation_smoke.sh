#!/usr/bin/env bash
# Fresh processes, fixed input plan and one foreground nonresetting watchdog.
set -euo pipefail
if [[ "${VISUALIZEIT_CLOUD:-}" != 1 || "$(uname -s)" != Linux ]]; then
  echo 'Requires an approved Linux cloud host and VISUALIZEIT_CLOUD=1.' >&2
  exit 2
fi
if [[ $# != 4 ]]; then
  echo 'Usage: bash scripts/cloud_adaptation_smoke.sh PLAN LABEL_PLAN CONFIG PRIVATE_OUTPUT' >&2
  exit 2
fi
plan=$(realpath "$1")
label_plan=$(realpath "$2")
config=$(realpath "$3")
output=$(realpath -m "$4")
cd "$(dirname "$0")/.."
cloud_python=${VISUALIZEIT_PYTHON:-python3}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 XFORMERS_DISABLED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec timeout --signal=TERM --kill-after=120 1800 \
  "$cloud_python" -m bench.quality_adaptation_smoke run --plan "$plan" \
  --label-plan "$label_plan" --config "$config" --output "$output" --cloud-execution
