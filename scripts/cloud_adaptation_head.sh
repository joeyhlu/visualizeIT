#!/usr/bin/env bash
# One foreground bounded invocation; no provisioning, installation or downloads.
set -euo pipefail
if [[ "${VISUALIZEIT_CLOUD:-}" != 1 || "$(uname -s)" != Linux ]]; then
  echo 'Requires an approved Linux cloud workspace and VISUALIZEIT_CLOUD=1.' >&2
  exit 2
fi
if [[ $# != 4 ]]; then
  echo 'Usage: bash scripts/cloud_adaptation_head.sh train|resume MANIFEST CONFIG RUN_DIRECTORY' >&2
  exit 2
fi
case "$1" in train|resume) action=$1 ;; *) echo 'Action must be train or resume.' >&2; exit 2 ;; esac
manifest=$(realpath "$2")
config=$(realpath "$3")
run=$(realpath -m "$4")
cd "$(dirname "$0")/.."
cloud_python=${VISUALIZEIT_PYTHON:-python3}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 XFORMERS_DISABLED=1
"$cloud_python" -m bench.quality_adaptation_train preflight --manifest "$manifest" --config "$config"
# Cooperative budget is <=24h; this independent hard watchdog also covers a
# stuck native op. TERM permits checkpoint save; KILL reaps after 120 seconds.
# This wrapper never backgrounds the worker or leaves a monitoring shell behind.
wall_seconds=$("$cloud_python" -c 'import json,sys,math; c=json.load(open(sys.argv[1])); print(math.ceil(c.get("max_seconds",900))+300)' "$config")
timeout --signal=TERM --kill-after=120 "$wall_seconds" \
  "$cloud_python" -m bench.quality_adaptation_train "$action" --manifest "$manifest" \
  --config "$config" --output "$run" --cloud-execution
echo 'Run ended. Inspect status.json, history.json and availability before any export or promotion.'
