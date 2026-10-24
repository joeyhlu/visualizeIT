#!/usr/bin/env bash
# Optional sequential SAM stage, using pre-existing assets and dependencies.
set -euo pipefail
if [[ "${VISUALIZEIT_CLOUD:-}" != 1 || "$(uname -s)" != Linux || $# != 4 ]]; then
  echo 'On approved Linux cloud: cloud_adaptation_sam.sh train|resume MANIFEST CONFIG RUN' >&2
  exit 2
fi
case "$1" in train|resume) action=$1 ;; *) exit 2 ;; esac
manifest=$(realpath "$2")
config=$(realpath "$3")
run=$(realpath -m "$4")
cd "$(dirname "$0")/.."
cloud_python=${VISUALIZEIT_PYTHON:-python3}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 SAM2_BUILD_CUDA=0
"$cloud_python" -m bench.quality_adaptation_sam preflight --manifest "$manifest" --config "$config"
wall_seconds=$("$cloud_python" -c 'import json,sys,math; c=json.load(open(sys.argv[1])); print(math.ceil(c.get("max_seconds",900))+300)' "$config")
timeout --signal=TERM --kill-after=120 "$wall_seconds" \
  "$cloud_python" -m bench.quality_adaptation_sam "$action" --manifest "$manifest" \
  --config "$config" --output "$run" --cloud-execution --enable-sam-adaptation
