#!/usr/bin/env bash
# Consume an audited cache produced from actual pipeline candidates. No downloads.
set -euo pipefail
if [[ "${VISUALIZEIT_CLOUD:-}" != 1 || "$(uname -s)" != Linux ]]; then
  echo 'This script runs only inside an approved cloud workspace.' >&2
  exit 2
fi
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/cloud_ranker_train.sh CACHE_DIRECTORY NEW_RUN_DIRECTORY' >&2
  exit 2
fi
cache=$(realpath "$1")
run=$(realpath -m "$2")
cd "$(dirname "$0")/.."
case "${VISUALIZEIT_MANAGED_ENV:-0}" in
  1) cloud_python=$(command -v python3) ;;
  0) source .venv-cloud/bin/activate; cloud_python=$(command -v python) ;;
  *) echo 'VISUALIZEIT_MANAGED_ENV must be 0 or 1.' >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1 PYOPENGL_PLATFORM=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
ranker_device=${VISUALIZEIT_RANKER_DEVICE:-cuda}
case "$ranker_device" in
  cpu) precision_args=() ;;
  cuda) precision_args=(--amp) ;;
  *) echo 'VISUALIZEIT_RANKER_DEVICE must be cpu or cuda.' >&2; exit 2 ;;
esac
mkdir "$run"
for head in linear mlp; do
  architecture=linear
  [[ "$head" == mlp ]] && architecture=mlp64_32
  "$cloud_python" -m bench.quality_ranker_train smoke --manifest "$cache/cache_manifest.json" \
    --output "$run/smoke-$head" --architecture "$architecture" --device "$ranker_device" "${precision_args[@]}" --cloud-execution
  "$cloud_python" - "bench/runtime/ranker-$head.json" "$run/config-$head.json" "$ranker_device" <<'PY'
import json
import pathlib
import sys

source, target, device = sys.argv[1:]
config = json.loads(pathlib.Path(source).read_text())
config.update(device=device, amp=device == "cuda")
with pathlib.Path(target).open("x") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  "$cloud_python" -m bench.quality_ranker_train train --manifest "$cache/cache_manifest.json" \
    --output "$run/$head" --config "$run/config-$head.json" --cloud-execution
  if [[ ! -f "$run/$head/best.pt" ]]; then
    echo "No validated checkpoint for $head. Inspect logs and resume last.pt before export." >&2
    exit 3
  fi
  "$cloud_python" -m bench.quality_ranker_train export --manifest "$cache/cache_manifest.json" \
    --checkpoint "$run/$head/best.pt" --output "$run/$head/model.json" --cloud-execution
done
echo 'Models exported. Complete validation comparison and sequential replay before any promotion.'
echo 'Check your provider console and stop billable compute; persistent storage may remain billable.'
