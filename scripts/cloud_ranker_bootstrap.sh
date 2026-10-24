#!/usr/bin/env bash
# Run only inside an already approved cloud machine. Does not create resources.
set -euo pipefail
if [[ "${VISUALIZEIT_CLOUD:-}" != 1 || "$(uname -s)" != Linux ]]; then
  echo 'Set VISUALIZEIT_CLOUD=1 inside your approved Linux cloud workspace.' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
export SAM2_BUILD_CUDA=0 XFORMERS_DISABLED=1 PYOPENGL_PLATFORM=egl
case "${VISUALIZEIT_MANAGED_ENV:-0}" in
  1)
    # Some providers permit only their existing environment. Explicit opt-in
    # avoids unexpectedly installing into an operator's default interpreter.
    cloud_python=$(command -v python3)
    runtime_dir=.cache/cloud-runtime
    ;;
  0)
    python3 -m venv .venv-cloud
    source .venv-cloud/bin/activate
    cloud_python=$(command -v python)
    runtime_dir=.venv-cloud
    ;;
  *) echo 'VISUALIZEIT_MANAGED_ENV must be 0 or 1.' >&2; exit 2 ;;
esac
"$cloud_python" -c 'import sys; assert sys.version_info[:2] in ((3, 10), (3, 11)), "Select Python 3.10 or 3.11 before bootstrap"'
mkdir -p "$runtime_dir"
"$cloud_python" -m pip install 'pip==24.3.1' 'setuptools==75.3.0' 'wheel==0.44.0'
"$cloud_python" -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu124
"$cloud_python" -m pip install -r bench/runtime/requirements-ranker.txt
# Existing source revisions and model SHA-256 pins remain authoritative.
"$cloud_python" -m bench.quality_assets
"$cloud_python" -m pip install --no-deps --no-build-isolation -e .cache/model-quality/sources/sam2
"$cloud_python" -m pip freeze > "$runtime_dir/resolved-requirements.txt"
"$cloud_python" -m bench.quality_cloud preflight --root . --output "$runtime_dir/preflight.json" --require-models
echo 'Bootstrap complete. Check provider GPU/storage billing state separately.'
