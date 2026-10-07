#!/usr/bin/env bash
# No Docker/WSL startup. Call only inside an available Linux user environment.
set -euo pipefail
cd "$(dirname "$0")/.."
quality_python="${QUALITY_BOOTSTRAP_PYTHON:-python3.10}"
quality_runtime=".cache/quality-runtime"
"$quality_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)'
if [[ ! -f "$quality_runtime/bin/python" ]]; then "$quality_python" -m venv "$quality_runtime"; fi
quality_python="$quality_runtime/bin/python"
"$quality_python" -m pip install pip==24.3.1 setuptools==75.3.0 wheel==0.44.0
lock="bench/runtime/requirements-resolved-linux.txt"
if [[ -f "$lock" ]]; then
  "$quality_python" -m pip install -r "$lock" --extra-index-url https://download.pytorch.org/whl/cu124
else
  "$quality_python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  "$quality_python" -m pip install -r bench/runtime/requirements-quality.txt
  "$quality_python" -m pip freeze > "$lock"
fi
"$quality_python" -m pip check
printf '%s\n' 'Runtime installed in workspace. Install system EGL/Mesa prerequisites in the Linux image if absent; then run tools/run-quality-linux.sh with QUALITY_PYTHON=.cache/quality-runtime/bin/python.'
