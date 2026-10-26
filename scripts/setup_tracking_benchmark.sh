#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
"$PYTHON" -m venv .venv-benchmark
.venv-benchmark/bin/python -m pip install -r bench/requirements-tracking.txt
.venv-benchmark/bin/python -m unittest bench.test_tracking_benchmark bench.test_tracking_synthetic_data bench.test_tracking_real_data bench.test_prototype_tracking bench.test_tracker_repair -q
