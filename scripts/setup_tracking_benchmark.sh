#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
if ! git cat-file -e df788084e1e0eb79950fe81cf87b603044c1a2d5:bench/tracker.py 2>/dev/null; then
  echo 'Historical tracker is absent. Fetch full repository history (git fetch --unshallow, or an ordinary full clone), then rerun.' >&2
  exit 1
fi
"$PYTHON" -m venv .venv-benchmark
.venv-benchmark/bin/python -m pip install -r bench/requirements-tracking.txt
.venv-benchmark/bin/python -m unittest bench.test_tracking_benchmark bench.test_tracking_synthetic_data bench.test_tracking_real_data bench.test_prototype_tracking bench.test_tracker_repair -q
