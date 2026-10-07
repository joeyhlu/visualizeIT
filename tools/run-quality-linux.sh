#!/usr/bin/env bash
# Run only inside an already configured isolated Linux environment.
# This script never starts, installs, resets or updates Docker/WSL.
set -euo pipefail
cd "$(dirname "$0")/.."
quality_python="${QUALITY_PYTHON:-python}"
quality_device="${QUALITY_DEVICE:-cuda}"
quality_cache=".cache/model-quality"
mkdir -p "$quality_cache/runtime-evidence"
"$quality_python" -m pip freeze > "$quality_cache/runtime-evidence/pip-freeze.txt"
for object in keyboard mug ranch; do
  bundle="$quality_cache/inputs/$object"
  output="$quality_cache/results/$object"
  mkdir -p "$output" "$quality_cache/smoke" "$quality_cache/banks"
  quality_stage() {
    local stage="$1"; shift
    if "$quality_python" -B -m bench.quality_runner "$stage" --bundle "$bundle" --device "$quality_device" "$@" > "$output/$stage-$quality_device.log" 2>&1; then
      return 0
    fi
    cat "$output/$stage-$quality_device.log"
    if [[ "$quality_device" == cpu ]]; then return 1; fi
    printf '%s\n' "$stage failed on CUDA; retrying the same models on CPU." | tee -a "$output/cpu-fallback.txt"
    quality_device=cpu
    "$quality_python" -B -m bench.quality_runner smoke --bundle "$bundle" --output "$quality_cache/smoke/$object-cpu.json" --device cpu
    "$quality_python" -B -m bench.quality_runner "$stage" --bundle "$bundle" --device cpu "$@" > "$output/$stage-cpu.log" 2>&1
  }
  # Verify loading, inference, rendering and units before expensive videos.
  if ! "$quality_python" -B -m bench.quality_runner smoke --bundle "$bundle" --output "$quality_cache/smoke/$object-$quality_device.json" --device "$quality_device" > "$output/smoke-$quality_device.log" 2>&1; then
    if [[ "$quality_device" == cpu ]]; then cat "$output/smoke-cpu.log"; exit 1; fi
    printf '%s\n' 'CUDA smoke failed; same models will run on CPU.' | tee "$output/cpu-fallback.txt"
    quality_device=cpu
    "$quality_python" -B -m bench.quality_runner smoke --bundle "$bundle" --output "$quality_cache/smoke/$object-cpu.json" --device cpu
  fi
  quality_stage banks --output "$quality_cache/banks/$object-foundpose.pt"
  quality_stage cnos-bank --output "$quality_cache/banks/$object-cnos.npz"
  quality_stage segment --output "$output/segmentation"
  for mode in controlled complete; do
    quality_stage pose --masks "$output/segmentation" --output "$output/$mode.json" --mode "$mode"
  done
done
"$quality_python" -B -m bench.quality_player
