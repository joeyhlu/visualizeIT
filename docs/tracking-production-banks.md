# Reusable exact production onboarding banks

`bench/tracking_production_banks.py` prepares or authenticates the exact five
arrays returned by the existing `prototype_benchmark.build_bank`. It never
calls a tracker update, opens scored RGB/depth/masks/poses, opens
`evaluator.json`, downloads weights, trains, or substitutes a different bank.

This is a preparation cache, **not** the simplified depth-backprojection bank.
The unchanged builder receives exactly the first ten authorized onboarding
records, `read_ply(...).original` and its original source-to-bench transform.
There is no additional feature cap. In particular, an exact mesh/raster bank can
be smaller than a 3,000-anchor depth bank; do not assume anchor-count ordering.

## Prepare, only while tracker timing is idle

Explicit object IDs are required. Start with a measured small preparation before
scaling; different meshes can have substantially different setup costs.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m bench.tracking_production_banks \
  --dataset .cache/tracking-benchmark/real-v1 \
  --cache .cache/tracking-benchmark/production-bank-cache-v1 \
  --object-ids 2 7 21 4 9 19 1 16 17 \
  --workers 4 --max-worker-rss 1610612736 --max-seconds 1800
```

Use a normal module/script entrypoint, not a stdin-only Python session, because
fresh-process `spawn` needs an importable main module. At most four preparation
workers are allowed. Each gets one OpenCV/BLAS thread. Default bounds are
1.5 GiB RSS per worker, at most 7 GiB total worker RSS, 256 MiB cache output,
10 GiB free disk reserve, and a finite preparation deadline. RSS is sampled
every 250 ms. A resource failure terminates owned workers and records unfinished
entries as failures; it does not retry with different parameters. The supervisor
uses the pinned CPython ProcessPool's owned process handles so even initialization
is supervised, and refuses an unsupported implementation.

No tracking or model speed measurement should overlap parallel preparation.
Cached `sourceBuildSeconds` measures the original builder/read-Ply stage, not
cache-load latency or a new tracking update. Report these separately.

## Immutable payload and loader

Each `object-NNNNNN/` holds `metadata.json` and, when available,
`reference-bank.npz` with exactly:

1. `referencePoints` (float64 N×3)
2. `referenceDescriptors` (uint8 N×32)
3. `previousGray` (uint8 H×W)
4. `seedPixels` (float32 M×2)
5. `seedPoints` (float64 M×3)

Metadata binds the full inference-manifest SHA-256, exact model and onboarding
specifications, original builder/renderer/PLY/model/vision source hashes,
helper-source hash, Python/NumPy/OpenCV/Pillow versions, every framed array hash,
the NPZ size/hash, legacy concatenated bank hash, support counts, build time and
origin. Model/onboarding payload hashes are rechecked by the loader. NPZ members,
NPY headers, expanded sizes, dtypes and shapes are checked before allocation;
`allow_pickle=False` is mandatory. Loaded arrays are read-only.

```python
from bench.tracking_production_banks import load_bank
entry = load_bank(dataset_root, cache_root, object_id)
# entry.arrays is the exact five-array tuple, or None for a cached failure.
# entry.metadata contains status/origin/counts/sourceBuildSeconds and bindings.
```

`ready` requires at least 12 total and 12 seed-view anchors. `limited` retains
low/zero-support arrays without manufacturing features; `failed` retains the
original exception without a payload. A later call returns the same cached
failure, not an automatic retry. Existing/incomplete files are never silently
overwritten. Changing any bound source, runtime, manifest or input invalidates
the cache rather than quietly accepting stale initialization.

## Reuse already measured exact ablation banks

`adopt_ablation(dataset, cache, object_id, run_directory)` validates a sealed
`production_mesh` run from the reviewed persisted-bank harness
`f2bb48a0ab875c84380ff7a1bbf91cd4715206c2830cd60b359413d85813f289`.
It checks the freeze/seal, inference identity, original builder source/runtime
bindings, NPZ hash, five array identities and support counts before byte-copying
the NPZ. A depth-budget or unknown-source bank is refused; there is no fallback
recomputation or silently changed initialization. Its original measured setup
time remains labelled as source-build time.

The first real three-object check supplied verified banks with total/seed counts
20: 1,893/148; 28: 1,085/92; 33: 161/16. Original build times were approximately
153.60, 228.54 and 124.39 seconds. These are one measured preparation each, not
hardware-independent setup guarantees or broader object-quality evidence.

Tests use manufactured arrays and tiny invalid-Ply failures, never expensive
production rasterization:

```bash
python -m unittest bench.test_tracking_production_banks -q
```
