# Persisted onboarding-bank geometry: quality proxies versus quantity

This supplement reads **only already persisted first-ten authorized-onboarding
banks** for HB objects 20 and 28. No ORB detection, PnP, tracking, neural forward,
rasterization, scored-frame data, evaluator labels or benchmark scores were
opened or recomputed. It does not replace the separate original-bank tracking
ablation.

Both the NPZ archive SHA256 and the ordered raw-array SHA256 were independently
recomputed and matched each sequence's `progress.json`. The ordered array digest
covers `referencePoints`, `referenceDescriptors`, `previousGray`, `seedPixels`,
and `seedPoints`, in that order, preserving their saved dtypes. Model hashes
match the original public HB acquisition lock. Exact source PLY vertices are read
independently in millimetres; saved anchor metres are multiplied by 1,000.

Full evidence and hashes: [`bank-comparison.json`](bank-comparison.json).
Reproducer: `tools/audit_tracking_bank_geometry.py` at the repository root.

## Counts and geometric proxy

**Distances below are to the nearest model vertex, NOT to the nearest triangle
surface and NOT independent physical ground truth.** A single-thread SciPy
`cKDTree` queries every saved anchor against all original model vertices—no
all-pairs distance matrix or sampled-model shortcut.

| HB object | Existing initialization | Anchors | Last-view seeds | Anchor vertex-distance P50 / P95, mm | Seed vertex-distance P50 / P95, mm |
|---|---|---:|---:|---:|---:|
| 20 | Simplified masked-depth budget | 3,000 | 300 | 1.725 / 4.946 | 2.399 / 5.395 |
| 20 | Original production mesh/ray bank | 1,893 | 148 | 0.387 / 0.618 | 0.386 / 0.667 |
| 28 | Simplified masked-depth budget | 1,016 | 73 | 1.189 / 3.609 | 1.108 / 2.610 |
| 28 | Original production mesh/ray bank | 1,085 | 92 | 0.383 / 0.606 | 0.364 / 0.601 |

The original bank does **not** uniformly add more anchors: object 20 has fewer,
while object 28 has slightly more. The banks also differ geometrically; this is
not a count-only comparison. Ray/triangle anchors are expected to lie on the
model surface, so their nonzero approximately 0.4 mm nearest-vertex median is
largely a **mesh-discretization floor**, not a measured surface error. Depth
anchors can differ from the scan through sensor noise, registration, model shape,
source-pose error or mixed-depth pixels; this audit does not isolate those causes.

Object 28's simplified bank contains **four of 1,016 anchors (0.394%)** more than
10 mm from any model vertex, with a maximum of 190.74 mm; its P95 remains 3.61 mm.
No other checked bank has a >10 mm nearest-vertex distance. These are descriptive
outliers, not proof that they were used in a failing pose fit or caused a tracker
failure; a robust solver could reject them.

## Spread and redundancy

| HB object | Initialization | Anchor RMS radius, mm | Whole-bank nonplanarity σmin/σmax | Exact-coordinate duplicates |
|---|---|---:|---:|---:|
| 20 | Simplified depth | 62.22 | 0.1716 | 0.3667% |
| 20 | Original mesh/ray | 67.37 | 0.2175 | 0.3698% |
| 28 | Simplified depth | 86.22 | 0.2698 | 0.4921% |
| 28 | Original mesh/ray | 102.24 | 0.2536 | 0.2765% |

No exact descriptor duplicates were found in any of the four banks. All four
seed sets have no exact coordinate duplicates. Exact equality does not establish
independent physical feature identities; near-duplicates can remain.
`bank-comparison.json` also records axis extents, population principal standard
deviations and occupied 1 mm source-grid cells. Grid occupancy depends on origin
and sample count, and outliers can inflate spread/nonplanarity. Whole-bank
nonplanarity is **not** the per-frame inlier support tested by a pose solver.

## Reproduce

Use an existing Python environment with NumPy 1.26.4 and SciPy 1.13.1 (the versions
recorded in this run). PyTorch and private checkpoints are not required. First
produce/preserve the two matched initialization-ablation runs and their bank
NPZ/progress files; this script never generates banks or runs their trackers.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tools/audit_tracking_bank_geometry.py \
  --models .cache/tracking-benchmark/real-v1/models \
  --source-lock .cache/tracking-benchmark/real-v1/sources.lock.json \
  --simplified artifacts/tracking-benchmark/real-prefix-simplified-v2 \
  --production artifacts/tracking-benchmark/real-prefix-production-v2 \
  --output artifacts/tracking-benchmark/bank-geometry-audit-rerun.json
```

Choose a new output JSON path. The actual lightweight audit took 0.493 seconds;
that duration is **not tracker performance**. Script SHA256:
`4e37c67afc4cdace44c45a8fc2b81890feaa190220b79408e96e7a2f8debc2f4`.

**No causal or fix claim:** anchor count, selected feature locations, spatial
coverage and geometry all change together. These proxies cannot isolate
“quality versus quantity,” establish improved pose accuracy, or diagnose why a
particular frame fails. They make the changed initialization concrete; the
separate controlled tracking results must be considered with their own limits.
