# VisualizeIT tracking regression dataset v1

This is an **initialized rigid-object tracking** dataset and benchmark, not an
object detector benchmark. No private phone recordings or user photographs are
included. Real recordings and synthetic stress cases are reported separately.

| Corpus | Object-sequences | Scored object-frames | Evidence |
|---|---:|---:|---|
| Synthetic stress v1 | 24 | 4,560 | Procedural RGB-D, exact geometric reference poses, eight stress scenarios × three fixed texture/geometry seeds |
| BOP HomebrewedDB (HB) v1 | 12 | 3,600 | Real public CC0 RGB-D, provided object models, camera calibration, reference poses and visible masks |

Every sequence begins with **ten explicitly authorized annotated onboarding
frames**. These initialize a fixed feature bank. Only RGB and camera calibration
are available in subsequent tracker updates. Evaluation poses, masks, depth,
visibility and stress metadata are held in a separate evaluator manifest, opened
only after predictions are sealed. Scored frames never refresh the reference
bank from evaluation annotations.

Synthetic sequences include smooth motion, fast motion, motion blur, changing
illumination, partial occlusion, repeated full occlusion, exit/reentry with scale
change, and low/repeated texture. These are controlled failure tests, **not real
footage or evidence of unseen-real-object generalization**. See
[the synthetic card](synthetic-README.md) for exact generation and visibility
limitations; [sources](sources.md) documents real-data provenance and license.

## What is actually stored in GitHub

- Complete compressed synthetic pose/visibility annotations and object models,
  corpus manifests, representative images, license and reproducible generator.
- Frozen real-data selection, source revision/license/byte/hash locks and
  acquisition code; representative licensed examples and the real annotation
  pack are included when acquisition completes.
- Benchmark source, regression tests and measured result/provenance summaries.

The full **RGB/depth/mask media archives are now published** in the
[dataset-only GitHub research prerelease](https://github.com/joeyhlu/visualizeIT/releases/tag/tracking-dataset-v1),
not embedded as large Git history blobs. Both uploaded byte counts and GitHub's
computed SHA-256 digests match the local archives; see
[`release-artifacts.json`](release-artifacts.json) for verified links, sizes and
checksums. The release points to benchmark commit `c10cb8b`; later benchmark-only
report commits do not change the frozen dataset or inference source hashes.
Regeneration/acquisition below remains available without private credentials.

## Reproduce on a CPU VM

Requires Python 3.12, git history containing the pinned original tracker, and
about 4 GiB free for datasets, archives and temporary output (8 GiB recommended).
No GPU, paid compute, Unity, Torch, private model weights or display server is
needed for this geometric benchmark.

```bash
bash scripts/setup_tracking_benchmark.sh
.venv-benchmark/bin/python -m bench.tracking_synthetic_data --help
.venv-benchmark/bin/python -m bench.tracking_real_data --help
```

Use each generator's documented command/flags. Keep generated media under the
ignored `.cache/tracking-benchmark/` directory, and reports under `artifacts/`.
After datasets exist, run inference and evaluation as separate processes:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-benchmark/bin/python -m bench.tracking_benchmark infer \
  --dataset .cache/tracking-benchmark/synthetic-v1 \
  --output artifacts/tracking-benchmark/synthetic-v1-run1 --repeats 3 --warmup 20
.venv-benchmark/bin/python -m bench.tracking_benchmark evaluate \
  --dataset .cache/tracking-benchmark/synthetic-v1 \
  --output artifacts/tracking-benchmark/synthetic-v1-run1
# Repeat both stages with real-v1 and a distinct output directory.
```

All runs require a fresh output directory; previous results are never silently
replaced. `--limit-sequences 1 --repeats 1 --warmup 5` is a smoke test, not the
full benchmark. The evaluator validates prediction hashes and dataset/model/RGB/
calibration binding before scoring. Failure frames remain in the denominator;
initialization failures do not inflate measured FPS.

## Compared systems and fixed initialization

1. `original_rigid`: **exact historical repository source** from commit
   `df788084e1e0eb79950fe81cf87b603044c1a2d5`, tracker SHA256
   `0d619def904edaf68ffb35e2bedee45b1dcafdb3d2053a4af59be7d40a5d89d1`.
2. `current_rigid`: `bench.tracker.RigidTracker` at the report's source hashes,
   including exact identity/support, flow/fresh fusion and solve guards.
3. `current_prototype`: actual `bench.prototype_tracking.PrototypeTrackerV1`
   stream wrapper, including calibrated packet checks and two-candidate recovery.

Tracker algorithms and update methods are not rewritten or trained. All systems
receive identical authorized RGB-D/pose initialization: ORB detects up to 2,000
features per view; the first 300 detector features per view form a maximum
3,000-anchor CPU bank. Mask erosion avoids depth boundaries. Object-space anchors
are backprojected from depth using onboarding calibration/poses. This is an
explicit shared **initialization adapter**, not the earlier production
mesh-raster/ray bank; runtime measurements must not be presented as an unmodified
full-service deployment latency. This budget was fixed before full benchmark
scores, without selecting parameters on test results.

## Metrics and boundaries

- All-frame public pose availability; visible-frame availability (`visibility >=
  0.5`); visible-frame coverage with **mean corresponding-model reprojection
  error <5 native pixels**. This last measure is not generic accuracy or an
  independent physical-attachment pass. Native pixels are resolution-dependent,
  so the 320×240 synthetic and 640×480 real scores are **not pooled**.
- Conditional mean-corresponding-point reprojection, ADD, translation and rotation
  error medians/P95; deterministic geometry-only sample of up to 512 vertices.
  Symmetries are not quotiented, so symmetric objects can receive large pose
  errors for visually equivalent orientations. Conditional errors do not erase
  failed/unavailable frames.
- Longest accurate visible run, public poses during fully hidden frames, and
  observable recovery delay to two consecutive reprojection-qualified outputs
  after each hidden episode. An unrecovered episode remains null, not a pass.
- Update-only CPU timing and RGB decode/hash-inclusive offline replay timing;
  initialization, warmup, scoring, rendering, networking and camera capture are
  excluded. Three measured fresh-state repeats follow a separate 20-frame warmup
  stream. OpenCV uses one thread; systems/sequences execute serially on the same
  VM. Shared-host timing is not a hardware-independent real-time guarantee.

The current native trained GoTrack bridge is a **different pipeline**. This VM's
readiness probe found no Torch, installed weights or reviewed operator admission,
and no NVIDIA GPU was detected. No trained-GoTrack, SAM, native iOS, live phone,
HTTP deployment or model-adaptation performance is claimed by these results.
