# Synthetic rigid-tracking stress corpus v1

**Every frame is procedurally rendered synthetic data.** This is not real footage,
independent human annotation, unseen physical-object validation, or a live-camera
performance measurement. It supplements—not replaces—the real corpus.

## Scale and coverage

- 24 sequences, 200 frames each: **4,800 RGB frames, 4,560 scored frames**.
- First 10 frames per sequence are explicitly authorized RGB-D/known-pose onboarding.
- Three disjoint procedural seed groups named `train`, `dev`, and `test`; each
  includes all eight scenarios. These are generator-seed splits, not proof of
  real-world generalization or no exposure in third-party pretraining.
- Eight scenarios: smooth motion/scale, fast motion, motion blur, varying
  illumination, partial occlusion, full occlusion, exit/reentry, and weak repeated
  texture. Synthetic backgrounds contain deterministic colored distractors.
- Varied asymmetric cuboids, tapered prisms, and sheared prisms, with distinct
  aspect ratios and textures. There are 24 sequence-specific geometries, not 24
  physical scanned object identities. Meshes have 54 vertices / 48 triangles.
- Resolution 320×240; analytic 30 Hz timestamps describe the generated trajectory,
  **not observed inference throughput**. CV axes are X-right, Y-down, Z-forward.

Trajectories, stress intervals, seeds, geometry, and selection rules were fixed
without consulting tracker outputs. The weak-texture condition is weak-textured
in onboarding and evaluation; it does not secretly change object identity.

## What is checked into GitHub

| Artifact | Contents |
|---|---|
| `synthetic-manifest.json` | Full 24-sequence inventory, fixed seeds, geometry parameters, split/scenario counts, and annotation-pack SHA256 |
| `synthetic-annotations.jsonl.gz` | All 4,800 frame poses, camera intrinsics, visibility estimates, phase/split/stress, and RGB/depth/mask SHA256 values |
| `synthetic-contact-sheet.jpg` | Three representative frames per sequence (72 thumbnails), explicitly synthetic |
| `synthetic-models/` | Exact generated source-object geometry in millimetre PLY format |
| `bench/tracking_synthetic_data.py` (repo root) | Deterministic complete-media generation recipe (actual file is `bench/tracking_synthetic_data.py` at repository root) |

The full 14,400 PNG RGB/depth/mask files are generated locally into the ignored
cache, **not included as Git blobs**. A complete-media `.tar.gz` release archive has been
[published as a GitHub research prerelease asset](https://github.com/joeyhlu/visualizeIT/releases/download/tracking-dataset-v1/visualizeit-synthetic-tracking-v1.tar.gz).
Its uploaded size and GitHub-computed SHA-256 match the local archive; see
`synthetic-release.json`. The annotations are the full dataset labels,
not placeholder/random score rows. Hashes pin media identity for the documented
runtime; PNG encoding can differ with different Pillow/zlib versions, so use the
benchmark's pinned environment and verify the regenerated manifest hashes.

## Regenerate and test

From the repository root, after installing `bench/requirements-tracking.txt` into
a Python 3.12 virtual environment:

```bash
python -m bench.tracking_synthetic_data \
  --output .cache/tracking-benchmark/synthetic-v1 \
  --artifact-dir datasets/tracking-v1
python -m unittest bench.test_tracking_synthetic_data -v
# Optional: prepare a complete local release asset (does not upload):
python -m bench.tracking_synthetic_data \
  --archive artifacts/tracking-benchmark/releases/visualizeit-synthetic-tracking-v1.tar.gz
```

Choose a fresh directory: incomplete existing output is not silently reused. A
complete dataset is reused only if its configuration digest matches. For a quick
harness check, `--frames 30 --sequences 1` creates 20 scored frames; smoke fixtures
are separate from, and must never be reported as, the 4,560-frame benchmark.

## Label and inference boundary

`inference.json` contains only the geometry reference, identity, intrinsics,
ordered RGB test frames, timestamps, and the 10 explicitly authorized onboarding
RGB/depth/mask/pose records. Evaluation poses, visibility, masks, depths, and stress
intervals appear in the separate `evaluator.json`, never in RGB test records.
The tracker harness should bind and seal predictions before opening that file.

`cameraFromObject` in the compressed label pack is a **3×4** matrix `[R | t]`, with
`t` in metres. PLY vertices and BOP-style `cam_t_m2c` are millimetres, matching the
existing project's loader. Depth PNGs are uint16 millimetres, zero outside the
visible object. Masks identify the pixel-exact rendered object after occlusion;
blur does not change the nominal object-pose or geometric visibility label.

`visibility` / `evaluationInfo.visib_fract` is visible raster pixel count divided
by projected convex-hull area, clamped to [0,1]. It is a reproducible geometric
visibility estimate including out-of-frame clipping, not measured human
visibility. Occluding rectangles are recorded in labels. Full occlusions and
out-of-frame intervals remain scheduled frames; do not discard tracker failures
from availability denominators. Report pose metrics conditional on published
poses **alongside all-frame accurate coverage**, so a tracker cannot look good by
withholding all difficult predictions. Source-object symmetry handling is not
claimed; these geometries and textures are intentionally mostly asymmetric.

## Rights and limits

Generated dataset media, models, and annotation artifacts are offered under
CC0-1.0; see `synthetic-LICENSE.txt`. No external imagery, private photos, scans,
or downloaded model weights are incorporated. This notice does not relicense the
rest of the repository or third-party software dependencies. Synthetic labels are
exact with respect to this renderer/analytic trajectory; the renderer and scorer
share some implementation conventions, so passing these controls alone is not
independent physical accuracy evidence.
