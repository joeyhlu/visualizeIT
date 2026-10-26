# Offline recorded SAM → metric pose dataset v1

**Real RGB input for the actual recorded-image pipeline, not cached neural
features, projected masks, synthetic footage, or an accuracy result.**

| Property | Frozen protocol |
|---|---|
| Real source | BOP HB / HomebrewedDB Primesense validation |
| Upstream revision | `985ebaef7b1bd4bffb2039f18dd81faa647fd060` |
| Coverage | All 12 selected objects, four scenes, 300 scored frames per object |
| Runtime inputs | Source frame 9, then source frames 10–309: 301 images per object |
| Unique RGB resources | 1,204 across shared scene images; 3,600 scored object-frames |
| Native frame IDs | `f000009` onboarding; `f000010`–`f000309` scored |
| Image geometry | Original 640×480; no resize, crop, invented physical FPS or timestamps |
| Time | Native `image_sequence` / `frame_index`; timestamp/media-time fields remain null |
| Calibration | Original BOP **dataset-reference** K, integer pixel centres; not measured phone calibration |
| Geometry | All 12 original metric models converted mm → m, source-object axes/topology/vertex RGBA preserved |
| Initialization | One authorized known pose at original frame 9; no scored-frame pose initialization |
| Segmentation prompt | One positive and one negative point, derived only from authorized frame-9 mask/RGB and visually reviewed by the assistant |
| Runtime masks | Must be actual observed SAM 2.1 outputs, never annotation masks or projected silhouettes |
| Evaluation | Separate evaluator manifest with scored poses, visibility and original depth/mask resource references |

## What is published versus reused

The Git branch contains the complete metadata, evaluation labels, review sheets
and model-conversion recipe. The newly prepared 37 MB GLB adapter archive is
**not uploaded**: its publication status is explicit in `release-asset.json`.
All original HB RGB/depth/masks/PLY models are already available in the earlier
verified dataset Release, so preparation can reproduce every GLB without that
new archive or private model weights.

- `inference/`: complete 12-object native `ClipSpec`, metric-bundle and pending
  observed-SAM case templates, resource index and ordered subset profiles.
- `onboarding-review/`: proposals and two full visual-review sheets. The proposal
  record intentionally remains `reviewed: false`; the finalized input index and
  clip selections separately attest the completed **AI** review against the exact
  review-evidence hash. This is annotation-assisted onboarding, **not a user tap,
  independent human annotation, or automatic object discovery**.
- `evaluator/evaluator.json.gz`: complete 3,600-frame dataset-reference pose and
  visibility annotations, with depth/mask resource hashes. Pixel masks/depth are
  referenced, not packed into JSON and never placed in the runtime input root.
- `LOCK.json`: original-source manifest seals, every input/model/metadata byte
  identity, and source-axis/metre conversion audits.
- The separately prepared adapter asset contains **all 12 metric GLBs**, complete
  metadata and model audits. It contains no private weights/configuration and no
  model predictions/results. Its final filename/size/hash is in
  `release-asset.json` after packaging.
- RGB/depth/mask media are already in the [full licensed HB release](https://github.com/joeyhlu/visualizeIT/releases/tag/tracking-dataset-v1).
  They are **not uploaded a second time**. Preparation reuses local frozen source
  bytes with hardlinks; there is no silent large-copy fallback.

No SAM or pose model is executed by dataset preparation. These files do not
grant workbench admission or imply that any model passed a quality gate.

## Prepare and inspect before acknowledging review

Use Python 3.12 with the existing pinned geometric/data-preparation requirements:

```bash
python -m pip install -r bench/requirements-tracking.txt
# Only if the frozen real corpus is not already present:
python -m bench.tracking_real_data acquire \
  --output .cache/tracking-benchmark/real-v1 \
  --recipe-output datasets/tracking-v1

python -m bench.tracking_offline_dataset prepare \
  --artifacts .cache/tracking-benchmark/offline-sam-pose-v1/reproduced-metadata
```

The first command prints `visual_review_required` and the hash of
`onboarding-review/review-evidence.json`. Inspect both review PNGs and verify that
every green positive point is inside the intended object, and every red negative
point is background. Only after actual inspection acknowledge that evidence:

```bash
python -m bench.tracking_offline_dataset prepare \
  --artifacts .cache/tracking-benchmark/offline-sam-pose-v1/reproduced-metadata \
  --reviewed-by '<truthful reviewer identity, e.g. assistant_AI_visual_review>' \
  --review-evidence-sha256 '<printed evidence hash>'
python -m bench.tracking_offline_dataset verify
```

Do not claim human review when review was performed by an assistant. The points
remain in native integer-centre coordinates; the unchanged SAM entry point
performs its documented `+0.5` corner-origin conversion itself.
The local `--artifacts` destination avoids attempting to replace the published,
immutable metadata with a different consumer's reviewer/provenance fields.

## Native registration and execution profiles

Profiles are selected by deterministic source order, never by evaluation error:
`smoke` = 10 scored frames, `pilot` = 30, `long` = 110, `full` = 300.
**The published dataset remains all 12 × 300**, even when a CPU run is bounded.

```bash
python -m bench.tracking_offline_dataset register \
  --workspace .cache/tracking-benchmark/offline-sam-pose-v1/workbench-pilot \
  --object-id 20 --profile pilot

# Full 301-frame native input with exact lossless RGB re-encoding:
python -m bench.tracking_offline_dataset register \
  --workspace .cache/tracking-benchmark/offline-sam-pose-v1/workbench-full-lossless \
  --object-id 20 --profile full --lossless-rgb
```

Use an isolated workspace per full object: the native **250,000,000-byte input
budget is global across saved clips**, not per clip. Registration preserves that
budget and fully counts the runtime encoded image bytes. It does not set counts
to zero or relax a gate. Original PNGs are about 0.92 MB each; legitimate lossless
PNG re-encoding makes a full first-object input 112,611,506 bytes. Every decoded
uint8 RGB array is checked for exact equality, with both encoded hashes and the
native dtype/shape/pixel-array SHA recorded under `runtime-lossless/proofs/`.
No pixel geometry, K, frame ID or calibration is changed. Original frozen inputs
and earlier registered pilots are untouched.

Registration writes consumer-local absolute bundle paths **only** in the ignored
native workspace. Published bundle templates use relative paths. `ClipSpec`,
`MetricBundle`, native RGB hash checks, and first/last `FrameReader` decodes are
validated without models. Inference roots contain RGB/model inputs only; the
evaluator and annotation-review mask trees are not registered roots.

## Actual SAM masks and native gate cases

The published `case-templates/` are explicitly blocked until real observed SAM
masks exist. They contain no fabricated mask path, hash or producer receipt.
They are not executable admitted cases and must not be passed directly as one.

After the actual native SAM step has produced its immutable `MaskStore`:

```bash
WS=.cache/tracking-benchmark/offline-sam-pose-v1/workbench-pilot
CLIP=offline-hb-scene000001-object000020-pilot
python -m bench.workbench_cli --workspace "$WS" segment "$CLIP" \
  --mode regression --mask-set sam2-observed-v1 --device cpu
python -m bench.workbench_cli --workspace "$WS" gate case "$CLIP" \
  --frame f000009 --mask-set sam2-observed-v1 \
  --metric-bundle "$WS/metric-bundles/$CLIP.json" \
  --out "$WS/cases/actual-sam-f000009.json"
```

This uses the unchanged native case generator to bind the actual observed SAM
mask and its producer manifest. Known pose initialization remains the single
authorized frame-9 seed; scored labels are not used as masks, seeds, resets or
fallbacks. The normal artifact authentication, geometry, parity and serving
admission stages remain separate prerequisites. Preparation does not bypass
them, import private weights, or assert that they passed.

## Licensing and provenance

Real footage, annotations and original object scans are **CC0 1.0 Universal**,
as listed by the [official BOP dataset catalog](https://bop.felk.cvut.cz/datasets/)
and [pinned HB dataset card](https://huggingface.co/datasets/bop-benchmark/hb/blob/985ebaef7b1bd4bffb2039f18dd81faa647fd060/README.md).
Credit: Roman Kaskman, Ivan Shugurov, Sergey Zakharov and Slobodan Ilic,
*HomebrewedDB: RGB-D Dataset for 6D Pose Estimation of 3D Objects*, ICCVW 2019;
BOP conversion by the BOP authors. Native conventions are OpenCV
X-right/Y-down/Z-forward camera axes, camera-from-source-object column-vector
poses and metre translations. No object normalization/recentring/axis reflection
is introduced by the GLB conversion.

HB was already used by this project and pretrained exposure is unknown. This is
not an unseen-object generalization benchmark or newly collected independent
human ground truth. The dataset publication includes no new model quality or
performance analysis.
