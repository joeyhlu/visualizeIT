# Model-based tracking experiment

Execution update (2026-10-02): a separate Windows Python 3.10 environment at
`.cache/quality-windows` now loads GoTrack and SAM 2.1 on the GTX 1660 SUPER.
The keyboard renderer/inference smoke passed: silhouette IoU 0.98297, median
depth difference 0.0000503 m, metric extent agreement. Both initialization modes
returned validated poses over the first 30 frames. The old tracker also accepted
that prefix, so it is not sufficient evidence of improvement. Full-window runs
and independent accuracy verification remain in progress. Authoritative measured
run summaries are in `artifacts/model-quality/measured-runs.json`.

Docker remains stopped at the user's request. No Docker repair, update or reset
was attempted. The old Python environment, original 720 frames and earlier
results are retained. Windows dependency versions are frozen in
`bench/runtime/requirements-resolved-windows.txt`; bootstrap and execution are
available through `tools/bootstrap-quality-windows.ps1` and
`tools/run-quality-windows.ps1`.

The keyboard complete run now accepts 203/240 frames versus 64/240 previously,
but remains a failed candidate: a late 174-degree rotation transition persists,
availability is below 90%, and a repeated 30-frame prefix differed despite
identical masks/settings/code. Deterministic CUDA and single-thread inference
settings have been added but not yet verified by repeated execution. Mug and
bottle runs are underway. No independent annotation gate has passed.

Execution/memory update (2026-10-02): FAISS now runs in a private CPU worker
process so its LLVM OpenMP runtime does not collide with Torch's runtime.
SAM's positional grid uses the exact `arange` equivalent of `cumsum(ones)`;
CPU equality was verified for square and non-square grids, and strict CUDA
execution succeeds. A render trace localized repeat differences to tiny color
differences before network inference; disabling GL dithering is being tested,
not claimed as a verified fix.

The optional `--model-memory` experiment keeps fixed model points and follows
their image appearance forward with bidirectional optical flow. It uses these
observations to seed GoTrack after loss and reject poses that contradict
observed motion. It never returns a flow-only display pose, invents geometry,
or uses evaluation annotations. Missing/hidden imagery invalidates this motion
evidence. Four synthetic motion tests and 18 existing quality checks pass;
real improvement remains unverified. `tools/test-model-memory-windows.ps1`
compares memory enabled/disabled on all 240 keyboard frames, reruns SAM on
short/full prefixes, and preserves results in a separate experiment directory.
The viewer's “Model memory experiment” mode stays empty until poses are saved.

## How object selection works

The comparison viewer switches between saved keyboard, mug and bottle results.
It plays precomputed masks and poses; it does not run inference on the phone or
select a new object from a live camera. Each test object already has a textured
3D model supplied by the dataset.

Each current setup image has six saved point prompts: three include points on
the object and three exclude points on the hand or background. SAM receives
these once, then propagates the selected object forward. There are no manual
correction clicks during scored frames. These setup prompts are separate from
the independent evaluation annotations, which remain incomplete.

Controlled tracking also receives the final onboarding pose. Complete automatic
tracking receives no supplied object pose and initializes with FoundPose and
GoTrack. A good visible mask identifies object pixels; accurate mapping still
requires the correct 3D pose and surface correspondences.

Selecting an arbitrary new object will require its own scan or model. That
capture workflow and native phone deployment remain deferred.

## What is present

- `quality_assets.py`: pinned GoTrack/SAM 2 sources, exact BOP/DINOv2 submodule
  revisions, four official checkpoints and hash checks. Dedicated model cache
  budget: 8 GiB. Acquired sources, checkpoints and input copies currently use
  approximately 3.31 GiB. Runtime installations are separate from model storage.
- `quality_dataset.py`: label-free source/asset bundles, original 240 consecutive
  scored frames per object, final onboarding image and controlled pose only.
  The complete mode initializes from selection and FoundPose on that setup
  image. No later reference pose is loaded by inference.
- `quality_sam2.py`: Hiera Base Plus, float32, forward video memory, initial
  positive/negative prompts, original MP4 pixels decoded lazily without another
  lossy JPEG encode. SAM weights are offloaded before automatic CNOS subprocess
  recovery; video memory resumes from a same-frame detection mask.
- `quality_gotrack.py`: strict upstream checkpoint namespace loading, pinned
  local DINOv2, upstream decoder/head and perspective crop, five 280 × 280
  refinements. A fixed textured GLB is rerendered each iteration. Dense matched
  target pixels are rejected against the current visible mask before PnP.
- `quality_foundpose.py`: 57 sphere directions × 14 in-plane rotations,
  model-only DINO features registered using rendered depth, PCA/visual words,
  upstream TF-IDF retrieval and cyclic buddies, five coarse hypotheses for
  refinement. CPU FAISS retains the upstream matching mathematics.
- `quality_cnos.py`: full-image FastSAM proposals and DINOv2-L template matching,
  batch size one, ambiguity rejection. Runs separately from SAM and GoTrack.
- `quality_contract.py`: image/frame/K input contract, metre SE(3) results,
  same-frame reprojection/support validation, private recovery state and two
  consecutive validations before rendering. Missing masks never become projected
  mesh masks. No visibility claim is based on an overlay merely being present.
- `quality_evaluate.py`: visible IoU, symmetric boundary distance, hand leakage,
  independently reviewed landmark errors, all-frame pose availability and
  separate render visibility. Corrected-mask controls cannot overwrite or pass
  the automatic benchmark. Dataset pose agreement is secondary only.
- `quality_player.py`: four synchronized panels and estimated reference control;
  actual 3D mesh UVs remain attached to the surface. Missing inference is visibly
  labelled **not run**, never replaced with reference poses.
- `quality_annotations.py`: original-image polygon/landmark editor and checked
  JSON import. All 120 annotation candidates remain **pending**. The ten difficult
  candidates per object still need image-only selection review. None is claimed
  as a human label, and no independent accuracy metric has been computed.

These adapters are a compatibility integration of the released Torch 2.5.1
runtime with upstream research networks. They are not claimed to reproduce
upstream results until strict loading, crop/render and inference checks pass.
Template rendering retains metric node transforms/textures without upstream
BOP millimetre rescaling. The smoke test compares its silhouette, depth and
extents against the existing CPU renderer. Template-bank rendering now uses 4×
SSAA. Lighting uses the upstream refiner's camera spotlight; upstream onboarding
uses multidirectional lighting. Exact onboarding appearance parity remains
unverified and this integration is an experimental baseline.

## Verify and inspect now (no Docker)

```powershell
./tools/run-quality.ps1 -Stage Verify
./tools/run-quality.ps1 -Stage Viewer
```

The last full check passed 55 existing tests and 16 new numerical/failure-path
tests, **71 total**. New tests cover unit/projection invariance, mask exclusion,
pose support, ambiguity, stale-pose suppression, recovery confirmation, wrapper
prefix invariance and synthetic 15-frame interruption. These use synthetic
correspondences, not pretrained-model inference or real-world accuracy results.

With the existing artifacts server on port 8876, open `/model-quality/` and
`/model-quality/annotate.html`. The viewer uses the preserved source videos and
old baseline poses. Complete/controlled panels are pending model runs.

## Resume in an already working isolated Linux environment

No script starts or repairs Docker Desktop. `tools/bootstrap-quality-linux.sh`
creates a workspace venv, pins released direct dependencies and retains the
first resolved transitive lock. Linux EGL/Mesa libraries are required for
offscreen rendering. `bench/runtime/Dockerfile` is optional and requires an
immutable base-image digest; no image was built or pulled in this experiment.

After setup:

```bash
QUALITY_PYTHON=.cache/quality-runtime/bin/python bash tools/run-quality-linux.sh
```

The runner preserves raw errors, checks renderer/inference smoke first, builds
template banks, caches segmentation, and runs controlled/complete pose modes
separately. GPU smoke or subsequent stage failure retries the same models on
CPU, after a CPU smoke check, and records the limitation. A failed CPU retry
stops execution. Failures must not be reported as a speed optimization or
hidden by skipping frames.

Initial positive/negative clicks were selected by visual review of the final
setup images and saved only in each input bundle. They still require SAM smoke
inspection for successful object/hand separation. No scored-frame correction
clicks are part of automatic inference.

## Evaluation and remaining release gates

Import independently reviewed editor exports with
`python -B -m bench.quality_annotations --import-labels <export.json>`.
Every labelled object point must be independently matched to a fixed metric
model point; entering dataset-projected image coordinates is not annotation.

Score with `python -B -m bench.quality_evaluate` using its required results,
annotation, mask, input and output paths. All original frame IDs must match,
including failures. Missing annotations leave independent gates unscored.
The boundary report conservatively reports the worst per-frame symmetric P95
at 720-pixel height. Leakage is reported independently from IoU. Availability
uses every source frame; errors during accepted tracking report missing labelled
landmarks separately.

The evaluator's `export_oracle_masks` creates explicitly diagnostic corrected
masks. Run those through the same pose settings into a separate output; never
overwrite `complete.json` or `controlled.json`. Compare only corresponding
reviewed image frames, keeping all other inputs and settings unchanged.

Run segmentation/pose on an identical 120-frame prefix and the full 240-frame
window in separate outputs; compare masks by hashes and poses/states via
`prefix_equal`. The synthetic wrapper check is insufficient for model memory
prefix invariance. The original PBR and subsequent unlit 30-frame real checks
failed despite matching input/code provenance. A single-sample render diagnostic
now passes two versus four actual keyboard frames; 30 versus all 240 original
frames is being tested separately. A short match does not pass the full gate.

For a forced interruption, use `--force-occlusion <source-frame>` on both
segmentation and pose stages into separate stress outputs. It replaces 15 input
frames with black pixels. Rendering decisions use the current image/mask, not
an evaluator visibility flag. `occlusion_recovery` checks hidden suppression and
reacquisition within 30 source frames. Physical hand-occlusion quality still
requires the original image annotations and visual review.

Additional windows were reserved before candidate inference:

| Object | Setup frames | Additional scored frames |
|---|---|---|
| Keyboard | 0–9 | 10–609 |
| Mug | 0–9 | 10–609 |
| Bottle | 250–259 | 260–859 |

Review setup visibility from source images before freezing these selections.
After tuning passes the original gates, record code/settings/checkpoint hashes,
then run these windows without changes. No held-out evaluation or 30 FPS live
claim exists. Unity, network training, C++ migration and mobile builds stay
deferred. GoTrack/SHOW3D remain noncommercial research baselines.

## Image evidence and renderer experiments

`tools/test-appearance-windows.ps1` produced a paired automatic bottle result:
unlit-template control 230/240 accepted, unlit plus texture check 235/240.
Estimated-reference P95 attachment error is respectively 19.34 / 19.09 px,
versus 44.13 px on the preserved earlier PBR run. The appearance adapter rejects
only two proposals; most improvement is associated with the changed template
rendering. The reference model family is related, settings were explored on the
original window, and no independent accuracy or held-out gate passes.

`bench.quality_photometric_probe` tests current-image-only local SE(3)
optimization against fixed texture and current visible-object masks. It loads
no reference poses. `bench.quality_photometric_evaluate` is separate and reads
estimated dataset controls. The six-frame diagnostic has five available poses;
secondary median/P95 worsens from 1.62/10.13 to 2.21/11.82 px. One difficult
orientation improves while others worsen. It remains unintegrated and is not
a benchmark or validated confidence decision.

`tools/test-render-stability-windows.ps1` tests the opt-in
`--unlit-templates --disable-multisampling` configuration without modifying the
installed pyrender source. A scoped GL hook restores state even on failure.
Two independent crop-render processes produce identical color/depth hashes for
30 renders each. The first real two-versus-four-frame neural check also matches
every overlapping trace record. `tools/test-stable-keyboard-windows.ps1` freezes
source, then compares 30 and 240 frames and reports attachment separately.
These experiments retain the original cached masks and FoundPose bank. The bank
was generated with the earlier PBR renderer; it is hashed and not regenerated.
The display viewer still uses its own surface renderer. Default tracking
settings and original complete results are unchanged.

`tools/test-quality-recovery-windows.ps1 -MaskAssociation` runs the complete
original keyboard window with black input at sources 1239–1253 in both SAM2
and pose processing. The unassociated control selected a chair upon return;
its full segmentation and incomplete 118-frame pose pass remain preserved.
The opt-in policy associates actual FastSAM proposals with the last observed
foreground mask, for at most 30 source frames. It retains the descriptor
minimum and ambiguity checks, rejects unsupported candidates and never emits
a prior/projected mask. Cached image-only diagnostic selection and the actual
rerun both choose the keyboard's full current proposal at 1254.

The associated full result is 224/240 accepted: exactly 15 suppressed hidden
frames and one suppressed confirmation frame, then automatic resumption at
1255. `quality_recovery_report.strict_recovery` also verifies pose/mask/render
consistency and two consecutive validated recovery frames. The mechanical
interruption check passes; independent accuracy remains unscored. Secondary
median/P95 are 4.02/10.95 px. Detector processing takes 30.3 s offline, so source
frame latency must not be described as live latency. The report is
`artifacts/model-quality/recovery-results.json`. Longer disappearance, faster
motion, other objects and physical hand masking remain unevaluated for this
association policy; the original tracking defaults remain unchanged.

`tools/test-recovery-prefix-windows.ps1` independently processes setup plus
120 scored frames, including the actual 15-frame interruption and recovery,
then compares it with the 240-frame associated run. The completed trial matches
all native cached masks and all scored poses/states/failures. Its source,
checkpoints, stage-specific bank hashes and settings match. The evaluator treats
the FoundPose bank as pose-stage metadata; SAM2 does not load that bank.
The report is `artifacts/model-quality/recovery-prefix-results.json`. Longer
prefixes, other objects and held-out windows remain separate requirements.

`bench.quality_surface_agreement` adds a secondary surface-area diagnostic
without changing legacy metrics. Stratified triangle-area samples and uniform
barycentric positions avoid selecting landmarks by vertex order. On the same
keyboard window, previous versus single-sample median/P95 are 6.73/40.68 versus
6.04/22.33 px. The associated interruption trial gives 6.60/22.59 px, with 92.8%
of estimated-reference-visible surface samples having an accepted prediction.
These results expose substantial remaining drift. They use related-model
estimated reference poses and geometric self-visibility, without physical hand
annotations, and cannot pass the independent attachment gate. All older scores
remain unchanged. Per-frame diagnostics stay in the local cache and a small
summary appears in the comparison viewer.

## Appearance coverage and bottle identity trials

Canonical keyboard views rendered without any video pose expose a model asset
limitation: its underside is 99.44% below 2/255 luminance and does not contain
the feet or markings visible in the actual video. This is a model appearance
audit, not an isolated proof of the cause of tracking drift. Two source-only
coarse mask labels are quarantined as a pilot requiring human audit; none of
the 120 frozen annotation candidates is promoted or supplied to inference.

A convex-contour pose proposal was tested on ten selected keyboard frames.
Paired additional GoTrack refinement has eight common validated frames; area
P95 worsens from 24.79 to 26.11 px. The proposal is rejected and unintegrated.

The full stable-rendering bottle control accepts 222/240 and passes its
separate 30-frame prefix check, but area P95 is 38.13 px and 32 accepted poses
have large estimated-reference orientation disagreement. It is rejected for
quality. A separate current-image texture diagnostic with unchanged thresholds
flags 30 of these 32 poses and no other accepted pose; reference orientations
are loaded only afterward by the evaluator.

`tools/test-stable-appearance-bottle-windows.ps1` runs a separate 30-frame prefix
and all 240 original frames with `--unlit-templates --disable-multisampling
--appearance-check`, the same inference source/checkpoints/bank and actual masks
as that control. The evaluator verifies provenance and masks and reports total
availability, suppression intervals, orientation diagnostics and surface error
on both all accepted frames and the common accepted frames. A smaller error
after suppressing failures alone does not establish improved attachment.
The phone viewer exposes this opt-in experiment and bookmarks its automatic
state transitions; tracker defaults remain unchanged.

The completed result is 218/240 accepted versus 222/240 control, with the first
30 predictions identical to the separate prefix and identical input masks and
inference provenance. Estimated-reference orientation disagreements above
90 degrees decrease from 32 accepted frames to 10. Area P95 across each run's
accepted frames improves from 38.13 to 23.97 px; on the 215 common accepted
frames it improves from 35.02 to 23.92 px. This paired result indicates more
than suppression alone, but remains a secondary measurement. The 10 px tail
target, independent accuracy and held-out gates remain unmet/unverified.
Eleven frames contain actual texture rejections; 22 frames are suppressed in
total, including recovery confirmation. All source frames remain counted.

JSON/HTML transport is losslessly compressed by the same route-limited preview
server. Keyboard JSON transfers 607,463 bytes for 2,409,758 decoded bytes; byte
ranges remain uncompressed and encoding-specific ETags prevent representation
mix-ups. No annotations, arbitrary cache paths or additional routes are exposed.

## Rejected competing-rotation diagnostics

`quality_texture_detail_probe` uses current RGB, cached actual masks and the
fixed model to render twelve axial hypotheses on ten exploratory original
frames. It does not load reference poses. The detail score removes slow shading
and excludes five-pixel silhouette/occluder borders. Three numerical checks
verify brightness invariance, missing/flat evidence and metric center-preserving
rotation. Raw highest-detail hypotheses worsen typical secondary surface error.

`quality_texture_detail_neural` compares five additional GoTrack iterations
from identical saved control poses and the selected hypotheses. All ten poses
in both branches validate against current learned correspondences, but secondary
area median/P95 changes from 5.88/45.91 to 8.75/45.35 px. A separate
`quality_texture_shape_probe` ranks the same hypotheses by visible-mask IoU;
paired refinement gives 11.38/36.68 px. Both proposals are rejected. Visible-mask
IoU also penalizes real occlusion, so it is not a valid stand-alone pose gate.

`quality_joint_seed` requires improvement in both detail and outline; it returns
the original seed on all ten frames, so another identical neural run is skipped.
Three additional checks cover one-cue conflicts, weak ties, texture absence and
foreground support. No new accepted tracking poses or full-window result are
published from these probes. The summary is
`artifacts/model-quality/bottle-texture-diagnostic-results.json`; raw results and
source snapshots remain in the cache. The phone page shows model/image evidence
in an expandable diagnostic section while retaining the better full-window
experiment. Independent image annotations remain pending.

The camera audit verifies original PinholePlane calibration, exact inference
intrinsics and native dimensions for all three objects. Released views are
undistorted per the source documentation; this is not a pose correctness claim.
