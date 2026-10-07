# Observed implementation status

2026-10-03: dataset product readiness remains unverified. Independent object/hand
and landmark annotations, reliable recovery and frozen additional-window tests
are still incomplete. The viewer labels its older run "Automatic baseline
(original)" to distinguish it from newer automatic experiments. The latest
rejected bottle patch calibration is now shown on the existing Cloudflare URL:
32 integrated UI/preview checks pass, local/public HTML matches and the expanded
summary was checked at 390 x 844 with no horizontal overflow. The 11 other
public assets and all canonical annotation files are unchanged. This is browser
evidence only; physical-phone validation and tracking gains are not established.
The bottle comparison using all matches versus source-textured
matches is now complete and rejected by its frozen pose gate; details appear
in the source-texture diagnostic section below. No new attachment gain is established.

2026-10-02 latest bottle trial: stable rendering plus the existing texture
identity check completes all 240 unmodified original frames. It accepts
218/240 (90.8%) versus 222/240 for its same-code control, with identical input
masks and source/checkpoint/bank provenance. Separate 30-frame predictions
match. Large estimated-reference orientation disagreements decrease from 32
accepted frames to 10. On the 215 commonly accepted frames, secondary surface
area median/P95 changes from 1.56/35.02 px to 1.56/23.92 px. Across its own
accepted frames, the candidate area P95 is 23.97 px and legacy P95 22.43 px.
This is measured partial progress, not independent accuracy or a passed gate.
No tracker default is promoted. The completed opt-in comparison, automatic
recovery bookmarks and aggregate results are on the authorized phone page.
Four targeted surface/orientation metric tests pass; viewer JavaScript parses.

Further bottle diagnostics (2026-10-02): ten exploratory original-window
frames compare a saved pose against axial rotations of the fixed model, without
reference/annotation inputs. Removing slow shading exposes weaker texture
detail support on several problematic back views. Raw texture-selected poses
worsen secondary median error. Paired five-iteration GoTrack refinement then
validates all ten frames in each branch, but area median/P95 changes from
5.88/45.91 px control to 8.75/45.35 px detail seeds and 11.38/36.68 px outline
seeds. Both worsen typical error, are rejected, and are not integrated.

An exploratory joint selector requires gains in both visible-silhouette IoU
and texture-detail correlation. None of the ten frames supports a changed
seed; no redundant neural run is started. Six targeted detail/geometry/joint
checks pass. All diagnostics, masks, source snapshots and their hash provenance
remain preserved; the initial neural source snapshot was reconstructed and
verified against its recorded hash after adding optional diagnostic variants.
These selected frames are not a full benchmark or held-out result. The public
viewer includes honest rejected summaries and a model/image comparison figure.

All three source calibrations use PinholePlane; inference intrinsics match the
source values exactly and native video dimensions match 1024 x 1280. The source
documentation confirms released views are undistorted. This audit rules out the
tested calibration-model/resolution mismatch; it does not prove pose accuracy.
No neural jobs remain active from these diagnostics. The better completed
218-frame bottle experiment and all previous results are unchanged. Next work
must verify image correspondence identity and ambiguity, not only fitted flow
residual, before accepting a new tracking change.

2026-10-02 latest update: full original-window pretrained experiments finished
for all three objects. The bottle's new texture recovery experiment accepts
235/240 frames versus 188/240 previously; secondary estimated-reference P95
attachment error improves from 44.13 to 19.09 px, still above the 10 px target.
The shared phone viewer includes both paired bottle experiments. Independent
annotations remain incomplete and no release gate passes. The motion-memory
candidate was rejected. A separate image-only SE(3) alignment diagnostic also
worsened aggregate error and remains unintegrated. Fixed crop renders now repeat
across two processes when multisampling is disabled. The real two-versus-four
keyboard frame test now matches poses, states and every overlapping traced
render/network output. The stronger 30-versus-240 check also matches with
identical settings/provenance. The full keyboard run accepts 240/240 versus 203
previously; secondary median improves 4.89 to 4.02 px, while P95 worsens 10.24
to 10.85 px and remains above target. All 87 offline regression tests pass.
Docker remains untouched.

2026-10-02 recovery update: a full 15-frame black-input keyboard interruption
exposed an identity failure in CNOS. At return frame 1254 it selected a distant
chair (score .492), despite a useful keyboard proposal existing. Full SAM output
is preserved; the wrong-mask pose control was interrupted after 118 scored
frames and is explicitly incomplete. Its 13 post-interruption frames stayed
lost. No missing frames are counted as a full benchmark result.

Unlit grayscale detector templates still chose the chair and were not
promoted. A separate opt-in association policy now ranks current-image
foreground proposals against the last observed mask, with a 30-frame expiry,
unchanged .25 appearance minimum and .95 ambiguity ratio. No projected
silhouette or remembered mask is emitted as a foreground prediction. At 1254
it automatically selects keyboard proposal 7, raw identity score .326 and
prior IoU .356, then resumes SAM video memory with that current-image mask.

The full associated interruption run counts all 240 original source frames:
224 accepted, 15 hidden/lost and one recovery-confirmation frame. Masks and
rendering are empty/suppressed during all 15 blank inputs; source 1254 remains
recovering and 1255 resumes after the second validated current-image pose.
The stricter evaluator verifies no published hidden poses, matching stage
provenance and consecutive recovery confirmation. This interruption check
passes. Secondary median/P95 are 4.02/10.95 px, no orientation disagreement
exceeds 90 degrees, and no independent or physical hand-occlusion gate passes.
The detector takes 30.3 seconds on this offline run; one source-frame recovery
does not mean live 1/60-second recovery. All 93 offline regression tests pass.
Both failed control and associated modes are in the comparison viewer with
their own masks, actual interrupted RGB input and recovery bookmarks.
All neural experiment processes from this recovery study are terminal.

Recorded 2026-10-01. The first mobile milestone is source-complete for an initial integration attempt; it is **not build-verified or device-verified**. No APK, IPA, TestFlight build, freshly captured phone AR footage, reference-device attachment benchmark, or reconstruction result has been produced. Unity setup is deferred at the user's request. The offline mathematical model and initialized public-footage tracking prototype are implemented and evaluated below.

## Evidence available now

| Check | Observed result | What it establishes |
|---|---|---|
| C# engine-independent geometry/visibility/mapping suite | 1,377 assertions passed | Dimensions, indices, normals, winding, metric UVs, seam relocation/repeat closure, local distortion, visibility loss/recovery logic |
| C# syntax parsing | 15 files passed with editor/Android/iOS conditional symbols | Syntax only; no Unity API/type checking |
| C++ coordinate boundary | Compiled with warnings treated as errors; tests passed | Coordinate transform, in-place round trip, rejection of invalid matrices/null, ABI version |
| Geometry export and plot | Plane 4 vertices/2 triangles; box 24/12; cylinder 390/384 | Exact source mesh data, displayed in `artifacts/geometry/geometry-report.png` |
| Attachment scorer unit tests | 6 synthetic tests passed | Percentiles, frame-based availability, threshold/loss behavior, malformed data rejection; no AR accuracy result |
| CPU reference renderer | 10 unit tests passed | Calibrated projection, perspective UV/depth agreement with independent ray intersections, clipping, depth ordering and synthetic occlusion |
| Virtual surface model | 24-view 720p cylinder orbit and 10 error conditions generated | Mathematical sensitivity with known inputs; no camera pose estimation or physical accuracy claim |
| Real object scan mapping baseline | Five YCB meshes, 81,920 triangles total; 40 known-pose orbit views; 7 new math/import/texture tests passed | Curved/concave/irregular geometry, local UV distortion and chart seams; existing public scans, no new reconstruction or physical tracking |
| Offline mapping improvements | Repeat seam-colour error reduced 45.0% mug, 52.4% bowl, 76.0% banana, 39.2% drill; bottle unchanged | Safeguarded chart-scale/orientation/phase candidates; single-artwork improvement gates fail |
| Initialized public-footage tracking | 330 held-out frames across three YCB-Video clips; separate supplied-pose controls, annotated visibility masks and pose scoring | Actual ORB/flow/PnP estimates on recorded RGB, annotated onboarding; not automatic recognition or phone AR |
| Offline regression suite | 30 renderer/import/mapping/tracker/archive tests + 6 scorer tests passed | Geometry/projection, outlier PnP, ambiguity/loss, chart overlaps, depth occlusion, repeat coordinates, rounded annotations, bounded ZIP decoding |
| Camera rendering, shader compilation, native image import | Pending | Requires the supported Unity editor and mobile builds |
| Attachment, appearance, occlusion, thermal performance | Pending | Requires physical reference devices and footage |

The exported box has bounds 0.10 × 0.20 × 0.15 m. The cylinder has diameter 0.10 m, height 0.20 m, and side UV width 0.314159274 m. Maximum vertex radial error is approximately 3.4e-9 m; this is a floating-point calculation check, **not real-world placement accuracy**. Exported normal length error is below 5.6e-8. UV cap/side overlaps are intentional independent face coordinates, with boundary seams still needing rendering review.

The attachment scorer has synthetic unit tests. Their results test the scorer's accounting, not the camera system. The app's JSON diagnostics always mark `physicalValidationPassed: false`; gates require reviewed footage and recorded annotations.

The mapping core now measures each triangle's UV-to-surface Jacobian to detect scale changes and anisotropic stretch. Tests detect an intentionally distorted 2:1 map and collapsed UV triangles. Fixture local scale/anisotropy is within 0.02% in the tested 10 cm examples; this is mesh mapping quality, not camera attachment accuracy. Cylinder repeats can be fitted to a whole repeat count at zero pattern rotation and the seam can be relocated. Arbitrary pattern rotation, image borders and cap/side boundaries are not claimed seamless. A shader adjustment preserves image aspect ratio for very wide designs; its compilation remains pending.

## Working virtual model

`bench/` uses the actual exported C# mesh/mapping data and checks a separate Python texture-coordinate implementation against it. Maximum parity difference in the generated fixtures is 5.6e-7 texture-coordinate units. The perspective-correct renderer agrees with an independent ray/triangle oracle to floating-point precision across 2,090 samples. The intentionally wrong affine interpolation has median error 0.390 texture tiles, producing visible checkerboard bending.

In the default 24-view 1280 × 720 cylinder orbit (`fx=fy=850 px`, horizontal camera radius 36 cm), injected 3 mm position error produces 5.43 px median/7.56 px p95 attachment error; 5 mm produces 9.05/12.60 px. A +2% geometry-scale error produces 9.40/10.75 px. A stale room anchor while the object moves produces 26.85/54.83 px. These are sensitivity measurements from exact synthetic references, not an evaluated tracker.

The 2 mm normal surface lift used by the app's current default contributes 4.65/5.45 px displacement in the synthetic test; 0.25 mm contributes 0.58/0.68 px. The app default has not been changed based solely on this simulation: real provider depth precision and self-occlusion need device testing. Both offset and pose errors matter to natural-looking alignment.

Three deliberately lost frames yield 87.5% availability and fail the provisional 90% simulated threshold despite zero error on visible frames. Synthetic results never pass a physical gate. Outputs include `artifacts/bench/index.html`, `mapping-model.png`, `camera-orbit.gif`, `error-sensitivity.png`, calibrated camera manifests, CSV annotations and `report.json`. See [rebuilding and assumptions](../bench/README.md).

## Real object mapping baseline

Existing YCB `google_16k` scans replace ideal shapes in this offline test. The original geometry/topology is retained through axis conversion and translation; nonfinite mug normals are repaired from the existing geometry (two expanded vertex entries). xatlas 0.0.11 generates metric charts for 4 cm pattern tiles / 5 mm checker squares. The camera uses 960 × 540 frames, `fx=fy=637.5 px`, and eight known poses per object; distance varies to frame each unchanged mesh. Source coordinates are treated as metres, without independent physical measurement.

| Object | Median local scale error | P95 local scale error | Surface area within provisional limits | Charts |
|---|---|---|---|---|
| Mug | 1.20% | 7.36% | 67.8% | 298 |
| Mustard bottle | 1.50% | 6.30% | 64.4% | 65 |
| Bowl | 1.64% | 6.83% | 58.6% | 229 |
| Banana | 0.62% | 5.03% | 82.4% | 431 |
| Power drill | 2.29% | 13.58% | 44.8% | 192 |

Independent UV-to-surface Jacobian singular values measure local scale error. Quantiles are weighted by valid physical triangle area; degenerate geometry and collapsed UV faces are counted separately. The provisional local limits are ≤2% scale error and ≤1.05 anisotropy. They are not a validated release acceptance threshold. Invalid measurable area is below 0.016% for each scan; those faces count against accepted area. All five contain chart seams with coordinate jumps, so locally acceptable regions do not establish a continuous whole-surface pattern. The drill has the largest measured P95 scale error; the banana has the most area within local limits.

Outputs include original photographic textures rendered on their source meshes, replacement patterns, quality masks, orbit GIFs, mesh arrays and per-view camera transforms, metrics/provenance, `artifacts/real-objects/real-object-results.png` and `index.html`. The enlarged contact-sheet crops are for viewing only; the saved full frames retain the original calibration. Seventeen renderer/import/mapping tests and six attachment scorer tests pass. See [rebuild, source attribution and limitations](../bench/YCB_SOURCES.md).

This baseline exposed a mapping problem before mobile installation: automatic UV chart packing alone does not solve local distortion or seam continuity. The subsequent offline prototype below attempts chart orientation/phase correction and preserves quality masks. No live AR or reconstruction gate passes from these renders.

## Implemented offline mapping and RGB tracking prototype

The [offline pipeline](../bench/OFFLINE_PIPELINE.md) now supplies repeat/artwork candidate comparisons, baseline/selected orbit renders, source/UV/chart assets, bounded public sequence acquisition, annotated-pose controls, initialized ORB/flow/RANSAC-PnP tracking, confidence/loss/recovery, measured-depth compositing, independent annotated visibility scoring and raw pose-error jitter diagnostics. `tools/run-offline.ps1` exposes verification, mapping, acquisition, control, tracking, scoring and interruption stages.

Physical seam length and eight-view visibility weight chart alignment. Promotion rejects worsened distortion/invalid area, inverted UVs and overlapping finite artwork. Four repeat-mode objects pass the provisional ≥30% seam-colour improvement gate; the bottle does not. Single-artwork seam alignment creates overlapping layouts and is rejected. Finite artwork is supported for diagnostic rendering, but none of the five passes its improvement gate. These are design-specific diagnostic reductions, not guarantees for arbitrary imported images.

YCB-Video uses matching BOP models, pinned dataset revision and hashed inputs. The first ten annotated frames create the reference bank; evaluation poses/masks are not passed to the tracker. Each clip has 110 held-out frames. RGB tracking is separate from measured-depth compositing and annotation scoring. Source footage is 640×480; errors below are 720p-equivalent, with native errors also retained.

| Object | Median attachment error | P95 error | Tracking availability | Qualifying motion clip | Provisional dataset gate |
|---|---|---|---|---|---|
| Mustard bottle | 2.14 px | 4.35 px | 100% | No: 10.6° | Failed selection |
| Mug | 2.56 px | 5.77 px | 61.8% | No: 7.4° | Failed selection and availability |
| Power drill | 2.74 px | 5.23 px | 100% | Yes: 15.1° | Passed |

The drill gate is for this initialized public sequence only. It does not establish mobile AR, arbitrary-object recognition, object pickup, or photorealistic rendering. The mug experiences a recovered six-frame loss and ends with 36 unrecovered frames. The failed states hide the overlay rather than continue a stale pose.

A separate drill test injects three black tracker-input frames while evaluation retains the original footage. It loses tracking for those three frames and recovers, then has a later seven-frame recovered loss and one final lost frame. Availability is exactly 90%; median/P95 error is 2.86/5.33 px during valid tracking. The final lost state remains visible despite meeting the numeric provisional availability boundary. This is an artificial interruption, not a physical hand-occlusion recording.

Source acquisition remains bounded at 384 MiB and new previews at 256 MiB, with a default 512 MiB disk reserve. Only archive directory/selected members are downloaded; the full 15 GB archive is not acquired. Data/source annotations and models are supplied by YCB-Video/BOP; this is an internal tracking study, not a BOP leaderboard result. Full contiguous annotations can contain pose errors. JPEG/GIF/WebM previews are presentation artifacts. During disk pressure, older BOP test captures, redundant encoding intermediates, downloaded Unity reference caches and copied verification DLLs were evicted; finished reports and native WebM previews were retained. Reacquire source captures/reference packages before repeating affected earlier checks.

Reports: `artifacts/mapping-improvements/index.html`, `artifacts/video-tracking/index.html`, `artifacts/video-interruptions/index.html`, saved tracking traces, independently regenerated CSV scores and versioned NPZ object assets. CPU tracking timing excludes rendering/loading; the CPU renderer is for offline inspection. No phone frame-rate or thermal result has been obtained.

## Higher FPS desktop experiments

Three paired studies replayed all 110 held-out native-HD frames for both the pudding box and corn can. They compare preprocessing order, LSH key size, expanding object-region detection, and ORB pyramid depth. All use the same 480 × 360 tracking size, 400-feature cap and common ten-frame initialization bank. Candidate order rotates; timings include CPU preprocessing and vision, exclude decoding/scoring/rendering, and omit five warmup frames only from timing. Reports include raw frame timings and separate accepted/lost/limited-state latency, since most frames fail tracking. The final 44 offline regression checks passed.

In the pyramid comparison, eight levels versus three levels measured:

| Object | Baseline median / P95 | Three-level median / P95 | Baseline / three-level availability |
|---|---|---|---|
| Pudding box | 13.41 / 17.72 ms | 11.79 / 22.75 ms | 6.4% / 10.9% |
| Corn can | 14.12 / 21.07 ms | 11.79 / 19.82 ms | 10.9% / 8.2% |

Median latency improved about 12–16%, but box tail latency worsened and can availability fell. Three levels passed the provisional 20 ms P95 vision budget only on the can. Every combined attachment/availability/speed gate failed. The faster settings remain opt-in; defaults are unchanged. Earlier preprocessing/index variants were generally slower, and region search did not improve both speed and attachment. No live phone FPS or sustained thermal result exists. Measurements are short desktop replays on an Intel Core i5-11400F with variable background load; compare candidates within each paired run.

Reports: `artifacts/video-hd/performance-higher-fps/index.html`, `performance-object-region/index.html`, and `performance-pyramid/index.html`. The synchronized mapping/reference-mask player is `artifacts/video-hd/visual.html`. Its 10 FPS presentation is independent of these vision measurements. Full native box tracking and lower-resolution box/can speed studies completed; the earlier full native can export did not complete during storage pressure.

## Genuine 60 FPS examples

The SHOW3D adapter and browser GPU player test three additional real moving objects: keyboard, white mug and dressing bottle. All 720 held-out frames were processed, with ten separate onboarding frames per object. Native footage is 1024 × 1280 / 60 FPS, verified against capture metadata and consecutive timestamps; the 640 × 800 WebM previews retain all frames at 60 FPS. One decoded source frame drives footage, reference silhouette and mapped-design views. The player exposes independent baseline/faster estimates and hides lost tracking. Browser drawing FPS is separate and can be lower than the recording rate.

| Object | Baseline median / P95 ms | Faster median / P95 ms | Baseline / faster availability | Faster accepted-state P95 ms (samples) |
|---|---|---|---|---|
| Keyboard | 11.28 / 21.07 | 10.03 / 20.92 | 26.7% / 23.3% | 27.57 (51) |
| White mug | 8.27 / 11.35 | 6.67 / 9.83 | 3.8% / 0% | No accepted frames |
| Dressing bottle | 11.11 / 16.61 | 9.60 / 17.21 | 10.4% / 10.8% | 25.95 (21) |

All-state median time improves roughly 11–19%, but mostly measures failed tracking. Accepted-state tail latency exceeds the 16.67 ms full 60 FPS frame budget on every candidate with accepted samples; other live-path costs are excluded. All combined gates fail. Three-level P95 error versus estimated references is 9.24 px for keyboard and 14.72 px for bottle at 720p-equivalent height; mug has no valid error samples. These numbers do not establish independently annotated physical accuracy. No defaults are promoted and no live phone performance is claimed.

SHOW3D supplies automatic estimated object poses, not human ground truth. Confidence ≥0.5 and repaired valid-camera flags filter references; the common moving-rig coordinate frame is explicitly converted into camera-relative metric poses. Canonical HOT3D GLBs are matched by model name and checked against physical dimensions. The geometric reference silhouette is not automatic segmentation, and external hand occlusion is neither scored nor suppressed. Original model UVs and simple checker shading do not prove whole-surface seam quality or natural lighting. Source revisions, hashes, licences, every pose, state-specific timing and environment are saved in `artifacts/video-60/report.json`. Reproduce through `tools/run-offline.ps1 -Stage SixtyFPS`; watch `artifacts/video-60/index.html`. The final 47 offline regression checks passed, including coordinate, unit, binary mesh and genuine-60-FPS timestamp checks.

## Mask and recovery retries (2026-10-01)

The original mask was a projected reference mesh, not an independent visible-object prediction. It cannot exclude a thumb. The feature tracker also discarded usable optical flow on weak-support frames. New experimental code retains private flow through short failures, validates planar pose continuity, and hides unvalidated poses. More detailed surface ray intersections and reference views generated from model textures improve correspondence initialization without using later video poses.

Classical GrabCut foreground, dense feature replenishment and silhouette alignment were tested on the same 240-frame windows. Longer visibility sometimes introduced large pose drift; the contour mug retry reached 39.6% availability with 38.3 px P95 error and was rejected. GrabCut bottle tracking reached 57.9% with 20.7 px P95 error and was rejected. None becomes the default tracker.

An optional learned-mask experiment runs MediaPipe MagicTouch v1 plus hand landmarks locally on CPU. Approximate hand cores remove some finger observations, but are not pixel-accurate hand masks. Independent optical flow moves the segmentation prompt; no evaluation poses enter this process. The Windows MediaPipe 1.0.1 DLL lacks the new `MpInteractiveSegmenterCreate` entry point, so the v2 model/API could not run. The reported results use the explicitly named legacy v1 model, not v2.

| Object | Returned learned masks | Learned-mask pose retry availability | P95 attachment error (720p px) | Summed mask + tracker P95 ms |
|---|---|---|---|---|
| Keyboard | 70.0% | 32.1% | 10.2 | 448 |
| White mug | 86.2% | 5.0% | 10.5 | 429 |
| Dressing bottle | 92.1% | 23.3% | 8.3 | 461 |

Returned-mask availability is not segmentation accuracy. Visual inspection found mask leakage onto the mug holder's forearm, missing regions and shape-jump rejections. No manual mask annotations or IoU/boundary metrics exist yet. Mask filtering improves bottle error on accepted frames, but the sparse-feature pose tracker still loses all objects too often. Every attachment/availability/performance gate fails. Separate offline model and tracker call times were summed; this is not a live concurrent measurement or a phone result. The browser replays source frames at 60 FPS while CPU inference runs much slower.

`artifacts/video-60/recovery.html` shows original footage, selectable predicted masks or reference silhouettes, and independent mapping retries. Orange shows approximate hand cores; green shows predicted object regions. It clips designs to the selected mask and hides unavailable overlays. Green/red points expose accepted/rejected pose correspondences. The original player remains intact. Full per-frame results and model sources are retained; 55 offline regression checks passed, including foreground exclusion before pose estimation and suppression when foreground is unavailable.

The next quality experiment should use a video segmentation model with temporal memory (SAM 2.1 is a candidate), object/hand correction prompts and independently annotated mask frames. Stronger pose correspondences or a dedicated model-based pose estimator must then address low-texture surfaces and recovery across new views; segmentation alone cannot establish object rotation. RGB-D estimators require depth and cannot be honestly benchmarked on these monochrome RGB-only windows. Training our own network, switching languages and optimizing toward live FPS should follow evidence that a candidate maintains attachment.

## Model-based quality pipeline (2026-10-01)

The replacement pipeline code, pinned source/checkpoint cache, Linux bootstrap and sequential inference stages are implemented. SAM 2.1 masks feed GoTrack refinement; FoundPose supplies recovery hypotheses and CNOS searches when segmentation is unavailable. Contracts enforce metric camera-relative poses, foreground correspondence exclusion, independent mask/pose/render states, two validated recovery frames and suppression on loss. The comparison viewer and source-only annotation editor are at `artifacts/model-quality/`.

All 71 existing and new numerical/regression checks pass. These checks do not load the pretrained networks. All 120 annotation candidates remain pending. No SAM/GoTrack video inference, independent mask/attachment score, real-model prefix test or recovery gate has passed. Linux runtime execution is deferred because the user asked to stop opening the broken Docker installation. No Windows ML environment was installed. Template rendering also needs parity validation against upstream supersampling and lighting before a quality claim. See [../bench/MODEL_QUALITY.md](../bench/MODEL_QUALITY.md) for commands, implemented boundaries and outstanding work.

## Pretrained execution and quality retries (2026-10-02)

The isolated Windows runtime loads Torch 2.5.1/CUDA 12.4, SAM 2.1 and GoTrack successfully on the GTX 1660 SUPER. Docker remains stopped. Keyboard renderer verification reached 0.98297 silhouette IoU with approximately 0.05 mm median depth difference versus the CPU renderer; this verifies implementation alignment, not real-object accuracy. SAM processed the full original 240-frame window automatically. Both 30-frame initialization modes ran successfully, and complete mode used FoundPose rather than a supplied object pose.

The first full complete run accepted 195/240 frames but contained a roughly 170-degree recovery flip and was rejected. The next run retains recent poses privately, excludes source correspondences behind current occluders, and rejects implausible pose jumps without displaying stale poses. It accepted 203/240 frames versus the original baseline's 64/240. Its secondary estimated-reference median/P95 were 4.89/10.24 pixels at 720p-equivalent height. On the common 64 accepted frames, the new P95 was 8.95 versus the previous 9.23, while median error was worse (4.70 versus 3.41). This does not prove independent attachment accuracy.

A late roughly 174-degree transition still occurred after a 20-frame loss. The 30-frame prefix comparison also failed despite identical masks and adapter hashes. Deterministic Torch/CUDA and single-thread CPU settings have now been added; their repeatability still requires execution verification. The candidate fails the 90% availability gate, and all independent annotation/recovery/additional-window gates remain outstanding. Original results and rejected runs remain preserved. Mug and bottle jobs have started. Four source/model key-centre correspondences form an explicitly agent-reviewed pilot requiring human audit; the 120 full annotation candidates remain pending.

## Appearance and memory diagnostics (2026-10-02)

Full pretrained complete-pipeline runs now include all 240 original frames for
each object: keyboard 203 accepted, white mug 240, dressing bottle 188. The
mug's secondary estimated-reference agreement is 2.46 px median / 8.73 px P95;
the bottle's is 1.48 / 44.13 px. The bottle therefore remains a failed candidate
despite better coverage. Independent annotations remain 0/40 for each object.

An evaluator-only orientation review found 44 accepted bottle frames more than
90 degrees from the estimated reference orientation. At source frame 235 the
disagreement is 179.60 degrees despite a 0.46 px network-correspondence residual
and 13,695 inliers. The comparison viewer now offers exact-frame review buttons
and an asymmetric UV color grid to expose flips hidden by repeating checkers.
These references share model lineage and are not independent truth.

The keyboard motion-memory diagnostic obtains evidence on 206/240 frames and
contradicts 30 saved poses, but misses the late flip: surviving surface points
drop from 27 to 22 at frame 1381. The separately tested ORB identity bank gets
evidence on only 13 keyboard and 11 bottle frames and also misses both late
flips. It is not integrated or promoted. The full with/without motion-memory
neural comparison rejected the candidate: 149/240 accepted with memory versus
209/240 without, and secondary P95 10.64 versus 10.75 pixels. Both runs count
all original frames. The first 30 poses fail the real prefix-invariance check
despite identical masks and provenance.

Fixed-texture grayscale/gradient checks were measured on all 240 bottle images.
Exploratory thresholds flag 27/44 large orientation disagreements, miss 17,
and flag no estimated-reference controls. This is a diagnostic ablation, not a
new tracking result. An optional appearance evidence adapter and synthetic
brightness/visibility tests are prepared for a separate automatic recovery run.
The thresholds were selected after inspecting original-window failures; held-out
validation is still required. No live speed or independent accuracy gate passes.

Setup-only renderer repetitions isolate color variation: shaded rendering gives
2–4 distinct color hashes for identical inputs, while depth is identical.
Unlit textured rendering produces one color/depth hash in each of three context
tests and identical hashes across two independent processes. This proves the
limited renderer check, not full network repeatability. The next experiment
uses optional unlit tracking templates, separate unlit appearance rendering,
the existing cached masks and fixed FoundPose bank, and no sparse-memory veto.
It tests keyboard 2-vs-4-frame repeatability before paired 30-prefix/full-240
bottle runs with and without the appearance check. Source, bank and checkpoint
hashes are recorded. All older defaults and results remain preserved.
Run `tools/test-appearance-windows.ps1` only when no other neural stage is active.
The viewer's "Texture recovery experiment" remains explicitly experimental.
All 83 offline regression tests passed after the additions; these are not model
accuracy tests.

The subsequent real 2-vs-4-frame prefix still fails. Its trace first differs in
cropped-render color, with identical camera/depth hashes;
single-thread math does not remove it. Nearest texture sampling also fails
cross-process color repeatability in a crop-only diagnostic and is not added to
the active tracker. The current paired bottle experiment therefore remains
unverified even if its attachment metrics improve. Unlit appearance needs
different exploratory thresholds (grayscale .7 and gradient .25); this flags
37/44 original orientation disagreements and no estimated-reference controls.
The interrupted .2-threshold setup is preserved as `appearance-unlit-v1`.
The fresh paired run completes 30/30 accepted prefix frames in both modes and
records matching provenance before continuing on both full 240-frame windows.

Both full paired bottle runs now completed: unlit control 230/240 accepted,
appearance-assisted 235/240. Secondary median/P95 are respectively 1.58/19.34
and 1.58/19.09 px, versus the earlier run's 1.48/44.13 px. Neither new run has
an accepted pose more than 90 degrees from the estimated reference orientation.
This reduces severe failures but leaves substantial tail attachment error.
Both real 30-frame prefix tests still fail despite matching provenance. The
appearance stage rejected two proposed poses; the major change versus the old
run is the unlit-template configuration, not proof that the extra veto solves
tracking. No default is promoted. Exact-frame browsing, larger phone views,
directional UV grids, completed experiment counts and bounded asset-fetch
retries are available through the approved comparison link.

The user-approved route-limited Cloudflare phone preview remains active while
the computer and processes are running. It exposes only comparison assets and
three public-dataset clips. Saved result files refresh in the viewer. Whole-clip
blob loading avoids stalled range seeks after transient tunnel disconnections.
Concurrent non-neural render diagnostics mean the completed motion-memory run's
timings must not be treated as an isolated performance benchmark.

Completed single-sample keyboard experiment (2026-10-02): all 240 original
frames have accepted poses, and no frame exceeds 90 degrees of orientation
disagreement from the estimated reference. The old final 174-degree jump is
absent. The first 30 predictions match a separately run 30-frame prefix with
identical provenance/settings. On the 203 frames accepted by both runs,
secondary median/P95 are 3.98/10.62 px versus the old 4.89/10.24 px. Coverage
and typical error improve; tail accuracy does not. Defaults remain unchanged.
The opt-in result and late-frame screenshot are in the shared viewer. The
full report is `artifacts/model-quality/render-stability-results.json`.
No neural jobs remain running from these experiments.

Completed keyboard recovery experiment (2026-10-02): the unassociated detector
selected a background chair after the forced 15-frame black-input interruption.
That pose control was stopped after 118 scored frames and preserved; it is not
a full-window benchmark. A separately cached unlit/grayscale detector template
bank still selected the chair and was not promoted.

The opt-in association policy combines current proposal identity with overlap
against the last observed object mask, expires after 30 source frames, and never
substitutes that remembered mask for current foreground. The new full automatic
trial processes all 240 original keyboard frames and accepts 224. All 15 blank
inputs are lost/suppressed, frame 1254 validates privately, and frame 1255 resumes
rendering after the second validation. The detector itself took 30.3 seconds
offline. Mechanical interruption checks pass; physical hand occlusion and
independent attachment accuracy do not yet have verified results. This mode is
visible as "Recovery with object association" in the approved phone comparison.
No tracker defaults are promoted. The original results remain preserved.

Supplementary area-based surface sampling exposes the legacy vertex-index
metric's tessellation bias. On the same original keyboard window, 128 points
sampled by triangle area give secondary median/P95 6.73/40.68 px for the previous
203-frame tracker and 6.04/22.33 px for the 240-frame single-sample tracker.
The associated interruption trial gives 6.60/22.59 px and 92.8% reference-visible
surface sample availability. These are estimated-reference/self-visibility
diagnostics, without hand ground truth, and cannot pass the independent gate.
Legacy scores are unchanged. The new result still has substantial surface drift.

A separate 120-frame prefix trial including the actual interruption/recovery
completed. All 121 masks (setup plus scored frames) and all 120 scored poses,
states and failures match the corresponding full-run prefix. Source, checkpoint
and stage-specific bank provenance and settings match. This is a limited real
prefix check, not independent accuracy or a test of other objects. The report is
`artifacts/model-quality/recovery-prefix-results.json`. All 97 offline regression
checks pass. No neural experiments from these runs remain active.

The former public preview returned Cloudflare 1033 despite local server health.
The same approved route-limited server was reconnected through the existing
cloudflared binary using IPv4/HTTP2; no routes or system network settings changed.
The replacement URL is
https://creations-council-cindy-prize.trycloudflare.com/model-quality/.
Its keyboard video, associated recovery frame 1255 and updated evidence notes
were verified in the browser. All three preview route-isolation tests pass.
Temporary URLs still depend on the computer and preview processes remaining on.

Model-appearance and contour diagnostics (2026-10-02): prescribed canonical
views load no video poses. The keyboard key-side unlit render has luminance
standard deviation 25.27/255; the underside is 99.44% below 2/255 and lacks the
rubber feet and printed markings visible in source frame 1314. The model/source
figure appears in the public viewer's expandable keyboard appearance section.
This exposes limited appearance coverage, not an isolated causal proof of drift.

Separate source-only, coarse agent mask labels were drawn at 1149 and 1314.
They remain quarantined in `.cache/model-quality/diagnostics/keyboard-mask-pilot`;
the frozen 40-frame files are unchanged, and the pilot requires human audit.
Approximate original-mask IoUs are .953 and .876, with uncertain thin borders
and finger gaps; these are not release-gate measurements or correction inputs.

An experimental fixed-3D convex-contour proposal uses actual current masks,
outward boundary normals, bounded SE(3) changes and broad spatial support. It
reads no evaluation labels. On ten selected keyboard frames, nine proposals are
available. A paired neural test then runs five additional GoTrack iterations
from either the saved automatic pose or its contour proposal. Both branches
validate 9/10, with eight common frames. Secondary area median/P95 worsens from
6.91/24.79 to 7.26/26.11 px. The experiment is rejected and not integrated or
displayed as successful tracking. All 100 numerical regression checks pass.

The bottle single-sample experiment completed all 240 original frames and its
30-frame repeatability check passes. Quality regressed: 222/240 accepted versus
the earlier 230/240 unlit control; secondary vertex P95 is 36.15 px and area P95
38.13 px. Thirty-two accepted poses have more than 90 degrees of disagreement
with the estimated reference orientation. This experiment is rejected for
quality and remains visible for inspection. The stopped missing-smoke bootstrap
attempt is preserved separately.

A current-image texture diagnostic with the existing frozen thresholds flags
30 of those 32 saved poses, and flags none of the other accepted poses. It reads
no reference poses during inference. This is a diagnostic, not a sequential
tracking improvement. A new same-code full original bottle trial enables this
check alongside stable rendering; all 240 frames now complete. It is published
as "Stable rendering + texture recovery". No correction inputs or scoring
annotations enter either trial. Final scoring includes accuracy on common
accepted frames as well as total availability; detailed results appear above.

The comparison now publishes only required pose/state fields; complete native
hashes, timings and validation diagnostics remain in cached result records.
The approved route-limited server supports lossless gzip for JSON/HTML, with
encoding-specific ETags and identity byte ranges. All four preview HTTP tests
pass. Actual keyboard JSON transport is 607,463 bytes versus 2,409,758 decoded
bytes (74.8% smaller), verified byte-for-byte after decompression. No additional
public routes were added. The existing IPv4 tunnel and public URL are unchanged.

## Deferred Unity build setup

Only Unity 2022.3.35f1 is installed. The intended project requires Unity 6000.3.25f1. The Unity 6 Windows editor requires approximately 8.4 GB installed. Earlier checks had less than 2 GB free; disk space was subsequently freed, with approximately 100 GB available before the 60 FPS study. Android/iOS support modules need additional space. Setup remains deferred at the user's request; the offline model runs independently.

An isolated supplementary shader check attempted to launch the existing Unity 2022 editor. It exited with `No valid Unity Editor license found`. The log is `.cache/shader-check/shader-check.log`. No shader compilation result was obtained. Even a successful Unity 2022 supplementary check would not substitute for Unity 6/mobile compilation.

iOS signing/device builds additionally require macOS, Xcode, a signing identity, and physical devices. These are not available in this workspace. No AR release gate has passed.

## Remaining milestone work

The improvement goal resumed after the agent setup. A fresh mug smoke using
unlit, single-sample templates passed on the existing GTX 1660 SUPER: checkpoint
loading and inference completed, silhouette IoU was 1.0, median depth difference
against the independent renderer was 0.000544 m, and metric extents agreed.
Evidence: `.cache/model-quality/smoke/mug-cuda-unlit-no-msaa.json`. This is a
setup check, not tracking, segmentation, independent accuracy or phone evidence.
The fresh ten-frame mug comparison uses the explicit local solver option and
preserved automatic masks/seeds. Both branches validate 10/10 selected frames;
secondary surface median/P95 is 2.456/7.570 px for the control and 2.449/6.503 px
without the solver's extrinsic guess. This reproduces the earlier selected-frame
diagnostic, with little median change; it is not new sequential or independent
accuracy evidence. All 50 sampled solver packets replayed on CPU with both
options. Evidence is preserved in
`.cache/model-quality/diagnostics/mug-local-pnp-v1` and its separate replay
directory. The full-retained/native-mask replay and paired chronological
follow-up are now complete, with results below. No prior result is overwritten.

Full-retained native replay now completes for all ten selected mug frames and
all five iterations each. The fresh capture is 49,379,941 bytes, below its
64 MiB packet/mask cap; all final control poses match the preceding capture
exactly. CPU replay matches all 50 actual control candidates, hashes, counts
and validation results within frozen tolerances; both solver options validate
50/50 on those control-conditioned inputs. Evidence is in
`mug-native-full-retained-v1` and `mug-native-full-retained-v2-replay` under the
diagnostics cache. An earlier replay stopped when a live Windows reader blocked
atomic replacement; that failed run is preserved. Compact report serialization
passed ten CPU tests and Sol review before the successful rerun. These checks
verify instrumentation, not chronological tracking or independent attachment.

The evaluator now reports each branch's all-source-frame availability and
own-accepted errors beside common-frame metrics, and rejects explicitly
unfinished records. Eight targeted CPU tests and Sol review pass; original
fields recompute exactly for all three preserved diagnostics. The phone
comparison now shows mask, pose, render and failure states separately, with
larger touch controls. The existing approved public URL was refreshed without
new routes or video copies; a 390×844 browser check is saved at
`.cache/ui-proof/mobile-tracking-states.jpg`. Physical-phone and independent
tracking-accuracy validation remain pending.

The strict CPU mask-cache audit completed for all three preserved original
windows. Every setup/scored mask artifact passed path, native-resolution and
consumed-hash checks; all 720 scored masks are marked available. This rules out
a recorded mask-loss transition as the sole explanation for disappearing
mapping in those runs, but does not prove mask identity or accurate boundaries.
Detection invocation remains unknown in the old records. All 120 annotation
candidates are still pending, so independent scores are null and no accuracy
gate passes. Reports are in `.cache/model-quality/diagnostics/mask-audit-v1/`.

The mug solver change did not generalize from selected frames to the full
chronological window. Six fresh complete-mode runs (30, 120 and 240 frames per
branch) completed with terminal exit-zero records and identical frozen inputs.
All six prefix comparisons pass for poses, masks, states, failures and inference
traces. The control accepts 226/240 frames and the candidate 235/240. On each
branch's own accepted frames, area-sampled median/P95 is 2.704/39.279 px versus
2.703/40.420 px. On the same 226 commonly accepted frames, it is 2.704/39.279 px
versus 2.627/39.284 px. The candidate additionally accepts frames 1062 and 1063
with at least 90 degrees of disagreement with the estimated reference. It fails
three frozen continuation checks and is rejected for advancement; the default
remains unchanged. Evidence is
`.cache/model-quality/diagnostics/mug-pnp-full-window-v2/report.json`, bound to
the closed stage outputs and traces. The earlier preparation failure is retained
separately. These are secondary estimated-reference metrics, not independent
attachment accuracy. Pose-stage medians are approximately 2.2–2.3 seconds on
this desktop GPU, with CPU test/review work concurrent; no live FPS claim follows.

The approved phone comparison now includes the two completed mug branches.
Choose White mug and No-extrinsic-guess ablation to show the same-code control
beside the candidate on the same timeline and observed masks. The rejection and
own/common-frame measurements are explicit; earlier modes remain available.
Nine frontend fixtures, final review, actual-report generation and four preview
route/range tests pass. The 390×844 browser view and object-switch reset were
verified; screenshots are `.cache/ui-proof/mug-pnp-paired-late-frame.png` and
`.cache/ui-proof/mug-pnp-mobile.png`. This is web-preview verification, with
physical-phone and independent attachment checks still pending.

The local annotation editor now supports explicitly reviewed fully hidden
frames without inventing object regions or landmarks. Empty visible reviews
are rejected, visibility edits reset pending status, and the evaluator reports
false foreground and visible-render leakage on reviewed hidden frames. CPU and
browser checks pass, but the actual 120 annotation candidates remain pending.
The annotation editor is not exposed through the public comparison routes.

A frozen bottle correspondence diagnostic completed all 64 synthetic conditions,
40 real-image conditions and four repeat checks. All repeats match, source and
packet hashes verify, and all eight known-correspondence geometry controls pass.
Grayscale templates do not improve synthetic identity availability or the
tested real-image texture support. The matching/patch qualification gates fail,
so these results do not justify a tracker change or independent accuracy claim.
With exact synthetic endpoints, patch support passes on three textured-side
carriers; learned-flow support passes only one. The reverse-side banks remain
too sparse for that test. The next bounded calibration compares identical-image
matching with and without observed-mask clipping before another tracking change.
Original failed preparation and CPU display runs remain preserved. The completed
audit is `.cache/model-quality/diagnostics/bottle-identity-v2-audit-r2/audit.json`.

A follow-up bottle diagnostic completed six GPU calls on three frozen synthetic
carriers: an identical rendered query, then that same query clipped by the observed
mask. All six pass unchanged identity, confidence and patch-support checks.
Correctness among confident visible sources ranges from 96.94% to 99.61%, with
confidence availability from 99.88% to 100%. The three stored eight-degree view
changes fail the combined gate when recomputed. This implicates view-dependent
correspondence limitations in this synthetic cohort; it does not validate the real
hand masks or improve live/video attachment. All packets and decisions were
independently recomputed. The closed record is
`.cache/model-quality/diagnostics/bottle-zero-view-calibration-v3/zero_view.json`.
The two earlier preflight failures are preserved with zero neural calls. A local
diagnostic image is `.cache/ui-proof/bottle-zero-view-v3.png`. The phone viewer
continues to replay precomputed poses and masks; its FPS is browser playback.

Unity setup is deferred at the user's request. Current work evaluates HD recorded-object tracking, CPU vision latency toward a 30 FPS live target, and synchronized mask/mapping previews. Reliable attachment and recovery remain necessary before live integration. For mobile integration later, resolve the supported editor/license/build environment, compile the project, repair platform integration issues, and run the device procedure. The shader, runtime-generated scene/UI, XR loader configuration and Android/iOS picker bridges all require actual integration verification.

After the stationary fixture gate passes: implement synchronized mobile capture, scan job service and reconstruction worker, then measured scanned-object attachment. The offline tracker can inform subsequent native reference-feature/tracking integration, but requires stronger recovery, phone-compatible onboarding and device benchmarks first. Skin and fabric deformation remain independent research milestones. The C++ module currently converts coordinates only; the server directory contains a specification only.
## Bottle source-texture pose selection diagnostic (2026-10-03)

The frozen CPU-only experiment completed all twelve cached synthetic/self-view
conditions and all 24 fitting arms with terminal exit zero. Source hashes stayed
fixed, immutable packet hashes matched, and all 24 deterministic repeats matched.
The exploratory prerequisite **failed**, so this candidate is not integrated.
Selecting only source texture rejected all three opposite-side proposals that
the all-match arm accepted (about 172 degrees wrong in this synthetic test).
However, the three positive rotation errors changed from 2.394/3.379/3.770 degrees
to 3.104/3.322/1.534 degrees, with translation trade-offs; self controls also
failed the frozen nondegradation/accuracy requirements. Positive full-surface
projection P95 improved from 1.797/2.077/1.644 to 1.672/1.792/1.273 pixels at
720p-equivalent height, but this report-only metric cannot override the failed
prerequisite. These are known synthetic/model-view tests, not new real-video
attachment results or independent mask/landmark accuracy. The 872,845-byte
private report is `bottle-source-observability-pose-ablation-v1/pose_ablation.json`
(SHA256 A394BF54F51DDF527549EE5B3B6C7E8F623546B2C277A49051DF2CE1C4D95259).

The phone comparison now includes this rejected diagnostic in an expandable
section, with readable metrics before the optional chart. Reviewed source and
23 focused tests passed; generation and generated JavaScript syntax passed.
Local and approved Cloudflare responses match, and the unchanged twelve public
routes total 29,724,534 bytes. A 390x844 browser check confirmed readable text,
a loaded chart and no horizontal overflow. This publishes evidence only; the
existing video predictions and tracker defaults remain unchanged.

The corrected chronological mug diagnostic is prepared in a fresh private
resource-v2 directory. Independent CPU review verified all 991 source bindings,
249 frozen files and 241 ordered native masks, plus the unchanged 201-frame
window and twelve diagnostic frames. The 320 MiB raw-copy budget has a proven
284,352,096-byte maximum; the earlier partial 64 MiB trial remains preserved.
The final parent driver passed independent review and ten CPU lifecycle tests,
including interruption immediately after worker launch. The two-stage neural
execution completed both 201-frame branches with exit zero. Their full ordered
output, settings, provenance and inference-trace comparisons match the earlier
run. The parent verification exited one because captured crop-to-native pose
matrices fail a separate packet-consistency check by roughly 3e-8. Exact replay
identified the cause: the verifier omitted the production rotation
canonicalization before unit conversion. All22 stored native poses reproduce
exactly when that production chain is used, at the unchanged strict tolerance.
A separate reviewed offline revalidator completed with exit zero. Both201-frame
outputs and setup decisions, all2286/2112 ordered trace records and22-pair
accounting pass. Sol independently verified all1943 report pins and31 archived
sources. The new299293-byte report is42BEF03D; the original failed report and
parent receipt remain preserved. This establishes capture consistency only,
not surface identity or an attachment improvement.

The twelve-condition bottle patch calibration is complete. A first execution
failed its imported-source binding check; a corrected full repeat binds1624
sources and reproduces every condition result exactly. The calibration itself
is rejected: the hardest positive has42.43% verified coverage against50%
required, and9.60% fixed-anchor hull against12%. All12 pose fits were skipped.
The accepted rejection is diagnostic evidence, not a new video tracker.

Additional ten-second test windows are now frozen for each object, using only
source hashes and video-header metadata. Each reserves600 scored frames and
ten earlier onboarding frames, disjoint from the original benchmark. No scored
RGB was decoded or predictions generated during the freeze. Independent human
annotations remain0/120, and no release-quality or live-FPS gate has passed.

The annotation editor now separates visible-mask review from landmark review
and explicit unobservable landmarks. The integrated editor/evaluator passed
55 independent Python regressions and an actual-template Node smoke test.
The local generated editor loaded original pixels in the browser and refused
incomplete mask and landmark review attempts. The canonical annotation files
remain unchanged; receipt12987EA2 records this UI check. Human reviews are
still required before independent accuracy can be scored. The editor remains
local-only and is excluded from the approved public preview.

