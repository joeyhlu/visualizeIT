# Mapping and real-footage prototype without Unity

Python 3.12 with `requirements-real.txt` runs the original offline pipeline. The workspace uses the bundled Python executable and an isolated `.cache/vision` installation of xatlas 0.0.11 and OpenCV Python 4.13.0.92. The original pipeline needs no GPU, Unity activation, model weights, or cloud service. The optional learned-mask experiment described below adds model weights and a separate runtime. Engine/device validation is still pending.

## Reproduce

On this Windows workspace, `powershell -ExecutionPolicy Bypass -File tools/run-offline.ps1 -Stage Verify` selects the bundled Python automatically. Stages are `All`, `Verify`, `Mapping`, `Acquire`, `Control`, `Track`, `Evaluate`, `Interruptions`, `HD`, `Performance`, `HigherFPS`, and `SixtyFPS`; override `-Python` in other environments. `All` runs the original study and can take several minutes. Failed research gates remain reported data, while execution/data errors stop the runner.

```text
python -m bench.mapping_experiment
python -m bench.mapping_experiment --design my-design.png --output artifacts/mapping-improvements
python -m bench.mapping_experiment --design my-design.png --tile-metres 0.04 --rotation-degrees 30
python -m bench.bop_assets --include-limited-motion
python -m bench.video_experiment --operation control
python -m bench.video_experiment --operation control --object 15 --design my-design.png --design-mode artwork
python -m bench.video_experiment --operation track
python -m bench.video_experiment --operation evaluate
python -m bench.video_experiment --object 15 --interruptions --render-stride 110 --output artifacts/video-interruptions
python -m unittest bench.test_renderer bench.test_scan_mapping bench.test_improvements bench.test_remote_zip
python tools/test_evaluate_tracking.py
python -m http.server 8876 --bind 127.0.0.1 --directory artifacts
```

`mapping-improvements/index.html` compares mapping candidates. `video-tracking/index.html` contrasts supplied-pose controls with estimated-pose overlays. `video-interruptions/index.html` shows an artificial three-frame interruption. Control-only output is `video-tracking/controls.html`. Opening files directly also works. No external scripts or accounts are required.

## Mapping behavior

Geometry and source correspondence remain unchanged. Candidates are the original metric atlas, chart scale correction, and corrected charts with adjacency-based rotation/translation alignment. Physical seam length and visibility weight the seam solver. Repeating designs allow whole-tile translations; single artwork uses absolute coordinates. The largest chart fixes the arbitrary global orientation. The solver retains the best of ten translation iterations.

Promotion rejects increased invalid area, inverted UVs, duplicated/overlapping artwork, P95 scale-error increases above 0.5 percentage points, and accepted-area decreases above one percentage point. Eligible candidates compete on mean asymmetric-design seam colour error. The provisional improvement target is ≥30%; a failed experiment retains its baseline and displays the reason. A connected UV layout can overlap after alignment: those single-artwork candidates are rejected rather than presenting duplicated content as a solution.

The report also measures checker, stripe and labelled-artwork seam sensitivity. Imported PNG/JPEG designs retain aspect ratio; transparent images receive a white diagnostic underlay. There is one shared artwork transform, with no per-chart image repetition in artwork mode. Nonperiodic tile borders and general curved-surface flattening can still create unavoidable discontinuities. Purple faces are seam-adjacent, not a confidence improvement. Teal/orange retain the local distortion test.

Version 2 NPZ assets contain positions, normals, triangles, metric UVs, source vertices/faces, chart IDs and trustworthy-face masks. Video assets additionally retain `benchFromSource`, reference 3D points and ORB descriptors. Coordinate/scale conventions are explicit: BOP meshes/translation use millimetres in source files; the tracker uses their original right-handed model axes in metres. The existing renderer uses the recorded axis reflection and translation. Matching model hashes are checked before use.

## Dataset and truthful evaluation

The public source is [BOP's YCB-Video conversion](https://bop.felk.cvut.cz/datasets/), supplied via [its official dataset repository](https://huggingface.co/datasets/bop-benchmark/ycbv). YCB-Video: Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, Dieter Fox, *PoseCNN*, RSS 2018. BOP lists YCB-Video as MIT licensed. Preserve attribution to the original YCB scanned models under [CC BY 4.0](YCB_SOURCES.md) as well. These are local internal experiments, not a BOP leaderboard submission.

HTTP-range ZIP/ZIP64 access reads the archive directory and selected members only. Each decoded member is CRC/size checked, the dataset revision is pinned on first acquisition, and completed input caches are SHA-256 verified on reuse. The tool refuses a full-download fallback. Source cache is capped at 384 MiB, preview outputs at 256 MiB including native HD derivatives, and writes retain a 512 MiB disk reserve. Full scene discovery annotations and unused model textures are not retained; selected metadata and hashes remain in manifests.

The intended selection requires 120 consecutive frames, at least 60 frames with ≥50% visibility, and ≥15° viewing change. Drill scene 000059 qualifies. Bottle scene 000052 and mug scene 000055 have approximately 10.6° and 7.4° respectively; they are explicitly failed-selection controls, enabled with `--include-limited-motion`. Strict acquisition does not silently accept them. Full contiguous annotations can contain errors; BOP's separately curated sparse benchmark avoids some erroneous poses. Our control previews and metrics therefore remain an internal study using supplied annotations, not independently annotated physical accuracy.

The first ten annotated frames generate reference 2D-to-3D correspondences through mesh ray intersections. Their poses and measured depth restrict onboarding features. No evaluation pose, evaluation mask, or depth enters the tracker after onboarding. On subsequent frames it receives RGB, K and frame ID only. Ground-truth poses and source depth are consumed separately by scoring/compositing. This is initialized tracking, not automatic object recognition.

The RGB tracker uses ORB (2,000 features), mutual Hamming matches with ratio 0.75, forward/backward optical-flow checks ≤1 pixel, descriptor refresh every five frames/on loss, RANSAC EPNP and LM refinement. It requires ≥12 inliers, ≥50% inliers, ≤3 native-pixel median reprojection error, positive depth, and ≥20% projected feature coverage. Underconstrained planar configurations become limited; there is no unverified planar ambiguity resolver. Invalid tracking hides the overlay and clears stale flow; relocalization uses the reference bank. No temporal smoothing conceals errors. Raw pose-error residual jitter relative to supplied annotations is measured on consecutive valid frames; it is not a sensor-jitter result.

Scoring projects held-out model landmarks using independent annotated poses and filters their visibility using separately acquired annotated visibility masks, normals and source depth. Evaluation masks are hash checked and never enter tracker or compositor inputs. Every frame remains in the tracker trace; all-frame and ≥50%-visible-frame availability are reported separately. CSV attachment errors are isotropically scaled to 720p-equivalent height, with native errors also reported. This is not native 720p footage. Near-rigid rounded annotation rotations are projected to SO(3), rejecting correction above 0.001 Frobenius norm.

Measured-depth compositing suppresses unknown-depth pixels and uses tolerance `max(5 mm, 1% of depth)`. It uses neutral diffuse checker shading, not photorealistic lighting. JPEG/GIF previews are presentation artifacts; measurements use original captures. Tracking is evaluated on all 110 held-out frames per object; normal GIFs show every tenth frame plus the final frame, with synthetic presentation timing. No dataset frame rate is inferred. Reported tracker timing excludes CPU rendering/loading and does not establish phone FPS.

The interruption run replaces tracker RGB input with black for evaluation indices 30–32 while preserving original evaluation images/depth. Reports retain the injected condition, lost states, recovery duration and final unrecovered loss. It is not presented as an actual hand occlusion.

## Native HD RGB-only study

Run `python -m bench.hd_experiment` to acquire two bounded, hash-verified HOPE static-object orbit clips and test the existing tracker at native 1920 × 1440 resolution. Outputs are in `artifacts/video-hd/index.html`: original, supplied-pose control and estimated-pose MP4s, every-frame pose traces, and evaluation CSVs. Each clip uses ten annotated initialization frames and 110 held-out frames. The first ten masks may initialize reference features; subsequent masks and poses are evaluator inputs only. Independent model ray intersections reject self-occluded scoring landmarks. No depth image is synthesized or represented as a sensor measurement. External occlusion is not implemented for these RGB-only renders.

The videos sample every second evaluation frame plus the final frame and use explicit 10 FPS presentation timing; source frame rate is unknown. They retain native dimensions with MPEG-4 compression. Reports include native and 720p-equivalent errors, availability, view span and CPU tracking time. These isolated-object orbit clips complement the earlier cluttered RGB-D footage and do not establish phone performance or object-pickup tracking. The bundled source metadata specifies **CC BY-NC-SA 4.0**, unlike the BOP website's CC BY-SA listing; retain the more restrictive source terms for these local research derivatives. Source: NVIDIA HOPE, Tyree et al., and BOP onboarding conversion, pinned Hugging Face revision `ddd0a26ca3460085e93b648748160fb25e4b5566`. Run `python -m unittest bench.test_hd_experiment` for independent ray visibility and lost-frame scoring checks.

## Genuine 60 FPS hand-held object study

Run `python -B -m bench.show3d_experiment` or `tools/run-offline.ps1 -Stage SixtyFPS`. The player is `artifacts/video-60/index.html`, served at `/video-60/`. It contains a keyboard, white mug and dressing bottle from [SHOW3D](https://huggingface.co/datasets/facebook/show3d-dataset), pinned at `720104c5793a55aab352ca933f1f73bde8582d3c`. Source metadata, MP4 rate and consecutive timestamps verify 60 FPS. Each object uses ten consecutive onboarding frames and all 240 following frames for evaluation. The first qualifying window has ten valid initialization poses and at least 90% reference coverage; selection does not depend on tracker success. Source is 1024 × 1280 monochrome; footage retains every frame at 640 × 800 / 60 FPS, without interpolation. Browser draw rate is displayed separately and can fall below source FPS.

Canonical GLBs come from [HOT3D](https://huggingface.co/datasets/bop-benchmark/hot3d), revision `30fe9674782f32e1e5edba98476b6ff4300132c5`. Models match names, not SHOW3D library indices: keyboard ID 28, mug_white ID 9 and bottle_ranch ID 15. Embedded geometry, normals and UVs retain node transforms; dimensions are checked against model metadata in metres. Source hashes and raw poses are retained. SHOW3D is CC BY-NC 4.0; the HOT3D model agreement and attribution are cached. This is a local research preview. Selected acquisition has a separate 384 MiB cache; derivatives share the existing 256 MiB artifact limit. Preview encoding is capped at 16 MiB per clip before atomic publication, with a 512 MiB disk reserve.

SHOW3D object poses are automatic FoundPose/GoTrack estimates, **not independent human ground truth**. Confidence ≥0.5 and valid camera flags are required. Its shared “world” frame moves with the back rig: `cameraFromObject = inverse(worldFromCamera) × worldFromObject`, with translations converted from millimetres to metres. Released undistorted PinholePlane intrinsics are used. Invalid reference poses remain null. Scoring uses mesh normals/self-visibility, without externally occluding hand masks or depth. Reference silhouettes are projected geometry, not predicted segmentation; onboarding may therefore include hand features. After onboarding the tracker receives only camera images, intrinsics and frame ID.

Eight- and three-level ORB candidates share the capped reference bank, 288 × 360 working images and 400-feature cap, with independent states and rotating execution order. Timings include monochrome resize and CPU vision, excluding decode, evaluation, encoding, rendering and acquisition. Five warmup frames are omitted only from latency. Every frame counts toward availability. All-state and accepted/lost-state latency and sample counts are reported separately. A provisional 60 FPS vision budget is 10 ms P95 within 16.67 ms total. No candidate passes the combined accuracy, availability and speed gate; no phone/live/thermal result exists.

One decoded video frame drives the original, reference silhouette and mapped-design panels. Frame timestamps select matching saved poses; no tracker runs in the player. Reference, baseline and faster-tracker modes are available; invalid/lost states hide the design. Patterns use original mesh UVs and neutral normal-based shading. Seam continuity, PBR lighting, hand occlusion and automatic scanning are not established by these controls.

## Mask / pose quality recovery

The original SHOW3D cyan silhouette uses reference geometry. New `surface_tracker.py` retains private flow during weak support, accepts validated planar pose solutions, and never displays the onboarding pose as a fallback. `model_references.py` generates descriptor views from model texture alone. GrabCut, contour and model-reference retries are research candidates: all combined quality gates failed. Contour/replenishment trials increased availability at the cost of drift; they were not promoted.

Run these quality experiments after the original SHOW3D acquisition:

```text
python -B -m bench.retry_experiment --mask --output artifacts/video-60/retry-mask
python -B -m bench.retry_experiment --mask --synthetic --output artifacts/video-60/retry-model-mask
python -B -m pip install --no-deps --target .cache/segmentation -r bench/requirements-segmentation.txt
python -B -m bench.segmentation_assets
python -B -m bench.learned_mask_probe
python -B -m bench.learned_mask_experiment
python -B -m bench.retry_experiment --learned --synthetic --output artifacts/video-60/retry-learned-mask
python -B -m bench.recovery_player
```

Optional runtime versions are pinned separately; the bundled Python supplies CFFI 2.1.1. Import the existing vision/NumPy runtime before MediaPipe so the supplementary installation does not replace baseline tracking dependencies. Models use official versioned Google URLs and pinned SHA-256 bytes. The MagicTouch v1 file is 6,227,884 bytes (`e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25`); hand landmarker v1 is 7,819,105 bytes (`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`). New MagicTouch v2 was downloaded separately, but the released Windows 1.0.1 library lacks its required native creation function; no v2 inference result is claimed. The working probe explicitly uses the legacy API/model.

`learned_foreground.py` predicts selected-object masks using image-only prompted segmentation, forward/backward image flow and hand landmarks. Hand cores are conservative geometric approximations, not exact semantic hand masks. Rejected predictions are not displayed; private state moves with image evidence for subsequent attempts. Returned mask availability is not accuracy: visual examples show forearm leakage, missing regions and prompt failures. No IoU or human-annotated boundary evaluation exists yet. The model is slow on this CPU and is not used inside the player.

The learned pose retry filters image correspondences with independently precomputed masks before solving pose. It uses the same initialization allowance and held-out windows, with no later reference poses entering update. All 240 frames count, including failures. Tracker timings exclude model inference; separate mask + tracker totals are reported by summing their recorded durations, with no concurrent-live speed claim. Results: 32.1% keyboard, 5.0% mug and 23.3% bottle accepted-pose availability; all gates fail. Fifty-five offline regression checks pass.

Serve `artifacts/` and open `/video-60/recovery.html`. It loads five pose variants, learned masks, original 60 FPS WebM and model UVs; source video is not duplicated. Mask selection controls renderer clipping, so unavailable masks hide the surface even if a reference or tracker pose exists. Reference controls remain visibly labelled. Keep mask prediction accuracy, pose validity, rendering occlusion and playback FPS distinct.

Next: benchmark temporal video segmentation such as [SAM 2.1](https://github.com/facebookresearch/sam2) with hand/background correction prompts and annotated mask boundaries. Then benchmark stronger learned correspondences or model-based pose estimation on the same data; do not assume a good 2D mask proves 6D attachment. RGB-D methods additionally need a depth-bearing dataset. Only optimize the successful pipeline toward 30 FPS live after attachment and recovery pass.

## Remaining work

For a paired comparison on the same inputs, run `python -m bench.speed_sweep`. It tests the current 480 × 360 grayscale / 400-feature baseline, resizing RGB before grayscale conversion, and the same resize-first variant with a 16-bit LSH key. All variants share the ten-frame onboarding bank and use independent tracker states; candidate order rotates per frame. All 110 held-out frames per object count toward attachment and availability. Latency includes preprocessing and vision, with five warmup frames excluded only from timing. Reports are saved under `artifacts/video-hd/performance-higher-fps/`. The experiments leave production/default tracker behavior unchanged. Faster descriptor lookup may lose matches, so inspect both timing and quality gates.

`python -m bench.visual_video` creates a bounded, small HTML player at `artifacts/video-hd/visual.html`, reusing the existing native WebM videos and hash-checked reference masks. It shows original footage, annotated masking and surface mapping in synchronized playback. The mask is an evaluation reference, not a predicted segmentation. Video-frame callbacks align masks to decoded images; presentation remains 10 FPS and does not establish camera/tracker throughput.

Run `python -m bench.speed_sweep --roi --output artifacts/video-hd/performance-object-region` for a paired object-region experiment. It initializes the search from onboarding features, updates it from accepted tracking points, expands after losses and returns to full-image search after three failed frames. Crop-local detections are restored to full working-camera image coordinates before pose solving. No evaluation mask or pose supplies the search region. Ground-truth visibility is computed once per frame and shared only by the independent evaluators. This reduces benchmark overhead without changing the measured tracker timing. Individual replay controls are `bench.performance_experiment --fast --gray --compact --height 360 --features 400 --roi`, with optional `--resize-first` and `--lsh-key-size 16`; all these are opt-in research candidates.

Run `python -m bench.speed_sweep --levels --output artifacts/video-hd/performance-pyramid` to compare eight, four and three ORB pyramid levels. Each variant keeps the same image size and 400-feature cap and uses the same original onboarding bank. The equivalent individual replay flag is `bench.performance_experiment --fast --gray --compact --height 360 --features 400 --levels 4`. Fewer pyramid scales can reduce detector cost while changing correspondence support, so they require the same full accuracy/availability evaluation. Defaults remain eight levels.

For the requested **30 FPS live target**, `python -m bench.performance_experiment --height 480` replays the native HD clips with smaller, calibrated tracking inputs; `--compact` keeps at most 1,500 reference descriptors, with at most two per 2 mm surface cell. Use different `--output` folders to compare variants. Every held-out frame is independently scored at native resolution. Latency includes image resize and vision, excludes image decode and rendering, and discards five warmup frames only for timing. A provisional P95 vision budget of 20 ms leaves part of the 33.3 ms full-frame budget for camera, GPU rendering and display. This is a desktop optimization study; phone FPS requires an actual camera/GPU implementation. RANSAC and flow thresholds remain in working-image pixels, so speed changes must be assessed together with accuracy and tracking availability.

Single-artwork seam continuity remains unsolved where aligned charts overlap. The mug tracker loses correspondence support; the current reference bank does not reliably relocalize it. More diverse onboarding views and stronger correspondence verification are next experiments. Phone capture, object pickup, live latency/heat, lighting, RGB-only hand occlusion, reconstruction and skin/fabric remain unvalidated and out of this prototype's claims.

