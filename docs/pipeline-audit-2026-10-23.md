# VisualizeIT pipeline findings and verification

Date: October 23, 2026 (Toronto).

Prioritize observation and supervision quality, then measure whether learned correspondences improve surface attachment. The phone camera preview is a separate experimental 2D translation path; it does not use the trained GoTrack head.

## Completed camera and studio connection

Use **Back to design studio** from the camera to return the generated artwork for editing, PNG download and saving after pairing. The one-shot transfer contains only a normalized recipe and its associated generated prompt. It excludes unsubmitted prompt drafts, camera images, masks, credentials and placement state. Returning does not reinterpret the prompt or save automatically.

Final checks: **23 browser transfer controls passed**, including exact artwork-pixel hashes across both directions, reset/discard consistency, invalid and oversized input rejection, storage denial and zero non-GET requests. The existing **27 manufactured live-camera checks passed**, as did **13 Node test units**, including twelve manufactured tracker controls. These checks do not establish physical-phone or real-camera model quality.

## Additional numerical findings

On an unoccluded analytic plane tilted 63.435 degrees, at 1.2 m depth and a 500-pixel focal length, nearest-pixel depth sampling with a 2 mm tolerance marked **1,558 of 4,141 visible points (37.6%) as not visible**. Flat and 45-degree controls had zero false-invisible points. Frequency in the real TRAIN cache remains unknown. Incorrect visibility can exclude flow supervision and, in recipes that train visibility, can create incorrect negative targets. The terminal-flow-only experiment freezes its confidence row.

For one continuous plane, inverse depth is affine in image coordinates. A separate interpolation check had maximum error 4.83e-8 m on that plane. This supports investigating geometry-aware sampling with an explicit unknown state; it does not validate interpolation across silhouettes, occluders, depth discontinuities or general curved surfaces.

One saved cracker-box observation-setup comparison improved the trained head median anchor error from **6.960 to 5.585 px720** and artwork frames from **54 to 109**. Both heads benefited. Opening prompt and segmentation seed changed together; this is one dataset-reference development case, not a general causal segmentation-only improvement. See [saved comparison](video-workbench.md#qualified-mask-cracker-replay).

## Source audit findings

### 1 Measure usable correspondences and attachment before scaling training

Visible mean flow EPE and confidence-vs-visibility counts do not measure solver-selected correspondence accuracy. Identical confidence is independent of changed flow errors. Earlier full-head adaptation lowered medians but produced translation outliers and reduced availability; this is a different checkpoint from the terminal-flow head.

Next check: Freeze inputs, masks, initial estimates and thresholds; log accepted support changes, accepted-visible residual tails, accepted-nonvisible counts, crop/native coordinate scale, normalized geometry conditioning, and downstream attachment plus availability.

### 2 Exploit the convex terminal regression

For fixed frozen features z, the two trainable 1x1 output rows give Wz+b. Bilinear interpolation is a fixed linear operator B, so flow=B(Wz+b) is affine in the 514 variables. Visible labels and normalization weights are fixed. SmoothL1 is convex; composing it with this affine map and summing with nonnegative weights preserves convexity. Confidence is frozen and visibility_weight=0. Therefore the terminal data-fit loss is convex. This says neither that it is strictly convex nor that AdamW reached an optimum; AdamW weight decay is not asserted to be an exact regularized minimizer.

Next check: TRAIN-only bounded convergence and feature-conditioning diagnostic versus pretrained initialization before adding capacity or extending epochs. Preserve ordinary export parity and evaluate closed-loop transfer separately.

### 3 Make loss recovery an explicit capability

Missing/rejected mask or solver failure clears the prior. The consumed single-use seed prevents later automatic restart. A one-frame dropout can therefore end the neural session.

Next check: Add explicit reselection/re-onboarding first; verify valid -> missing -> returning observations. Automatic reacquisition must use current observed evidence, never stale or reference poses.

### 4 Phone current-visibility and motion model

A 168-pixel occluder within the 8000-pixel object mask leaves 107 inliers in four quadrants and valid=true, while the reference mask remains opaque there. The renderer translates the old masked artwork. This demonstrates tracker acceptance under partial occlusion, not physical-phone output accuracy. Rotation/scale and evolving mask shape are not represented by translation.

Next check: Add current-visible-support/occlusion control; then separately evaluate similarity or planar homography tracking for planar objects. Neither model establishes curved-surface attachment.

### 5 Bound curved-chart distortion

The chart is tangent projection. On a cylinder u=R sin(theta), while surface arclength s=R theta. ds/du=sec(theta) for abs(theta)<pi/2: local physical pattern spacing can stretch 2x at 60 degrees. This is a structural example, not a measured distortion value for the current meshes.

Next check: Manufactured known-distance grid checks and a local distortion bound/patch-size gate before claims of faithful wrapping; consider cylindrical coordinates for known cylinders.

### 6 Independent evaluation and latency

VAL is 20 available views of one repeatedly inspected object. 167 TRAIN rows did slightly worse than 95 rows. Saved paired workbench timings are seconds per frame, with roles serial on a shared host; they are not phone FPS or causal head-speed comparisons. Independent attachment/recovery labels remain missing.

Next check: Use paired frozen baseline/candidate sequences, object/scene-level independent groups, equal-update matched controls and all failures in denominators; measure attachment tails, wrong-surface events, recovery and end-to-end device timing. Do not create frame-level pseudoreplicate confidence intervals or open sealed TEST for tuning.

## Evidence and next action

Follow the [detailed implementation plan](pipeline-improvement-plan-2026-10-23.md). Local machine-readable evidence is retained under `.cache`: `audit-20261023-summary.json`, `audit-20261023-label-plane.json`, `audit-20261023-depth-mechanism.json`, `audit-20261023-partial-occlusion.json`, and `studio-return-20261023-final-validation.json`.

Existing YCB dataset-reference diagnostics provide useful pose/anchor measurements but do not replace independent human attachment, wrong-surface or recovery labels. Multi-point footprint/orientation checks and physical-device gates remain necessary.

The audit used no training or neural inference and opened no sealed TEST payloads. Finishing browser tests substituted camera frames and segmentation masks. Existing research and graph schedules remain paused. No commit or public deployment was made.
