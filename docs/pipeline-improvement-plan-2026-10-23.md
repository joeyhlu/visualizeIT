# VisualizeIT improvement plan

Date: October 23, 2026 (Toronto).

Improve observation and label quality first, then use paired downstream evidence to select training and tracking changes. Camera-to-studio artwork return is implemented and browser-verified. The following research stages are proposed work, not a resumed execution queue.

## Implementation order

| Stage | Work | Depends on |
| --- | --- | --- |
| P0 | Freeze current baseline and decision metrics | Baseline inspection |
| P1A | Selection quality and current visibility | P0 |
| P1B | Audit supervision before new fitting | P0 |
| P2 | Run an attribution experiment before choosing next model | P1A, P1B |
| P3 | Convex terminal-fit diagnostic, then targeted training | P1B, P2 |
| P4 | Explicit recovery and orientation identity | P0 |
| P5 | Surface-aware artwork mapping | P2, P4 |
| P6 | Device performance and rollout | P1A, P2, P4, P5 |

## P0 Freeze current baseline and decision metrics

Scope: `docs/video-workbench.md`, `bench/workbench_evaluate.py`, `tools/workbench_comparison.py`.

Deliverables:

- One source/checkpoint/mask/seed/split evidence index.
- Baseline scoreboard including dataset-reference anchor metrics and independent-label gaps.
- Object/source-grouped development and untouched final evaluation manifests.

Acceptance: Every metric has units, eligible denominator, uncertainty scope and evidence category; no historical report overwritten.


## P1A Selection quality and current visibility

Scope: `bench/web/design-studio/phone-tracker.js`, `bench/web/design-studio/phone-camera.js`, `tools/test_phone_tracker.js`, `tools/check_live_camera_browser.js`.

Deliverables:

- Partial-occlusion control preserving distributed feature support.
- Tap/negative-point correction and observed-mask quality diagnostics.
- Current visibility handling, explicit uncertainty and reselect states.

Acceptance: Opaque artwork does not cover controlled new occluders; no stale mask/pose presented as current; full occlusion/loss/reentry accounted for. Rotation/scale support separately validated.


## P1B Audit supervision before new fitting

Scope: `bench/quality_adaptation_prepare.py`, `bench/quality_adaptation_data.py`, `bench/test_quality_adaptation_contract.py`.

Deliverables:

- Manufactured plane/occluder/silhouette/depth-hole controls.
- TRAIN-only depth-gradient/scale/edge label audit.
- Versioned positive/negative/unknown visibility policy if prevalence warrants correction.

Acceptance: True-visible sloping-plane controls pass, actual occluders remain rejected, discontinuities become unknown when unresolved; all label changes have new manifests and no test payload reads.


## P2 Run an attribution experiment before choosing next model

Scope: `bench/workbench_evaluate.py`, `tools/workbench_ycb_dataset_evaluate.py`, `tools/workbench_mask_replay_evaluate.py`.

Deliverables:

- Paired baseline/adapted heads crossed with original/qualified observation policies on development clips.
- Per-frame solver-selected support, residual tails, spatial conditioning, pose and multi-point footprint errors.
- Object/source-level summaries retaining unavailable and failed frames.

Acceptance: Predeclared attachment and availability checks; do not infer whole-suite or independent-phone gains from one dataset-reference case; unchanged artifacts reused where valid.


## P3 Convex terminal-fit diagnostic, then targeted training

Scope: `bench/quality_adaptation_flow_terminal.py`, `bench/quality_adaptation_learning_curve.py`, `bench/quality_adaptation_refinement_states.py`.

Deliverables:

- TRAIN-only optimization/feature-conditioning diagnostic of the affine 514-parameter terminal head.
- Bounded convergence certificate for an explicitly defined constrained/regularized diagnostic objective.
- Equal-update iteration-zero versus later-state control followed by closed-loop replay.
- Parity-verified ordinary export, frozen confidence preservation where required.

Acceptance: Lower training loss alone never promotes; retain candidate only if downstream errors/tails and failure-aware availability meet predeclared guards. Increase capacity only if measured representation limitation remains.


## P4 Explicit recovery and orientation identity

Scope: `bench/workbench_neural.py`, `bench/test_workbench_serving_lifecycle.py`, `bench/workbench_contract.py`.

Deliverables:

- Explicit tracking/uncertain/lost/reselect or reacquiring states.
- Separate accepted pose from provisional candidates.
- Fresh observed evidence for reseeding and asymmetric footprint/orientation checks.

Acceptance: One-frame dropout and reentry sequences behave explicitly; stable wrong orientation is not accepted solely because two candidates agree; historical seed/reference pose never silently reused.


## P5 Surface-aware artwork mapping

Scope: `bench/workbench_geometry.py`, `bench/test_workbench_geometry.py`, `bench/test_workbench_geometry_adversarial.py`.

Deliverables:

- Planar calibrated/homography path.
- Cylindrical angle-height chart for verified cylinders.
- Distortion-bounded local mesh charts and footprint checks for other surfaces.

Acceptance: Known-spacing grid respects declared stretch budget and orientation; no overlap/fold; suppress unsupported regions. Arbitrary curved global isometry is not promised.


## P6 Device performance and rollout

Scope: `bench/web/design-studio/phone-camera.js`, `tools/phone_runtime.py`, `docs/phone-camera.md`, `docs/validation.md`.

Deliverables:

- Actual phone segmentation startup, frame age, processing, rendering, memory and thermal traces.
- Evaluate current approximately15Hz loop and fast-tracking/keyframe-segmentation scheduling.
- Trusted HTTPS, iOS/Android checks and ten-minute sustained device sessions.
- Versioned release/admission and roll-backable feature gates.

Acceptance: Meet selected product frame-rate and attachment goals on real devices; existing30FPS target remains unmet until measured. Preserve no-camera-upload policy.

## First bounded implementation batch

1. Register the label and partial-occlusion controls and freeze the current evidence index.
2. Audit TRAIN label ambiguity by depth gradient, scale and silhouette distance; preserve original labels and version any revised policy.
3. Add current-visibility controls and correction behavior to the phone preview after exclusive file handoff.
4. Freeze the paired original/qualified observation and frozen/adapted head comparison. Schedule new inference only when research execution resumes.

## Quality gates

- Provisional attachment targets remain median below 5 px and P95 below 10 px at 720-pixel image height, with at least 90% pose availability. Report all failures and each sequence separately.
- Use directional landmarks and multiple footprint points. One accurate anchor does not establish correct orientation or pattern spacing.
- Group splits and uncertainty by independent object/source; do not treat neighboring frames as independent replications. Previously inspected validation remains development data.
- Compare equal-update training controls, then evaluate each candidate along its own sequential trajectory.
- A lower EPE, preserved confidence, passing unit suite or single improved clip is insufficient for promotion.

## Device and resource boundaries

- Preserve no-camera-upload behavior and measure actual iOS/Android startup, frame age, attachment, thermal behavior and sustained performance.
- Keep the existing 1 GiB free-space reserve plus output allowance, 8 GiB model-cache and 256 MiB public-artifact budgets.
- Preserve checkpoints, original/failed trials, immutable source receipts and train/evaluator separation.
- One neural stage runs at a time when authorized. This plan does not resume the research/graph schedule, Docker or Unity/mobile setup.

See [findings and verification](pipeline-audit-2026-10-23.md) for the supporting evidence.
