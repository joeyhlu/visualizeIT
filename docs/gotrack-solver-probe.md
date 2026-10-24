# GoTrack same-input solver probe

This development diagnostic investigates the expanded head's pose outliers from
the [saved-output audit](gotrack-expanded-failure-diagnosis.md). It performs
inference with existing weights. It does not train or select a new checkpoint.

## Comparison

Six fixed, already inspected frames cover each large failure and its preceding
frame. Both heads receive identical recorded RGB, observed automatic SAM masks,
calibration and automatic estimated seeds. Seeds come from the adapted run's
accepted-pose memory. This is an adapted-trajectory seed replay with a frozen
same-input counterfactual, rather than another full causal tracking replay.

For each case and head, the driver compares ordinary refinement with capture
disabled and enabled. Pose, points, image correspondences and weights must agree.
The captured run uses the existing ordinary integer-coordinate refinement path.
Its observer records flow, confidence, retained correspondences, deterministic
subset indices, returned RANSAC inliers and the poses returned by RANSAC and LM
at each refinement iteration.
OpenCV's internal random draws and internal LM optimization steps are unavailable.
Correspondence reprojection residuals measure internal consistency; they do not
establish pose correctness. Centered 3D singular values describe point spread,
not calibrated pose uncertainty.

Reference poses and evaluation errors are excluded from the plan and inference.
The existing inference validator authenticates synthetic generation provenance
metadata, including identities and source hashes; it ignores label references
and never opens pose sidecars. These inspected development objects do not supply
independent test evidence.

## Cloud command

Use a separate source directory containing the reviewed probe runtime and the
small pinned plan package. Reuse the existing cloud input and model paths.

```bash
export VISUALIZEIT_CLOUD=1 PYOPENGL_PLATFORM=egl XFORMERS_DISABLED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SAM2_BUILD_CUDA=0
python -B -m bench.quality_gotrack_probe supervise \
  --plan .cache/cloud-prepare/expanded-same-input-probe-plan-v2.json \
  --plan-sha256 6b715983d67efba72d841b0ba3a1c4c6ee921db835fa6966e374b22ec0a2867b \
  --output NEW_OUTPUT_DIRECTORY --cloud-execution
```

The cloud process requires the existing pinned Linux environment. The initial
implementation uses CPU. It downloads no weights or data. `supervise` reuses the
existing offline process watchdog to enforce the 1,800-second child limit,
sample process memory, check storage and terminate a blocked child. The inner
`run` entry point has cooperative checks only. Watchdog cleanup can add time
after the limit. Ending the process does not prove that a cloud machine or its
billing has stopped.

The separate probe runtime authenticates the intentionally instrumented source.
Historical runtime files remain unchanged: use their preserved source directory
to reproduce old runs. They must reject changed source rather than silently
accept it. The ordinary pipeline's numerical thresholds and default head stay
unchanged.

The observer capability is a private admission mechanism for this pinned driver,
not a general authorization system for arbitrary inference callers.

Watch `probe.log` and individual `capture/row-*.json` files while the child runs.
The supervisor writes `supervision.json` when execution ends.
Successful completion requires matching candidate/input identities and an
authenticated child receipt. Missing, failed and unrun trajectories remain in
the 24-row inventory, with five iteration states each. Keep raw capture files in
cloud storage; download only the small summaries for review.

## Execution status

The source, command and pinned six-case plan are implemented. The combined
105-test suite passed without importing Torch, including 22 probe tests. Checks
cover manufactured observer parity, mutation and exception isolation, units and
inlier mappings, all 24 mocked trajectories, failed decoding/model loading and
cleanup, source drift, rejected annotation inputs and a bounded non-neural child
timeout. Process inspection is doubled in that local timeout test because the
science runtime lacks `psutil`; it does not establish Linux descendant cleanup.
The cloud environment must supply the existing `psutil==6.0.0` requirement.

## Actual development run: 2026-10-15

The fixed diagnostic completed on the existing Lightning Free 4-CPU studio.
Both heads ran sequentially in evaluation/inference mode, with seed 0,
deterministic execution and one Torch thread. There were no optimizer updates,
new weights or dataset downloads. The execution's 105 model-free tests passed
in 5.763 seconds without importing Torch; the separate earlier preflight also
passed all 105 tests. Source and input authentication passed after inference.

All 24 trajectories are accounted for: 22 returned candidates and two returned
`solver_unavailable`. The two unavailable trajectories are the frozen head's
capture-disabled and capture-enabled runs for TRAIN object 17, pitch +25 degrees,
frame 9. All 12 observer parity comparisons passed for pose, points, pixels,
weights and availability; all six cross-head initial input/seed comparisons
passed. The observer therefore preserved the measured outputs in this run.
This is finite evidence for this pinned CPU execution, not universal parity.

The captured half contains 60 planned iteration states: 55 refined, one with
insufficient retained correspondences and four unrun after that failure.
Its 167 private array artifacts remain in the cloud (228,730,552 raw bytes).
Only a 247,847-byte report ZIP was downloaded. All 40 ZIP members and the 39
manifest-listed payload hashes were authenticated locally and independently.

### Where the heads diverge

For all six first iterations, query crop, rendered template RGB, rendered depth,
calibration, crop transform and initial pose identities match exactly across
heads. Both predicted flow and confidence arrays differ before correspondence
selection and PnP. This locates an observable difference before the solver; it
does not isolate flow versus confidence as the cause of a pose outlier.

| Development case | First-iteration retained pairs: frozen | Adapted |
| --- | ---: | ---: |
| TRAIN object 17, pitch -25, frame 8 | 10,000 | 10,000 |
| TRAIN object 17, pitch -25, frame 9 | 593 | 595 |
| TRAIN object 17, pitch +25, frame 8 | 10,000 | 10,000 |
| TRAIN object 17, pitch +25, frame 9 | 16 | 29 |
| Previously used VAL object 27, pitch -25, frame 10 | 1,098 | 1,116 |
| Previously used VAL object 27, pitch -25, frame 11 | 3,757 | 3,754 |

The frozen failure has 16 retained pairs, below the ordinary minimum of 20.
The adapted head has 29 and proceeds. More retained pairs do not establish
correctness. Some support is nearly planar despite numerical rank 3; point
spread is not an uncertainty estimate. Across the 55 successful captured
iterations, LM did not increase the internal inlier reprojection RMS. A small
residual can still accompany an incorrect pose. Shared seeds come from the
adapted trajectory, and later crops may diverge between heads.

### Runtime and provenance

The supervised stage took 333.255 seconds with sampled peak process-tree RSS
3,018,383,360 bytes (about 2.81 GiB). The operator recorded 356.641 seconds before
final archive completion, including installation and the earlier preflight.
Provider startup and manual idle/collection time are outside that accounting.
These timings are a diagnostic workload, not a production FPS benchmark.

The sealed report records no remaining model workers and no reference-pose,
evaluation-sidecar or held-out test-object payload reads. The separate final
542-byte operator-completion file was visible in the cloud UI but was not
downloaded; its exact bytes and final elapsed time are not locally authenticated.
The downloaded report and capture completion receipt establish successful
inference and capture. The provider was then visibly verified **sleeping**, with
Machine `-` and 3.94 displayed credits. Exact credit consumption is unknown.

Local evidence is under `.cache/expanded-same-input-probe-v1/`:

- `report.zip`: SHA-256 `b216c72f2376cc810e98f1c80c25d42a0f9b48f31f26d158a5a46033c4630bf4`.
- `sealed/operator-report.json`: `202fad42fc0dea345826b07bd6421cf233ba58eb93e18e8b2bb14582a296f873`.
- `sealed/solver-boundary-summary.json`: `9f120ae448fce8a7a8350d8e9a6d863139b804ca910f375f7d3254e7b99fd821`.

The cloud source manifest is
`fea35312410e035a94f8e54fc4c61eba47ef71fe39030bc0a2606a365ce4d9fb`;
the plan hash is pinned in the command above. Parent and independent review
checked the numerical summaries and availability counts.

### Consequence for training

The old frozen head remains the default because the preceding full-video
development evaluation regressed. This diagnostic does not replace that
evaluation or establish improved accuracy, recovery, surface attachment,
generalization or phone readiness. No additional training was performed.

Before another training run, define a bounded experiment that tests the suspected
gap between iteration-0 training crops and later refinement states. A diagnostic
that independently exchanges flow and confidence would help separate their
effects. Neither a coverage gap nor these internal residuals prove the cause;
keep held-out test objects untouched and assess any new checkpoint on complete
causal replays before considering promotion.
