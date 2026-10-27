# Experimental within-frame input preparation reuse

P1-A is one opt-in candidate for the offline 15 processed FPS investigation.
It removes repeated full-frame RGB normalization and mask tensor preparation
within one existing `GoTrackMetricBackend.estimate` call. It is not a general
cache and does not establish neural parity or a measured speedup.

The existing loop passes the same RGB, observed mask and calibration through
five refiner iterations. Only the full-frame input preparation is invariant.
Each iteration still recomputes its pose-dependent crop, rendered template,
network forward, correspondences and PnP solution with its original RNG seed.
Five iterations remain the default. Current masks, pose transforms, quality
thresholds and public tracking/recovery decisions are unchanged.

## Flag and rollback

The backend constructor has `preparation_reuse=False`. Set it to `True` only in
an explicitly admitted experimental configuration. Enabled mode requires the
actual `WorkbenchIntegerRefiner` protocol; incompatible custom refiners refuse
rather than silently selecting another tracker. Disabled mode follows the
original preparation path and retains its existing identity/limit fields.

Enabled backend identity extras and limits identify
`experimental_estimate_local_immutable_bytes_v1`. The benchmark driver must also
pin the policy in its run configuration, use matching current-source admission
and prevent reuse of disabled-arm cached results. This source change does not
modify old receipts, automatically admit either arm, or enable the flag in the
normal command-line workflow. Roll back by disabling the flag.

## Ownership and invalidation

The preparation scope exists only around one estimate's refinement loop and is
cleared on every exit, including a solver loss or exception. The next frame
starts empty even if all its bytes happen to match the preceding frame. A new
estimate/re-onboarding invocation cannot inherit the previous preparation.

Within the scope, the certificate binds actual RGB/mask content, dimensions and
dtypes, native calibration, surface identity, network/session and model/head
provenance, device, role, policy/configuration, state identity and available
declared epoch. Changed inputs or context cause recomputation. Epoch metadata
is a declaration; missing epochs are not fabricated. The estimate lifetime
provides the hard frame boundary.

The retained representation contains immutable bytes plus dtype, shape, strides
and a checksum. It retains no live tensor or NumPy array. Preparation uses the
original Torch operations; it does not substitute a potentially different NumPy
normalization. Every iteration receives a new writable backing buffer with the
same values, layout and CPU float32 dtype. Upstream mutation therefore cannot
poison a subsequent iteration's retained input. Primary and baseline backends
keep their own frame-local state.

Restoration checks layout, byte bounds and integrity. Corrupt or incompatible
prepared data refuses safely, with scope cleanup. This does not authorize
fallback to older weights, stale masks or predicted poses. Original decoded
inputs and saved results are not rewritten.

Hashing, serialization and fresh-buffer copies have a cost. They can erase the
benefit of avoiding repeated conversion. No performance gain follows from fewer
preparation calls in a software test.

## Software checks and promotion

Model-free checks execute the actual parent backend/refiner with deterministic
injected components. They compare enabled and disabled outputs, tensor values,
layouts, calls, RNG seeds, exceptions and state transitions; verify five forwards
and pose-dependent work; and check scope cleanup, invalidation and immutable
buffer isolation. They do not execute the neural model.

The independent module also loads the exact accepted parent neural source from
Git commit `1c4248c964ee0d10116d3628ee6059f76caa00f5` and verifies its file
SHA-256 before execution with injected components. Both candidate arms are
compared against that implementation, including two-frame tracking, solver loss
and a network exception. The parent Git object must be available; a missing or
different parent fails this control rather than silently skipping it.

```text
python -B -m unittest tools.test_workbench_preparation_reuse
python -B -m unittest tools.test_workbench_preparation_reuse_independent
python -B -m unittest bench.test_workbench_refiner_timing bench.test_workbench_refiner_timing_independent bench.test_workbench_neural
python -B -m unittest tools.test_workbench_offline_export_integration
```

Before real comparisons, retain the proposed, unvalidated promotion gates:
at least 10% preparation-stage improvement and 5% genuine whole-flow improvement,
with quality gates satisfied; no predeclared slow-object throughput regression
above 2%, and P95 latency regression no greater than 15%. Preserve failures and
disable/park the candidate if these gates fail. Do not multiply hypothetical
component gains.

The P0 acceptance workload is 300 consecutive authenticated scored frames with
current-frame SAM2, trained primary GoTrack, supported real artwork and drained
encoding in at most 20 seconds after legitimate initialization. Scheduled and
genuine-work throughput must both reach 15 FPS; every outcome stays visible.
Suppressed, predicted or recovering frames are not successful measured overlays.
Cold whole-job/startup/preparation costs are mandatory separate reporting. No
artificial warmup or dropped scored frames can manufacture success. Missing 300
eligible frames or authorized initialization means the P0 is not evaluated.

Heavy comparisons remain blocked until safe executor ownership, complete exact
source/model/runtime and asset admission, verified current hardware and resource
guards are established. Software parity is not actual neural parity, independent
attachment quality or a 15 FPS result.
