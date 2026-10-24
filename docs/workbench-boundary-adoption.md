# Main-pipeline adoption of the portable frame boundary

Ordinary runs (`bench/workbench_cli.py` `Pipeline.execute`, and so the service's `execute_run`) now
execute every **cold** primary-neural, frozen-baseline and relative-geometry frame through the
accepted portable boundary. The boundary is `bench/workbench_frame_boundary.py` `be5dbf04…`;
the opt-in replay tool `tools/workbench_frame_replay.py` `4536eb69…` is unchanged and is **not**
called by the pipeline.

**Not changed:**
- the stage-cache format, the contract schemas, renderer math, the numerical algorithms, the
  service, the installer and upstream code;
- input re-verification, reconstruction and its cache, backend/model construction, and
  `workbench_telemetry.gpu_stage` ownership (one `ExitStack` per neural role);
- causal warm-up then scored order, cancellation, atomic neural overlay writes (`write_png_rgba`),
  the artifact-failure policy and final complete accounting (`_finish`).

There is no model-quality, phone or continuous-camera claim.

## Execution path

| Role | Cold path |
| --- | --- |
| Primary / baseline (`_neural`) | Bundle, seed plan and artwork plan (unchanged), then `neural_binding` and `FrameContextV1.build`. The boundary seed plan must equal the pipeline plan, otherwise `boundary_context_unavailable`. Then the cache lookup; on a miss, `gpu_stage` → backend → `NeuralSession` → **one** `FrameBoundary` with `NeuralSessionAdapter`, then the warm-up through `boundary.process`, then the scored frames through `boundary.process`. |
| Relative (`_relative` → `_track`) | Reconstruction or its cache hit (unchanged) produces `package_dir`. Then `_track(reader, masks, package_dir, identity, cache)` builds `RelativePackageProof.from_package_dir(package_dir)` (authenticated bytes, loader checks repeated), then `relative_binding`, then the context, then the cache lookup; on a miss, **one** `FrameBoundary` with `RelativeTrackerAdapter(RelativeTracker(), proof)`, then the scored frames through `boundary.process`. |

**Direct calls removed.** No direct `NeuralSession.process` or `RelativeTracker.process` calls
remain in the CLI; only the boundary reaches them.

**Frame input.** `Pipeline._boundary_input` builds each `FrameInputV1` from `FrameReader.read`
and `MaskStore.load`.
- A decoder or read failure becomes an explicit RGB-unavailable packet: counted, and
  tracking is invalidated until an explicit restart.
- A mask load failure takes the safe missing-mask path.
- The backend is never reset or re-armed implicitly.

**Overlays.** Only the boundary's owned, current-frame `overlay_rgba` is written, through one shared
policy, `Pipeline._write_overlay`:
- **Writers:** neural overlays use the atomic `write_png_rgba`. Relative overlays keep their
  existing cv2 BGRA encoder (`_write_relative_png`). It refuses an existing file and removes only
  a partial file it created itself.
- **`write_overlays=False`:** nothing is written and no overlay path is recorded. Rendered rows
  stay `rendered` (the RAM render happened) and carry
  `overlay_artifact: "not_written: write_overlays disabled"`.
- **Write errors:** any mkdir, convert, encode, write or collision error keeps the CURRENT computed
  row, its pose and render state, and records `artifact_write_error` with no overlay path.
  Accounting continues, stage `<role>_artifacts` / `relative_tracking_artifacts` fails, and **no
  cache commit** is made.

## Lifecycle and cleanup

**Neural:**
- If the `NeuralSession` constructor fails, the backend is closed exactly as before.
- Once `boundary.initialize` is called, the boundary **owns the session and its backend**,
  including an injected `RunOptions.neural_backends` backend. Any initialization refusal closes
  them, records stage `<role>_boundary: failed` with the cleanup errors, and gives `_neural_block`
  rows for every scheduled frame.

**Relative:** the tracker is closed on every early return (context refusal, input re-verification
failure, cache hit, invalid cache entry). Initialization failure closes it through the boundary.

**Loops:** rows already produced are kept. `boundary.close()` runs in `finally` on every path:
cancel, decoder failure, artifact failure, adapter failure. Its cleanup errors are recorded as
stage `<role>_cleanup` / `relative_tracking_cleanup: failed` and **block the cache commit**.

**GPU stage (neural):** the `ExitStack` holding `gpu_stage(role)` is released **exactly once**, in
`finally`, **before** any commit is queued. A release failure:
- keeps every computed row;
- records stage `<role>_gpu_release: failed`, so the run fails;
- forbids the commit.

If the body itself raised, the original error is preserved; the release failure is recorded as
well but never replaces it.

**Deferred commits:** tracking and reconstruction cache entries are only queued
(`_defer_cache_store`) while roles run. `Pipeline.execute` closes the decoder first; a decoder
close failure is no longer swallowed. Then `_settle_pending_commits`:
- **Clean close:** the queued entries are committed.
- **Failed close:** records `reader_cleanup: failed` and `cache_commits: skipped`, and commits
  nothing. Rows are kept and the run fails.

Historical entries are never overwritten. There is no automatic neural retry or fallback.

**Single final report:** every path that created a decoder finishes only after it is closed and
pending commits are settled. A source-verification failure no longer returns from inside the
`try`. It records `source: failed`, runs no role, falls through to the common cleanup (decoder
close, `reader_cleanup` if that fails, then settle with no commit) and then calls the single
`_finish(blocked_reason="source_unverified: …")`. The returned and saved reports are identical
and contain both the source error and any decoder close error, with every planned row blocked
and no model constructed. Only the budget refusal still returns early, before any decoder exists.

**Exceptions:** a `BoundaryFailure` or other exception inside a loop propagates to `_run_role`.
`_finish` then fills every remaining scheduled frame with `role_failed: …`.

## Cache identity and provenance (tracking stages)

**Module constants** (`bench/workbench_cli.py`):
- `ADOPTED_EXECUTION_PATH = "workbench_frame_boundary_v1_main_pipeline"`;
- `BOUNDARY_SOURCE = "bench/workbench_frame_boundary.py"`.

**Helpers:**
- `boundary_context_digest(context)`: the canonical SHA-256 of the context, independent of the
  run ID. It covers schema, `clip_sha256` (frames, timestamps, media times, K, dimensions),
  role, source mode, warm-up, scheduled and permitted order, K, width, height, `binding_sha256`,
  the prefix attestation, `run.primary`, artwork kind and onboarding.
- `boundary_provenance(context)` returns `{execution_path, boundary_version, boundary_source_sha256, context_digest, binding_sha256}`.

**Neural keys (`<role>` stage).** All existing inputs and versions are kept:
- media and frame file hashes, masks and the mask manifest, clip, onboarding, intrinsics and
  dimensions, warm-up IDs;
- the metric bundle file SHA, the artwork record SHA or reason;
- the verified head or base bytes, the base pin, the admission body;
- the configuration `device` **and `write_overlays`**, so entries with and without overlays never
  share an entry;
- the module hashes, now **plus** `workbench_frame_boundary`.

Added:
- `inputs.boundary = {context_digest, binding_sha256}`. The binding carries the complete bundle
  record: SE(3) seed and its hash, object ID, GLB path and SHA, label, origin, provenance, seed
  target and warm-up.
- `versions.execution = boundary_provenance(context)`.

**Relative keys (`relative_tracking` stage).** All existing inputs and configuration are kept
(`tracker` cfg, artwork, `write_overlays`), plus:
- `inputs.package_sha256` = the `surface.json` bytes SHA;
- `inputs.boundary`, whose binding includes per-component file SHA, bytes, semantic array SHA,
  dtype and shape, the prefix, cameras, freeze frame and artwork SHAs;
- `versions` = `workbench_geometry`, `workbench_cli`, `workbench_contract`,
  `workbench_frame_boundary` and `workbench_reconstruction` hashes, plus `execution`.

The reconstruction key is unchanged in structure. It already includes the `workbench_cli` hash, so
this CLI change conservatively invalidates it.

**Payload.** `"execution": {**provenance, "computed_by": "fresh_boundary_execution"}`.

**Row evidence:**
- fresh rows: `execution_provenance: "fresh_boundary_execution"`;
- replayed rows: `execution_provenance: "cache_replay_of_boundary_computed_rows"` plus
  `computed_execution_path`.

**Stage records:** `execution` carries the provenance with `computed_by` (fresh) or
`replayed` (cache).

**Hit rules.** `_adopted_lookup(store, key, role, provenance, *, context=None)` validates the
**whole entry before any materialization or append**:
- **Provenance:** the payload `execution` equals the current provenance exactly.
- **Row coverage:** the rows are exactly the scheduled frames of this role.
- **Per row:**
  - it passes `FrameResult` contract validation;
  - its clip ID, source index and timestamp equal the manifest (wrong clock is refused);
  - its media time equals the manifest. Neural rows may carry no other media time; relative rows
    may carry the decoder time when the manifest has none;
  - its pose units are `metric` for neural or `object_unit` for relative;
  - an observed mask belongs to this frame;
  - `execution_provenance` is `fresh_boundary_execution`, so legacy or unlabelled rows are never
    promoted;
  - it has no `artifact_write_error`.
- **Overlays:**
  - an overlay path must be exactly `evidence/<role>/<frame>.png`; a basename alone is not enough,
    and cross-role paths are refused;
  - it is allowed only with `write_overlays` on and a `rendered` row;
  - with `write_overlays` on, every rendered row must have one;
  - the entry's file set must equal exactly the referenced overlays, each with `sha256` and size
    metadata.
- **Backend identity:** one constant identity across rows, with:
  - the base checkpoint pin and `quality_claim: false`;
  - the current admission `contract_sha256`;
  - **primary:** installed and export head equal to the pinned 27e06;
  - **baseline:** no installed head and `frozen_base_pose_head`.
- **With `context` (required for any hit):** production callers in `_neural` and `_track` pass the
  validated `context=context`. `_check_lookup_context` refuses a missing context (an otherwise
  valid entry is still refused with `the validated boundary context is required…`) or a context
  of another role, clip or schedule.
  - **neural with a bound bundle:** every field in `NEURAL_BUNDLE_IDENTITY_FIELDS` is
    **required** (`principal_point_origin`, `initialization`, `seed_frame_id`,
    `seed_target_frame_id`, `glb_sha256`, `metric_units`, `effective_K`); deleting one fails
    exactly like changing it. The values must equal the bound bundle, the seed plan and `metres`.
    `effective_K` must be 9 finite numbers equal to `native_K(context.K, origin)`.
  - **relative:** the identity must equal exactly `{role, tracker, quality_claim, package_id,
    package_sha256}` from the bound proof.

  Consistency across rows alone is never enough; an entry whose every row carries the same wrong
  GLB, seed target, K or package SHA is refused.

Per-frame image and mask pixel hashes are not re-derived at lookup. The key already binds the
current frame files, masks and manifest; the per-row check is on identity form only.

**Replay (`_replay_adopted_hit`):** materializes the entry (each file re-hashed after copy, and
the materialized set must equal the entry). It then appends rows with the current run ID,
`execution_provenance: "cache_replay_of_boundary_computed_rows"`, `computed_execution_path` and
`cached_source_run_id` (the legitimate original run).

**Replay without construction:** a genuine hit is materialized and replayed **without
constructing a model, entering a GPU stage or building a boundary**.

**Fail closed:** a lookup integrity error or ANY validation or materialization failure marks stage
`<role>` / `relative_tracking` **failed** with `cache_entry_invalid: …` and gives blocked rows for
every scheduled frame. There is no recompute and no legacy fallback.

**Earlier direct-session entries never qualify.** Their keys lack `inputs.boundary`,
`versions.execution` and the boundary module hash, and their payloads lack `execution`. They are
neither overwritten nor deleted; existing cache entries, historical results and failures are kept.

**What changes invalidate an entry.** Changing the boundary source or version, the CLI, the
context (calibration, time, order, masks, seed, bundle, surface, artwork), the head or base
bytes, the admission body or any source module changes the key, giving a new cold computation.
Changing only the run ID does not.

## Serving admission dependency (intentional invalidation)

`bench/workbench_neural.py` `NEW_SOURCES` now includes `bench/workbench_frame_boundary.py`, because
the CLI executes every neural frame through it. Together with the changed `workbench_cli.py`
hash, this changes the serving source closure. The existing serving admission (07) **must fail
closed for fresh actual runs** until the parent earns a fresh full-image gate on the final
source closure. No hashes, contracts, admission receipts or gates were edited or relaxed. The
earlier full six-case comparison remains historical evidence of the previous closure.

The main head remains `27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311`.

## Telemetry

**Neural:**
- `tracking_ms` = the measured `boundary.process` wall time (with GPU synchronisation, and
  including boundary validation) minus the measured render time;
- `rendering_ms` = the measured artwork render;
- new `boundary_process_ms` = the boundary's own measured call time;
- `end_to_end_latency_ms` includes the read and mask load.

**Relative:** the adapter tracks and renders inside one boundary call, so `tracking_ms` and
`rendering_ms` are `null` (unmeasured, never divided heuristically).
`boundary_process_ms` and `end_to_end_latency_ms` are measured.

**Cold versus cached:** cache-hit runs reuse the original timings under
`cached_original_timings`. Phone performance remains unmeasured.

## Public API and test impact

**Signature change:**
- `Pipeline._track(reader, masks, pkg, arrays, K, identity, cache)` is now
  `Pipeline._track(reader, masks, package_dir, identity, cache)`.

**New members:**
- `Pipeline._boundary_input`, `_adopted_lookup`, `_replay_adopted_hit`, `_fail_closed`;
- module-level `ADOPTED_EXECUTION_PATH`, `BOUNDARY_SOURCE`, `boundary_context_digest` and
  `boundary_provenance`.

**Repair 1 API changes:**
- `_adopted_lookup(store, key, role, provenance, *, context=None)`: the new keyword-only `context`.
  Positional callers keep working; with `context=None` the pin, row, unit, clock and overlay checks
  still apply, and only the bundle, seed, origin, K and package checks need it. The pipeline
  passes it.
- New members:
  - `_validate_adopted_entry(record, role, provenance, context)`;
  - `_expected_identity_checks(role, identities, context)`;
  - `_write_overlay(role, fid, overlay_rgba, evidence, artifact_errors, files)`;
  - static `_write_relative_png(path, rgba)`;
  - `_defer_cache_store(cache, key, payload, files=None, *, stage)`;
  - `_settle_pending_commits(cleanup_failure)`.
- `_cache_store` is unchanged and is now called only from `_settle_pending_commits`.
- **Repair 2:**
  - `_adopted_lookup(..., *, context=None)` now **requires** the context for a hit; a
    positional call without it only reports the basic reasons and never returns a record.
  - New `_check_lookup_context(role, context)`.
  - New class attribute `Pipeline.NEURAL_BUNDLE_IDENTITY_FIELDS`.
  - `execute` returns through one final `_finish` after cleanup.
  - **Test impact:** neural backends used with a bound bundle must emit the full metric identity
    extras (the parent's `make_backend` does). Positional `_adopted_lookup` callers can no longer
    obtain a hit.
- Replayed rows gain `cached_source_run_id`. Rendered rows written with `write_overlays=False` gain
  `overlay_artifact`.
- New stage names: `<role>_gpu_release`, `reader_cleanup` and `cache_commits` (`skipped`).

**Behaviour changes that affect tests:**
- **Primary identity required.** A neural role in a run whose `RunSpec.primary` is not the pinned
  primary is now `boundary_context_unavailable` (`build_run_spec` already sets the pinned primary).
- **Real bundle required.** A metric-bundle plan must hold a real `MetricBundle` with a
  `controlled_init:`/`onboarding:` seed label.
- **Reader identity.** Reader identities must carry `source_frame_index`.
- **Deferred commits.** Pipelines driven through `_neural` or `_track` directly (not `execute`) only
  queue commits; nothing is stored until `_settle_pending_commits` runs.
- **`tools/test_workbench_image_io.py`.** The parent has updated
  `test_neural_overlay_failure_preserves_all_computed_rows_and_blocks_cache` to a real
  `MetricBundle` and complete reader identities, and it passes.

## Worker development checks (not acceptance)

**Repair 2:**
- **Wiring fix:** a repair-1 rewrite that was meant to pass `context` at both production call
  sites had not actually landed in the source. It is now wired at both.
- **Suites:** `bench.test_workbench_boundary_adoption` (your 9 tests, including the
  ordinary-run cache poisoning of GLB, seed target or K and deleted GLB or K, the combined
  source-verification plus decoder-close report, and the constant wrong relative package SHA)
  **all pass**. With 15 further suites (`cli`, `serving_lifecycle`, `source_binding`, `neural`,
  `frame_boundary`, `frame_replay`, `image_io`, `cache_regression`, `cache`, `telemetry`, `service`,
  `clip_controls`, `reconstruction`, `geometry`, `contract`), **205 tests pass** (2 skipped).
- **Genuine repeat:** the unchanged repeat still replays with zero estimator forwards and no model,
  GPU-stage or boundary construction.

**Repair 1:**
- **Suites:** `bench.test_workbench_boundary_adoption` (your 6 tests: cold, then repeat with zero
  model, GPU or boundary work; no-overlay honoured and not sharing the overlay-enabled entry; GPU
  release failure; the cached-row checks; the relative artifact exception) plus 13 existing suites
  (`cli`, `serving_lifecycle`, `source_binding`, `neural`, `frame_boundary`, `frame_replay`,
  `image_io`, `cache_regression`, `cache`, `telemetry`, `service`, `clip_controls`,
  `reconstruction`): **180 tests pass** (2 skipped).
- **Development probes** (your fixture with the test-double estimator):
  - **decoder close failure at the end:** `failed`; rows kept (2 tracking);
    `reader_cleanup: failed` and `cache_commits: skipped`; `_cache_store` was called 0 times.
  - **body error plus GPU release failure:** the role is failed with the original
    `RuntimeError` reason; `primary_neural_gpu_release: failed`; 2 accounted rows; 0 commits;
    the backend closed once.
  - **context-dependent lookup:** an unmodified entry is valid. Wrong `effective_K`, a moved
    seed target, a different admission contract, a legacy row without provenance and an extra
    cached file are all refused. The test-double backend exposes no `effective_K` or seed fields,
    so those checks apply only when the real backend supplies them.

**Initial delivery (earlier source, kept for history):**
**Existing suites** (CPU):
- `bench.test_workbench_cli`, `serving_lifecycle`, `source_binding`, `neural`,
  `frame_boundary`, `tools.test_workbench_frame_replay`, `test_workbench_image_io`,
  `test_workbench_cache_regression`: 101 of 102 pass. The one failure is the image-I/O test
  described above.
- `bench.test_workbench_cache`, `telemetry`, `service`, `clip_controls`, `reconstruction`:
  all 72 pass (2 skipped).

**Manufactured end-to-end check through `Pipeline.execute`.** It used:
- a manufactured image sequence with real `FrameReader`, `MaskStore` and the real stage cache;
- a real `MetricBundle` object (GLB never read);
- a counting numerical test double in place of `GoTrackMetricBackend`, with `gpu_stage` counted;
- a manufactured authenticated plane package returned in place of COLMAP `reconstruct`.

Results:
- **Cold:** all three roles completed through the boundary: 3 boundary initializations, 2 backend
  constructions, 6 neural forwards (1 warm-up + 2 scored per role), 2 GPU-stage entries. Relative
  gave f2 `recovering`, then f3 `tracking` + `rendered`. Every row carried
  `fresh_boundary_execution` with path `workbench_frame_boundary_v1_main_pipeline`.
- **Unchanged repeat:** **0** constructions, **0** forwards, **0** GPU-stage entries and **0**
  boundary initializations. All rows were `cache_replay_of_boundary_computed_rows`.
- **Corrupted primary entry:** the primary stage failed closed (`cache_entry_invalid: …
  CacheIntegrityError`) with no recompute or construction; the other roles were replayed.

## Still unverified

- No real `GoTrackMetricBackend`, weights, GPU, COLMAP reconstruction or real video was
  exercised by the worker.
- A fresh full-image serving gate on the new source closure has not been earned. Until it is,
  fresh actual neural runs are expected to be blocked by admission.
- The parent's acceptance tests have not run: cache-key mutation matrix, old-path-entry miss,
  cancel/decoder/init/artifact cleanup, seed pristine, mask freshness, units and current
  overlays.
- The service inherits this path through `execute_run` but was not separately exercised.
- Model quality and phone performance are not measured.
