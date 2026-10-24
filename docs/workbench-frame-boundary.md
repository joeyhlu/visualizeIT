# Portable frame boundary and recorded replay (opt-in)

- `bench/workbench_frame_boundary.py`: the boundary (`BOUNDARY_VERSION = "2"`).
- `tools/workbench_frame_replay.py`: the recorded-replay library and a safe `inspect` CLI.

One shared lifecycle for a finite recorded video today and, later, a reviewed camera source:

`initialize(context, adapter)` → `process(frame_input)` (repeated) → `reset(reason="replay_restart")` → `close()`

**Scope:**
- **Authoritative records:** the existing `ClipSpec`, `RunSpec`, `FrameSpec`, `SurfacePackage` and
  `FrameResult`.
- **Real classes do the work:** `NeuralSession`, and `RelativeTracker` with `render_marker`. No
  metric math or readiness signal is added.
- **Additive only.** `bench/workbench_cli.py` and its admitted, cached stages are unchanged and do
  not call this boundary; full adoption is a separately reviewed source, admission and cache
  migration.
- **No claims.** This path is **not cached** and makes no phone, continuous-camera, accuracy or
  performance claim.
- **No live neural cameras.** Live or camera-style neural sources are **unsupported**.

## What is verified, and how far

| Area | Verified | Not verified / limits |
| --- | --- | --- |
| **Context** | `initialize` re-derives a private canonical context from the validated clip, run and binding (`canonical_context`). It refuses any `FrameContextV1` whose schema, K, dimensions, mode, warm-up, scheduled or permitted order, hashes or prefix attestation differ, whether built with the public constructor or `dataclasses.replace`. It fingerprints the caller's context object and raises `ContextMutated` (closing the adapter) if that object, including nested clip/run dictionaries, changes before a later `process` or `reset`. | Python frozen dataclasses are not deeply immutable; this is detection, not prevention. |
| **Run primary** | Any run that requests a neural role (a baseline comparator included) must have `RunSpec.primary` equal to the pinned `workbench_neural.PRIMARY` artifact ID, export SHA and bytes. | Relative-only runs do not use `primary` and it is not checked for them. |
| **Neural binding** | `neural_binding` stores a complete `bundle_record`: family, GLB path and SHA, object ID, origin, seed label, seed frame, the **validated SE(3) seed matrix and its canonical hash**, and metric provenance. It also stores the seed target and the warm-up (derived as in the legacy pipeline). **Seed labels:** whenever a seed pose exists, its label must start with `workbench_neural.SEED_PREFIXES` (`controlled_init:` or `onboarding:`) followed by non-empty text. An `evaluator:` label, any other label, an unlabelled seed or a bare prefix is refused in `bundle_record`, in binding validation and in the adapter's open and drift snapshots, so it never reaches the estimator. This is slightly stricter than `MetricBundle.from_file`, which accepts a bare prefix. **Pristine sessions:** `NeuralSessionAdapter.open` refuses a session with a non-empty `_state`, a consumed `_last_index`, a retained `last_overlay`, or a backend `_seed_used` that is not `False`. It never resets or re-arms implicitly; pass a new session. An explicit `reset("replay_restart")` after initialization remains supported. At open, and before and after every `process` and `reset`, the adapter compares a full snapshot: session class, role, run, clip hash, renderer bound, backend class, install-provenance hash, the complete bundle record, seed target, the **entire admission** (`admitted`, `reasons`, `body`) and the session identity. Any drift closes the session **before estimation**; there is no implicit reset or seed re-arm. | Drift is detected at those check points, not between them. The GLB bytes are not re-hashed here; `MetricBundle.from_file` does that when loading. Pristine checks read `NeuralSession`'s private fields, which a future core refactor could rename. |
| **Relative geometry** | Only a `RelativePackageProof` is accepted: the surface record bytes plus raw component bytes. `verify()` repeats the existing loader's checks: FileRef size and SHA, plain `.npy` with no pickle, dtype and columns, finite values, `validate_mesh`, counts, face-support shape, and a descriptor bank present. Read-only arrays are decoded **from the proven bytes**. The binding stores the record's canonical SHA, the `surface.json` bytes SHA and, per component, the file SHA and bytes, the semantic `array_sha256`, the dtype and the shape. The package must be `object_unit` (metric packages are refused, never relabelled), available and frozen. Its prefix, cameras and freeze frame must lie within the run's reviewed onboarding keyframes and before the first scored frame. `run.surface_package_id`, if set, must match. The regenerated built-in artwork must equal the package artwork SHA and the marker's declared artwork SHA. | A `from_memory` proof is labelled `manufactured_memory_bytes`; it is a byte-consistency proof only, not evidence of a real reconstruction. |
| **Frame input** | `validate_input` runs at every public `process`, and `create` only pre-checks. It covers the schema, a full `FrameSpec` contract round-trip, array types, `rgb_unavailable_reason` when RGB is missing, the decoder time (`None` or a finite non-negative float), identity hash and ID formats, bounded reasons, and canonical strict metadata (at most 32 keys and 16 KiB, no reserved keys such as `K`, `intrinsics`, `family` or `admission`). | |
| **Clock** | The order, the exact manifest record, strictly increasing `timestamp_seconds` (or null for `frame_index`) and an advancing index. **Video** frames whose manifest declares `media_time_s` must carry a decoder PTS (a mismatch of more than 1 ms is counted and invalidates). **Image sequences** must carry no decoder PTS. | |

## Records

| Record | Contents |
| --- | --- |
| `FrameContextV1.build(clip, run, role, binding, *, source_mode="recorded_finite", warmup_frame_ids=None)` | The validated context. |
| `FrameInputV1.create(...)` / `.from_reader(frame, rgb, ident, mask, mident, ...)` | Owned read-only RGB and mask copies plus declared identities. |
| `FrameOutputV1` | A re-validated `FrameResult`, an owned overlay copy (rendered rows only), `epoch`, `phase`, `position`, a measured `process_ms` and `boundary_reason`. |

## Per-frame policy

**Refused with `BoundaryRejected`; the cursor and backend are untouched:**
- a frame that is not the next one in the permitted order (duplicate, reversed or skipped);
- a frame differing from the manifest record;
- clock or PTS rule violations;
- any input that fails structural revalidation.

**Counted with no inference.** Tracking stays invalidated until an explicit
`reset("replay_restart")`, and the backend is never reset implicitly:
- missing RGB;
- a wrong dimension or dtype;
- a pixel hash differing from the declared identity;
- decoder PTS more than 1 ms from the manifest.

**Masks:**
- A missing mask goes through the session's own safe guard (pose and artwork suppressed
  immediately).
- A stale or mis-sized mask is withheld, so the session takes the same safe path, and the row is
  relabelled `rejected`.

**Output validation.** Rows must match:
- run, clip, frame, index, timestamp and role;
- manifest media time, or the decoder media time when the manifest has none;
- pose units (`metric` for neural, `object_unit` for relative);
- the initialized backend identity, the verified image identity, and the verified mask for an
  observed row.

An invalid row, an adapter exception or drift closes the adapter (state `failed`) and raises
`BoundaryFailure`.

**Lifecycle:**
- `initialize` is allowed once and **takes ownership of the adapter**. On ANY initialization
  failure (context mismatch, wrong adapter kind, `open` failure, an unserializable record) the
  adapter is closed and close errors are kept in `boundary.cleanup_errors`.
- `close()` is idempotent and returns cleanup errors.
- `reset("replay_restart")` restarts from the first frame of the original frozen warm-up/seed
  order. The seed target never moves, and there is no automatic reacquisition.

## Adapters (exact classes only)

| Adapter | Wraps |
| --- | --- |
| `NeuralSessionAdapter(session)` | An exact `workbench_neural.NeuralSession` (its admission/family/head/base/provenance/observed-mask gates stay authoritative) and a `ServingAdmission`. |
| `RelativeTrackerAdapter(tracker, proof)` | An exact `workbench_geometry.RelativeTracker`, `builtin_artwork`, `marker_patch` and `render_marker`, mirroring the legacy relative stage. The tracker configuration is checked every frame. |

## Recorded replay and ownership

**`replay_recorded(context, adapter, reader, masks, *, on_output=None, verify_media=True)`**
- Owns `adapter` and `reader` and closes **each exactly once** on every path.
- **Before anything else**, inside that ownership, the context is canonicalized (schema, binding,
  prefix attestation and every nested field) and the result skeleton is built. An invalid context
  (for example `binding_json=b"not-json"` or a malformed `prefix_attestation_json`) closes the
  adapter and reader and raises `ReplayRefused`, a subclass of `BoundaryRejected` whose
  `cleanup_errors` lists any close failures. It never invents rows for an invalid context.
- An adapter that was never opened is closed too.
- Media or initialization failure gives `blocked`; later failures give `failed`. Every planned
  warm-up and scored row is always present.
- Read, mask and callback failures are counted, and the remaining rows are blocked.
- Warm-up and scored rows and their measured timings are kept separate.
- `cleanup_errors` includes the boundary's close errors.
- It returns the in-memory `workbench_frame_replay_v1` result, with source-file SHAs for future
  reports.

**`blocked_replay(context, reason, *, cleanup_errors=None)`** gives every planned row blocked.

**`build_relative_replay(workspace, clip, run, *, package_dir, mask_set, tracker=None)`** uses
`RelativePackageProof.from_package_dir` (the loader's containment, size and aggregate rules), then
the proof binding, then the context. On builder failure it **closes the tracker** (and any reader
it created).

**`build_neural_replay(workspace, clip, run, session, *, role, family, mask_set, bundle=None)`**
**takes ownership of the caller-built session**: on builder failure it closes the session (and any
reader it created) and re-raises.

**`run_neural_replay(workspace, clip, run, *, role, metric_bundle_path, admission_dir, head_path, mask_set, device="cuda", factories=None)`**
is the documented end-to-end sequence. Every component is **real by default**:
1. `MetricBundle.from_file` (re-hashes the GLB).
2. `neural_binding` and the context (seed target and warm-up).
3. `load_admission` (recomputed; not admitted gives `blocked_replay`).
4. `MaskStore` and `NeuralArtwork` from the reviewed selection and its observed mask. If the
   artwork is unavailable, rows stay unrendered and the reason is recorded.
5. `workbench_telemetry.gpu_stage(role)` (busy gives `blocked`), held around backend construction
   and replay.
6. `GoTrackMetricBackend`, then `backend.seed_target_frame_id` set from the binding, then
   `NeuralSession(..., renderer=artwork.render)`. A failure closes the backend and gives `blocked`.
7. `FrameReader`, then `replay_recorded`, then `gpu_stage` released **exactly once**. If releasing
   the stage fails, the completed or blocked result, with all its rows and its provenance, is kept.
   The failure is appended to `cleanup_errors`; `completed` becomes `completed_with_cleanup_errors`,
   and `blocked` or `failed` keep their classification. If the body itself raises unexpectedly, the
   stage is released with that exception and the original error is re-raised.

`NeuralFactories(...)` may replace any component **for tests only**. Every replacement is listed
in `result["provenance"]["injected_factories"]`, with
`evidence_class: "test_double_execution"`, so it is never promoted to real admission or model
evidence.

**Parent-scheduled real use** (loads weights; never run by tests or `inspect`):

```python
from pathlib import Path
from bench import workbench_contract as wc
from bench.workbench_cli import Workspace
from tools.workbench_frame_replay import run_neural_replay

ws = Workspace()
clip = ws.load_clip(CLIP_ID)
run = wc.RunSpec.from_dict(RUN_DICT)          # primary must equal workbench_neural.PRIMARY
result = run_neural_replay(ws, clip, run, role="primary_neural",
                           metric_bundle_path=Path(BUNDLE_JSON), admission_dir=Path(ADMISSION_DIR),
                           head_path=Path(HEAD_PT), mask_set=MASK_SET, device="cuda")
# result["status"] in {"completed", "completed_with_cleanup_errors", "blocked", "failed"};
# all planned rows are present; the session, backend, decoder and GPU stage are released.
```

**CLI:** `python -B tools/workbench_frame_replay.py inspect` prints static capabilities and source
SHAs only. It reads no data, model, run or GT, imports nothing GPU-related, and writes nothing.
There is no neural CLI runner.

## Exact signature changes from the first version

| Before | Now |
| --- | --- |
| `relative_binding(package, package_sha256, *, artwork_kind)` | `relative_binding(proof: RelativePackageProof, *, artwork_kind)` |
| `RelativeTrackerAdapter(tracker, package, arrays, package_sha256)` | `RelativeTrackerAdapter(tracker, proof)` |
| Relative binding keys | `package_record_sha256`, `surface_json_sha256`, `proof_source`, `status`, `frozen`, `metric_scale_available`, `prefix_frame_ids`, `camera_frame_ids`, `package_artwork`, `generated_artwork_sha256`, `marker_artwork_sha256`, `components` |
| Neural binding `bundle` | Now the complete `bundle_record` (adds `glb_path`, `seed_pose_m`, `seed_pose_sha256`) |
| `FrameInputV1.create` | Calls `validate_input`; a decoder time that is a boolean, negative or non-finite is refused |
| `FrameBoundary` | Adds the `cleanup_errors` attribute |

New: `RelativePackageProof` (`from_package_dir`, `from_memory`, `verify`, `source`,
`surface_json_sha256`), `bundle_record`, `canonical_context`, `validate_input`, `ContextMutated`
(a subclass of both `BoundaryRejected` and `BoundaryFailure`), `blocked_replay`,
`NeuralFactories` and `run_neural_replay`.

**Added by the final repair:**
- `ReplayRefused` (a `BoundaryRejected` with `cleanup_errors`);
- seed-label enforcement;
- pristine-session refusal at open;
- the GPU-stage release failure kept on the result.

No public signature changed in the final repair.

**Parent fixture impact:** the existing relative fixture (placeholder FileRefs, caller arrays and
`'d'*64`) is intentionally refused. It must be rebuilt with `RelativePackageProof.from_memory`
using real `.npy` bytes and their FileRef SHA and sizes.

## Worker development checks (manufactured, CPU, in memory; not acceptance)

**Final repair:**
- **Parent suites:** your current suites (45 methods, including the four new regression groups,
  real `RelativeTracker`/`render_marker` geometry, a tiny local video, a proof directory and the
  real telemetry lock on the CPU) **all pass**.
- **Seed labels:** `onboarding:…` and `controlled_init:…` are accepted. `controlled_init:` with
  nothing after it, `None`, `evaluator:ground_truth_pose` and `gt:pose` are refused.
- **Explicit reset:** still works after initialization (epoch 1, next frame `f1`, one backend
  reset).
- **Malformed context with a failing reader close:** `ReplayRefused` with the close failure in
  `cleanup_errors`, and the reader and backend each closed exactly once.
- **Blocked setup with a failing stage release:** a backend factory failure plus a failing stage
  release stays `blocked`, with all 3 rows, the provenance, one enter/exit pair and the release
  error in `cleanup_errors`.

**Earlier repair (unchanged):**
**Parent tests:** the existing parent tests ran 29: 27 passed, including forged K, unsupported
schema and the real `FrameReader`/`MaskStore` replay. 2 errors are the expected
old-relative-fixture `TypeError`s.

**Review probes, all refused before any estimator call:**
- the caller's context mutated after initialize (`ContextMutated`, adapter closed);
- seed pose or object ID changed before open;
- seed pose mutated in place after open;
- admission contents changed after open;
- seed target moved before reset (no `backend.reset` call);
- a wrong `RunSpec.primary`;
- an input with a replaced schema, `K` in metadata, NaN or negative decoder time, or missing PTS
  on a video with manifest media time (cursor 0);
- an early role mismatch (adapter closed).

**Relative tracking and rendering on manufactured package bytes.** The package was a textured
plane, a real `build_descriptor_bank` bank of 600 points and a real marker. Through
`replay_recorded` with the actual `RelativeTracker` and `render_marker` it gave:
- f2 `recovering`, f3 `tracking` + `rendered` (`object_unit`), f4 `lost` (empty mask), f5
  `recovering`;
- an RGBA 480×640 overlay of 4,186 pixels, **all inside the observed mask**, centroid
  (335, 224) for an anchor at (319.5, 239.5).

These are placement and clipping observations only, not accuracy.

**Refused for relative packages:**
- metric relabelling;
- a prefix of f2/f3 overlapping scoring;
- an artwork SHA mismatch;
- descriptor bytes changed versus the FileRef;
- an adapter proof differing from the context binding.

**Docs sequence with stubs** (`NeuralFactories`; no network, model, weights or GPU):
`run_neural_replay` completed with warm-up `[f1]` and scored `[tracking, tracking]`,
`evidence_class: test_double_execution`, and the backend closed once. An unadmitted
`ServingAdmission` gave `blocked` with all 3 rows.

**Coverage status:** a `FrameReader` over a real (tiny) video file, `from_package_dir` on a
manufactured proof directory, an unserializable adapter open record and the real `gpu_stage`
lock (CPU, temporary lock directory) are now covered by **your parent tests**, not by the
worker. None of these is a real reconstruction, real model weights or a GPU run.
