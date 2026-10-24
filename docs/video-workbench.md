# Video validation workbench

Local research workbench for checking object attachment on real video. It runs
on `127.0.0.1` only and is separate from the phone app and the published phone
viewer. It installs nothing, downloads nothing and starts neural work only when
an operator explicitly asks for it.

## Quick start

Reviewed six-case configuration (one command; local UI at http://127.0.0.1:8765/):

```powershell
./tools/run-workbench.ps1 -ReviewedSixCase          # start the loopback service
./tools/run-workbench.ps1 -ReviewedSixCase -Plan    # print the exact command; start nothing
```

`-ReviewedSixCase` only uses cached local assets that must already exist. The launcher never
downloads models, never substitutes missing assets or other weights, and starts no training,
segmentation or inference just to open the UI. It uses:

- the pinned GPU Python `.cache/quality-windows/Scripts/python.exe`;
- the head `.cache/video-workbench/models/flow-terminal-20261021/head.pt`. Its SHA-256 must
  be `27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311` before anything starts;
- serving admission `.cache/video-workbench/admission/attempt-08`;
- the three historical metric bundles (`bundle-{mug,keyboard,ranch}-v1.json`), mapped per clip;
- the three execution-only supplemental controls in
  `.cache/video-workbench/derived/execution-controls/v1/object-0000{02,04,18}`;
- six mask sets: `dev-keyboard=sam2-workbench-02`, and `sam2-workbench-01` for the other five
  clips;
- COLMAP 3.11.1 at `.cache/video-workbench-tools/colmap-3.11.1/bin/colmap.exe`;
- `--device cuda`.

The service then re-validates every asset, bundle, execution control and the admission itself.

Generic mode is still available:

```powershell
python -m bench.workbench_cli --help
python -m bench.workbench_cli inspect              # budgets, COLMAP, neural + SAM prerequisites
python -m bench.workbench_cli suite                # register the six development cases (read-only sources)
./tools/run-workbench.ps1 -Port 8765               # local UI at http://127.0.0.1:8765/
```

Workspace: `.cache/video-workbench/` (`clips/`, `imports/`, `masks/`,
`derived/`, `scratch/`, `runs/`, `suite/`, `service/`). Source videos and image
sequences are referenced in place through registered roots
(`init --root NAME=DIR`); they are never copied.

Budgets: new imported inputs ≤ 250,000,000 bytes (decimal); derived + scratch +
cache ≤ 1 GiB; at least 1 GiB of free disk must remain. The 8 GiB model cache
and 256 MiB public-artifact limits are unchanged. A run that would break a
budget is recorded as `blocked` with every scheduled frame unavailable.

## Launcher (`tools/run-workbench.ps1`)

**Parameters:**
- Existing: `-Port`, `-Workspace`, `-Root`, `-Colmap`, `-Head`, `-Admission`,
  `-MetricBundle`, `-Masks`, `-Device`, `-Python`, `-ExportSuite`, `-Inspect`.
- New, matching the service flags:
  - `-MetricBundleFor CLIP=PATH` → `--metric-bundle-for`
  - `-ExecutionControlFor CLIP=DIRECTORY` → `--execution-control-for`
  - `-MasksFor CLIP=SET` → `--masks-for`
  - `-MetricBundleClip CLIP` → `--metric-bundle-clip`

**Passing several mappings:** PowerShell does not allow repeating a parameter. Pass an array
in-session (`-MasksFor @("dev-mug=sam2-workbench-01", "dev-keyboard=sam2-workbench-02")`) or a
comma list (`-MasksFor dev-mug=sam2-workbench-01,dev-keyboard=sam2-workbench-02`, which also
works with `powershell -File`). Mapping values therefore cannot contain commas. Every item
becomes its own native argument; there is no command-string composition, so paths with spaces
stay intact.

**Checked before launching:** the service stays authoritative, but the launcher first rejects:
- malformed `CLIP=VALUE` items and invalid clip IDs or mask names;
- duplicate clips within one option;
- a clip with both a metric bundle and an execution control;
- `-MetricBundleClip` without `-MetricBundle`, or on a clip that already has a per-clip
  geometry mapping;
- ports outside 1–65535.

The host is always `127.0.0.1`.

**`-ReviewedSixCase`:** it rejects explicit `-Python`, `-Head`, `-Admission`, `-Colmap`,
`-Device`, `-Workspace`, `-Metric*`, `-ExecutionControlFor`, `-Masks` and `-MasksFor` instead of
silently overriding them. `-Port`, `-Root`, `-ExportSuite`, `-Inspect` and `-Plan` remain
allowed. Missing assets are listed and the launcher exits with code 2. A head hash mismatch
refuses to start.

**`-Plan`:** prints `{mode, preset, head_sha256_verified, loopback, commands:[{purpose,
executable, arguments}]}` as JSON and exits 0, without starting the server, segmentation or
inference. It is a dry run, not approval.

## Conventions

- OpenCV camera axes; integer pixel centres; poses are camera-from-object 4x4
  matrices on column vectors.
- `FrameSpec.source_frame_index` is authoritative. `clock_kind
  timestamp_seconds` requires strictly increasing `timestamp_s`; `frame_index`
  clips keep `timestamp_s = null` and cannot attest the 10-second onboarding
  bound. `media_time_s` comes only from actual decoder timestamps.
- Selection points are source pixels. The UI corrects `object-fit` letterboxing
  and converts only at that boundary.
- Geometry units are explicit: `object_unit` (relative reconstructions, never
  treated as metres) or `metric`.

## Commands

| Command | Purpose |
| --- | --- |
| `inspect` | Workspace budgets, roots, clips, COLMAP preflight, neural preflight |
| `init [--root NAME=DIR]` | Create the workspace and register media roots |
| `import video PATH --clip-id ID (--K fx,fy,cx,cy \| --calibration-unavailable)` | Register a local video in place; decoder timing is probed by streaming |
| `import images DIR ...` | Register an image sequence (frame-index clock) |
| `import masks DIR --clip-id ID --mask-set NAME --producer TEXT` | Register externally produced observed masks |
| `segment CLIP --mask-set NAME` | Tap-to-mask: SAM 2.1 prompted with the reviewed points, causal frame release |
| `suite [--verify-images]` | Export the deterministic six-case development suite |
| `run CLIP --mode smoke\|focused\|regression` | Execute the pipeline; `--dry-run` executes nothing (exit 4) |
| `report RUN_ID [--frames]` | Print a run report |
| `gate bundle\|case\|contract\|geometry\|parity\|verify` | Earn or verify neural serving admission |

Exit codes: 0 completed, 1 failed, 3 blocked (a real prerequisite is missing), 4 dry run.

## Pipeline

1. Source: media hash/size check; frames decoded on demand by exact source index.
2. Observed masks: from a `segment` run or an imported mask set. A projected
   silhouette is never used as an observed mask.
3. Relative geometry (no CAD model): installed COLMAP on the causal onboarding
   prefix (≤ 24 frames) with masks → frozen `SurfacePackage` (object units)
   → masked ORB + forward/backward LK + absolute PnP each frame → curved
   marker rendered with depth testing and observed-mask clipping.
4. Neural primary (new flow-terminal head `27e06a51…`, 80,165,526 bytes) and the
   frozen baseline as a separate same-frame comparator. Both stay blocked
   until a verified serving admission and metric bundle exist. The baseline
   never replaces the primary model.
5. Report: every scheduled frame counted per backend, with separate mask/pose/render states.
   Accuracy remains `unavailable_no_independent_annotations` without human labels.

Stage results are cached through `bench.workbench_cache` (owned separately).
The cache is sized to whatever remains of the shared 1 GiB derived budget,
minus a 1 MiB margin for lock and temp files. Before every lookup the
pipeline re-verifies its inputs:

- the media hash, or every image-sequence frame file;
- every observed-mask file;
- the head bytes or base checkpoint bytes.

Each key binds:

- the full `ClipSpec` hash and per-frame identities;
- onboarding and selection, intrinsics, dimensions and the mask manifest;
- the COLMAP version and executable;
- module source hashes;
- for neural roles, the metric bundle and the complete admission record.

Reconstruction packages and tracking overlays are stored *inside* the immutable
cache entry. On a hit they are copied into the new run and re-verified. Cached
rows never rerun inference, and an overlay that cannot be materialized is
dropped from the row's evidence rather than left pointing at another run.
Failed and blocked stages keep their receipts under the run directory, and
earlier runs are never overwritten.

Neural rows carry `backend_identity.installed_head_sha256`, taken from the
validated installer provenance: the export hash for the primary, and `null` for
the frozen baseline, whose head is the base checkpoint's own. Both rows also
carry `installed_head_state_sha256`, a digest of the actually installed
pose-head tensors.

## Earning neural serving admission (`workbench_integer_metric_v1`)

As of 2026-10-21, the weights are present locally and hash-verified in
`.cache/video-workbench/models/flow-terminal-20261021/`. For a fresh machine, the operator copies
`run/head.pt` and `run/best.pt` from the private backup
`achitaan/visualizeit-flow-terminal-20261021-27e06a51` v1 into that folder. The workbench and its
launcher never download them. Then, to re-earn admission:

```powershell
python -m bench.workbench_cli gate bundle keyboard --principal-origin <documented origin> --out bundle-keyboard.json
python -m bench.workbench_cli segment dev-keyboard --mask-set sam2-a     # observed masks
python -m bench.workbench_cli gate case dev-keyboard --frame s001149 --mask-set sam2-a --metric-bundle bundle-keyboard.json --out case-keyboard.json
# repeat for one calibrated YCB case (each intended family needs a case)
python -m bench.workbench_cli gate contract --cases case-*.json --head HEAD --reference BEST --out ADM/contract.json
python -m bench.workbench_cli gate geometry --production --out ADM/geometry.json
python -m bench.workbench_cli gate parity --contract ADM/contract.json --cases case-*.json --head HEAD --reference BEST --out ADM/parity
python -m bench.workbench_cli gate verify --admission ADM
python -m bench.workbench_cli run dev-keyboard --mode smoke --masks sam2-a --admission ADM --metric-bundle bundle-keyboard.json --head HEAD
```

- The contract authenticates the export, reference checkpoint and base
  checkpoint, metric GLB node transforms and textures, RGB/mask hashes, K,
  segmentation identity, initialization source, runtime packages and the
  current source closure. The closure includes the workbench CLI and telemetry
  sources and the review map. Historical training hashes are kept unchanged.
- `bench/quality_gotrack.py` and `bench/quality_runner.py` were already modified
  before the workbench existed. They are accepted for this profile only through
  `REVIEWED_REPLACEMENTS`:
  - each entry pins the historical training hash, the exact reviewed current
    hash, the scope and the reason;
  - the entries are bound to the review diff
    `.cache/workbench-historical-source-replacement.diff`.
- Any other changed historical file, or any further change to those two, is
  reported as `unreviewed` and blocks admission. Admission records both
  reviewed and unreviewed lists. It means execution compatibility only, never
  model quality.
- Geometry validation measures the actual renderer and crop sampler on
  manufactured scenes (crop sampling error must be < 0.001 px). Deliberate
  half-pixel and ×1000 unit controls must fail.
- Parity runs RGB → backbone → decoder → head twice: `best.pt['head']`, then the
  atomically installed export. The run uses fixed seeds with TF32/AMP off and a
  pre-declared tolerance of 1e-6 absolute, with exact matching for discrete
  outputs. Forward execution is observed through hooks, and every compared
  array is stored.
- Each parity variant runs in its own function scope. The network, runner,
  hooks and renderers are released, and garbage is collected (plus a CUDA
  cache release when CUDA is in use), before the next checkpoint loads.
  Traces are compared from the saved, hash-checked files.
- Every phase appends one fsynced line to `<parity dir>/phase-events.jsonl`
  (RSS, CUDA memory, free disk), so a native crash still shows the last
  completed phase.
- A failed parity attempt is never resumed or counted as a pass; a source
  change requires a fresh output directory.
- `verify` recomputes the bindings and comparisons from stored evidence. It
  never trusts `accepted: true`. Admission means experimental execution
  compatibility only, never model quality. Relative object-unit
  reconstructions are never admitted to the metric profile.
- The principal-point origin of the historical and YCB intrinsics is not
  documented in their manifests. The operator must declare it; a corner origin
  subtracts 0.5 and integer centres leave K unchanged.
- Initialization uses the labelled `controlled_init` onboarding pose once. After
  that, only the previous validated pose seeds the next frame. After a loss,
  recovery is reported unavailable because automatic reacquisition is not
  integrated. Two consecutive consistent poses are needed before a pose is
  public.

## COLMAP

`colmap` must already be installed (`PATH`, `VISUALIZEIT_COLMAP` or `--colmap`). Preflight reads
`colmap help` and each command's `--help`.

The exhaustive matcher uses a pinned CPU resource policy,
`cpu_exhaustive_matcher_one_thread_v1`: `use_gpu 0` and `num_threads 1`. The
thread flag is discovered from the executable's own help: `--SiftMatching.num_threads`
on 3.11.1, and `--FeatureMatching.num_threads` only if help lists it. If either
the GPU or the thread control is missing, COLMAP is reported unsupported. There
is no fallback to the default thread count.

The policy is pinned from measured evidence. The default-thread CPU matcher
aborted natively (0xc0000409). A copied database with one thread completed
(`derived/diagnostics/matcher-cpu-one-thread-v1`). No memory cause, quality or
mesh-readiness claim follows from that. Only `feature_extractor` (PINHOLE,
single camera, masks, CPU SIFT), `exhaustive_matcher`, `mapper`,
`model_converter` (TXT), `image_undistorter`, `patch_match_stereo`,
`stereo_fusion` and `poisson_mesher` are run. The workbench parses only the
sparse TXT model, the dense `.bin` depth maps (with the undistorted cameras)
and PLY files.

A face survives only if all three vertices and the centroid lie in the observed
mask and agree with the dense depth in at least 3 registered views. Admission
gates are development heuristics: at least 8 registered views; at least 100
points seen in 3 or more views; median triangulation angle of at least 2°; and
median reprojection error of at most 2 px. Each stage runs in its own process
group with a timeout, cancellation, bounded logs and budget checks before,
during and after the stage. Each subprocess also holds the repository GPU stage
from before it is spawned until it has exited and been cleaned up. CPU-only
commands are serialized too, deliberately. If the stage is busy, that step
records `gpu_stage_busy`, nothing is spawned and the package becomes
`unavailable`. Preflight `help` calls are not locked.

Every COLMAP step, whether it completed or failed (including busy and spawn
failures), immediately writes an fsynced `<stage>.stage.receipt.json` beside its
raw logs. The receipt records:

- exit code, argv, timing and GPU-stage policy;
- log references (path, bytes, SHA-256) to the full bounded `.stdout.log` and
  `.stderr.log`, which are kept unchanged;
- control-free, escaped tail excerpts of at most 2048 characters.

All reconstruction metadata is sanitized to control-free strings of at most
4096 characters before the SurfacePackage is serialized. The run pipeline turns
an ordinary exception in any role into a typed error receipt under
`errors/`. Already produced rows are kept, and only the missing (role, frame)
pairs are filled, without duplicates. The run is reported `failed`, never
completed. Any failure produces an
`unavailable` package with the original image and mask preview. A mesh is never
invented.

## Telemetry and the GPU stage lock

`bench/workbench_telemetry.py` records only measured values. Every run report
carries `telemetry`:

- `run_wall_ms`, and `run_cache_mode` (`cold` / `cached` / `partially_cached`);
- stage durations (source verification, reconstruction, initialization);
- per-frame `tracking_ms`, `rendering_ms` and `end_to_end_latency_ms` for each
  backend role (also copied into each row's `metrics`);
- `process_peak_rss_bytes`;
- GPU peak allocated/reserved bytes, only when CUDA was actually initialized
  (otherwise `null`).

GPU work is synchronized before it is timed. Cache lookup time is recorded as
`lookup_ms` and is never reported as inference latency. On a cache hit, the
original run's measured timings are attached under `cached_original_timings`,
labelled as cached evidence.

Each report also carries `dependencies`:

- installed package versions, read from metadata without imports;
- workbench and production source hashes;
- per-role head identities (`installed_head_sha256` is `null` when no head was
  loaded).

SAM segmentation manifests record initialization time, per-frame segmentation
time, and peak memory.

Segmentation, each neural role, `gate parity`, `gate geometry --production` and
every COLMAP subprocess all run inside one GPU stage. The stage is an in-process lock plus a single
repository-wide lock file, `.cache/gpu-stage-lock/gpu-stage.lock`, shared by
every workspace, so passing a different `--workspace` cannot bypass it. An
existing foreign lock is refused, never stolen. An unreadable or partly written
lock is never deleted. Cleanup removes the lock only when it still holds this
holder's own token. Service jobs are also serialized, so segmentation and pose
never overlap.

## Zero-sample-storage renderer (`bench/workbench_render.py`)

The serving profile renders through `WorkbenchZeroStorageRenderer`
(`workbench_zero_sample_storage_v1`). It wraps the unchanged legacy
`TexturedRenderer` (`legacy_v1`, integer-centre, unlit), which otherwise
resolves 4x multisampled depth. A scoped hook redirects pyrender's two 4x
renderbuffer requests to ordinary single-sample storage, and the adapter then
checks the actual GL state:

- the logged allocations;
- the draw/read FBO attachment object type (`GL_RENDERBUFFER`) and names;
- completeness and the viewport;
- internal formats, with `GL_RENDERBUFFER_SAMPLES == 0` and `GL_SAMPLES == 0`;
- that a reused framebuffer was not reallocated and its attachments did not
  change.

Bounds: each edge ≤ 1280 and ≤ 2,000,000 pixels. The historical 1120 capture
guard and all pins are unchanged. Rendering requires the repository GPU stage
and the stability `_render_lock`. A failed renderer is disposed and poisoned,
with no fallback to multisampled depth.

`WorkbenchIntegerRefiner`, `production_renderer_probe` and
`measure_production_crop` all use this adapter. Its factory identity (policy,
pinned sources, adapter hash) is bound into the contract, the geometry
receipt, the parity receipt and neural cache keys, so a changed policy
invalidates earlier admission and cache entries.

Admission enforces the fixed depth gate (relative error < 1e-4, IoU > 0.995)
independently for the native plane and the template crop. The gate is
recomputed from stored float64 depth evidence in `geometry-evidence/`.

Metric GLBs must be self-contained: every buffer and image must be embedded
(the binary chunk, buffer views or `data:` URIs). Any external URI is rejected
by both `glb_inventory` and `glb_mesh`, because its content would not be
covered by the GLB hash.

## Neural artwork, seeding and evaluation

- Both neural roles render the same frozen artwork, anchored once on the pinned
  metric GLB at the first reviewed positive point on the seed frame, in metres.
- Rendering uses the current validated pose and this frame's observed mask, with
  depth testing and suppression of unsupported, folded or back-facing surface.
  Overlays are served by `GET /api/runs/{run}/frames/{frame}/overlay?role=primary_neural|frozen_baseline|relative_geometry`.
- The `controlled_init` seed applies only to the frame immediately after the
  bundle's `seed_frame_id`. Focused runs that start later replay the intervening
  frames causally first. That replay is not scored, and it is part of the cache
  identity. The seed is never reused after a loss.
- Neural rows record `principal_point_origin` and `effective_K` (integer-centre)
  from the actual bundle so the evaluator can apply the same convention.
- Parity runs only on images and masks decoded from the authenticated case
  bytes, and on the authenticated GLB. Admission re-verifies runtime package
  versions, case files and GLB bytes.

Post-run evaluation and the progress graph are owned by Codex. They run
separately from inference:

```powershell
python -m bench.workbench_evaluate --workspace .cache/video-workbench --run-id RUN_ID --output eval.json [--annotations labels.json]
python tools/workbench_progress.py --records RECORDS --output-dir artifacts/video-workbench/progress --evaluation-report eval.json --checks CHECKS
```

## Local service API (selected)

`GET /api/status`, `/api/clips`, `/api/clips/{id}`, `/api/clips/{id}/media` (byte ranges),
`/api/clips/{id}/frames/{frame_id}/image` (exact source frame), `/api/runs/{run_id}`,
`/api/runs/{run_id}/frames/{frame_id}/overlay|mask`, `/api/jobs/{id}`, `/api/jobs/{id}/results`;
`POST /api/import` (headers `X-Filename`, `X-Clip-Id`, `X-Intrinsics`), `/api/selections`,
`/api/segment`, `/api/runs`, `/api/jobs`, `/api/jobs/{id}/cancel`.

Jobs are serialized and move through `queued → running → blocked|failed|completed|cancelled`.
Every result row is revalidated through `FrameResult` before it is published.

## Per-clip prepared metric controls (service)

The operator configures prepared controls when starting the service. The API can't choose a
bundle path.

```
python -m bench.workbench_service --admission .cache/video-workbench/admission --head <head.pt> \
    --metric-bundle-for dev-mug=.cache/video-workbench/admission/bundle-mug-v1.json \
    --metric-bundle-for dev-keyboard=.cache/video-workbench/admission/bundle-keyboard-v1.json \
    --metric-bundle-for dev-ranch=.cache/video-workbench/admission/bundle-ranch-v1.json \
    --masks-for dev-mug=sam2-workbench-01 --masks-for dev-keyboard=sam2-workbench-02
```

- **Which settings apply:**
  - `--metric-bundle-for CLIP_ID=PATH` and `--masks-for CLIP_ID=SET` are repeatable. A
    duplicate clip id, a malformed pair, a missing file or invalid bundle content stops the
    service at startup.
  - A per-clip mapping always wins.
  - The legacy `--metric-bundle PATH` is used **only** when it is explicitly bound with
    `--metric-bundle-clip CLIP_ID`. It then applies to that exact clip ID only, and the
    following checks still apply:
    - the family's frame size equals the clip size;
    - the clip has calibration;
    - the bundle has a seed, and its seed frame is a frame of the clip and equals the reviewed
      selection frame.

    These checks are necessary but never sufficient: clip identity is never inferred from
    frame size or seed frame. An unbound `--metric-bundle` is reported as unavailable for every
    clip, with the action "configure `--metric-bundle-for CLIP=PATH` or bind the legacy bundle
    with `--metric-bundle-clip CLIP_ID`". This intentionally narrows the earlier
    backward-compatible behaviour so that identity fails closed.
  - The principal-point origin is the bundle's own declaration. The clip record stores no
    origin, so readiness does not claim that it was independently validated.
  - The global `--masks` set is the default only for clips that have that set with a manifest.
    A failed set that has only `failure.json` is never chosen.
- **Hash binding:** each bundle file's SHA-256 is bound at startup and rehashed when the run
  is requested, when the job is created and immediately before the pipeline starts. If a
  mapped bundle has changed, is missing, is invalid or does not fit its clip, the job is
  **blocked before any runner or model is constructed**. Nothing falls back to the global
  bundle or another clip's geometry.
- **No bundle for the clip:** a clip without an explicitly bound prepared control runs with no
  bundle at all. The pipeline then blocks the neural roles explicitly, while automatic relative
  geometry may still run as a separate, auxiliary experiment.
- **What the API returns:** `GET /api/status` (`metric_controls`, including the legacy
  bundle's `bound_clip_id`) and `GET /api/clips/{id}` (`prepared_control`, `masks`) return only
  clip IDs, statuses, sources (`per_clip` or `global_bound`) and hashes, never file paths. The
  possible statuses are:
  - `prepared_metric_cad_control`
  - `pending_selection`
  - `unavailable`: automatic preparation is a separate experimental path
  - `blocked`

  Readiness is not admission. Serving admission and the GLB hash are re-verified when the job
  runs.
- **UI:**
  - *Open clip* shows the clip's control status and replaces the mask-set field with that
    clip's own default.
  - The run panel has two explicit role choices: the paired neural comparison (new head plus
    frozen baseline, always together) and automatic relative geometry (auxiliary, not a new
    head).
  - The job status lists the scheduled roles, the mask set and the neural control actually
    used.
  - YCB candidates are not configured and are never promoted automatically. The reviewed
    execution-only YCB controls below are a separate, explicitly configured class.

## Supplemental execution-only YCB CAD controls (service)

The emitted execution-only controls are directories written by the Sol-reviewed adapter
`tools/workbench_ycb_execution_control.py` (`57a08ff8…`) from the fixed reviewed map
`tools/workbench_ycb_execution_controls.reviewed.json` (`c5ae3b00…`). Each holds
`model.glb`, `metric-bundle.json`, `execution-control-sidecar.json` and
`emission-receipt.json`. The operator maps them per clip at startup; there is no browser or
API path or approval. The service must run in the environment that satisfies the adapter's
runtime pins (`.cache/quality-windows`).

```
.cache\quality-windows\Scripts\python.exe -m bench.workbench_service --admission <admission dir> --head <head.pt> \
    --execution-control-for dev-ycb-003-cracker-box=.cache/video-workbench/derived/execution-controls/v1/object-000002 \
    --execution-control-for dev-ycb-005-tomato-soup-can=.cache/video-workbench/derived/execution-controls/v1/object-000004 \
    --execution-control-for dev-ycb-040-large-marker=.cache/video-workbench/derived/execution-controls/v1/object-000018 \
    --masks-for dev-ycb-003-cracker-box=sam2-workbench-01 --masks-for dev-ycb-005-tomato-soup-can=sam2-workbench-01 \
    --masks-for dev-ycb-040-large-marker=sam2-workbench-01
```

- **Adapter loading:** the adapter is imported lazily, only when a mapping exists. Before each
  use, the adapter file and the reviewed map must equal their pinned SHA-256, and the imported
  module must be the project file.
- **Validation:** the bundle is resolved only through the adapter's `validate_output`
  launch gate. It re-derives:
  - the prepared proof;
  - the runtime pins;
  - the GPU evidence;
  - the reviewed-map entry;
  - the qualitative review;
  - the exact bytes of all four files.

  The service then requires the control to belong to **exactly** that clip. All of these must
  agree:
  - dataset, object ID and canonical YCB ID;
  - inference-manifest SHA and scene;
  - 640x480 family against the clip size;
  - calibration present, and the clip K equal to the manifest's final onboarding `cam_K`;
  - the seed frame, which is the final onboarding frame (its source index and RGB SHA are
    checked).

  A wrong object, clip, seed, size, calibration or selection frame fails closed, and no
  foreign control is used as a fallback.
- **Startup identity:** the result is bound once at startup and never changes for the life
  of the service. It covers the bundle, GLB, sidecar, receipt, adapter and map SHA-256, plus
  the object, seed and flags.
  - Startup refuses: invalid controls, unregistered clips, duplicates, paths that are not
    directories or lie outside the derived root, and a clip that also has a `--metric-bundle-for`
    or bound legacy bundle (conflicting geometry is never silently overridden).
- **Full re-validation:** the full launch gate runs again at run request, at job creation and
  immediately before the pipeline. Each takes about 8 s per clip in the CPU checks. The result
  must equal the startup identity.
  - The run request writes the server-derived identity into the run config
    (`config.metric_control`). Caller-supplied `config` or `metric_control` fields are ignored.
  - The job and the pre-pipeline gate require that bound identity to equal the current one.
    Otherwise the job is **blocked before any runner or model is constructed**.
  - After the run, the control files are rehashed. If they changed, the job is marked failed;
    its outputs are kept but not attributed to the control.
  - Status and clip-detail views only rehash the files against the startup identity.
- **Cache identity:** the pipeline's neural cache key still binds the metric-bundle SHA-256,
  and the bundle pins the GLB. The sidecar, receipt, adapter and map hashes are recorded in
  the run config and job record, not in the stage key; the CLI is unchanged.
- **Honest labels** (`prepared_control.labels` and the UI):
  - execution-only known metric CAD;
  - parent qualitative placement reviewed (not independent annotation);
  - independent correspondence and attachment accuracy unverified;
  - raw-PLY vs loaded-GLTF relative convention passed on both opening frames, with the full-UV
    diagnostic unchanged;
  - automatic RGB geometry is a separate experiment;
  - the 640x480 serving admission is still required;
  - main head `27e06a51…` and the frozen baseline are unchanged.

  The ready status is `execution_only_metric_cad_control`. Serving admission is never bypassed:
  the pipeline still blocks the neural roles without it. The historical `--metric-bundle-for`
  and legacy bundle behaviour is unchanged.

## Status snapshot (2026-10-21; counts are a dated snapshot, not live values)

- **Model:**
  - The pinned flow-terminal head
    (`27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311`) and its reference
    checkpoint are downloaded locally and hash-verified.
  - Serving admission attempt-06 passed actual full-RGB parity on all six cases. The contract
    is `83c041f9198161ff3ba3d5f6c477a3e7955eb24c20e451c014e62e01dced6cf0` and the parity record
    is `40746b0d017b545747fa45c269a45fa175270ffb74983b8dfb4ad9a1d1a7a040`.
  - Admission is execution compatibility only, not a quality claim.
- **Paired full-video runs (new head vs frozen baseline):**
  - The three historical cases (mug, keyboard, ranch) are complete, with 240 scheduled frames
    per role each.
  - Supplemental cracker box is complete: 110 frames per role, 109 with a pose and 54 with
    rendered artwork per role.
  - Soup can and marker are complete: 110 frames per role, with 109 pose and artwork rows
    per role. All six original comparisons contain 1,050 frames per role and 2,100 result rows.
  - Pose availability, for example around 99% tracking rows, counts frames with a validated
    pose. It is **not** accuracy, and the render fraction is separate.
- **COLMAP 3.11.1 is installed.** An actual 24-keyframe, 10-second RGB mug preparation completed
  feature matching, but the mapper failed with "No good initial image pair found". This is
  recorded as an honest failed experiment: no mesh was invented, and it makes no claim about the
  new head.
- **Supplemental YCB controls:** the three supplemental objects (2, 4, 18) are admitted as
  separate **execution-only** CAD controls by the exact reviewed map. Independent full-UV and
  landmark correspondence remain **inconclusive**.
- **No claims:** there is no independent attachment-accuracy claim and no phone/30 FPS claim. The
  portable frame boundary is not evidence of phone performance.
- **Software checks:** 396 checks pass and 2 are skipped, under the source-bound receipt 16,
  plus 3 actual negative tamper controls. Passing software checks does not prove tracking
  quality.
- **Budgets:** new inputs are 24,162,286 B of the 250,000,000 B limit. All new derived output
  is conservatively about 515 MB of the 1 GiB limit at this snapshot. Use `inspect` for current
  values.
- **Segmentation:** the SAM 2.1 checkpoint and sources are present; segmentation runs only when
  an operator triggers it.
  A separate opening-only cracker-box prompt trial removed the large mask flips but failed its
  frozen qualification check. It was retained without promotion or a neural rerun.
  A separate original-prompt trial isolated forward and reverse SAM state while keeping both
  states at the original 120-frame temporal normalization. All 120 masks were written and
  verified, but 52 scored masks were undersized and seven consecutive transitions failed
  the frozen continuity threshold. The original seed remained byte-identical. This trial
  also remains unpromoted, with no neural replay; state isolation did not resolve coverage.
  A subsequent single whole-object box prompt, selected from opening RGB370 only and frozen
  before inference, passed all development consistency checks: seed component fraction 0.9953,
  zero undersized scored masks and zero low-IoU transitions. The fixed nine-frame preview was
  accepted by parent and independent model review. This is not independent segmentation accuracy.
  The new seed has its own identity; equality to the old point-only seed is inapplicable.
  A stale inherited description of that old equality rule is superseded by an immutable,
  hash-bound clarification; mask pixels, source and trial outputs are preserved.
  Its qualified masks are registered privately under `sam2-opening-box-v1-qualified` for one
  separate paired 110-frame cracker diagnostic. That replay completed all 220 rows: both models
  rendered on 109/110 frames, compared with 54/110 under the original masks. Both retained
  109 tracking and one recovering frame. This improves display availability, not established
  alignment accuracy or head superiority; the post-run anchor evaluation remains pending.
  The viewer now shows matching-frame rendered and rejected pixel counts below each pane;
  missing statistics appear as unavailable rather than zero.
- **Development cases:** the six cases are development cases, not unseen objects, and there are
  no independent annotations.
- **Artwork:** limited to the fixed checkerboard and directional patterns plus an import
  interface (≤ 1024 px). Artwork generation is deferred.
# Post-run dataset-reference diagnostic

The three original supplemental YCB comparisons now have a separate, sealed post-run
evaluation against cached BOP reference poses, visible masks and depth. It uses the
fixed `27e06a51…` head and the frozen pretrained baseline. No scored reference labels
were opened until the original inference jobs had completed and the evaluation plan,
sample points, source/runtime identities and parent terminal seal were frozen.

| Clip | New head median / P95 (px720) | Baseline median / P95 (px720) | Measured / scheduled |
| --- | --- | --- | --- |
| Cracker box | 6.96 / 10.31 | 6.76 / 10.32 | 109 / 110 per role |
| Soup can | 8.69 / 11.07 | 8.67 / 11.03 | 109 / 110 per role |
| Marker | 9.50 / 12.40 | 9.42 / 12.40 | 109 / 110 per role |

These are conditional errors for the fixed artwork anchor, normalized to 720-pixel
image height. Each missing-pose frame remains in the failure-aware coverage denominator.
All 110 reference anchors per clip passed the reference-only visibility checks. The
mask IoU medians against dataset visible masks are 0.515, 0.915 and 0.850 respectively;
the masks are shared inputs to both model roles.

All three new-head diagnostics miss the provisional median <5 / P95 <10 px targets.
Small differences from the baseline do not establish improvement. These development
controls share CAD geometry/calibration and use reference-derived onboarding poses;
they are not independent human attachment measurements or phone-performance evidence.
Human wrong-surface and recovery-correctness metrics remain unverified.

Implementation and the two-phase reproduction procedure are documented in
`docs/workbench-ycb-dataset-evaluation.md`. The immutable result is
`.cache/video-workbench/derived/dataset-reference/eval-v1/dataset-reference-projection.json`
(SHA-256 `83bccd35f98f70a46730d99bb15ec33883e6e6f3d421f13dc5c9410a451250b1`).

## Synthetic dynamic pose control

A separate GPU control exercises the actual crop, PnP and pose-update path with
synthetic query images and exactly known synthetic poses. It uses the three prepared
YCB geometries, exact and fixed perturbed initializations, and five iterations per
branch. Perfect geometric flow passes all 30 checks before the installed new head
runs. No scored video labels or frames enter this experiment.

| Object | New-head final checkpoint P95, exact initialization | Perturbed initialization |
| --- | --- | --- |
| Cracker box | 0.836 native px | 0.786 native px |
| Soup can | 0.691 native px | 0.770 native px |
| Marker | 0.766 native px | 0.716 native px |

The oracle's worst checkpoint P95 is 0.00000646 native pixels. All six learned
branches complete; the actual counters are 63 renders and 30 learned forwards.
Input, source, runtime and admission hashes agree before and after execution, with
no cleanup errors. The marker still has about 2.4 degrees of rotation error, despite
subpixel projected errors; the metrics measure different properties.

This supports the tested conventions on synthetic inputs. It does not establish
real-video accuracy, correct physical initialization, model improvement or phone
readiness. No production convention correction was made. Original bundle seeds
remain unchanged; slightly rounded rotations are projected onto SO(3) only when
constructing the synthetic targets.

See `docs/workbench-dynamic-pose-control.md`. The immutable result is
`.cache/video-workbench/derived/dynamic-pose-control/attempt-01/result.json`
(SHA-256 `de1a93311e813083072f93a3d922711adb388aa8cb3e42e536b4cb8eb424fdcd`).

Each loaded replay displays its recorded mask set and prompt SHA-256. Switching runs clears
stale provenance; missing recorded values appear explicitly. The create-run controls retain
their own settings. See `docs/workbench-run-provenance.md` for the data sources.

## Qualified-mask cracker replay

A separate replay of cracker frames 371–480 uses one opening RGB370 box plus the original
points and the unchanged trained head, baseline, prepared geometry, camera and artwork anchor.
The original six-case suite is preserved. The sealed CPU evaluation reuses the frozen reference
anchor pixels and first reproduces all 220 original rows within 1e-8 px.

| Role | Original median / P95 (px720) | Qualified-mask median / P95 (px720) | Artwork frames, original → qualified |
| --- | --- | --- | --- |
| Trained head | 6.960 / 10.307 | 5.585 / 8.666 | 54 → 109 |
| Frozen baseline | 6.760 / 10.321 | 5.692 / 8.722 | 54 → 109 |

Each cell has 110 eligible, reference-visible frames, with 109 measured tracking poses and one
recovering frame. With qualified masks, failure-aware coverage is 39/110 at ≤5 px720 and
108/110 at ≤10 px720 for both roles. Both conditional medians still exceed the provisional
5 px target. These cached dataset-reference anchor diagnostics establish no human attachment
accuracy, orientation/footprint correctness, phone readiness, whole-suite improvement or
trained-head superiority. The changed prompt and seed benefit both models in this case.

The independent output audit reproduced all 440 old/new rows within 1.15e-13 px720. Eight
actual copied-metadata fault controls rejected before reference parsing. Model and mask physical
closure is parent-attested; the evaluator reads no model, mesh, raw-label or mask pixels.
See `docs/workbench-mask-replay-evaluation.md` for the frozen plan, seal and reproduction boundary.

The source-bound old/new mask graph is `artifacts/video-workbench/progress/cracker-mask-replay-v1.png`, with matching JSON and CSV. Codex verified all four cells, missing-frame denominators, source hashes and visual layout. It introduces no new inference or independent labels.

The prepared-CAD budget audit independently counts GLB triangles and matches them to
the serving records: mug 31,606; keyboard 3,876; ranch 8,316; cracker, soup and marker
15,728 each. The mug exceeds the initial 20,000-triangle target. Each recorded artwork
texture is 256 pixels per edge. Each GLB plus its metric-bundle record is below 20 MB,
but this excludes other references and does not verify a complete `SurfacePackage`
size. Geometry has not been simplified, and attachment-preserving simplification and
phone performance remain unverified. The audit is retained in
`.cache/video-workbench/derived/validation/prepared-cad-target-inventory-v1.json`.

The cache-only regression command (`tools/workbench_cache_regression.py`) replayed six
unchanged cases through the current pipeline: 1,840 frame results, 1,284 overlays and six
artwork records matched their originals exactly. No new model inference or GPU work ran.
Pipeline-call wall times were 10.29–13.10 seconds, excluding outer command closure hashing.
The graph `artifacts/video-workbench/progress/cache-regression-turnaround-v2.png` uses
`report.telemetry.run_wall_ms` for both cold and cached runs (10.23–13.04 seconds cached),
which excludes final report-writing overhead as well. Shared-host timings are descriptive.
The original mug cache is correctly ineligible for three source changes; a separate current
serving revalidation uses the same pinned models and frozen inputs. See
`docs/workbench-cache-regression.md` for guards and failure handling.

A read-only alignment audit matched 660 completed YCB rows to source-frame metadata and found no source-supported frame-index or seed-target shift. Exploratory lag checks did not support a common correction. Official lag-zero scores remain unchanged; the residual attachment error is unresolved.

The replay provenance strip now labels each role’s timing origin from recorded telemetry, including cached historical measurements, fresh-run measurements and unknown fields. Its pipeline wall excludes outer command validation. Cached → cold → cached browser checks and eight mixed/missing/invalid telemetry controls passed; no inference or phone performance is inferred from a cached replay.

The original current-source mug revalidation processed all 480 neural frames, but its run
remains **failed** because two baseline overlay writes were missing during cache publication.
Both sets of poses and states exactly match the preserved original mug result; this does
not supply independent accuracy. The failed run is sealed and preserved. Its separately
reviewed CPU recovery copies 369 images and reconstructs only the two missing images,
with exact recorded render statistics. The derivative is labeled
`completed_evidence_recovery`, with zero new inference. The checked PNG writer now
preserves computed rows and reports an artifact failure when an image cannot be published.

Source diversity is also tracked explicitly: the frozen six object cases use four media
sources because all three YCB objects share sequence 000059. Two existing SAM gallery videos
(cups and alphabet blocks) are now registered as input-only diagnostics, with calibration
explicitly unavailable and selection/prepared geometry pending. They increase the available
library to six media sources; they are **not** neural accuracy evaluations. No source video
was downloaded or copied. Counted inputs are 28,514,828 bytes, including the two reused references.


Fresh serving admission07 passed six full-image cases. This verifies execution
compatibility, not attachment accuracy or phone readiness.


The actual prepared-control portability audit is independently accepted. Mug's 31,606
triangles exceed the 20,000 target. Keyboard remains unverified because its required
`KHR_texture_transform` extension is unsupported. Ranch, cracker, soup and marker pass
the byte, triangle and artwork targets. This accounts for prepared CAD controls; it does
not establish a complete phone runtime or attachment-preserving simplification.

The cached HB preflight is also independently accepted as input feasibility only. It
checks a 120-frame inventory, previews only opening frames 0 and 9, and inspects the
explicit object-1 PLY header. That header does not bind object 1 to the visible target.
The frozen opening is frames 0–9, with planned seed 9 and scoring frames 10–119. Physical
FPS and timestamps remain unknown. No HB tracking or ground-truth pose extraction ran.
Instance binding, an isolated reviewed seed, geometry conversion and neural admission
remain separate requirements. See `docs/workbench-hb-input-preflight.md`.

The existing two-hour graph generator now displays all eight requested metrics and
completed scoring-window coverage. Its CSV separates cached-token development rows from
per-clip video rows. Unknown metrics stay blank in CSV and visibly unmeasured or unverified
in the graph. Memory is host process peak RSS; frame latency excludes model loading and
segmentation preparation. Neither measures phone performance. The reviewed snapshot is
`artifacts/video-workbench/progress/results-20261021T200517029825Z.png`, with matching
JSON/CSV. See `docs/workbench-progress-reporting.md` for reproduction and evidence limits.
