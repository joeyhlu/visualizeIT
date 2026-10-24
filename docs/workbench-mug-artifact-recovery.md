# Mug artifact recovery and checked PNG writer

Development diagnostic only. This makes no accuracy, phone or speed claim.

## Why

The current-source mug run `run-20261021T071542-8205a748` completed both neural
roles: 480 unique rows, with 225 and 224 frames tracking. It still exited 1.

- Two `frozen_baseline` overlays, `s000974.png` and `s000975.png`, were never
  written.
- The old CLI ignored the `False` returned by `cv2.imwrite` and registered the
  paths anyway.
- The stage-cache commit then failed on the first missing file.

The libpng "Write Error" messages don't show the underlying OS or disk cause.

The failed run, its parent seal (`f87d3620…`) and the original mug run
`run-20261021T002156-1b2de77d` are not modified.

## Part A: checked writer (`bench/workbench_image_io.py`)

| API | Behaviour |
|---|---|
| `check_rgba(a)` | HxWx4 `uint8` only. |
| `encode_png_rgba(a)` | RGBA→BGRA, `cv2.imencode` result checked; raises `ImageWriteError`. |
| `decode_png_rgba(data, expected_shape=)` | Strict: PNG signature, 8-bit, 4 channels; returns RGBA. |
| `inspect_png_rgba(data, expected_shape=)` | Strict decode plus `alpha_binary` and `alpha_on_pixels`. |
| `publish_new_bytes(path, data, check=)` | Writes an exclusive owned temp file, then publishes with `os.link` (see below). Never overwrites. |
| `write_png_rgba(path, rgba, max_bytes=)` | Encode, check the decoded pixels equal the input, then publish. Returns `{path, sha256, bytes}`. |

`publish_new_bytes` writes the temp file in full and checks it, then flushes,
fsyncs and reads it back, and runs the optional `check`. Only then does it
`os.link` the temp file to the final path. That fails if the final path already
exists, so nothing is overwritten.

The owned temp file is **kept** until the published path has been confirmed as
the same file (same identity as the temp), re-read and compared. If that check
fails, the writer removes only its own final link, and only when the path still
has the temp's file identity. A replaced or pre-existing file is never touched.
It then raises `ImageWriteError`, saying what happened to the link. The owned
temp file is always removed last.

OpenCV is imported lazily, and no model runtime is imported.

**CLI change** (`bench/workbench_cli.py`, neural branch only):

- The unchecked `cv2.imwrite` is replaced by `write_png_rgba`.
- **On a write failure:**
  - the computed row (pose, states, `render_stats`, metrics) is kept and
    appended;
  - `evidence.overlay` is withheld, and `evidence.artifact_write_error` records
    the reason;
  - processing continues with the next frame, and the neural network is not
    rerun.
- **Stage results after a failure:** the role stage stays `completed`. A new
  `{role}_artifacts` stage is `failed`, listing the frame IDs and the first 20
  reasons. The run status is therefore `failed`, and that role's cache entry is
  **not** committed.
- `workbench_image_io` is added to the report's `dependencies.sources` and to the
  neural stage-key module list.

Changing the CLI and adding the new module changes the module hashes, so earlier
neural stage keys no longer match. The parent regenerates admission before any
future neural execution.

## Part B: recovery tool (`tools/workbench_mug_artifact_recovery.py`)

```
prepare --out FRESH-plan.json
run --plan FILE --expected-plan-sha256 HEX
```

**`prepare`** is read-only apart from writing the plan, which is ≤ 256 KiB,
fresh, atomic and budget-checked. It binds and verifies:

- the seal (SHA, exit 1, `failed`, both roles completed, 480 rows, the two
  expected missing overlays, the exact single `stage_cache_store_06` failure,
  head/base identities). Types are checked exactly: `exit_code` and `rows` must
  be integers rather than booleans, and `worker_exited`,
  `original_run_immutable` and `no_new_neural_execution` must be boolean
  `true`;
- the launcher plan (`ca23fe7e…`) and the terminal log;
- all 10 producer source snapshots, kept separate from the current exporter.
  They must equal the source report's `dependencies.sources`, with
  `workbench_render` taken from the launcher pins;
- the failed run's directory: exactly the 380 sealed files, each with the sealed
  hash and length, plus canonical `results.jsonl`, a RunSpec equal to the
  launcher's, and the window s000827..s001066 for both roles;
- the report's known failure: these exact stage statuses, a cold run, GPU used,
  480 telemetry frames;
- the rows:
  - 480 unique rows, 240 per role;
  - observed masks, with `mask_frame_id == frame_id`;
  - head `27e06a51…` for primary and base `f7d127ab…` for both;
  - the admission contract self-hash from `attempt-06`;
  - overlay present ⇔ `rendered`;
  - the exact tracking and rendered counts;
  - the two missing rows have `rendered_pixels` 29 and 38;
- the clip record, the mask manifest, every referenced mask file (seed plus 240)
  and each row's **semantic** mask digest;
- the metric bundle and physical GLB, and the `attempt-06` contract, geometry
  and parity file pins;
- the computational sources, which must equal the producer pins:
  `workbench_neural` 86d0…, `workbench_render` da9d… and
  `workbench_geometry` 80a6…;
- the current exporter modules and tool, and the runtime versions (recorded);
- the original mug run's four JSON files (its `run.json` against the launcher
  pin). Its PNGs are never read.

It also freezes:

- a fresh derivative run ID and directory;
- the derivative RunSpec: same window, object, masks, artwork and weights, with
  config `{mask_set, evidence_recovery_only: true, development_diagnostic_only:
  true, source_neural_run_id, source_run_config, recovery_note}`. There is no
  `segmentation_diagnostic` key;
- the prefix attestation;
- the recorded stats to reproduce;
- a byte reserve of ≤ 8 MiB.

**`run`:**

1. **Load and check:** the plan SHA and tool hash must match. Then it enters
   the guards and re-runs the complete binding, which must equal the plan
   (closure before).
2. **Budget:** the derivative must not exist, and the reserve must fit the 8 MiB
   cap, the workspace's 1 GiB derived budget and the 1 GiB free reserve. These
   are re-checked before each stage.
3. **Rebuild the artwork:** `NeuralArtwork(bundle, reviewed selection on
   s000826, checkerboard, clip K/size, seed mask)`. Its record SHA must equal
   `d20f8b89…`, it must have 2585 patch faces, and its canonical bytes must
   equal the source `neural-artwork.json`. This happens before anything renders.
4. **Re-render in memory:** for `s000974` and `s000975`, call
   `artwork.render(saved pose, observed mask)`. The mask must match the row's
   semantic digest. **Every** `render_stats` field (including `reason`, after
   the session's own normalisation) must equal the recorded one. The result
   must be RGBA `uint8` with binary alpha and 29 or 38 alpha-on pixels.
5. **Create the derivative directory**, then write `run.json` and
   `neural-artwork.json`.
6. **Copy the 369 source PNGs.** Each must be strict RGBA with the frame size,
   binary alpha and an alpha count equal to the recorded `rendered_pixels`, and
   its bytes must equal the seal.
7. **Publish the two re-renders** with `write_png_rgba` (pixel-exact decode,
   ≤ 256 KiB each), then decode them again from disk.
8. **Write `results.jsonl`.** The rows equal the source rows with only `run_id`
   changed, the `FrameResult` round-trip must be exact, and the rows are
   re-read after writing.
9. **Check the file set:** exactly `run.json`, `neural-artwork.json`,
   `results.jsonl` and the 371 overlays.
10. **Closure after:** the complete binding must still equal the plan (seal,
    runs, masks, models' source inputs, computational sources, exporter and
    runtime).
11. **Publish `report.json` last.** The final derivative size and the returned
    summary are computed before publication. After the report is published,
    nothing touches the filesystem or can fail; the prepared summary is
    returned.

**Guards** apply for the whole run and for the binding in `prepare`, and are
restored on every path. They raise `RecoveryViolation` (a `BaseException`) for:

- `GoTrackMetricBackend`, `NeuralSession`, `segment_prompted`;
- `gpu_stage`, `execute_run`, `Pipeline.execute`;
- `StageCache.store`;
- any import of torch, torchvision, sam2, pyrender, OpenGL or the
  `quality_gotrack*` modules. Their presence in `sys.modules` beforehand is also
  rejected.

Models are not opened or rehashed. Their identities are the ones the parent
launcher recorded, as quoted from the seal.

**Exit codes:** 0 on success. 1 means the recovery started and failed:
`recovery-failure.json` (status `failed_evidence_recovery`) is written only
into a derivative directory created by this run, otherwise a stderr record is
printed. The partial derivative is kept and no report is published. 2 means a
plan, usage or `prepare` error.

## Derivative report

- `status` is `completed_evidence_recovery`, with stages
  `closure_before … closure_after`.
- **`telemetry`:** `run_cache_mode: "artifact_recovery"`, `frames: []`,
  `cache: {}`, `gpu_used: false` and `run_wall_ms: null`. Old frames are not
  copied, so the viewer cannot call them fresh.
- **`dependencies`:**
  - `versions`: the current runtime;
  - `sources`: the current exporter modules and tool;
  - `heads`: reused from the source report, as the note says.
- **`recovery`:**

| Field | Value |
|---|---|
| `source_neural_run_id` | The failed source run |
| `source_report_status` | `failed` |
| `source_exit_code` | 1 |
| `source_role_stages` | As recorded |
| `reused_saved_predictions` | true |
| `new_neural_frames` | 0 |
| `new_neural_execution` | false |
| `stage_cache_hit` | false |
| `stage_cache_writes` | 0 |
| `gpu_used` | false |
| `overlays_copied` / `overlays_regenerated` | 369 / 2, with per-file SHA and label |
| `source_neural_dependencies`, `source_neural_telemetry` | Historical, labelled |
| `producer_source_snapshots`, `computational_sources`, `exporter_sources`, `runtime` | As bound |
| `current_cpu_wall_ms` | Labelled "not inference" |
| `plan_sha256`, `seal_sha256`, `launcher_plan_sha256`, `artwork_record_sha256` | Pins |
| `model_verification`, `claims` | Quoted and stated |

## Viewer (`bench/web/workbench/workbench.js`)

When `RunSpec.config.evidence_recovery_only === true`, the provenance strip adds
these lines, read only from the loaded report's `recovery` block:

- **Evidence recovery:** the source run, with its recorded status and exit.
- **Neural source check:** "no new neural inference, no stage-cache hit, no GPU
  (0 new neural frames); head identities are reused from the source rows". This
  appears only when the report states all four.
- **Overlay evidence:** copied and regenerated counts.
- **Recorded CPU recovery wall (not inference).**

If the report doesn't confirm a completed recovery, the strip shows "status
unknown". `artifact_recovery` is a recognised cache mode, and per-frame timings
show as `unknown`.

## Limits

- The two regenerated PNGs are reconstructions that agree with the recorded
  state. They are not the historical bytes.
- Recovery does not repair the source run or its counts, and does not
  write or migrate the cache.
- It is not a neural execution or a cache hit.

## Parent test suggestions

- **Writer:**
  - encode failure (`imencode` → `False`);
  - temp write or link failure;
  - an existing destination is not overwritten;
  - a decoded-pixel mismatch makes `check` raise;
  - only owned temp files are removed.
- **CLI:** with a writer patched to fail on one frame:
  - the row is kept, with `artifact_write_error` and no `overlay`;
  - later frames continue;
  - the `{role}_artifacts` stage is failed and the run is `failed`;
  - `_cache_store` is not called for that role.
- **Tool, pure helpers:**
  - `check_seal`, `check_source_report` and `check_source_rows` with a mutated
    field, count, identity, extra failed stage or a third missing overlay;
  - `check_regenerated` with one stat changed (including `reason`), or with
    non-binary or miscounted alpha;
  - `check_copied_png` with an alpha mismatch;
  - `rows_equal_except_run_id`;
  - `budget_reserve` stays ≤ 8 MiB;
  - `recovery_guards` blocks and restores.
- **Tool, flows:**
  - a mutated pin, drift, or an admission file mismatch makes `bind_inputs`
    fail;
  - the closure-after mismatch path writes the receipt and publishes no report.
