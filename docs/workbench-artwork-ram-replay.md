# RAM-only artwork replay from a saved pose

| File | Purpose |
| --- | --- |
| `bench/workbench_artwork_replay.py` | The helper |
| `bench/workbench_service.py` | Endpoint and descriptor |
| `bench/web/workbench/workbench.js` | UI |

When a finalized neural row was rendered but its artwork PNG is genuinely absent, the workbench
can re-render that frame's artwork **in memory on the CPU** from the **saved** metric pose. A
typical cause is the sandbox denying the atomic PNG publication (`WinError 5`).

**What a replay is:**
- labelled `cpu_replay_of_saved_pose`;
- **not** the original artwork file (`historical_png_bytes_verified: false`);
- **not** new tracking (`new_pose_estimation: false`, `new_neural_inference: false`);
- never written to disk (`disk_files_written: false`);
- no quality claim.

**Not changed:** the existing recovery tool, the existing PNG endpoint, the CLI, neural,
contract, geometry, image-I/O and boundary code. No model, GPU, cache, memoization or output file
is involved.

## HTTP

`GET /api/runs/{run_id}/frames/{frame_id}/artwork-replay?role=primary_neural|frozen_baseline`

Every response has schema `workbench_artwork_ram_replay_v1`, the run, role and frame, and the
fixed flags `source`, `historical_png_bytes_verified`, `new_pose_estimation`,
`new_neural_inference`, `disk_files_written` and `quality_claim: false`.

| Status | HTTP | Meaning |
| --- | --- | --- |
| `ok` | 200 | `label`, `original_run_status`, `original_artifact {recorded: false, reason}`, `png {sha256, bytes, width, height, base64}`, `render_stats`, `provenance`, `limits` |
| `busy` | 429 (`Retry-After: 1`) | Another replay is rendering (one at a time, non-blocking, no queue) |
| `not_ready` | 409 | The run is still executing, the report is missing or non-terminal, no registered bundle, or no admission metadata |
| `original_available` | 409 | The original PNG is recorded and present; use the existing `/overlay` endpoint, which always takes precedence |
| `refused` | 422 | Any authentication, identity, input, statistics or bound failure; the reason is included |
| `not_found` | 404 | No such run, or no row for this role and frame |

**`provenance`** contains: `results_sha256`, `report_sha256`, `run_json_sha256`,
`artwork_record_sha256`, `artwork_file_sha256`, `binding_sha256`, `context_digest`,
`mask_manifest_sha256`, `mask_sha256`, `image_sha256`, `bundle_sha256`, `glb_sha256`,
`admission_contract_sha256`, `producer_sources` (report dependencies, boundary source, admission
closure) and `replay_sources` (current hashes of the replay helper and the modules it uses).

**`run_detail`** (`GET /api/runs/{id}`) adds a small adjacent descriptor, `artwork_sources`
(`workbench_artwork_sources_v1`). For every rendered neural row it lists `role`, `frame_id`,
`original_png` (`original_recorded`, `original_missing` or `not_recorded`), `ram_replay`
(`eligible` or `ineligible`) and `reason`, plus `replay_route` and `label`. The `FrameResult`
schema and `frame_results` are unchanged. Eligibility there is static; the endpoint
re-authenticates everything.

## What is authenticated before rendering (fail closed)

1. **Finalized run.**
   - `report.json` must exist with a terminal status, otherwise `not_ready`; a run the service is
     currently executing is also `not_ready`.
   - Size bounds: `run.json`, `report.json` and `neural-artwork.json` at most 4 MiB each;
     `results.jsonl` at most 16 MiB. Every bounded reader stats first, then reads **at most
     limit + 1 bytes** and rejects overflow; there is never an unbounded allocation. Every **bounded
     hash** (`_sha_file(path, limit)`, used for the clip JSON, bundle file and GLB) also enforces a
     cumulative read limit. Each read asks for at most the remaining allowance plus one detection
     byte, and a file that grows past its cap after the stat check is refused with no digest
     returned. Unbounded source hashing (`limit=None`) remains an ordinary streaming hash.
   - Run IDs must agree across files. The report, `run.json`, the RunSpec and **every row** must
     name the run's clip. There must be exactly one row per scheduled role and frame.
2. **Row.**
   - Role `primary_neural` or `frozen_baseline`, with `tracking` + `rendered` + `metric` units and
     a contract-valid saved pose.
   - An observed mask of the same frame.
   - **Strict recorded statistics:** `rendered_pixels` must be an exact `int` greater than 0
     (`True` is never a count). Every recorded count (`mask_rejected_pixels`,
     `overlap_rejected_pixels`, `backfacing_rejected_pixels`, `patch_faces`, `folded_faces`) must be
     an exact non-negative `int`, so `False` and `0.0` are refused. `reason` must be a string or
     null.
   - **Exact frame metadata,** before any artwork construction or render: the row's `frame_id`,
     `source_frame_index`, `timestamp_s` and `media_time_s` must equal the current clip manifest
     `FrameSpec` in value and type. A decoded image hash is never a substitute.
   - Lost, recovering, blocked and suppressed rows are refused **before any render**.
3. **Original PNG precedence.**
   - **Recorded and present:** `original_available`.
   - **Recorded but missing or corrupt:** `refused`, never replaced.
   - **Replay allowed:** only when no overlay is recorded, the row records why
     (`artifact_write_error` or `overlay_artifact`), and no file exists at
     `evidence/<role>/<frame>.png`.
4. **Producer computational source closure** (metadata only; `load_admission`, weights and parity
   arrays are never read):
   - `report.dependencies.sources` for `workbench_cli`, `workbench_neural`, `workbench_geometry`,
     `workbench_contract` and `workbench_image_io` equal the current files;
   - the role stage's `execution.boundary_source_sha256`, `boundary_version` and
     `execution_path` equal the current boundary;
   - the configured admission `contract.json` is parsed as metadata. Its self-hash must equal the
     row's `admission_contract_sha256`, and its `source_closure.current_files` for cli, neural,
     geometry, contract, **render** and **boundary** must equal the current files. With no
     admission directory configured the result is `not_ready`.
5. **Pinned identities.**
   - Primary installed and export head = `27e06a51…`; base = `f7d127ab…`.
   - Baseline has no installed or export head and `frozen_base_pose_head`.
   - All of `principal_point_origin`, `initialization`, `seed_frame_id`, `seed_target_frame_id`,
     `glb_sha256` and `metric_units` are required and must equal the bundle, the seed plan and
     `metres`.
   - `effective_K` must be 9 finite values equal to `native_K(clip K, origin)`.
6. **Bundle.**
   - It comes **only** from the server's registered mapping (execution control, `--metric-bundle-for`
     or the bound global bundle), never from the browser.
   - The bundle file SHA is re-hashed, and the GLB must be at most 32 MiB **before**
     `MetricBundle.from_file`, which re-hashes the GLB.
   - `neural_binding` and `FrameContextV1` are recomputed, then `boundary_provenance`: binding SHA,
     context digest, boundary source, version and path must equal the role stage's `execution`.
7. **Masks.**
   - The current mask manifest identity equals the recorded mask stage.
   - The requested mask's identity equals the row's.
   - The selection mask is loaded exactly as the pipeline loaded it.
8. **Source frame.** The registered media is verified (`verify_media`, `verify_frames`), and the
   exact frame is decoded through `FrameReader`, whose `image_sha256` must equal the row's. The
   reader is closed on every path.
9. **Artwork.** A fresh `NeuralArtwork(bundle, saved reviewed selection, artwork kind, clip K,
   dimensions, selection mask)` must reproduce `neural-artwork.json` exactly: the complete
   canonical record, `record_sha256`, effective K, anchor and marker.
10. **Render.** `NeuralArtwork.render(saved metric pose, current observed mask)` on the CPU.
    Checks:
    - the re-rendered statistics must equal the recorded ones as **canonical JSON** (type-exact:
      `False` ≠ `0`, `1.0` ≠ `1`), and every computed count must itself be an exact `int`;
    - uint8 RGBA of the clip dimensions;
    - binary alpha, the alpha count equal to `rendered_pixels`, and every alpha pixel inside the
      observed mask.

    The PNG is encoded only with the existing `encode_png_rgba` (at most 2 MiB; the response is at
    most 4 MiB).
11. **After the render, before responding:**
    - run files, clip JSON, the bundle file and GLB;
    - a **fresh** `MaskStore` (never the earlier cached manifest), whose full identity must equal
      the recorded mask stage, with the requested and selection mask bytes re-verified;
    - the media and frame file;
    - the producer, admission and replay source hashes;
    - the original-artifact eligibility, **repeated immediately before responding**. A newly
      appearing original, or an ambiguous file at the original path, prevents the RAM
      substitution (`original_available` or `refused`).

    Any change gives `refused` with nothing returned. Original rows and status are never
    modified.

## Bounds and resources

**Caps:**

| Item | Cap |
| --- | --- |
| Frame edge | 1280 px |
| Frame area | 2 MP |
| Mesh faces | 200k |
| GLB | 32 MiB |
| `results.jsonl` | 16 MiB |
| Encoded PNG | 2 MiB |
| HTTP response | 4 MiB |

Dimensions and file sizes are checked before expensive processing.

**Memory:** `limits.ram_working_allowance_estimate_bytes` = 512 MiB is a prospective **estimate,
not a measured peak**.

**Concurrency and side effects:**
- One replay at a time per service (non-blocking `busy`, no queue).
- No prefetch, no geometry or image cache, no memoization.
- No files are written; only the service's existing request log lines.
- There is no global monkeypatching or import blocking inside the concurrent service.

## UI (`workbench.js`)

**When it appears:** a rendered neural row without an original overlay, and with
`artwork_sources` marking it `eligible`, shows the adjacent label *"CPU replay from saved pose;
original artwork file unavailable. Historical bytes cannot be verified."* It then requests the
replay after a 300 ms debounce. Scrubbing is never treated as real time.

**Sequential requests:** the current frame's eligible panes (primary and baseline) are requested
**one at a time**, matching the server's single render lock, so they never make each other busy.
A `busy` reply caused by another client gets at most three retries, 500 ms apart and
cancel-aware.

**Stale-overlay protection:**
- Every frame change and run change first calls `resetReplay()`: the generation increments,
  in-flight fetches **and retry delays** are aborted, the remaining sequence is dropped and every
  blob URL is revoked.
- Overlays are already cleared at the start of `show()`.
- A response is applied only if the generation, run, current frame and role still match, and the
  response's `run_id`, `frame_id`, `role`, `source` and `status` are correct.
- On refusal or busy, the pane shows the reason and no image.
- A successful replay is captioned "Not the original file and not new tracking."

**Unchanged:** the existing overlay path and all suppression rules (lost, recovering, suppressed or
no mask shows no artwork).

## Public API

- `ArtworkRamReplay(workspace, *, bundle_resolver, admission_dir, repo_root=ROOT).replay(run_id, frame_id, role) -> (http_status, body)`
- `eligibility(row, run_dir) -> (state, reason)`
- `recorded_stats_problem(stats) -> str | None` (repair 1; strict count types)
- `artwork_sources(run_id, rows, run_dir) -> dict`
- Constants: `SCHEMA`, `SOURCES_SCHEMA`, `LABEL`, `FLAGS`, `HTTP`, `COUNT_STATS` and the cap
  constants.

**Service:**
- `WorkbenchService.artwork_replay(run_id, frame_id, role)`;
- `WorkbenchService._artwork_bundle(clip_id)`, server-registered geometry only;
- `run_detail(...)["artwork_sources"]`.

## Worker development checks (synthetic, CPU, temporary; not acceptance)

**Repair 2:** `_sha_file` now enforces a cumulative cap. Your growth test (stat reports 2 bytes,
the stream yields 100, cap 4) refuses after reading at most 5 bytes for both `_read_bounded` and
`_sha_file`. All RAM replay tests and the earlier regression suites pass (results below).

**Repair 1** (same manufactured fixture):
- **Sequential requests:** primary then baseline both returned `ok` (no self-induced busy).
- **Refused before any `NeuralArtwork` was constructed** (0 constructions across these cases):
  `rendered_pixels: True`, `mask_rejected_pixels: False`, `mask_rejected_pixels: 0.0`, and a
  changed row `timestamp_s`, `source_frame_index` or `media_time_s`.
- **Refused after the render:** an original PNG that appeared mid-render, and a mask manifest
  changed mid-render (detected by the fresh `MaskStore`).
- **Bounded read:** an oversize file was refused.
- **Restored inputs:** `ok` again.
- **Suites:** your RAM replay tests plus the service, serving lifecycle, clip controls, execution
  controls and boundary adoption suites: 55 tests pass, including the earlier boolean
  pixel-count failure.
- **Not exercised by the worker:** the JS sequencing and retry in a browser (`node --check`
  only).

**Initial delivery:**
**Setup:** a manufactured workspace with a 64×48 image sequence, an observed-mask manifest and a
minimal GLB plane. It had a real `NeuralArtwork` and `neural-artwork.json`, a finalized two-role
run whose rows record `artifact_write_error`, a metadata-only admission `contract.json`, and a real
`WorkbenchService` with a per-clip bundle mapping.

**Results:**
- **Descriptor:** `artwork_sources` marked all 4 rendered rows eligible.
- **Replays:** primary and baseline both returned `200 ok`. The PNG decoded equal to the in-memory
  render, and its SHA matched. The file tree was byte-for-byte unchanged afterwards; nothing was
  written.
- **Concurrency:** a held render lock gave `429 busy`.
- **Refused:**
  - a changed `render_stats` value;
  - a recorded-but-missing original PNG;
  - a changed installed head;
  - a lost/suppressed row (before rendering);
  - a moved seed target;
  - a changed artwork record anchor;
  - changed observed-mask bytes;
  - a changed admission render closure;
  - the relative role.
- **Not ready:** a missing report.
- **Restored inputs:** replay was `200 ok` again.
- **Imports:** torch was not imported.
- **Existing suites:** service, serving lifecycle, clip controls, execution controls, boundary
  adoption, CLI and image I/O: 61 tests pass.

**Not exercised by the worker:**
- any real run, real mug video, real GLB or real render;
- the existing-PNG `original_available` path end to end;
- a run executing while a replay is requested;
- the browser UI (syntax-checked only).

The parent's actual CPU comparison against two existing known PNGs comes after source acceptance.
