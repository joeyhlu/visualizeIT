# HB scene 000003 CPU media preparation

Tool: `tools/workbench_hb_scene3_media.py`. It prepares **media only**, on the
CPU. It does not:

- run the GPU, neural inference or segmentation;
- read GT, poses, masks or depth;
- reopen the raw camera file;
- choose a seed pose, shift the principal-point origin, place artwork or
  assemble a clip;
- publish a ClipSpec or SurfacePackage, register anything, or grant neural
  admission;
- claim tracking quality, mobile readiness or phone performance.

The scene-2 tools, receipts and contracts and the frozen six-case suite are
unchanged. The old scene-2 stages (`load_allowlist`, `prepare_model`,
`prepare_video`) are **never called**, and no helper global is mutated. The
reused low-level helpers of `workbench_hb_metric_inputs` are listed below; their
documented assumptions are asserted at every stage start
(`_assert_helper_assumptions`):

- frame count 120, seed index 9, opening 0..9 and scoring 10..119;
- 640×480, mm→m factor 0.001, 200 000-face execution cap, 20 000-triangle
  portability target;
- the archive names, dataset revision and 30 fps presentation rate.

The reused helpers:

- index-row validation and re-validation;
- the payload stream;
- the PNG checks;
- the strict PLY parser;
- the GLB writer, reader and verifier;
- the pinned FFmpeg encoder;
- reopen-and-decode;
- the guard, budget and containment helpers.

## Inputs and pins

| Input | Production pin |
|---|---|
| Scene-3 profile `.cache/video-workbench/derived/hb-metric-control-000003-v1/profile/scene3-profile.json` | `de57a486f9014b97d8c00427a7fd97101d02c8ee8e5bce67282f9ade51072fb8` |
| K derivative `…/camera-k/camera-k.json` | `5b542672f25baa23e703132c4f4825ff190cb380df4b56a7d750de020ae1ba78` |
| Parent opening association `.cache/video-workbench/derived/validation/hb-scene3-opening-cad4-association-v1.json` | `1e719262a94ef8b9ee9f0f441c056fb9b0a66d749acb7162c3e4050699dc8986`, reviewed CAD id **4** |
| Scout plan / preview receipt | `d76aaec4…` / `a78e0c79…` (through the profile pins) |
| Three historical anchors | `workbench_hb_metric_inputs.ANCHORS` |
| Preparation scope | `.cache/video-workbench/derived/hb-metric-control-000003-v1` (shared with the immutable profile and K outputs) |
| Accepted input base | 187 459 695 bytes: all 10 CADs and the 1 MiB camera are already charged |

Every input is hashed (bounded) against its pin **before** parsing. The
caller-supplied expected hashes must equal the pins, or the stage is blocked.

Trust is `historical_local_integrity` with `upstream_authenticated: false`.

The association is the parent's **qualitative** opening review: frame
s000009, positive point (320, 340), ROI (285, 270, 370, 440), CAD 4, verdict
`positive_visual_match`, `instance_ordinal: null`, and every GT, pose,
scored-image, mask and NN boundary false. It is **not** an independent
attachment-accuracy annotation.

The CAD id comes **only** from this pinned review. The profile's unbound CAD-5
*candidate hint* never selects 5 and never excludes 4. No object id, ordinal
or pose is inferred from it. A future association (ids 1..10) needs its own
separately accepted, explicitly pinned review; it can't be repinned silently.

**Test-only pins.** `SyntheticMediaPins` (inside
`MediaContext.test_only_synthetic_pins`) replaces the production pins for
manufactured fixtures only. The CLI cannot set it. Outputs are labelled
`label: "TEST_ONLY"` and `pins_kind: "synthetic_test_only"`, and production
validation refuses them, so a TEST_ONLY allowlist can't pass as production.

## Stages and API

```python
MediaContext(root, workspace, preparation, test_only_synthetic_pins=None)
build_allowlist(ctx, *, profile, expected_profile_sha256, camera_k, expected_camera_k_sha256,
                association_review, expected_association_review_sha256, out)
prepare_model(ctx, *, allowlist, expected_allowlist_sha256, out)
prepare_video(ctx, *, allowlist, expected_allowlist_sha256, out)
check_prerequisites(ctx, *, allowlist, expected_allowlist_sha256, model_receipt,
                    expected_model_receipt_sha256, video_receipt, expected_video_receipt_sha256, out)
```

Each stage returns a dict with the published path, its SHA-256, the label,
`audited_opens_total` (actual read counts after the post-publication closure;
added in repair 1) and `exit: 0`.

Pure helpers are also exposed for tests: `validate_association`,
`validate_k_doc`, `derive_rows`, `validate_allowlist_doc`,
`validate_model_receipt`, `validate_video_receipt`, `logical_usage`,
`sealed_sha256`, `MediaLedger` and `publish`.

The CLI equivalents are `build-allowlist`, `prepare-model`,
`prepare-video` and `check-prerequisites`, run with
`python -B -S tools/prototype_bootstrap.py tools.workbench_hb_scene3_media STAGE …`.
Exit codes: 0 completed; 1 failure; 2 usage; 3 blocked.

### `build-allowlist`: metadata only (`media-allowlist.json`, `workbench_hb_scene3_media_allowlist_v1`)

1. The profile and K must lie in the scope, named `scene3-profile.json` and
   `camera-k.json`. The review must be a `.json` file outside the scope, at
   most 64 KiB. They are located without being opened.
2. The whole stage runs in the shared guard: empty payload allowlist, no
   process or network access, no forbidden imports. Prospective budgets are
   checked before any read.
3. **Profile.** Read pinned, then `validate_profile_doc`. It must name this
   scope, and its input total must equal the base.
4. **K.** Read pinned. The contract, profile binding and camera row are
   checked, and the raw K is re-validated from its **lexemes**. The camera
   file is not reopened.
5. **Review.** Read pinned and checked against the exact 13-key schema:
   - an opening frame only (scored frames refused);
   - a PNG hash equal to the profile's;
   - integer native point and ROI, with the point inside the ROI;
   - a CAD id equal to the pinned id (booleans refused);
   - all boundaries false and evidence bound to the accepted preview.
6. **Profile re-authentication.** The accepted profile authentication runs
   from the scout outputs and anchors: pinned before parse, anchors as exact
   **read-only** exceptions, camera checked with `lstat` only. The rebuilt
   profile core must be byte-identical to the profile.
7. **Row derivation.** One more pinned index read derives:
   - the complete RGB 0..119 rows (`validate_index_row`: revision, archive,
     role, location token, CRC, method and flags, sizes, both SHA-256 values),
     each compared field by field with the profile's compact identity and the
     scout plan row;
   - **only** the reviewed CAD row, validated the same way, plus the index
     mesh binding and the review's seals.

   No scene-2 rows are copied or read.
8. **Payload stats** use `lstat` only. The stage asserts zero payload opens
   and zero camera opens.
9. **Allowlist.** Written as compact JSON, at most 128 KiB. It records:
   - the inputs, anchors and source;
   - the frozen split and `FRAME_INDEX` clock;
   - native 640×480 (no crop, rescale or rotation);
   - the full RGB rows and the single model row;
   - the association block (qualitative, ordinal `null`, not attachment
     accuracy);
   - the raw K (no origin shift), with the principal point **unverified**;
   - the unresolved list;
   - the input budget: base plus association bytes, with cached payloads
     never double-counted;
   - the payload stats, read counts, sources, runtime and closure.

### `prepare-model`: `hb_obj_00000N.glb` + `model-receipt.json`

1. **Re-validation and re-authentication before any grant.** The allowlist
   is checked against the pins and fully re-validated, including every row
   via `_revalidate_row`. Then, with the payload grants still **empty**,
   `_reauthenticate` re-reads the pinned profile, K, association, scout
   outputs and anchors from the paths the allowlist names. It re-derives the
   allowlist core from the historical index and requires byte equality. A
   caller who rehashes an allowlist with a contradictory row is refused here,
   before any payload is opened, and the payload rows used afterwards are the
   **re-derived** ones.
2. **Streaming.** Exactly the one reviewed CAD payload is granted and
   **streamed once**, through `verified_payload_stream` and `AsciiPlyStream`.
   The header refuses more than 200 000 **original** faces before any body
   allocation.
3. **GLB.** `build_glb` applies mm × 0.001 **exactly once**; indices, winding
   and vertex colours are preserved. Vertex colours are required.
   Explicitly given normals are renormalised and kept; absent normals stay
   absent, and none are fabricated. There are no UVs and no decimation.
4. **Verification.** The GLB must be at most `GLB_CAP` (5.5 MiB). It is
   verified with `verify_glb` (independent reader plus producer decoder), and
   its accessor bounds are checked against the converted source.
5. **Receipt.** It records the source seals and header, the units, the GLB,
   geometry and bounds, and the portability targets (20 000 triangles,
   reported **separately**; a FAIL is not a geometry failure). It also
   records read counts and `seed_pose: null`.
6. **Closure.** The pre-publication pass re-hashes the inputs and the GLB
   before the receipt's read snapshot is taken. After publication, the CAD
   payload is fully re-sealed once more (a separately counted closure open),
   and the inputs, GLB and receipt are re-hashed.

### `prepare-video`: `video.mp4` + `video-receipt.json`

1. **Checks.** The video stage repeats the same re-authentication and
   re-derivation before any grant (see `prepare-model` step 1). Then:
   - the frozen split 0..9 / 9 / 10..119 and exactly 120 re-derived rows are
     asserted;
   - the 120 RGB payload sizes are checked with `lstat`;
   - the pinned FFmpeg executable is verified (its exact bytes) before any
     payload. Both candidate executable paths are watched by the read counter
     **before** the first hash.
2. **Encoding.** Frames are streamed one at a time and decoded as native
   640×480 RGB8 (no crop, rescale or rotation). They are piped into
   `FfmpegPipeEncoder`, which has:
   - an exact single-use process grant;
   - bounded stdout and stderr, and short-write retries;
   - a deadline watchdog and kill/reap/close cleanup.

   Nothing is dumped to disk except the encoder's single owned staging file.
3. **Decode check.** `reopen_and_decode` must find 120 sequential frames,
   then end-of-video, with exact dimensions and an exact seek to s000010. The
   source PNG and source-pixel hashes are recorded **separately** from the
   lossy derivative pixel hashes; pixel identity is not claimed.
4. **Clock.** `FRAME_INDEX`, with physical FPS and timestamps `null`. The
   30 fps presentation rate is synthetic and does not attest a physical
   10-second prefix.
5. **Closure.**
   - Encoder cleanup must be complete (reaped, return code 0, pipes closed,
     no threads alive, staging handle closed).
   - Before publication, the RGB payloads are checked by `lstat` identity.
   - **After** the receipt is published, **every** RGB payload is re-sealed
     by its actual bytes (`lstat` size first, then SHA-256 and CRC-32 within
     the expected size). A 4-byte change with a restored mtime is therefore
     caught, and so is corrupt DEFLATE (`zlib.error`). The failure is
     reported as a closure mismatch, and the owned `video.mp4` and receipt
     are invalidated.

### `check-prerequisites`: `prerequisites-receipt.json` (`workbench_hb_scene3_media_prerequisites_v1`)

1. **Re-derivation.** The allowlist is re-authenticated and its core
   re-derived (as above). The model and video receipts must be named exactly
   `model-receipt.json` and `video-receipt.json`, lie inside the scope, and
   are read pinned before parse.
2. **Actual reconstruction:**
   - the CAD is re-streamed and the GLB rebuilt **byte-identical**, then
     re-verified (`verify_glb`) and its bounds recomputed;
   - all 120 RGB payloads are re-streamed and decoded, giving the PNG and
     source-pixel hashes;
   - the video is re-decoded (120 frames plus EOF, the exact s000010 seek)
     for the derivative hashes.
3. **Strict receipt validation.** Both receipts must have exactly their
   known key sets. Every declaration must equal the reconstruction or the
   fixed contract with exact JSON types (`_same`):
   - **Model:** source seals and header, unit factor and "exactly once",
     vertex and face counts, bounds, roundtrip, normals note, appearance, UV,
     portability, flags, `decimated: false`, `seed_pose: null`, and the
     producer sources, runtime, scope and closure.
   - **Video:** clock (physical FPS and timestamps `null`, presentation 30),
     image (no crop, rotate or rescale), split, codec settings, encoder
     cleanup, decode block, and every frame in order (PNG, source-pixel and
     separate derivative hashes, and the identity flag).
   - **Video process record:** exit 0, `stdout_bytes` equal to the actual
     file size, and a bounded stderr.
   - **Dynamic fields:** read counts, budget and ledger records are
     validated for exact shape, limits, ownership and staging accounting.

   Any rehashed false claim fails **before** a success receipt exists. The
   producers' source closure and runtime must equal the current ones, so a
   receipt from a different code version is refused.
4. **Receipt.** `status: "prepared_media_only"`, with the unresolved list
   `instance_ordinal`, `seed_pose`, `principal_point_origin`,
   `artwork_placement`, `assembly`, `segmentation` and `admission`, plus the
   raw K and the qualitative association. Nothing is published as a ClipSpec
   or SurfacePackage, and nothing is registered or admitted.

## Ledger, publication and failures

- **Budgets before every encoded write (and before every failure write):**
  - the substage share (`mi.STAGE_CAPS`: allowlist 192 KiB, model
    `GLB_CAP + 32 KiB`, video `VIDEO_CAP + 160 KiB`, prerequisites 96 KiB);
  - the **whole** preparation scope within 12 MiB total and 512 KiB JSON.
    This counts the immutable profile and K outputs (38 771 bytes initially),
    retained failed siblings, staging partials and aliases. Usage is the
    larger of the accepted `mi.scope_usage` and this module's bounded,
    link-refusing `logical_usage`. The latter counts **every** name
    containing `.json` as metadata, including `.x.json.staging-*` partials,
    `invalidated-*.json` and failure records; the accepted helper sees only a
    final `.json` suffix.
  - a **fresh** global 1 GiB derived check and the 1 GiB free floor. This
    also applies to **every** FFmpeg stdout append, through the encoder
    adapter;
  - the 250 000 000-byte input limit.

  Normal writes keep room for a failure record. No caller value relaxes any
  budget.
- **Publication** is exclusive and never overwrites:
  - Data is written to an owned `O_EXCL` staging file, fsynced and verified.
  - On **Windows** it is committed with `os.rename`, which refuses an existing
    destination (verified: `WinError 183`), so there's no second logical name
    and the peak is 1×.
  - On POSIX it is committed with link plus unlink, and every check counts
    **2×**.
  - `os.replace` is never used, and a foreign name is never deleted or
    replaced.
- **Encoder trap.** `FfmpegPipeEncoder.__exit__` calls `ledger.record('video.mp4')`
  early. The stage passes an `_EncoderLedger` adapter: its `check` accounts
  1× growth of the single staging file and runs the full ledger including
  the fresh global check, and its `record` is kept as staging accounting
  only. Final ownership comes **only** from `_commit`.
- **Commit ownership.** `_commit` captures the owned staging file's identity
  (`st_dev`, `st_ino`) **before** the exclusive commit. It assigns that
  identity to the final name the moment the commit succeeds, with no lookup
  after the rename that could fail. A foreign collision is refused before
  the commit and never owned.
- **Receipt last; closure after.** Each stage's receipt is published last.
  After publication, the stage re-hashes (bounded):
  - the allowlist and the input files;
  - its own output names and bytes;
  - the sources and runtime;
  - for the model stage, a full re-seal of the CAD payload;
  - for the video stage, a full byte re-seal of all 120 RGB payloads.
- **Failure.** On any error after the output directory exists:
  - Every final name this stage created, if still that same file, is
    renamed to `invalidated-<name>`, and its bytes are kept.
  - Staging partials, foreign names, and earlier stages' outputs (including
    the immutable profile and K) are **never** renamed or deleted.
  - A bounded, valid-JSON `failure.json` (`completed_receipt_published:
    false`, at most 16 KiB) is written after fresh global and scope checks.
    If it can't be written, nothing is written to disk; stderr gets
    `NOT published` and the record as one-line JSON.

## Reads accounting

An audit-hook counter, inert outside these stages, counts every Python-level
open per label. Every label is attached **before** its first read: the
inputs, anchors (paths taken from the profile), payload root, the stage's
own output directory, and both FFmpeg executable candidates.

Each receipt's `reads` holds:

- `snapshot`: the counts **after** the pre-publication closure pass and
  before the receipt is written;
- the stream opens;
- `post_publication_planned` counts, which happen after the receipt exists.

Each stage's **return value** adds `audited_opens_total`, the actual counts
after the post-publication closure (for example, video payload reads
120 + 120). Camera opens must be 0.

OpenCV's native `VideoCapture` opens are invisible to audit hooks and are
counted by construction (2 per decode check). No absolute peak is claimed
from these counts.

The synthetic chain measured:

| Stage | Counted reads |
|---|---|
| `build-allowlist` | profile 1, K 1, association 1, scout plan 1, receipt 1, receipt and integrity anchors 1 each, index 2 (authentication + row derivation), payloads 0, camera 0 |
| `prepare-video` | receipt snapshot: FFmpeg executable 2 (pre-check + encoder), payloads 120; total: payloads 240 (stream + re-seal), camera 0 |
| `check-prerequisites` | snapshot: 121 payload reads (CAD + RGB); total: 242 after all 121 source payloads are re-sealed after publication; camera 0 |

Successful model/video receipt snapshots use a closed set of integer audit
counters. Each records two reads of the allowlist, profile, K, association,
scout plan/receipt and acquisition/integrity anchors, and three of the index
anchor. Model snapshots include one CAD read and four output reads/one write;
video snapshots include 120 RGB reads, two output reads/one write and two
FFmpeg-executable reads. No camera or unrecognized labels are accepted.
Planned budget allowances and accounting methods must match the producer
contract. Historical free/derived observations may vary within validated bounds.
Ledger totals must cover the owned output and the three mandatory metadata
files, with metadata no greater than total bytes and exact integer caps.

The prerequisite receipt is invalidated on a final source SHA/size/CRC mismatch;
failed outputs and failure records remain accounted. These checks do not infer
an annotation ordinal, a seed pose or an independently verified camera origin.

## Limitations

- **Real data:** the stages were checked on manufactured fixtures only. No
  real RGB, CAD, camera, GT or NN data was opened, and the actual metadata
  and media runs are the parent's.
- **Pins:** the production pins are taken from the brief. The association's
  binding of the profile path and hash, and the real CAD-4 row and mesh
  binding, are confirmed only by the parent's actual run, which fails closed
  if they differ.
- **POSIX:** commits use link plus unlink, and the prospective allowlist
  check (2 × 128 KiB) can't fit the 192 KiB allowlist share, so that stage
  refuses on POSIX. Windows, where rename is exclusive, is the supported
  platform.
- **Video pre-publication check:** before publication, the video stage
  checks the RGB payloads by `lstat` identity only. Their actual bytes are
  re-sealed after publication, and a mismatch invalidates the outputs.
- **Producer code must match:** `check-prerequisites` requires the
  producers' source closure and runtime to equal the current ones. Receipts
  from an earlier code version must be regenerated, not accepted.
- **Encoder budget cost:** the per-append global derived check walks the
  derived, scratch, runs, masks and suite trees for every 64 KiB stdout
  chunk. This is correct but slower with large workspaces.
- **Qualitative association:** it is not attachment accuracy. The instance
  ordinal, seed pose, principal-point origin, artwork placement, assembly,
  segmentation and admission remain **unresolved**.
- **Video is lossy:** decoded pixels differ from the source PNG pixels by
  design; only the derivative hashes are bound.
