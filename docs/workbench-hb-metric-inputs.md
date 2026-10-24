# HB prepared-metric control: trust-gated CPU inputs

Tools: `tools/workbench_hb_metric_inputs.py` (input preparer, tool version 3) and
`tools/workbench_hb_opening_seed.py` (isolated opening seed).

These modules are inputs only. Together they prepare one already-cached HB
`val_primesense` scene, 000002 (120 RGB PNGs), together with explicit CAD,
for a **future** prepared-geometry control of the main head (`27e06a51…`).

They do not run a model, the GPU or inference. They don't register a clip,
mutate an admission or make a mobile claim. This is not an automatic-geometry
experiment, and automatic geometry still forbids all GT.

HB keeps its own family, `hb_primesense_640x480`. It is never renamed to YCB
and never inherits a YCB admission. The execution cap is 200 000 triangles; the
20 000-triangle portability target is reported separately.

## Trust

| Field | Value |
|---|---|
| `trust_level` | `historical_local_integrity` |
| `upstream_authenticated` | `false` |
| `expectations_established_before_current_conversion` | `true` |

The three anchors are pinned in source, and the caller must also supply each
expected hash:

| Anchor | SHA-256 |
|---|---|
| Acquisition receipt | `064d3c21…` |
| Resources index (metadata index of ZIP members, not poses) | `a05b65a6…` |
| Integrity record | `0ef84395…` |

`docs/prototype.md` records the historical (Oct 11) verification of 2569
members. Local historical integrity is **not** upstream authentication.

**Allowlist derivation.** The caller's `--expected-allowlist-sha256` must
identify a **parent-accepted** `filter-allowlist` output. Every later stage
fully re-validates the allowlist, but a re-hashed, structurally plausible
allowlist does not by itself prove that it was derived from the historical
index.

## Stages and CLI

Run as `python -B -S tools/prototype_bootstrap.py tools.workbench_hb_metric_inputs STAGE …`.
`-B` is required, so no bytecode is written.

| Stage | Arguments | What it does |
|---|---|---|
| `filter-allowlist` | `--acquisition-receipt/--resources-index/--integrity-record` with `--expected-*-sha256`, `--object-id N …`, `--out NEW_DIR` | **Metadata only** (see below). Writes `allowlist.json`. |
| `prepare-model` | `--allowlist P --expected-allowlist-sha256 H --object-id N --existing-input-bytes N --out NEW_DIR` | One allowlisted PLY → `hb_obj_00000N.glb` (metres) plus `model-receipt.json`. |
| `prepare-video` | `--allowlist P --expected-allowlist-sha256 H --existing-input-bytes N --out NEW_DIR` | 120 PNGs → `video.mp4` (pinned FFmpeg, pipes only) plus `video-receipt.json`. |
| `assemble` | Allowlist, model/video receipts, seed, selection and placement reviews and calibrations, each with its expected SHA, then `--out NEW_DIR` | Refused unless the seed and both reviews are present; otherwise independently re-validates everything. |

**`filter-allowlist`** hash-verifies the three anchors against the caller's
expected values and the source pins **before** parsing them. It then copies
only scene 000002 RGB 0..119 and the explicitly selected CAD ids (1..10).

- **Not copied:** no other row is copied or opened.
- **Zero payload opens:** the payloads are only `stat`ed.
- **Closure:** sources and runtime are recorded at stage start and compared
  again immediately before publication. A change aborts the success output.

**Exit codes:** 0 completed; 1 failure; 2 usage; 3 blocked (missing
prerequisite, unsupported input, or over a cap).

**One preparation scope.** Every stage accepts `--preparation-root DIR`. This
is **one explicit fresh root** inside `.cache/video-workbench/derived` for the
whole control, with one subdirectory per stage. The `Context.preparation`
field defaults to `derived` itself, which is meant for isolated fixtures.

- Each `--out` must be a new directory **inside** that scope.
- The allowlist and every receipt record their scope (`preparation_scope`).
- Each later stage, and assembly, refuses an allowlist or receipt produced for
  a different scope, so attempts can't be detached from the accounting.
- Source inputs stay outside the scope and remain in the input budget: the
  anchors, payloads, reviews, calibrations and scene annotation.

The receipt is published last. On failure the stage publishes a bounded
`failure.json` instead of a receipt. Retained partials and failure records are
never deleted and stay counted.

## Full allowlist re-validation (`validate_allowlist_doc`)

This runs inside `load_allowlist` **before any payload permission** is
granted. It requires:

- **Top level:** exactly the known fields, no more and no fewer, with the
  current schema and tool version.
- **Anchors:** exactly the three pinned anchors, each recorded as
  `verified_before_parsing: true`.
- **Fixed records:** the trust, flags and exclusion records must equal their
  exact values.
- **Source identity:** corpus, revision, scene and archives match, the
  selection digest is a hash, and the payload root sits beside the anchored
  index.
- **Source seals:** the acquisition source hashes must be a non-empty
  `name → sha256` map.
- **Frozen split:** exactly opening 0..9, seed 9, scoring 10..119.
- **Rows:** every RGB row (exactly 120, in order) and every model row (unique,
  sorted ids) must be exactly what `validate_index_row` produces:
  - integer types (booleans are invalid);
  - method 0 or 8, flags limited to `2048`;
  - sizes that are non-zero and bounded;
  - CRC-32 ≤ `0xFFFFFFFF`;
  - both SHA-256 values;
  - the role, payload location token, index/frame/split and identity note.
- **Totals:** the RGB totals are recomputed.

Missing, unknown or conflicting fields are rejected.

## Guard (every stage)

The guard is a process-wide audit hook that is active only inside a stage and
is reset on every exit path.

- **Refused at any time inside a stage:**
  - network access;
  - forbidden imports (torch, sam2, pyrender, trimesh and similar,
    `quality_gotrack*`, `workbench_render`, and the seed module);
  - any open of GT, label, index, mask, depth, evaluator, checkpoint or
    scene-metadata paths.
- **Payloads** can be opened only after the trust and budget checks pass, and
  only the exact allowlisted files, read-only.
- **Read exceptions are read-only.** The filter's anchors and the seed's single
  GT file cannot be opened for writing, removed, renamed, truncated, linked or
  have their mode changed.
- **Processes:** any process creation is refused, except one exact,
  single-use grant for the pinned FFmpeg. The grant names the executable path
  and the exact argv; on Windows the audited form is `list2cmdline(argv)`. It
  allows one `subprocess.Popen` and the `_winapi.CreateProcess` that follows,
  and is consumed by any attempt. Arbitrary `subprocess` calls fail.
- The stdlib platform cache is warmed before the guard, so no stage spawns
  `ver`.
- **Naming rule:** review files must not reuse denied names such as
  `selection.json`, `inference.json`, `evaluator.json` or `scene_gt*`.

## Strict metadata JSON

`strict_json` rejects all of the following anywhere in a parsed document:
duplicate keys, NaN and Infinity tokens, numeric overflow (`1e400`,
`-1e400`), and integer literals longer than 64 digits.

## Model (`prepare-model`)

**Payload checks.** The bounded payload stream checks the exact compressed
size and SHA-256, the exact expansion with no over-expansion, truncation,
trailing bytes or padding, the CRC-32 and the expanded SHA-256.

**PLY parsing.** The **streaming ASCII PLY parser** reads actual geometry. It
accepts `format ascii 1.0` with exactly vertex then face elements; `x y z`
(float), optional `nx ny nz` (float) and `red green blue [alpha]` (uchar);
and one `list uchar int|uint vertex_indices` per face with exactly 3 valid
indices.
- Every line must have an exact token count; values must be finite, with no
  trailing data or truncation.
- Counts are capped **before** allocation (at most 200 000 triangles, at most
  2 000 000 vertices).

**Vertex colours are required.** A colourless source is **blocked**; there is
no white-appearance fallback.

**Units.** The source is millimetres, following the BOP convention
(<https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md>);
a header declaring any other unit is refused. `mm * 0.001` is applied
**exactly once**, in float64. The float32 metre positions must be finite
**before** serialisation; overflow is refused.

**Normals.** Explicit normals are normalised in an overflow-safe way: each is
divided by its largest component first, so `1e308` works. The result must be
finite unit float32. Zero-length normals are **blocked**; nothing is fabricated
or silently omitted.

**GLB.** The GLB preserves origin, basis, winding and vertex colours, with a
white `KHR_materials_unlit` material and no UVs or texture.

**Round trip.** The GLB is checked with an independent reader (bitwise) and
with the producer `workbench_neural.glb_mesh`. Positions must match exactly,
and the triangles must equal the source minus its degenerate faces, which are
counted. Nothing is decimated.

**Caps and targets.** The GLB must be ≤ 5.5 MiB. The portability targets
(20 000 triangles, 20 000 000 bytes) are reported separately from the
execution cap. Object 1's header shows 197 212 faces, so its portable `FAIL` is
expected.

## Video (`prepare-video`)

**Encoder.** The pinned local CPU FFmpeg is
`.cache/quality-windows/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe`:

| Field | Value |
|---|---|
| Bytes | 64 654 336 |
| SHA-256 | `404fdd54…0d375ac` |
| Version | 4.2.2 |
| Build | gcc 9.2.1 20200122 |

Its size and SHA are verified before every use. Nothing is installed, and the
version comes from the hash-bound pin; the binary is not re-executed to read
it.

**Fixed profile** (`VIDEO_PROFILE`, frozen before any outcome):
- input: `rawvideo rgb24 640x480` on stdin;
- encoder: software `mpeg4` with `-pix_fmt yuv420p -q:v 5 -g 30 -bf 0 -threads 1`
  and bitexact flags;
- output: fragmented MP4 (`frag_keyframe+empty_moov+default_base_moof`) on
  stdout;
- `-protocol_whitelist pipe` on input and output, with no audio, subtitles,
  data or metadata;
- presentation rate 30 fps (synthetic, not a capture rate);
- no hardware or network.

**Pipe handling.** The main thread feeds the verified PNG pixels (native
640×480 8-bit RGB) to stdin. `write_frame` delivers **every byte**: a raw pipe
may accept fewer bytes than offered, so the remainder is offered again until
the whole 921 600-byte frame is written. Progress of zero, `None`, or more than
was offered is refused. The drain error, watchdog and deadline are checked
between writes, and a broken pipe reports how many bytes were delivered plus
the stderr tail.

One thread drains stdout to a single staging file. The 6 MiB cap, the
aggregate scope cap and the free-disk reserve are checked **before every
write**, and the staging writes themselves must report valid progress. Another
thread drains stderr into a bounded 8 KiB tail.

**Cleanup.** A watchdog kills FFmpeg at its 900 s deadline. On every path,
including overflow and errors, cleanup:
- kills and reaps the process (with a timeout);
- closes all three pipes, which also unblocks a stuck reader;
- joins the drain threads;
- closes the staging handle **only if the stdout drain thread no longer uses
  it**.

The result is recorded in `cleanup` (`reaped`, `returncode`, `threads_alive`,
`staging_fd_closed`, `pipes_closed`). Incomplete cleanup is raised when no
other error is in flight, and recorded in `failure.json` otherwise. A success
receipt carries `encoder_cleanup`. If startup itself fails, the same cleanup
runs.

**Decode check.** The derivative is reopened and decoded sequentially: exactly
120 frames, then end of video, at 640×480. It also gets the runtime
`FrameReader` access pattern: the first read seeks to `s000010`, which must be
frame-exact. Per frame, the receipt records:
- the source member and PNG SHA-256;
- the source pixel hash and the decoded pixel hash;
- the frozen split.

The output is lossy, so no pixel identity is claimed.

**Clock and runtime.** The clock is `FRAME_INDEX`, and physical fps and
timestamps are `null`. The runtime provenance includes the FFmpeg pin, and the
receipt includes the profile.

## Opening seed (`workbench_hb_opening_seed.py`)

This module is the only place that reads raw scene annotations. It is a
CONTROLLED_INIT exception, and the preparer never imports it.

**Before reading any annotation byte**, it requires:
- frame 9 only, which is checked before any path is touched;
- the expected GT, allowlist and selection-review hashes;
- the explicit object and instance, which must match the review and an
  allowlisted CAD id.

**Annotation file.** The GT file gets one read-only guard exception. It is
pre-hashed in full and then scanned. The **lexical** top-level scanner captures
only key `"9"`; other values are skipped without converting any number or
building any structure, and no full-file `json.load(s)` is used. It is a
lexical scanner, not a full JSON grammar validator for the skipped values.
Both the pre-hash (`scene_gt.sha256`) and the scan hash (`scan_sha256`) are
recorded and must match.

**Closure.** The seed module, sources, runtime, the allowlist and review files
and the scene annotation are recorded before the stage. They are compared
again before `seed.json` is written, and **again after `seed.json` is published,
immediately before `seed-receipt.json`**. Any change, even whitespace, aborts
the stage: the failure record is kept and no success receipt is written. The
seed binds the **expected** allowlist and review hashes, never a fresh rehash.
The scene annotation and the review must lie outside the preparation scope.

**Pose output.** R must be a proper SO(3), and t is converted mm → m once.
The output holds only the selected pose, with label
`controlled_init:dataset_onboarding:HB:…:s000009`, applied once to `s000010`
and never on recovery. The seed receipt records the closure (including the seed
module hash) and the ledger.

## Assemble

`assemble` is refused unless the seed and both reviews are present. It then
independently re-validates every prerequisite and does not trust status
strings. Each receipt's **nested structure is validated first**, so a
contradiction is rejected on its own merits rather than only because a receipt
sits outside the scope. The actual artifacts are re-checked after that. The
parent-accepted expected receipt hashes remain the trust boundary; the PLY is
not re-opened.

- **Model receipt:**
  - exact fields, producer tool version, source closure and runtime;
  - the object, member and hashes against the allowlist and the reviewed
    object;
  - units exactly `millimetres → metres`, factor 0.001, applied once;
  - the source header must contain positions and colours; `geometry.vertices`
    must equal the header vertex count, and the triangle count must equal the
    header face count (≤ 200 000);
  - the round-trip flags (positions, winding, colours and producer positions)
    must all be `true`, `normals_equal` must equal whether the source has
    normals, and `decimated` must be `false`;
  - the producer triangle count must equal the triangles minus the degenerate
    ones;
  - the portable result must be consistent with the count.

  Then the **actual GLB** is checked: bytes and size; exactly `POSITION`,
  `COLOR_0` and `NORMAL` when the source has normals (no UVs); no images,
  textures or samplers; one white unlit material without a texture; and
  `COLOR_0` as normalised RGBA8. Position, colour and index counts must match,
  every index must be in bounds, and the normals must be finite and unit
  length. A producer re-decode follows.
- **Video receipt:**
  - exact fields, producer, codec profile and pin, runtime and frozen split;
  - the clock, exactly: `FRAME_INDEX`, presentation 30, null physical fps and
    timestamps, and the fixed note that no 10-second prefix is attested;
  - the geometry, exactly: 640×480, no crop, no rotation, K unchanged;
  - the decode record, exactly: 120 sequential frames, end of video after 120,
    640×480, an exact `s000010` seek, no pixel-identity claim and the fixed
    note;
  - the process: exit code `0`, `stdout_bytes` equal to the video size, and a
    bounded stderr record;
  - the encoder cleanup: complete;
  - exactly 120 ordered frames with exact fields and allowlist PNG hashes.

  Then the **actual video** is checked: bytes and size, its workspace path, and
  a full re-decode whose pixel hashes must equal the receipt.
- **Seed and its receipt:** they bind the review, the allowlist, the current
  sources and runtime, the seed-module hash and the scope. The receipt must
  show both the pre-publication and post-publication closure.
- **Placement review:** it binds the seed, GLB, video and selection hashes and
  K. `physical_proof` must be `false`, and the principal-point evidence is
  `reviewed_assumption` or `documented_dataset_convention`.
- **Locations** (checked after the structure): the receipts sit beside their
  artifacts, all stage outputs lie inside the bound scope in distinct
  directories, and the reviews and calibrations lie outside it.
- **Aggregate accounting:** the whole preparation scope, including retained
  failed siblings, is checked before each assembly write.
- **Closure:** inputs, artifacts, sources, runtime and the seed module are
  compared before writing and again before the receipt; any change aborts.

**Outputs.**
- **Bundle:** a `workbench_metric_bundle_v1` bundle (family
  `hb_primesense_640x480`), checked with the core `MetricBundle.from_file`.
- **Clip proposal:** checked with the core `ClipSpec.from_dict`.

Both are **unadmitted and unregistered**.

## Caps and budget

**Per preparation scope:**
- all preparation outputs: **12 MiB**;
- all JSON metadata: **0.5 MiB**.

**What counts.** Before **every** encoded disk write (each publication and
each FFmpeg stdout chunk), `OutputLedger.check` walks the **whole preparation
scope** with `lstat` and never follows links; a link or non-regular entry
inside the scope is refused. It counts every regular file present:
- published outputs;
- **retained failed or partial sibling attempts**, including staging partials;
- `failure.json` records.

Existing usage plus the proposed bytes must stay ≤ 12 MiB in total and
≤ 0.5 MiB of `.json`. Normal writes keep 16 KiB (`FAILURE_RESERVE`) free so a
failure record can still be written.

**Transient aliases.** The owned temp of a publication and the video staging
file are transient aliases of the same bytes. Publication checks the final
size before the temp exists and removes the temp right after linking. The
video staging name is unlinked immediately after `video.mp4` is linked, so
later checks never count an owned alias twice. A *retained failed* staging
file is a real partial and is counted.

**Global budget, separate from the 12 MiB.** Unrelated workspace content is
**not** part of the per-preparation aggregate. Prospectively, before any
payload is opened, and again at every publication (owned temp plus published
name, so 2 × the bytes):
- derived + scratch + runs + masks + suite plus the stage allowance must stay
  ≤ 1 GiB;
- free disk must stay ≥ 1 GiB.

The ledger also checks the free-disk reserve on every write.

**Inputs:** the parent-declared existing inputs, plus the allowlisted
candidate payloads, plus 12 MiB must be ≤ 250 000 000. Anchors, payloads,
reviews, calibrations and the scene annotation are inputs; they must live
outside the scope and are never counted as outputs.

## Limitations

- **Upstream authentication:** none; trust is historical local integrity.
- **Receipt trust:** the parent-accepted expected receipt hashes remain the
  trust boundary. Assembly re-validates structure and the actual artifacts,
  but cannot prove a receipt was produced by a reviewed run.
- **Encoder pixel format:** `yuv420p` is requested and the parent verified the
  encoder supports it, but the encoder's internal format is not otherwise
  inspected.
- **Video size:** a real 120-frame encode might exceed 6 MiB at `q:v 5`. If so,
  the stage blocks; the profile is not tuned after the fact.
- **Object 1's GLB** may exceed 5.5 MiB, which blocks with no decimation.
- **Seed scanner:** for skipped values it checks lexical structure only.
- **Principal point:** a reviewed assumption or documented convention; never
  physical proof.
- **Stuck drain thread:** if the stdout drain thread survives cleanup, its
  staging handle is deliberately left open, and the stage reports incomplete
  cleanup rather than success.

## Pure APIs (for parent tests)

| Function | Purpose |
|---|---|
| `validate_allowlist_doc`, `build_allowlist`, `validate_index_row` | Allowlist construction and full re-validation |
| `verified_payload_stream` | Bounded payload verification |
| `parse_ply_header`, `AsciiPlyStream` | PLY parsing |
| `metre_positions`, `unit_normals`, `build_glb`, `read_glb_document`, `read_glb_arrays`, `verify_glb` | Geometry conversion and round trip |
| `png_rgb8_header`, `decode_png_rgb`, `codec_settings`, `ffmpeg_executable`, `FfmpegPipeEncoder` (`write_frame`, `_drain_stdout`, `_drain_stderr`, `cleanup`), `reopen_and_decode` | Video stage |
| `select_calibration`, `validate_selection_review`, `validate_seed`, `validate_placement_review`, `validate_model_receipt`, `validate_video_receipt` | Assembly checks |
| `Context(preparation=)`, `apply_preparation_root`, `preparation_scope`, `scope_usage`, `OutputLedger`, `publish_bytes`, `workspace_budget`, `input_budget` | Scope, caps and budget |
| `guarded_stage`, `allow_payloads`, `add_read_exception`, `strict_json` | Guard and parsing |
| Seed module: `TopLevelFrameExtractor`, `extract_frame_value`, `select_pose`, `extract_seed` | Opening seed |
