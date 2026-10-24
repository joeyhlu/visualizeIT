# HB cached-input preflight (`tools/workbench_hb_input_preflight.py`)

This tool checks whether **one** distinct, calibrated, already-cached RGB source could later
become a prepared-metric development control: HB `val_primesense` scene `000002`, at dataset
revision `985ebaef7b1bd4bffb2039f18dd81faa647fd060`. It checks only whether the **inputs** are
feasible.
- No head, phone, tracking, automatic-geometry or quality claim.
- The source object/instance binding and the opening seed stay **unverified**.
- No GT is read, nothing is admitted, and no geometry is verified.

**Execution status:** source only. Real-data stages are **withheld** until the 1 GiB free-disk
reserve and the stage allowance are restored (free space was about 100 MB at the time of
writing). Every stage checks the budget up front and refuses to run below the reserve.

## What the integrity checks do and do not show

- **Local consistency only.** The expectations come from the **locally captured** acquisition
  index (`acquisition-plan.json`): the ZIP central-directory CRC-32 and exact compressed and
  expanded sizes. That index is metadata captured on this machine, not an independently trusted
  source. Its `status` (`planned_only`) is recorded as declared and never trusted.
- **SHA-256 is observed only.** The index pins no SHA-256 for payload rows. Every SHA-256 here is
  **observed**, not independently expected. Every plan, mesh and preview output carries
  `cryptographic_source_identity_verified: false`, and none of them authenticates the source.
- **Future model controls** need a separately reviewed, independently sealed **SHA-only
  allowlist**. This tool does **not** look for one: it never opens `resources.json` or
  `evaluator.json`.

## Boundaries and guard

**Never read:**
- `scene_gt*`, `scene_camera`, `evaluator.json`, `inference.json`, `resources.json`,
  `selection.json`, `models_info.json`;
- results, depth, masks, checkpoints and raw meshes (`.ply`, `.obj`, `.glb`, `.gltf`);
- any non-selected payload.

The tool contains no GT arguments, discovery or extraction. Object identities come only from
member names in the local index or an explicit operator argument, never from scene GT.

**Path checks happen before any open:**
- Every input (including `--plan`) must be an existing, regular, non-link file inside the
  repository root. URIs and traversal are rejected.
- Any input that looks like a GT, label or model file (for example a `scene_gt.json` given
  directly as `--plan`) is rejected **before it is opened**.
- Output paths may not look like GT, label, model or payload paths, so the tool's own writes are
  never blocked by its guard.

**Whole-stage guard.** Each stage runs entirely inside `guarded_stage`, a process-wide audit hook
that is only active during a stage. An already active guard is refused: no nesting or re-entry.
- It begins with an **empty payload allowlist**. `allow_payloads` widens it only after the plan
  check and the prospective budget check pass, and only by the selected payload paths.
- The final closure check and the publication also happen inside the guard.
- **Refused:**
  - imports of torch, torchvision, SAM, OpenGL, pyrender, trimesh, open3d, pytorch3d, nvdiffrast
    or kaolin;
  - subprocess, `os.system`, exec/spawn, socket, `urllib` and `webbrowser` events;
  - **any-mode** opens (including `r+b` and `os.open(O_RDWR)`) of a payload not on the
    allowlist. Any `payloads/*.bin` counts, even before its payload root is known;
  - **any-mode** opens of forbidden GT, label, mask, depth, checkpoint or raw-mesh paths inside
    the repository root.

Each output lists the payload files it opened. OpenCV runs on the CPU only
(`cv2.ocl.setUseOpenCL(False)`).

**Source pins** cover only this tool, `bench/__init__.py`, `bench/vision.py` and
`bench/workbench_geometry.py`.

**Data handling:** no download, no source copies (the 46 MB of RGB stays in place), no changes to
original data, nothing expanded written to disk, and no video, mesh, GLB or seed conversion,
registration or admission.

## Frozen future candidate (fixed before anything is viewed)

| | Frames |
| --- | --- |
| Opening | indices 0..9 |
| Planned seed | index 9 (`seed: none`, unverified) |
| Scoring | indices 10..119 (110 frames) |

- **Clock:** `FRAME_INDEX`. The physical timestamp is `null`, and no FPS is assumed.
- **Calibration:** dataset-reference K for 640×480 (fx 537.4799, fy 536.1447, cx 318.8965,
  cy 238.3781, checked to within 1e-3). It is not a phone calibration. Registry identity fields
  are not copied.

## Stages and CLI

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Verified data preflight only |
| 1 | Failure |
| 2 | Usage error |
| 3 | Incomplete or unsupported |

Integer arguments accept decimal digits only: no sign, spaces, underscores or exponents.

```
python -B tools/workbench_hb_input_preflight.py plan \
    --acquisition-plan .cache/prototype-sprint/hb-compact-v1/acquisition-plan.json \
    --calibrations .cache/prototype-sprint/calibrations-hb-v1.json --scene 000002 \
    [--object-id 1..10] [--existing-input-bytes N] --out .cache/video-workbench/derived/<fresh>/hb-input-plan.json
python -B tools/workbench_hb_input_preflight.py inspect-mesh --plan <plan.json> --expected-plan-sha256 <hex> \
    --object-id 1..10 --out .cache/video-workbench/derived/<fresh>/mesh-header-N.json
python -B tools/workbench_hb_input_preflight.py preview-opening --plan <plan.json> --expected-plan-sha256 <hex> \
    --frames 0 9 --out .cache/video-workbench/derived/<fresh-dir>     # PARENT ONLY, after review
```

### `plan` (metadata only; zero payload opens)

1. **Prospective budget** before anything is read: a stat-only workspace check for two 256 KiB
   JSON outputs plus a 64 KiB directory reserve.
2. **Reads** the acquisition index (at most 8 MiB, strict JSON) and the calibration registry (at
   most 256 KiB).
3. **RGB checks.** The scene's members must be exactly `val_primesense/000002/rgb/000000.png` to
   `000119.png`. Each row must have:
   - revision `985ebaef…` and archive `hb_val_primesense.zip`;
   - the member name, role `rgb`, and location kind `payload`;
   - `payloads/<sha256("hb_val_primesense.zip:" + member)>.bin` as its location;
   - method 8 (raw DEFLATE) or 0 (stored) matching `sourceMethod`, the compressed size equal to
     `sourceCompressedBytes`, and no flags other than UTF-8;
   - an expanded size of at most 4 MiB;
   - a contained, regular, non-link payload whose **stat** size equals the indexed compressed
     size.

   The totals must be 46,130,618 encoded bytes and 110,835,240 expanded bytes.
4. **Model inventory** (`model_inventory`): the **required** IDs are **1..10**. The output
   reports `required_ids`, `listed_ids`, `cached_ids`, `missing_required_ids`,
   `extra_cached_ids` and `listed_not_cached_ids` separately.
   - `inventory_complete` is true only when **every** required ID is cached.
   - An extra ID (for example 11) never substitutes for a missing one, so {2..11} is **not**
     complete.
   - If the inventory is incomplete, the plan is still written and the exit code is 3.
   - `--object-id` must be a required, cached ID.
5. **New-input limit** (`input_budget`), **enforced**:
   - **Candidate check:** the candidate bytes (encoded RGB plus the candidate CAD) must be at most
     250,000,000. The candidate CAD is the explicit `--object-id` model, or else every cached
     required model. Exceeding the limit gives exit 3 and no plan.
   - **With `--existing-input-bytes`:** existing plus candidate must also be at most 250,000,000
     (`within_limit_with_parent_declared_existing`); otherwise exit 3.
   - **Without it:** `combined_compliance` is `unverified_parent_required`. Existing usage is
     never estimated or read from clip files.
   - The existing figure is stored in the plan and carried into every later rebuild.
6. **Output:** a plan under 256 KiB (`workbench_hb_input_preflight_plan_v2`) containing the input
   SHAs, sources and runtime, per-frame metadata and split, the inventory, the calibration and the
   input budget. Its flags include `compressed_content_verified: false`,
   `cryptographic_source_identity_verified: false`, `seed: none`,
   `instance_binding: unverified`, `neural_admitted: false` and `geometry_verified: false`.

### `inspect-mesh` (one required candidate CAD payload; header only)

`--object-id` must be in 1..10. Everything below runs inside the guard, which starts with no
payloads allowed.
1. Checks the plan: the external SHA, then **exact equality with a fresh metadata rebuild**.
2. Takes a closure snapshot: the plan, acquisition index and calibration SHAs, sources, runtime
   and payload stat.
3. **Prospective budget** check, then allows only the selected payload.
4. **Bounded stream:** reads 1 MiB chunks, expands at most 1 MiB + 1 byte per step and keeps only
   the first 64 KiB. It fully consumes the member to check:
   - the exact compressed size;
   - no over-expansion, truncation or trailing bytes;
   - the exact expanded size and the CRC-32.

   It records both SHA-256 values as observed. No arrays are built and nothing is written to disk.

   **Memory:** the receipt's `memory` block lists the chunk limits and
   `payload_buffer_budget_estimate_bytes = 6·(CHUNK + 1) + HEADER_CAP + 256 KiB`. This is a
   conservative bookkeeping **estimate** covering the stream chunks, the header buffer and an
   allowance for zlib. It is **not** a certified absolute peak and not an RSS measurement. The
   original chunk, a copied unconsumed tail, the expanded output, temporaries and zlib state can
   overlap. It also excludes the Python interpreter, the runtime and other stage memory
   (`absolute_peak_measured: false`).
5. **Parses the PLY header only:**
   - `ply` magic, then the format line, then `end_header` within 64 KiB, in ASCII;
   - counts must be decimal non-negative integers;
   - duplicate elements or properties and unknown keywords are rejected.
6. **Feasibility for a future reviewed conversion:**
   - encoding `ascii` or `binary_little_endian`, version 1.0;
   - only `vertex` and `face` elements;
   - float `x`/`y`/`z` and an integer `vertex_indices` face list (omitting faces cannot bypass the
     cap);
   - at most 200,000 faces and non-empty geometry;
   - appearance from uchar vertex RGB. Texture coordinates or a texture file give
     `texture_image_not_in_cached_inventory`; no appearance gives `missing_appearance`;
   - units from the header, or `assumed_dataset_convention` (millimetres, source evidence only,
     not converted). An unparseable unit comment gives `unknown_units`;
   - a malformed header is not feasible.

   The result is never admission or verified geometry. Exit 0 if feasible, 3 if not.
7. Takes a second closure snapshot, then the remaining-growth budget recheck and the exclusive
   publication, all inside the guard.

### `preview-opening` (parent-run only after review)

1. **Before any file is opened:** `--frames` must be 1..10 unique indices within 0..9. Index 10
   or higher gives exit 2.
2. Inside the guard (empty allowlist): plan check and closure snapshot, then a **prospective
   budget** of 2 MiB for outputs plus 64 KiB for a failure receipt plus 64 KiB for directory
   entries, then only the selected opening payloads are allowed.
3. For each frame:
   - the bounded stream (CRC-32, sizes, observed SHAs), buffering at most the indexed size and
     never more than 4 MiB;
   - a strict PNG check: the signature, IHDR first with a valid CRC, **640×480, 8-bit, colour
     type 2**, standard compression and filter, ends with IEND;
   - `cv2.imdecode(IMREAD_UNCHANGED)`, which must give exactly `(480, 640, 3)` uint8 BGR,
     converted to RGB;
   - the semantic hash `array_sha256(rgb)`.
4. Builds one labelled contact sheet: 192×144 tiles at an exact 0.3 scale, at most 1024 px wide,
   with these labels: candidate input only, not reviewed; FRAME_INDEX clock; physical time
   unknown; neural unavailable; no GT/masks/poses/object identity; `*` = planned seed index
   (unverified); dataset K, not a phone calibration.
5. Takes a second closure snapshot, then checks the remaining-growth budget: the actual bytes,
   the failure reserve and the directory reserve, against the 2 MiB aggregate.
6. Stages `opening-contact-sheet.png`, then `preview-receipt.json` (written **last**), and renames
   the staging directory to `<out>` only if the target is still absent.

The receipt records per-frame integrity, the PNG structure, the decoded RGB hash, the
source-to-thumbnail geometry (`x_t = x0 + (u + 0.5)·0.3 − 0.5`), the contact-sheet SHA,
`payload_opens` and the labels. No native PNG frames, masks, poses, object names, GT or timings
are written, and no automatic selection or object identification is made.

**Failure.** Before the staging directory exists, nothing is written. After it exists, a bounded
`failure.json` is written (`status: failed`, `completed: false`, the stage, the error and the
staged file names) and the directory becomes `<out>.failed-<token>`. If that fails, the staging
directory is kept as evidence. A failed attempt is never promoted or retried over.

## Outputs and budgets

- Outputs go to **new** paths inside `.cache/video-workbench/derived`.
- **JSON outputs** (at most 256 KiB): written to an owned temporary file with `O_EXCL`, fsynced
  and SHA-verified, then exclusively linked (or renamed only if the target is absent). The
  temporary file is always removed.
- **Workspace budget** (stat only; no clip JSON or other contents read): derived, scratch, runs,
  masks and suite plus the planned bytes must stay within 1 GiB, and at least 1 GiB must remain
  free after the write. It is checked **prospectively, before any payload open**, and again for
  the actual remaining growth before publication.

## Public API (fixture-friendly)

**`Context(root, workspace)`** sets explicit roots for synthetic fixtures.

**Stage functions:**
- `plan_stage(ctx, acquisition_plan, calibrations, scene, out, explicit_object_id=None, existing_input_bytes=None)`
- `inspect_mesh(ctx, plan, expected_sha256, object_id, out)`
- `preview_opening(ctx, plan, expected_sha256, frames, out, decode=None)`

**Helpers:**

| Function | Notes |
| --- | --- |
| `build_plan(ctx, acquisition_plan, calibrations, scene, explicit_object_id, existing_input_bytes)` | |
| `model_inventory(models)` | |
| `input_budget(candidate_bytes, existing_input_bytes)` | |
| `select_calibration(registry)` | |
| `validate_member_row(...)` | |
| `verified_member_stream(path, entry, sink=None)` | |
| `parse_ply_header(prefix)` | |
| `evaluate_header(header)` | |
| `validate_frames(frames)` | |
| `check_png_structure(raw)` | |
| `decode_rgb(raw, decode=None)` | |
| `build_contact_sheet(frames, scene)` | |
| `load_plan(...)` | |
| `closure(...)` | |
| `workspace_budget(ctx, planned)` | |
| `check_output(...)` | |
| `publish_file(...)` | |
| `contained_input(...)` | |
| `safe_member_path(...)` | |
| `denied_path(path)` | |
| `guarded_stage(ctx)` | |
| `allow_payloads(payload_root, paths)` | |
| `strict_nonnegative_int(text)` | |

**Exceptions:** `PreflightError` (exit 1), `UsageError` (2), `Unsupported` (3), and
`GuardError` (a `PreflightError`).

**Changes from v1:**
- schema and tool version 2;
- `guarded_stage(ctx)` no longer takes a payload root;
- the observed SHA fields are renamed `compressed_sha256_observed` and
  `expanded_sha256_observed`;
- outputs carry `payload_integrity` instead of `payload`.

## Not in this packet

Seed extraction, mesh decode or conversion, video encoding, registration, admission and any
sealed SHA allowlist each need their own reviewed packet.

**Worker verification:** a syntax and import check only. No stage was run, and no acquisition
index, registry, payload, GT or model was opened.
