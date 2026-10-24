# HB scene 000003 cached-prefix scout

Tool: `tools/workbench_hb_prefix_scout.py`. It covers inputs only, runs on the
CPU, and is a **candidate scout**: no object binding, admission, registration,
neural or GPU run, GT, quality claim or mobile claim.

## Why scene 000003

The actual scene-000002 opening frames 0 and 9 show two hand tools and an
iron, but converted model 1 is a bear, so the visual instance binding was
**refused**. Previewing the six allowlisted CAD candidates (1, 3, 4, 5, 6, 9)
found no match. Candidates 2, 7, 8 and 10 exceed the 200 k execution cap and
were not previewed. The scout does not guess ids, relax caps, decimate,
extract poses or judge model quality. The frozen six-case suite and head
`27e06a51…` are unchanged.

Scene **000003** was chosen *prospectively*: it is the first unreviewed scene
in ascending order and has the smallest cached encoded footprint, 120 RGB
frames totalling **35 632 589 bytes**. This is a new source scope, not a retry
of the closed scene-000002 metric, prefix or boundary tools. The scout never
changes those tools' constants and never calls their stages. It reuses only
their pure helpers: `validate_index_row`, `verified_payload_stream`,
`strict_json`, the path-containment helpers, the PNG checks, the guard and the
budget functions. It never imports the opening-seed module.

## Trust

The three historical anchors are pinned (`workbench_hb_metric_inputs.ANCHORS`)
and hash-verified **before** any parsing:

| Anchor | Path | SHA-256 |
|---|---|---|
| Acquisition receipt | `.cache/prototype-sprint/hb-compact-v1/acquisition-receipt.json` | `064d3c21…` |
| Resources index | `.cache/prototype-sprint/hb-compact-v1/resources.json` | `a05b65a6…` |
| Integrity record | `.cache/prototype-sprint/hb-integrity-v1.json` | `0ef84395…` |

The receipt, index and integrity record must bind each other: status, revision,
index hash, selection digest, the 2569-member count and the source seals. This
is local historical integrity, **not** upstream authentication.

## Stages and CLI

Run with `python -B -S tools/prototype_bootstrap.py tools.workbench_hb_prefix_scout STAGE …`
(`-B` required).

```
plan --acquisition-receipt P --expected-receipt-sha256 H
     --resources-index P --expected-index-sha256 H
     --integrity-record P --expected-integrity-sha256 H
     --existing-input-bytes N --scout-root DIR --out DIR/NEW
preview-opening --plan P --expected-plan-sha256 H [--frame 0] [--frame 9]
                --scout-root DIR --out DIR/NEW
```

Exit codes: 0 completed; 1 failure; 2 usage; 3 blocked.

### `plan` (metadata only, zero payload opens)

1. The expected anchor hashes must equal the pins, or the stage is blocked.
   The anchors must be regular, contained files without links, outside the
   scout scope.
2. The guard starts with an **empty payload allowlist**. The anchors are exact
   **read-only** exceptions; they can't be written, removed, renamed or
   truncated.
3. Prospective budgets are checked (see Budgets).
4. The anchor bytes are hashed against the pins, then parsed strictly.
5. The rows must be **exactly** `val_primesense/000003/rgb/000000.png` …
   `000119.png` in `hb_val_primesense.zip`, with no extra or missing scene-3 RGB
   members. Each row must pass `validate_index_row`:
   - revision, archive, member and role;
   - method 0 or 8, flags limited to `2048`;
   - non-zero, bounded sizes and a valid CRC;
   - both SHA-256 values and the payload location token.
6. The encoded total must equal 35 632 589.
7. The payloads are checked by `lstat` only: each must be a regular file of
   the planned size.
8. The plan records:
   - the frozen split: opening 0..9, seed index 9, future scoring 10..119, and
     preview authorisation for indices 0 and 9 only;
   - the clock: `FRAME_INDEX`, with physical fps and timestamps `null` and no
     ten-real-seconds claim;
   - the calibration member `val_primesense/000003/scene_camera.json`, listed
     but **not opened**, with status
     `PENDING_scene_specific_K_only_verification`. An object-labelled
     calibration registry row is not proof of this scene's K.

   It records no masks, poses or object id. GT, depth, masks and model rows are
   never copied or opened.
9. The anchors, sources and runtime are compared again before the plan is
   published as `scout-plan.json` (≤ 192 KiB), and **again after
   publication** (see Post-publication closure).

### `preview-opening`

1. **Frame check first.** The request must be **exactly** the prospectively
   authorised pair 0 and 9, in either order and each once. A subset such as
   `[0]` or `[9]`, a duplicate, a non-integer and above all 10 are refused
   before any path is touched or payload opened. With no `--frame`, the CLI
   asks for the pair.
2. **Plan location, unread.** Before anything is read, `--plan` must be an
   existing, regular, link-free file inside the repository that is named
   exactly `scout-plan.json` and lies inside the scout scope. A payload, an
   anchor or any other file is refused at this point without a byte being
   read.
3. **Plan check, inside the guard.** The whole-stage guard starts with an
   **empty payload allowlist and no read exceptions**. The location check is
   repeated, then the plan is read (bounded) and must match
   `--expected-plan-sha256` (a parent-accepted plan) before it is parsed.
   Only after the plan's structure is validated are its three anchor paths
   added as exact **read-only** exceptions (`add_read_exception`). The plan is
   then fully re-validated:
   - its exact fields, schema, flags, pins, split, clock and notes;
   - every row against `validate_index_row`, and the totals;
   - its scout scope and location;
   - its source closure and runtime, which must equal the current ones;
   - a **fresh rebuild** of the metadata core from the re-verified anchors,
     which must be byte-identical to the plan;
   - the payload stats.
4. **Budgets** are checked prospectively, as the sequence sheet (≤ 768 KiB),
   then receipt (≤ 128 KiB), with transient peaks. Only then are payload
   permissions widened, to exactly the authorised frames.
5. **Each frame** is streamed and verified: exact compressed size and SHA-256,
   exact expansion with no trailing data, CRC-32 and the expanded SHA-256
   against the historical expectation. It must be an 8-bit RGB PNG of 640×480
   ending with IEND, and its pixels are decoded. At most two frames are held in
   memory, and no frame or payload is written to disk.
6. **One contact sheet** is written, `opening-contact-sheet.png`, 1008 px wide
   (≤ 1024), lossless PNG at compression level 9, capped at 768 KiB.
   `build_contact_sheet` itself accepts **only** frames 0 then 9 (anything
   else raises `InputsError`), so the banner's claim that both are shown is
   always true. The sheet has:
   - two thumbnails at scale 0.75 (480×360, `INTER_AREA`);
   - each labelled with its frame id and index, and frame 9 marked as the
     planned seed;
   - a banner stating, wrapped to fit and **never clipped**: scene 000003;
     the opening frame identities with no future frames; `FRAME_INDEX` with
     physical time and FPS unknown and no ten-real-seconds claim; K pending;
     candidate only, with no binding, admission or quality claim. A word that
     can't fit is an error.
7. **Receipt.** `preview-receipt.json` is published last. Per frame it records
   the index, `frame_id`, split, member, PNG SHA-256, compressed SHA-256,
   CRC-32, native 640×480 size and the decoded RGB semantic SHA-256
   (`array_sha256`). It also records:
   - the sheet's SHA, bytes, size and semantic SHA;
   - the thumbnail mapping (`sheet = offset + (native + 0.5) * scale - 0.5`
     per axis, with each frame's offset);
   - the banner lines;
   - the opened-payload count, which must equal the authorised frames;
   - the closure: plan, anchors, sources and runtime at the start, a
     pre-publication match, and the post-publication rule;
   - the budgets.

## Post-publication closure

The closure interval ends **after** the final publication, not before it.
Once `scout-plan.json` or `preview-receipt.json` has been published, still
inside the guard, the stage recomputes:

- the anchor hashes, the source closure and the runtime (plus the plan hash
  for the preview);
- the exact set of names in its output directory;
- the current bytes of each of its own outputs.

Everything must equal the values at the start and the published hashes. On
any mismatch, and on any other error after the output directory was created,
the stage fails as described below. This includes an error raised **inside the
publisher after its hard link created the final name**, such as its final hash
read, and an error raised when the guard closes.

- **Ownership** is established when the name is actually created, never by
  pre-registering it. `mi.publish_bytes` calls `ScoutLedger.record` right
  after its own `os.link` succeeded and before its final hash check. At that
  moment the ledger records the name together with the file identity
  (`st_dev`, `st_ino`). If a foreign file already holds the name, the
  publisher refuses before linking, so that name is never recorded.
- **An owned authoritative output** (`scout-plan.json` or
  `preview-receipt.json`) is renamed to `invalidated-scout-plan.json` or
  `invalidated-preview-receipt.json`, but only if it is still the very file
  this stage created. Its bytes are kept, but it is no longer the completed
  artifact. If its hash can't be read, the error is recorded and the rename
  still happens. Only if the rename fails is the owned file removed, and that
  outcome is recorded.
- **Foreign files are never renamed or deleted.** A file that already held
  the name, or that replaced our file under it, is left untouched and listed
  in `foreign_names_untouched`.
- **Other artifacts.** Non-authoritative artifacts (the contact sheet) and
  every other retained file stay as they are.
- **Failure record.** A `failure.json` (`workbench_hb_prefix_scout_failure_v1`)
  records the reason, `completed_receipt_published: false`,
  `post_publication_failure`, the invalidated outputs with their SHA-256 (or
  the hash error), `foreign_names_untouched`, and the retained files.
- **No success.** The stage re-raises the original error (`InputsError` for
  drift); it never returns success.

The earlier pre-publication closure checks are unchanged.

The failure record is always valid JSON and never larger than 16 KiB: the
details are shortened, never cut mid-JSON. Before it is written, the stage
runs a **fresh global check**: `workspace_budget` with 2 × the record's bytes
(temporary file plus published name), covering derived, scratch, runs, masks
and suite against 1 GiB plus the free-disk reserve. This check is separate
from the scope-ledger check that `publish_bytes(failure=True)` performs. If
the record can't be published because of any cap, an existing name or a
write error, nothing is written to disk. Stderr then gets an explicit
`failure.json NOT published (…)` line followed by the full record as
single-line JSON, and the original exception is still raised.

## Budgets

These are checked before any payload is opened, and the ledger checks are
repeated before every encoded write.

- **Scout scope:** a fresh `--scout-root` inside
  `.cache/video-workbench/derived`. Every output is a new directory inside it.
  The whole scope is walked with `lstat` and links are never followed. The
  total, including the plan, the preview, and retained failed or sibling
  attempts and their `failure.json`, must stay **≤ 2 MiB**, of which JSON
  metadata **≤ 512 KiB**.
  - **Transient peak.** `publish_bytes` keeps an owned temporary file and its
    hard-linked published name side by side until the temporary is removed,
    and both are counted. Every ledger check therefore requires
    `usage + earlier outputs + 2 × bytes` to fit, for normal and failure
    publication alike. Prospective checks use the stage caps (plan 192 KiB;
    sheet 768 KiB, then receipt 128 KiB), and every actual write is
    re-checked.
  - **Failure reserve.** Normal writes keep 2 × 16 KiB free, the transient
    peak of the largest failure record.
  - **No cap is relaxed.** In practice one scope holds one plan, one preview
    and a few failure records. Further attempts are refused once the
    metadata cap would be crossed and then need a fresh scope.
- **Global:** derived, scratch, runs, masks and suite plus the stage allowance
  (2 × the stage's prospective outputs) must stay ≤ 1 GiB, and actual free
  disk minus the allowance must stay ≥ 1 GiB. These are real filesystem checks
  and are never skipped.
- **New inputs:** `existing_input_bytes` (required; a non-negative integer,
  booleans refused; the actual figure is 136 098 466) plus the scene-3
  candidate (35 632 589) plus the scout cap (2 MiB) plus the future 12 MiB
  preparation reserve must be ≤ 250 000 000. The value is frozen into the plan
  and re-checked by the preview.
- **Unchanged limits:** the 12 MiB metric-preparation cap, the 8 GiB model
  cache and the 256 MiB public limit are not touched.

Outputs are exclusive and fresh, never overwritten, and failure records are
kept with their reason. Apart from the budget numbers, outputs are
deterministic, and the source closure is versioned. Missing or refused data is
never written as zeros.

## Library API (for parent tests)

| Item | Purpose |
|---|---|
| `ScoutContext(root, workspace, scout_root=None, synthetic=None)` | Injectable roots; `.mi()` gives the guarded `workbench_hb_metric_inputs.Context`. |
| `SyntheticPins(anchors, integrity_members, rgb_encoded_bytes, rgb_expanded_bytes=None)` | **Test-only** replacement for the historical pins. The CLI never sets it, and outputs record `pins.kind = "synthetic_test_only"`. |
| `plan_stage(ctx, *, receipt_path, expected_receipt_sha256, index_path, expected_index_sha256, integrity_path, expected_integrity_sha256, existing_input_bytes, out)` | The plan stage. |
| `preview_stage(ctx, *, plan, expected_plan_sha256, frames=(0, 9), out)` | The preview stage. |
| `validate_anchor_documents`, `build_plan_core`, `validate_plan_doc`, `check_preview_frames` (exactly {0, 9}; returns `(0, 9)`), `check_existing_input_bytes`, `scout_input_budget`, `wrap_text`, `build_contact_sheet` (exactly frames 0 then 9) | Pure helpers. |
| `ScoutLedger(mctx, out_dir, stage)` with `.check(nbytes, *, metadata, failure=False)`, `.check_sequence([(nbytes, metadata), …], *, failure=False)`, `.record(name, nbytes, *, metadata)` (records ownership and file identity after the publisher's link), `.owned_file(name)` and `.owned` | Scope ledger with 2× transient accounting and created-name ownership. |
| `AUTHORITATIVE_NAMES = (PLAN_NAME, RECEIPT_NAME)` | The only names a failing stage may invalidate, and only when owned. |
| `main(argv, ctx)` | The CLI. |
| Constants `PLAN_CAP = 192 KiB`, `SHEET_CAP = 768 KiB`, `RECEIPT_CAP = 128 KiB`, `FAILURE_RESERVE = 16 KiB`, `SHEET_PNG_COMPRESSION = 9`, `INVALIDATED_PREFIX = "invalidated-"`, `SCHEMA_FAILURE` | Caps and names. |

**Fixture schema** (as in `tools/test_workbench_hb_prefix_scout.py`): a resources
index with `schemaVersion 1`, corpus `HB`, the dataset revision, a selection
digest, `storageKind original_zip_members`, `resources` and
`acquisitionSourceHashes`.

- `resources` holds 120 rows `val_primesense/000003/rgb/NNNNNN.png` (raw
  DEFLATE or stored payloads at `payloads/<sha256(archive:member)>.bin`), plus
  any number of never-opened sentinels such as `scene_gt.json` and
  `scene_camera.json`.
- The receipt and integrity record bind the index SHA, revision, selection and
  member count.
- Tests pass `SyntheticPins` with the fixture's anchor hashes, member count and
  totals.

## Limitations

- **Trust:** historical local integrity only; no upstream authentication.
- **Calibration:** K stays pending; `scene_camera.json` is listed, never opened.
- **What it shows:** the preview is a visual candidate scout. It establishes no
  object identity, CAD association, pose, mask or seed, and grants no
  admission.
- **Clock:** `FRAME_INDEX` only, with physical time unknown.
- **Pixels:** the sheet thumbnails are resampled (`INTER_AREA`, scale 0.75);
  the native pixel identities are the recorded PNG and semantic hashes.
