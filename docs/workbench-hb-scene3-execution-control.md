# HB scene 000003 execution control: unique-instance controlled seed

Tool: `tools/workbench_hb_scene3_seed.py`. Tests: `tools/test_workbench_hb_scene3_seed.py`
(written separately).

This is a **controlled-init ground-truth seed**. One tracker pose for opening
frame 9 is taken from the historical BOP `scene_gt.json`. It is **not**:

- automatic RGB geometry, segmentation or neural execution;
- an evaluator-free run;
- independent accuracy.

It never assembles, registers or admits anything, never publishes a
ClipSpec, SurfacePackage or bundle, never places artwork or runs SAM, never
chooses a principal-point origin, and never operates on K. The raw K's
principal-point origin stays **UNVERIFIED**.

The accepted inputs are immutable, and so are the old scene-2 tools. Only
two pieces of `tools/workbench_hb_opening_seed.py` are reused:
`TopLevelFrameExtractor('9')` and `select_pose()`. The old `extract_seed`,
`mi.load_allowlist` and the assembler are not called. No global is mutated.
Helper assumptions are asserted at every stage: frame key `"9"`, at most 64
instances, nesting depth at most 64, a 1 MiB capture limit, mm→m 0.001,
scene 000003 and the media assumptions.

## Pins

| Input | Production pin |
|---|---|
| Prerequisites receipt `.cache/video-workbench/derived/hb-metric-control-000003-v1/prerequisites-v1/prerequisites-receipt.json` | `f11b9a3c41da9dc499c8e2d5fff766e018facc73af1ce3620ca9e8842b108681` |
| Frozen parent policy `.cache/video-workbench/derived/validation/hb-scene3-controlled-seed-policy-v1.json` (1 332 bytes) | `063d4f59cc2293bfd5dae264e6957125e3c283dac8c294a5311228f70255ff72` |
| Accepted media allowlist / model receipt / video receipt (bound through the prerequisites receipt) | `e2dfa6f7…` / `106048f7…` / `67ce007e…` |
| Profile, K, opening association, scout outputs, three anchors | the accepted media pins (`workbench_hb_scene3_media.pins`) |

The opening association stays unchanged, with `instance_ordinal: null`. The
policy was frozen **before** any annotation access. It authorises exactly
one `obj_id == 4` record in frame 9, and otherwise the stage refuses. The
source-order ordinal is derived only after uniqueness is established. There
are **no** object, frame or ordinal overrides in the API or CLI.

### Test-only pins

```python
SyntheticSeedPins(media: workbench_hb_scene3_media.SyntheticMediaPins,
                  prerequisites_sha256: str, policy_review_sha256: str)
SeedContext(root, workspace, preparation, test_only_synthetic_pins=None)
```

With synthetic pins, the allowlist, model-receipt and video-receipt hashes
are taken from the pinned prerequisites receipt. Outputs are labelled
`TEST_ONLY` / `synthetic_test_only`, and production refuses them; a
production context refused a TEST_ONLY prerequisites receipt in the checks.
The CLI can't set test pins.

## API and CLI

```python
plan_seed(ctx, *, prerequisites, expected_prerequisites_sha256, policy_review,
          expected_policy_review_sha256, out)        # -> {plan, plan_sha256, label, annotation_opens: 0, audited_opens_total, exit: 0}
extract_unique_seed(ctx, *, plan, expected_plan_sha256, out)
                                                     # -> {seed, seed_sha256, binding, binding_sha256, seed_label,
                                                     #     instance_ordinal, label, annotation_opens: 4, audited_opens_total, exit: 0}
```

```
python -B -S tools/prototype_bootstrap.py tools.workbench_hb_scene3_seed plan-seed \
    --prerequisites P --expected-prerequisites-sha256 H --policy-review P --expected-policy-review-sha256 H \
    --out .cache/video-workbench/derived/hb-metric-control-000003-v1/controlled-seed-v1/plan
python -B -S tools/prototype_bootstrap.py tools.workbench_hb_scene3_seed extract-unique-seed \
    --plan …/controlled-seed-v1/plan/seed-plan.json --expected-plan-sha256 H \
    --out …/controlled-seed-v1/seed
```

Running without `-B` is refused. `--out` must be a **new** directory directly
inside `controlled-seed-v1/` of the preparation scope. A refusal prints a
one-line JSON exit record to stderr: `{stage, status:
usage|blocked|error, exit: 2|3|1, reason}`.

**Pure helpers (exposed for tests):**

- `validate_policy`, `validate_prerequisites`;
- `validate_annotation_row(row, *, root, canonical, max_bytes=8 MiB)`;
- `decode_frame_value(captured_bytes)`, `select_unique_seed(frame)`;
- `annotation_pass(path, descriptor, *, scanner=None)`;
- `SeedLedger`.

## `plan-seed`: metadata only (`seed-plan.json`, `workbench_hb_scene3_seed_plan_v1`)

1. **Pins first.** The expected hashes must equal the pins, or the stage is
   blocked. The prerequisites receipt (named `prerequisites-receipt.json`,
   inside the scope) and the policy (`.json`, outside the scope) are located
   without being opened.
2. **Guard and budgets.** The whole stage runs in the shared guard (empty
   payload allowlist, no annotation exception). Budgets are checked
   prospectively **before any read**.
3. **Prerequisites receipt** (read pinned, hash before parse) must have the
   exact key set, a `prepared_media_only` status and the media flags, trust,
   scope, K origin and unresolved list. It must record its reconstructions
   and bind the pinned profile, K and association (plus, in production, the
   pinned allowlist and model and video receipts). Its association must be
   CAD 4 with ordinal null.
4. **Policy** (read pinned) must match the exact frozen schema: CAD 4,
   frame 9, the s000010 single use, no reuse on recovery, no scored values,
   no automatic access, the bound association and prerequisites, caps of
   8 MiB and 64 KiB, and no quality or phone claims. Booleans are refused as
   integers.
5. **Re-authentication through the accepted media helpers.**
   `media._load_allowlist` and `media._reauthenticate` re-read the profile,
   K, association, scout outputs and anchors and re-derive the allowlist
   core. The model and video **receipt JSON** files are re-hashed (bounded);
   the GLB, video and RGB are **never** reopened.
6. **Annotation descriptor.** The descriptor comes only from one more pinned
   index read. The `val_primesense/000003/scene_gt.json` reuse row must have:
   - the revision and archive `hb_val_primesense.zip`;
   - exactly one accepted **original cached role**: `metadata` (what the
     pinned historical index actually records for this member, confirmed by
     the parent's metadata-only audit), or `evaluation`/`annotation`
     (manufactured fixtures);
   - stored method 0, with stored size = expanded size and stored SHA =
     expanded SHA;
   - a valid CRC-32, flags and original ZIP provenance;
   - an expanded size of at most **8 MiB**;
   - the canonical path `<reuse_root>/val_primesense/000003/scene_gt.json`.
     The real index records it as an absolute path, which is accepted only
     if it resolves to exactly that canonical file.

   **The role** is preserved verbatim as `role`, and `semantic_role:
   controlled_ground_truth_annotation` is added for every accepted role,
   with a `role_disclosure`. The original `metadata` value does **not** grant
   generic metadata access or automatic RGB access. The basename GT denial
   in the guard stays, and only this one exact canonical file is ever granted
   (in `extract-unique-seed`, after policy, prerequisites and index
   authentication). Any other role (`rgb`, `model`, `depth`, `calibration`,
   a case variant, or two roles) is refused. No global role changes.

   No caller hash can create trust here.
7. **Annotation file check.** The file is checked with `lstat` **only**: it
   must be regular, link-free, outside the scope, of the index size, and at
   most 8 MiB. No read exception is granted, and the stage asserts zero
   annotation, camera and payload opens.
8. **Plan contents.** The plan records:
   - the policy, inputs and bound outputs (hashes only; `reopened: false`);
   - the anchors and the annotation descriptor, with the four planned passes;
   - trust and flags;
   - the input budget, unresolved list, sources, runtime, budget, reads and
     closure.

   Trust is `historical_local_integrity` with `upstream_authenticated:
   false`. The flags record `annotation_opens: 0` and
   `initial_ground_truth_read: false`.

   **Input budget:** the allowlist's bound total (base 187 459 695 plus
   association 2 822) plus the actual policy bytes plus the actual
   annotation bytes, which must stay ≤ 250 000 000. Cached payloads are not
   double-counted.
9. **Closure.** Before and after publication, every input is hashed
   (bounded), along with the sources and runtime and the annotation's
   `lstat` identity.

## `extract-unique-seed` (`controlled-seed.json` + `seed-binding.json`)

1. **Guard before reading.** The guard is active **before** any metadata is
   read. Budgets are checked prospectively for the seed (8 KiB) and binding
   (16 KiB).
2. **Plan check.** The plan must be named `seed-plan.json` and lie in the
   seed subtree. It is read pinned, and its label and pins kind must match
   (a TEST_ONLY plan can't pass as production). Its sources and runtime must
   equal the current ones **type-strictly** (`true` vs `1` differ). Then the
   **whole plan core is re-derived** from the paths it names under the pins
   (steps 3–7 above, which re-authenticate all 13 input files) and must be
   byte-identical.
3. **Plan evidence** (`validate_plan_evidence`). The plan's non-core claims
   are checked strictly, **before** the annotation grant:
   - **reads:**
     - the exact key set, the exact snapshot text, and
       `annotation_opens`/`camera_opens`/`payload_opens` exactly integer 0;
     - `audited_opens` exactly 2 reads each of prerequisites, policy,
       allowlist, model receipt, video receipt, profile, `camera_k`,
       association, scout plan, scout receipt, the acquisition-receipt anchor
       and the integrity anchor, plus 4 reads of the resources index. No
       camera, annotation, payload or stage-output label is allowed.
     - `post_publication_planned` exactly 13 input hashes and 1 output hash;
   - **budget** (`media._check_budget`): exact keys, an exact-int fixed
     allowance `TRANSIENT_FACTOR × PLAN_CAP + DIR_ENTRY_RESERVE`, the exact
     limits and method, with historical derived and free values allowed to
     vary only within valid bounds;
   - **closure:** exact keys, `before_files` equal to the authenticated
     pinned hashes, the fixed annotation-description string, and
     `pre_publication_matches: true`.

   A rehashed plan claiming, for example, 99 annotation opens, 0 planned
   bytes, a false closure or empty read counts is refused **before** the
   annotation grant.
4. **One exact grant.** Only after all of that is **one** exact read-only
   exception granted, for the annotation file. There are four bounded
   passes, each limited to the index size and 8 MiB per pass,
   independently of the extractor's 64 MiB. Each checks `lstat` size first
   (a size change is refused before reading), then the full SHA-256, CRC-32
   and size:
   1. full pre-hash;
   2. lexical scan: the whole file goes through `TopLevelFrameExtractor('9')`
      while being re-hashed. Only frame 9 is captured, with a 1 MiB cap,
      duplicate top-level keys refused and nothing else materialised;
   3. pre-publication closure;
   4. final closure **after both** outputs.
5. **Decoding frame 9** (`decode_frame_value`): strict JSON with **opaque**
   number tokens (`parse_float`/`parse_int` keep the text). Unselected
   `1e400` values or huge integers are therefore **never converted**.
   Duplicate keys and NaN/Infinity constants are refused. There must be 1..64
   entries of exactly `{cam_R_m2c[9], cam_t_m2c[3], obj_id}`, where `obj_id`
   is an integer lexeme (booleans, `4.0` and strings are refused) and pose
   values are number tokens.
6. **Selection** (`select_unique_seed`):
   - exactly **one** `obj_id` lexeme `"4"` (zero or several gives
     **blocked**);
   - its **source-order position** is the derived `instance_ordinal`;
   - **only** that record's 12 R/t values are converted (bounded lexemes,
     finite), while other entries keep opaque R/t and only their bounded
     `obj_id` lexemes are converted to integers, as `select_pose` requires;
   - `select_pose` checks for a proper SO(3) R and finite values and converts
     translation mm × 0.001 **exactly once**, with no inversion, recentering
     or rotation scaling;
   - an improper or reflected pose is refused, and the error message carries
     no annotation values.
7. **Outputs** are published exclusively from a fresh stage directory, with
   the binding as the receipt, published last. They use the reviewed media
   publisher: staged, hash-verified, then an exclusive Windows `os.rename`.
   It never overwrites or deletes, and ownership comes from the staged
   identity.

   The **seed** holds only the selected pose:
   - `seed_label` =
     `controlled_init:ground_truth:HB:<revision>:scene000003:obj000004:instance<N>:s000009`;
   - `seed_pose_m` (4×4, metres) and the convention notes;
   - `applies_once_to_frame_id: s000010`, `reuse_on_recovery: false` and
     `other_pose_values_converted: 0`;
   - the flags: `ground_truth_seed`, `controlled_init` and
     `initial_ground_truth_read` true; `automatic_geometry_annotation_access`
     and `scored_pose_values_materialized` false; quality, accuracy, mobile,
     neural, GPU, assembled, registered and admitted false;
   - an explicit **disclosure**: **all** bytes of the annotation file are
     read and hashed, in four bounded passes. **Only** the frame-9 value is
     captured, and **only** the selected `obj_id 4` record's pose tokens are
     interpreted, to initialise one pose.
   - an `annotation_access` record: `whole_file_bytes_read_and_hashed: true`,
     `bounded_passes: 4`, `frame_values_captured: ["9"]`, selected-record-only
     interpretation, `other_frames_values_materialized: false`,
     `other_pose_values_converted: 0` and the semantic role;
   - `principal_point_origin: null` and `K_operation: null`;
   - the unresolved list.

   The disclosure makes no privacy or quality claim.

   The **binding** records:
   - the seed hash, the plan binding and the association (ordinal `null`,
     unchanged);
   - `derived_instance_ordinal` and its derivation;
   - a frame-9 summary (counts only, never values: records, matches, 12
     selected values converted, 0 other pose values converted);
   - the annotation source descriptor (original role plus semantic role) and
     the `annotation_access` record;
   - the completed passes 1–3 with the final pass *planned*;
   - reads, budgets, sources, runtime, the unresolved list and the closure.
8. **Final closure.** After both outputs, the stage checks the annotation
   bytes (pass 4), all input metadata, sources and runtime, and the output
   bytes and names. It also asserts exactly **4** annotation reads and zero
   camera and payload opens. On any mismatch, or any error after the output
   directory exists (including an error inside the publisher), every
   **owned** output is quarantined (see Failures) and the stage fails.
   Accepted artifacts are never touched.

## Failures and quarantine

**Quarantine** (`_quarantine`). Each owned output is moved only if it is
still the very file this stage created (identity re-checked immediately
before the move):

- The move is exclusive to a **unique** name: `invalidated-<name>`, or
  `invalidated-<token>-<name>` when that name is taken, with at most 8
  attempts.
- It uses Windows `os.rename`, which refuses an existing destination; on
  POSIX it uses link plus unlink, with the alias budget-checked.
- Bytes and file identity are kept.
- A **foreign** name (for example a pre-existing `invalidated-controlled-seed.json`)
  is never overwritten, renamed or deleted.
- **It never raises.** A denied move, a budget refusal (`Blocked`), a
  lookup error or anything else is recorded in that output's entry (`error`),
  and the next output is still processed.
- **Post-move inspection is best-effort.** If the moved file can't be
  looked up, `identity_preserved` is `null`, with `post_move_lookup_error`
  set. An unknown identity is reported as unknown, never as proof of
  anything.

**Failure handler** (`_failure`). It processes every owned output
independently, even if a quarantine call itself raises. It then takes an
explicit inventory of each authoritative name with `_presence`:

| State | Meaning |
|---|---|
| `present_owned` | still the very file this stage created |
| `absent` | `FileNotFoundError` only |
| `replaced` | another file now holds the name; it is not owned and is left untouched |
| `unknown` | the lookup failed; this is **never** treated as proof of disappearance |

It never raises, so the **original stage exception always propagates**.

**Failure record.** `failure.json` (`workbench_hb_scene3_seed_failure_v1`,
at most 4 KiB, valid JSON, containing no annotation data or file dumps)
lists `quarantined_outputs`, `authoritative_states`,
**`authoritative_files_remaining`** (`present_owned`),
**`authoritative_files_uncertain`** (`unknown`) and `retained_files` (an
unreadable size is recorded as `null`).

- **Status:** `failed_authoritative_outputs_remain` if any output remains or
  is uncertain, otherwise `failed`. If the inventory itself fails, a
  minimal record with status `failed_authoritative_outputs_uncertain` keeps
  the original reason.
- **`completed_receipt_published`** is always `false`.
- **Publication** uses fresh subtree, scope, free-space and global checks.
  If it fails, the record goes to stderr as one line, and the original
  exception still propagates.

## Budgets

- **Seed subtree:** the **whole** `controlled-seed-v1/` subtree is at most
  **64 KiB**, counting the plan, seed, binding, failure records, staging
  partials and aliases of every attempt. `media._usage` counts any
  `*.json*` name as metadata, and the caps are plan 24 KiB, seed 8 KiB,
  binding 16 KiB and failure 4 KiB.
- **Shared limits:** the existing scope limits of 12 MiB total and 512 KiB
  metadata, a fresh global 1 GiB derived budget, and the 1 GiB free floor
  are checked before **every** write, including failure writes.
- **Input caps:** annotation input is at most 8 MiB (not the old 64 MiB),
  and new inputs stay within 250 MB.
- **Failure records:** these are valid JSON, at most 4 KiB, and carry no
  annotation data or file dumps. If one can't be published, it goes to
  stderr as one line.

## Reads accounting

The media audit-hook counter labels every input **before** its first read:
prerequisites, policy, allowlist, the receipts, profile, K, association,
scout outputs, anchors, the annotation, the camera, the payload root and the
stage output.

| Stage | Annotation reads | Camera reads | Payload reads |
|---|---|---|---|
| Plan | 0 | 0 | 0 |
| Extraction | exactly 4 (pre-hash, scan, pre-publication, final) | 0 | 0 |

No GPU is used. The binding holds a snapshot taken before it was written,
and the return value reports the totals.

## Developer checks (manufactured fixture only)

The checks used the parent's media fixture with a manufactured
`scene_gt.json` reuse row. In frame 9, records were [obj 1, obj 4, obj 7].
Unselected values included `1e400` and 400-digit integers, and the selected
pose was a 90° rotation about z with t = (100, −250, 1200) mm. The real
media chain ran: allowlist, model, a synthetic 120-frame FFmpeg video, and
prerequisites.

- **Plan:** 9.3 KB with 0 annotation opens and the `evaluation` role
  preserved. Wrong pins were blocked, and production refused TEST_ONLY
  inputs.
- **Rehashed plan** (changed annotation SHA): refused before the grant.
- **Extraction:** `…:obj000004:instance1:s000009`, with pose
  `[[0,-1,0,0.1],[1,0,0,-0.25],[0,0,1,1.2],[0,0,0,1]]`, ordinal 1,
  0 other values converted, and 4 annotation and 0 camera reads.
- **Attack after both outputs:** a same-size, restored-mtime annotation
  change was refused by the final closure. Both outputs became
  `invalidated-*` and `failure.json` was written.
- **Retained staging sibling** (40 KiB): blocked before any read.
- **Frame-9 refusals:** no CAD 4 and two CAD 4 (blocked); a boolean,
  `4.0` or string token; a duplicate key; NaN; a selected `1e400`; a
  reflected R. Decoding performed no float conversion at all.

Repair 1 added these checks:

- **The 22 parent tests pass.**
- **Actual audited index row** (from the parent's metadata-only audit; pure
  data, path comparison only, with the annotation file **not** opened): it
  validates with `role: metadata`, `semantic_role:
  controlled_ground_truth_annotation`, 288 020 bytes, SHA `4e2f4409…`, CRC
  3094127165, original method 8 / 103 647 bytes / offset 334036341, and an
  absolute canonical path.
- **Roles:** `evaluation`, `annotation` and `metadata` are accepted.
  `rgb`, `model`, `depth`, `calibration`, `evaluation_mask`, `Metadata` and
  two roles are refused.
- **Quarantine refused by the OS** (`PermissionError`): status
  `failed_authoritative_outputs_remain`, with both names listed in
  `authoritative_files_remaining` and `completed_receipt_published: false`.
- **Repeated foreign collisions:** the outputs moved to unique
  `invalidated-<token>-…` names with identity preserved. The foreign bytes
  were untouched and nothing authoritative remained.
- **Runtime type strictness:** a plan runtime with `dont_write_bytecode: 1`
  instead of `true` was refused before the grant.

Repair 2 (failure path only) added these checks:

- **The 26 parent tests pass** (test file `1c19359f…`).
- **Lookup error on the first quarantine name** (the parent's test): both
  outputs were still quarantined, `failure.json` was written with the
  original reason, and nothing authoritative remained.
- **Budget refusal raised by the quarantine** (the parent's test): both
  outputs were reported in `authoritative_files_remaining`, with status
  `failed_authoritative_outputs_remain`, and the handler completed.
- **Move denied for one output only:** the other output was still
  quarantined, and the denied one was listed as remaining.
- **Denied moves plus an unreadable state:** one output was listed as
  remaining and the other as uncertain (`unknown`), never as absent.
- **Post-move lookup failing for every quarantine:** both moved, with
  `identity_preserved: null` and the lookup error recorded, and nothing
  remained.
- **`failure.json` publication refused:** no exception escaped, the record
  went to stderr, and the quarantined bytes were kept.
- **Inventory crash:** a minimal record was written with status
  `failed_authoritative_outputs_uncertain` and the original reason.
- **End to end** (a binding publication error plus a denied move of the
  binding): the original `InputsError` propagated, `failure.json` listed
  the remaining binding, and the seed was quarantined.

## Limitations

- **Annotation content:** the real `scene_gt.json` content was not read.
  The parent's metadata-only audit confirmed the historical row (role
  `metadata`, an absolute reuse path, stored method 0 / original method 8,
  288 020 bytes, SHA `4e2f4409…`, CRC 3094127165), and that row validates
  here as pure data. The source content, CRC and the uniqueness of CAD 4
  in frame 9 remain **unverified** until the parent's actual run; the four
  bounded passes then verify the exact SHA, size and CRC, and the run fails
  closed otherwise.
- **Not accuracy:** this is controlled ground-truth initialisation, not
  independent accuracy.
- **Unresolved:** assembly, assembly selection, principal-point origin,
  artwork placement, segmentation, registration and admission remain
  unresolved.
- **Windows-first publishing:** exclusive rename is the reviewed path. On
  POSIX, link+unlink counts 2× against the 64 KiB subtree.
