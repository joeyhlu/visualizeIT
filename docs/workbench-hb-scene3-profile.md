# HB scene 000003 metric profile and camera K

Tools:

- `tools/workbench_hb_scene3_profile.py` contains the stages, the CLI and the
  publication ledger.
- `tools/workbench_hb_camera_k.py` contains pure helpers: the reuse-row
  validator, both read passes, the streaming scanner and the K-window check.

Both work on metadata only and run on the CPU. Together they produce an
immutable scene-3 **profile** and a **K-only** derivative. They do **not**:

- prepare a model or video;
- extract a seed, assemble, register or admit anything;
- run a network, the GPU or inference;
- read GT, poses, masks or RGB pixels;
- associate an object or claim tracking quality, phone performance or
  mobile readiness.

Scene-2 tools, receipts and contracts are unchanged. Pure helpers of
`workbench_hb_metric_inputs` and `workbench_hb_prefix_scout` are reused but
never mutated, and their stages are never called. Neither module imports the
opening-seed module, the resource store, or any neural or render module.

## Trust

Trust is historical local integrity only (`upstream_authenticated: false`).

| Input | Pin |
|---|---|
| Accepted scout plan `.cache/video-workbench/derived/hb-prefix-scout-000003-v1/plan/scout-plan.json` | `d76aaec4dabd33543aad06d330f520e95bfad394db84c147b47e56616e044201` |
| Accepted preview receipt `…/preview/preview-receipt.json` | `a78e0c79e986df6a4ba8890550ef45ec91fac62d1c43f3f51fb0f11468b0b346` |
| Acquisition receipt / resources index / integrity record | `064d3c21…` / `a05b65a6…` / `0ef84395…` (`workbench_hb_metric_inputs.ANCHORS`) |
| Camera member `val_primesense/000003/scene_camera.json`, expanded SHA-256 | `09f292d9305b3003a6356410adbce688548ed3c10c5b19d27c242d0ead7c809d` |
| Accepted input accounting (scout plan `input_budget.total`) | 186 411 119 bytes |

Every file is hashed **before** it is parsed. The caller-supplied expected
hashes must equal these pins, or the stage is blocked. `SyntheticPins` is a
**test-only** replacement: the CLI never sets it, and outputs record
`pins_kind = "synthetic_test_only"`.

## CLI

Run with `python -B -S tools/prototype_bootstrap.py tools.workbench_hb_scene3_profile STAGE …`.

```
build-profile --scout-plan P --expected-scout-plan-sha256 H
              --scout-receipt P --expected-scout-receipt-sha256 H
              --expected-receipt-sha256 H --expected-index-sha256 H --expected-integrity-sha256 H
              --preparation-root DIR --out DIR/NEW
extract-k     --profile DIR/…/scene3-profile.json --expected-profile-sha256 H
              --preparation-root DIR --out DIR/NEW
```

Exit codes: 0 completed; 1 failure; 2 usage; 3 blocked.

`--preparation-root` must be a fresh, exclusive scene-3 scope: its own
directory strictly inside `.cache/video-workbench/derived`, link-free, and
neither equal to, inside nor containing the scout scope.

## `build-profile`

1. **Before reading anything.** The given hashes must equal the pins. The
   scout plan and receipt must be existing, link-free, regular files named
   exactly `scout-plan.json` and `preview-receipt.json`, outside the
   preparation scope. They are located but not read.
2. **Guard.** The whole stage runs in the shared audit guard with an empty
   payload allowlist and no read exceptions. Prospective budgets are checked
   before any read (see Budgets).
3. **Scout outputs.** The plan is read (bounded), hashed against the pin and
   only then parsed. It is validated with `workbench_hb_prefix_scout.validate_plan_doc`
   and must hold the accepted input total. The receipt is hashed, parsed and
   bound to the plan: schema, status, flags, trust, pins, plan path and hash,
   frames 0 and 9 only, PNG, compressed and CRC identities equal to the plan
   rows, 640×480, calibration pending and the `FRAME_INDEX` clock.
4. **Anchors.** The three anchors named by the plan are added as exact
   **read-only** exceptions and hashed against the pins before parsing. The
   scout metadata core is rebuilt from them and must be byte-identical to the
   plan.
5. **Camera row.** The camera row is taken from the index and validated by
   `validate_camera_row`, not the RGB row validator:
   - revision, archive `hb_val_primesense.zip`, member, and
     `allowedRoles == ["metadata"]`;
   - location kind `reuse`, whose path (absolute, or relative to the
     repository) must equal the canonical
     `.cache/prototype-sprint/hb-scene-metadata-v1/val_primesense/000003/scene_camera.json`;
   - stored method 0, stored size equal to the expanded size, stored SHA
     equal to the expanded SHA equal to the pin, a valid CRC and a size of at
     most 1 MiB;
   - the original ZIP method 0 or 8, flags, offset and compressed size
     checked and kept as separate `original_zip` provenance.

   The camera file is only checked with `lstat` (link-free, regular, of the
   index size) and **never opened** here.
6. **Profile.** `scene3-profile.json` is written as compact JSON, at most
   48 KiB. It records:
   - the scout and anchor bindings, revision and scene;
   - all 120 exact RGB identities (columns: index, member, expanded and
     compressed SHA-256, CRC-32, sizes) and the RGB totals;
   - the image size 640×480 (PNG headers verified only for frames 0 and 9,
     by the scout);
   - the split: opening 0..9, seed 9, scoring 10..119;
   - the clock: `FRAME_INDEX`, with physical FPS and timestamps `null`;
   - the object: `object_id` and `instance_ordinal` are **unset**, and CAD 5
     is a **candidate only**, with no association;
   - the principal-point origin, marked **UNVERIFIED** (separate review);
   - the camera block, `PENDING_K_extraction` with `K: null`;
   - the input accounting, plus the sources, runtime and budget.
7. **Closure.** The scout outputs, anchors, sources and runtime are compared
   before publication and **again after it** (see Publication).

## `extract-k`

1. **Locate the profile.** It must be named exactly `scene3-profile.json` and
   lie inside the preparation scope; it isn't read yet.
2. **Read and check the profile** inside the guard. It is read (bounded) and
   hashed against `--expected-profile-sha256` before parsing. Its structure
   is validated, along with its preparation scope and its source closure and
   runtime, which must equal the current ones.
3. **Re-authenticate.** Everything in `build-profile` steps 3–5 is repeated
   from the paths the profile names. The rebuilt profile core must be
   byte-identical to the accepted profile.
4. **Budgets** are checked before the camera is opened. Only then is the
   exact camera path added as a **read-only** exception.
5. **Pass 1** hashes the whole bounded file (size, SHA-256, CRC-32) against
   the row **before any parsing**.
6. **Pass 2** streams the file through `StructuralScanner` while re-hashing
   it; any change since pass 1 fails. The scanner:
   - validates the full JSON grammar: UTF-8 split across chunk boundaries,
     no byte-order mark, string escapes, no control characters, duplicate
     keys at **every** level, the numeric lexical grammar across chunk
     boundaries, literals and a depth of at most 64;
   - requires canonical decimal top-level frame ids;
   - captures **only** the `cam_K` arrays of source indices 0..119.
     Everything else (R, t, `depth_scale`, frames outside the window, strings,
     literals) is validated and skipped, with no numeric conversion and no
     array construction. A skipped `1e400` is accepted and never converted.
     The camera document is never materialised.

   **String domain.** Keys are compared *decoded*:
   - Escapes decode to the same code points as literal UTF-8, so `"😀"` and
     `"\ud83d\ude00"` are the same key and a repeat is a duplicate. This also
     holds when a chunk boundary falls inside the pair.
   - A valid escaped surrogate pair (high `D800–DBFF` immediately followed by
     the escape of a low `DC00–DFFF`) is combined into one code point.
   - A lone high or low surrogate escape, a reversed pair, or a high
     surrogate followed by anything other than a low-surrogate escape is
     **unsupported and fails closed**, in keys and values alike.
   - Literal (unescaped) surrogates can't occur, because strict UTF-8
     decoding rejects encoded surrogates.

   **Requested frames.** The requested list is checked element by element
   **before** any sorting or de-duplication, both in the scanner and in
   `validate_k_window` (`check_requested`). Each element must be a plain
   non-negative `int`; booleans, floats, strings and repeats are refused
   with `UsageError`, so `[0, False]` or `[1, True]` can't hide behind
   de-duplication. A string, mapping or non-iterable is refused as a whole.
7. **K check.** `validate_k_window` requires, for every requested frame:
   - exactly 9 finite numbers (booleans, `null`, strings or nested values in
     `cam_K` fail, and so does an overflowing `5e400`);
   - zero-skew pinhole form `[[fx,0,cx],[0,fy,cy],[0,0,1]]` with fx, fy > 0
     and 0 < cx < 640, 0 < cy < 480;
   - identical values across the whole window.

   Missing, duplicate, varying, malformed or incompatible K **fails closed**.
8. **Output.** `camera-k.json` is written, at most 32 KiB. It records:
   - the raw K values, their exact lexemes, the 3×3 matrix and fx, fy, cx, cy;
   - the window, and whether the raw lexemes are identical across it;
   - the camera source row with its `original_zip` provenance, and the camera
     reads (see Camera reads and closure);
   - scanner statistics;
   - the adjustments, all `null` (no crop, rescale or origin shift);
   - the principal point marked unverified, and the metadata-only fixed-K
     control note;
   - the sources, runtime and budgets.

## Budgets

Each of these is checked before any source is opened and again before every
publication or failure write.

- **Substage:** each output directory (profile, K) is at most **128 KiB**,
  counting the transient peak, temporary file plus hard-linked final name,
  as 2× the bytes. Normal writes keep 2 × 16 KiB for a failure record.
- **Preparation scope:** the existing 12 MiB total and 512 KiB metadata
  aggregates, including retained failed siblings and temporary aliases.
- **Global:** derived, scratch, runs, masks and suite must stay ≤ 1 GiB, and
  actual free disk must keep a 1 GiB reserve (`workspace_budget`, plus a fresh
  check before every failure record).
- **Inputs:** the accepted total of 186 411 119 bytes (which already includes
  the 12 MiB future preparation reserve) plus a conservative 1 MiB camera
  input must stay ≤ 250 000 000. The camera input itself is at most 1 MiB.
- **Unchanged limits:** the 8 GiB model cache and the 256 MiB public limit
  are not touched.

## Camera reads and closure

The camera is **never** hashed with the generic file hash. Every pass over it
is bounded by the index-bound size and the 1 MiB cap, and checks size,
SHA-256 and CRC-32. Size, type and link changes are refused from `lstat`
before the file is opened. If the file grows after the scan, it is refused
without reading the changed file, and the read is stopped at the expected
size. A refusal in a closure pass is reported as a closure mismatch.

A successful `extract-k` opens the camera **4 times**, and records them
honestly in `camera_source.reads`:

| Pass | When | Count |
|---|---|---|
| `extraction_passes.hash_pass` | before parsing | 1 |
| `extraction_passes.scan_pass` | structural scan, re-hashing | 1 |
| `closure_passes.pre_publication_completed` | before the record is written | 1 |
| `closure_passes.post_publication_planned` | after the record is published | 1 |

`total_camera_opens_on_success` is 4.

The post-publication pass necessarily runs after the record is written, so
the record labels it *planned*. If it fails, the record is invalidated. The
stage's return value reports the counted opens
(`camera_opens: {"extraction_passes": 2, "closure_passes": 2, "total": 4}`).
On failure, fewer passes happen and no success record remains. `build-profile`
never opens the camera.

## Publication and failures

Outputs are published exclusively into a new directory and never
overwritten. The closure interval ends **after** publication: inputs,
sources, runtime, and the current names and bytes of the stage's own outputs
are re-checked.

On any error after the output directory exists, including an error inside the
publisher after its hard link was created:

- **Owned output:** if the profile or K output was created by this stage
  (ownership is recorded by `PrepLedger.record` right after the publisher's
  link, with the file identity) and is still that same file, it is renamed
  to `invalidated-<name>` and its bytes are kept.
- **Foreign files** are never renamed or deleted.
- **Failure record:** a bounded, valid-JSON `failure.json`
  (`workbench_hb_scene3_profile_failure_v1`, `completed_receipt_published:
  false`) is written after a fresh global budget check. If it can't be
  published, nothing is written; stderr gets a `NOT published` line and the
  record as one-line JSON, and the original error is still raised.

## Library API (for parent tests)

| Item | Purpose |
|---|---|
| `ProfileContext(root, workspace, preparation=None, synthetic=None)` | Injectable roots; `.mi()` requires the preparation scope. |
| `SyntheticPins(anchors, integrity_members, rgb_encoded_bytes, rgb_expanded_bytes, scout_plan_sha256, scout_receipt_sha256, camera_expanded_sha256, input_accounted, reuse_root)` | **Test-only** pins. |
| `build_profile(ctx, *, scout_plan, scout_receipt, accepted_scout_hashes={"plan","preview_receipt"}, historical_anchor_hashes={3 anchors}, out)` | Profile stage. |
| `extract_camera_k(ctx, *, profile, expected_profile_sha256, out)` | K stage. |
| `authenticate_inputs`, `validate_scout_receipt`, `build_profile_core`, `validate_profile_doc`, `PrepLedger` (`check`, `check_sequence`, `record`, `owned_file`, `owned`), `main(argv, ctx)` | Profile module helpers. |
| `ck.validate_camera_row(row, *, member, archive, revision, expected_sha256, max_bytes, root, canonical)`, `ck.canonical_reuse_path(root, reuse_root, member)` | Camera row checks. |
| `ck.hash_bounded(path, *, expected_bytes, cap)`, `ck.scan_camera_file(path, *, expected_bytes, expected_sha256, requested, chunk)` | The two read passes. |
| `ck.StructuralScanner(requested)` (`feed`, `finish`), `ck.scan_camera_json(chunks, requested)`, `ck.validate_k_window(captured, requested, image_size)`, `ck.check_requested(requested)` | Scanner (any chunk boundaries), K checks and the requested-frame validation done before normalisation. |

**Manufactured fixture.** Start from the scout fixture
(`scene_three_catalog`). Replace the camera row with a reuse row:

- `allowedRoles ["metadata"]`, `method 0`, and `compressedBytes ==
  expandedBytes == len(camera bytes)`;
- `compressedSHA256 == uncompressedSHA256 == sha256(camera bytes)` and the
  CRC-32;
- any original `sourceMethod`, `sourceCompressedBytes` and `sourceOffset`;
- `location {"kind": "reuse", "path": <root>/<reuse_root>/val_primesense/000003/scene_camera.json}`.

Then rewrite the receipt and integrity `indexSHA256`, recompute the anchors,
run the scout plan and preview with the scout's `SyntheticPins`, and pass the
profile `SyntheticPins`, using the scout plan's `input_budget.total` as
`input_accounted`. Camera JSON follows the BOP form
`{"<id>": {"cam_K": [9], "cam_R_w2c": [...], "cam_t_w2c": [...], "depth_scale": …}}`.

## Limitations

- **Unseen real camera row.** The real camera row was not read by the
  implementer. The archive (`hb_val_primesense.zip`), the reuse-path form
  and the `["metadata"]` role follow the brief and `bench/prototype_resource_store.py`.
  The parent's metadata run confirms them or fails closed.
- **Strict K form.** Pinhole form requires zero skew, and the principal
  point is only bounds-checked. Its origin convention stays unverified.
- **Image size.** The size is the HB native one, header-verified only for
  frames 0 and 9.
- **Profile size.** The profile cap (48 KiB) and K cap (32 KiB) are sized
  from fixtures (31 KB and 6 KB); a larger real profile is refused, not
  truncated.
- **Shared code fingerprint.** The source closure includes shared helpers;
  legitimate concurrent edits by other workers cause a refusal.
