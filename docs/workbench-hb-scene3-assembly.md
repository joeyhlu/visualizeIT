# HB scene 000003 / CAD 4 assembly (prepared metric control)

Tool: `tools/workbench_hb_scene3_execution_control.py`. Tests:
`tools/test_workbench_hb_scene3_execution_control.py` (written separately).

This assembles exactly **four JSON files** for the fifth distinct recording,
HB val_primesense scene 000003 / CAD 4. It consumes the **already accepted**
media and the separately **accepted actual controlled seed**, all named by one
parent-frozen registry. It is a **prepared metric control** for head
`27e06a51…`. It is **not**:

- automatic geometry or an evaluator-free run;
- head accuracy;
- mobile or phone proof.

The assembly **never**:

- opens GT, the raw camera file, RGB PNGs or the source PLY;
- decodes the GLB geometry or the video;
- copies or links any asset;
- runs a model, GPU or SAM;
- registers or admits anything;
- searches for an artwork ray or changes the point, pose, K or origin.

It never calls the old `mi.assemble`, the scene-2 stages, the old seed
validator, the YCB assembler, the media preparation stages or the seed
extraction stages. No global is mutated. Every stage runs inside the shared
audit guard: empty payload allowlist, no process or network access, and
forbidden imports and denied GT/camera names refused.

## Trust: the frozen registry

**Production registry:**
`.cache/video-workbench/derived/validation/hb-scene3-assembly-inputs-v1.json`,
SHA-256 `c9ea5036252308cb795bf02cf9199a8fb22388c0f3b914547a20aef92bb01a21`,
9 156 bytes (source constants `REGISTRY_PIN`).

**Registry checks.** The registry is read bounded and hashed **before**
parsing, then validated exactly:

- schema, `label: PRODUCTION_PINNED`, scene, object, family and head;
- `historical_local_integrity` trust;
- the full constraints block (see `EXPECTED_CONSTRAINTS`);
- the preparation scope and `output_directory = <scope>/assembly-v1`;
- the 21 inputs, each with path, SHA, size, cap and kind, and each on the
  correct side of the preparation scope;
- the nine accepted source versions, which must equal the **current**
  helper hashes.

**Input checks.**

- **Metadata inputs** (19): contained, link-free from the repository root
  through every ancestor, regular, of exactly the registered size within its
  cap, hashed before strict parsing.
- **Derivatives** (GLB and video): sealed by bounded streaming SHA-256 at the
  registered size, never decoded.

**No caller trust.** No caller-supplied hash, object, frame, ordinal, K,
origin or seed can create trust. There is no force or overwrite option.
`AssemblyContext.test_only_registry_pin` (API only) requires a registry
labelled `TEST_ONLY`; production refuses it. Every TEST_ONLY output carries
`label: TEST_ONLY` **and** the unadmitted synthetic family
`TEST_ONLY_hb_primesense_640x480` (`output_family`), consistently through the
bundle, clip provenance, sidecar, flags and receipt. A synthetic bundle
detached from its sidecar therefore still cannot pass as production. The
closed `MetricBundle` schema is unchanged, and the production family stays
`hb_primesense_640x480`.

### Relations re-verified (`_check_relations`)

- **Media chain:** the prerequisites receipt (`prepared_media_only`) binds
  the allowlist, model and video receipts, profile, K, association, GLB and
  video exactly.
  - **Model receipt:** GLB SHA, bytes and path; 118 136 triangles; not
    decimated; portable `FAIL`.
  - **Video receipt:** video SHA, bytes and path; 120 sequential frames plus
    EOF; exact s000010 seek; 640×480; physical FPS and timestamps `null`;
    presentation 30; frame order.
- **Profile, K and association:**
  - the raw K (no origin shift, crop or rescale);
  - the association's CAD 4, frame s000009, point (320, 340) and ROI
    (285, 270, 370, 440), with **original `instance_ordinal: null`**.
- **Seed chain:**
  - **Policy and plan:** both bind the association and prerequisites.
  - **Controlled seed:** an exact label, an ordinal, single use at s000010,
    no reuse, 0 other values converted, origin and K operation `null`, the GT
    flags, and a finite, proper SE(3) pose already in **metres**, used
    verbatim.
  - **Binding:** the seed hash, the derived ordinal, `obj_id_4_matches: 1`,
    and association ordinal `null` unchanged.
  - **Binding alone is insufficient.** The **seed source acceptance**
    (`accepted`, 0 Codex failures, the seed source pin) is required, plus the
    **actual plan and extraction execution records**. Both must show
    `completed`, `final_closure_returned_successfully: true`, exit 0, the
    accepted label, and hashes matching the registry: plan 0 annotation
    opens, extraction 4. The **independent actual seed acceptance** must show
    `accepted`, `final_closure_accepted`, unique match 1, the same ordinal,
    metres, and evidence hashes matching the registry.
- **Actual media execution:** the tool SHA, no neural run, no GT/camera
  read, triangles, and artifact hashes and sizes (Windows path keys are
  normalised).
- **Origin:** `integer_centers` as a **reviewed assumption only**:
  `verified_dataset_convention: false`, `physical_calibration_proof: false`,
  shift `[0, 0]`, raw K = effective K, and the independent review
  `accepted_as_reviewed_assumption_only`. `native_K(raw, "integer_centers")`
  must equal the raw K.
- **Opening placement:** `frozen_opening_ray_hit; observed_mask_review_pending`,
  with `observed_mask_checked: false`, `selection_reviewed: false`, no point
  change or search, the point, ROI and frame, and all input hashes. The
  frozen `workbench_neural_artwork_v1` record (checkerboard, 256 px,
  `fixed_builtin_v1`, anchor pixel (320, 340) on s000009, the GLB SHA, the
  seed label, the raw K, `anchor_mask_checked: false`) is **re-hashed** and
  must equal `artwork_record_sha256`. The placement sources must equal the
  accepted versions.
- **Complete placement and artwork validation**
  (`_validate_placement_geometry`; booleans are never numbers):
  - **Artwork record:** exactly the `NeuralArtwork.record` keys, the CAD
    `support_source`, and a strict-int `patch_faces` in (0, mesh faces].
  - **Marker:** must pass `wc.Marker.from_dict` (barycentric simplex,
    orthonormal right-handed chart, positive size). The placement's
    `face_id` (a strict int in [0, 118 136)) and `barycentric` (finite,
    ≥ 0, summing to 1) must equal the marker's **exactly**.
  - **Artwork digest:** the artwork and marker digests must equal the
    **recomputed** semantic digest of the existing fixed checkerboard
    (`wg.array_sha256(wg.builtin_artwork("checkerboard", 256))` = `d0c29333…`;
    pure NumPy, no renderer or model).
  - **Depth and residual:** `depth_m` is a finite non-bool > 0. The
    recorded residual is a finite non-bool in [0, 1e-3] px and must match
    the recomputed residual within 1e-6.
- **Frozen-point consistency only (no GLB read).** The placement's
  object-space hit point, transformed by the seed pose, must lie at the
  recorded depth (within 1e-6 m). Projected with the raw K, it must land on
  the frozen pixel (320, 340) within 1e-3 px, reproducing the recorded
  residual. This is a **one-point consistency check**. It does **not**
  independently verify units, origin or a unique pose: a different proper
  rigid pose rotating about that point passes too. Units and pose provenance
  are inherited from the pinned seed and model, and physical calibration of
  the origin remains an assumption.
- **Video path binding.** The accepted video receipt's `workspace_relative`
  must be a traversal-free POSIX path that resolves (workspace contained and
  link-free from the repository root) to **exactly** the sealed registry
  video, whose hash and size are preserved. Any other path, for example
  `derived/other/unsealed-video.mp4`, is refused.

## Outputs (`--out` = `<scope>/assembly-v1`, or a new attempt directly inside it)

1. **`metric-bundle.json`**: exactly `workbench_metric_bundle_v1`:
   - `family: hb_primesense_640x480` (TEST_ONLY runs:
     `TEST_ONLY_hb_primesense_640x480`), `obj_id: 4`;
   - `principal_point_origin: integer_centers`;
   - `metric_provenance: dataset_metric_cad`;
   - `seed_frame_id: s000009`, with the exact accepted `seed_label` and
     `seed_pose_m` (metres, verbatim);
   - `glb_path`: the **repository-relative**
     `.cache/video-workbench/derived/hb-metric-control-000003-v1/model-v1/hb_obj_000004.glb`
     (never `../model-v1`), with SHA `bf167639…`.

   The GLB is not copied or linked. Containment and link refusal (with
   ancestors) are enforced here, because the core loader doesn't. After
   publication, the core `MetricBundle.from_file` parse must agree field by
   field.
2. **`clip-proposal.json`**: validated by `wc.ClipSpec.from_dict` and
   `check_onboarding`, and **not registered**.
   - **Media and frames:** a native 640×480 `video_file` with a
     `frame_index` clock; media `root_id: workbench` with the
     workspace-relative video path, SHA and bytes. Exactly 120 frames
     `s000000…s000119` with `timestamp_s` and `media_time_s` both `null`.
   - **Onboarding and selection:** onboarding is s000000–s000009, and scoring
     is s000010–s000119. The selection is frame s000009, positive
     `[[320, 340]]`, negative `[]`, **`reviewed: false`**; the observed SAM
     tap gate is pending.
   - **Intrinsics:** `dataset_reference`, raw K.
   - **Provenance** discloses:
     - the frozen ROI (provenance only, not the Selection schema);
     - physical FPS `null`, with 30 fps presentation stated as synthetic;
     - the ten-second prefix **not attested**;
     - the origin as a reviewed assumption with raw K unchanged;
     - the association's original ordinal `null` and the derived ordinal 0
       (from the new binding);
     - the **prior controlled ground-truth initialisation**: this is not
       evaluator-free. `evaluator_inputs_read: false` refers only to this
       clip's own input path.
3. **`execution-control-sidecar.json`**
   (`workbench_hb_scene3_execution_control_sidecar_v1`):
   - **Artwork:** the frozen checkerboard NeuralArtwork configuration (edge
     256, `size_fraction` 0.25, `max_faces` 200 000) with the exact accepted
     placement and artwork-record hashes. **`primary_neural` and
     `frozen_baseline` share the same anchor.**
   - **Seed use:** the seed is intended **once at s000010 in each
     independently initialised role**, never re-armed on loss. The runtime is
     not executed or tested here.
   - **Placement:** a frozen CAD ray hit only; the CAD is **not** observed
     foreground.
   - **Gates:** the observed-SAM-review gate is separate and pending, with
     `selection_reviewed: false`.
   - **Origin:** `integer_centers` as a reviewed assumption, with
     `verified_dataset_convention: false`, `physical_calibration_proof:
     false`, raw K, shift `[0, 0]`, no crop or rescale, and K operation
     `null`.
   - **Mesh:** 118 136 triangles (portable target 20 000: **FAIL**),
     3 073 172 bytes, texture 256.
4. **`assembly-receipt.json`** (published **last**, status
   `assembled_unadmitted_unregistered`) records:
   - the registry and 21 inputs (path, SHA, bytes), and the **three other
     outputs' hashes** (it never hashes itself);
   - the constraints and flags: controlled-GT seed, not automatic, not
     evaluator-free, no accuracy or phone claim, 0 annotation and raw opens,
     no NN, SAM, segmentation, registration or admission, no asset copy;
   - the disclosure, derived ordinal, seed label and prefix attestation
     (`false`), the GLB facts and the unresolved list;
   - the input budget, global budget, reads snapshot, sources, runtime and
     closure.

   `assemble` returns the receipt hash **only after** the post-publication
   closure succeeds.

## Budgets, publication and failures

- **Budgets:**
  - **`assembly-v1` subtree:** at most **96 KiB** including every attempt,
    failure, quarantined output and staging alias (the seed subtree has its
    own 64 KiB).
  - **Preparation scope:** at most 12 MiB, with metadata at most 512 KiB.
  - **Global:** the 1 GiB derived budget and the 1 GiB free floor. All are
    recomputed fresh before **every** write, including the failure write.
  - **Prospective check:** the exact sizes of the three built outputs plus
    the receipt cap (24 KiB), with transient factor and failure reserve, are
    checked **before any write**.
  - **Output caps:** bundle 4 KiB, clip 24 KiB, sidecar 8 KiB, receipt
    24 KiB, failure record 4 KiB.
  - **Inputs:** the seed plan's bound input total plus the registry and the
    eight new review and execution records must stay ≤ 250 MB.
- **Publication:** the reviewed media publisher writes exclusive staging,
  verifies hashes and sizes, then makes an exclusive Windows rename. It never
  overwrites, and ownership comes from the staged identity. A second
  assembly into an existing directory is refused.
- **Closure:** for every input (metadata and derivatives) and the registry,
  `_closure` re-checks containment, regularity and the **whole ancestor
  chain** (link refusal from the repository root) **before** each bounded
  re-seal at the registered size. It then re-checks the full source closure
  and the runtime. Refusals are reported as mismatches.
  - `inspect_assembly` finishes this closure before reporting readiness.
  - `assemble` runs it before publication and again after the receipt,
    along with the output names and bytes.
  - `validate_output` runs it and then a fresh output reseal before
    reporting `valid`.
- **Output ancestry:** the output directory and every ancestor (inside the
  repository, link-free, a real directory) are re-checked (`_safe_dir`)
  before **every** publication and before **every** cleanup move.
- **On failure:** after the output directory exists, every **owned**
  output is quarantined independently using the reviewed seed mechanics
  (`sd._quarantine`/`_presence`): unique exclusive `invalidated-*` names,
  identity re-check, foreign names untouched. A bounded `failure.json`
  (`workbench_hb_scene3_assembly_failure_v1`) lists remaining and uncertain
  outputs, and the original error always propagates. If the output ancestry
  is unsafe, **nothing** is moved or written through it: the uncertain state
  is reported on stderr only.

## API (for parent tests)

```python
AssemblyContext(root=ROOT, workspace=ROOT/".cache/video-workbench", test_only_registry_pin=None)
SyntheticRegistryPin(path: str, sha256: str, bytes: int)          # TEST_ONLY; registry label must be TEST_ONLY
inspect_assembly(ctx) -> {status: "ready_to_assemble", label, family, registry_sha256, planned_outputs{name: sha},
                          seed_label, derived_instance_ordinal, prefix_attestation, input_budget, audited_opens, exit: 0}
assemble(ctx, *, out) -> {status: "assembled_unadmitted_unregistered", label, family, out, receipt_sha256,
                          outputs{name: sha}, audited_opens_total, exit: 0}
validate_output(ctx, *, out) -> {status: "valid", label, family, receipt_sha256, outputs, audited_opens_total, exit: 0}
```

**Pure helpers:** `validate_registry(doc, *, label)`,
`_check_relations(reg, docs, *, label, root)`,
`_validate_placement_geometry(pl, art, *, pose, raw_k, mesh_faces, fixed_xy)`,
`checker_digest()`, `output_family(label)`, `output_flags(label)`,
`expected_receipt_reads(reg)`, `build_outputs(reg, facts, *, label)`,
`check_outputs(outputs, *, root)`, `AssemblyLedger`, and the constants
(`REGISTRY_PIN`, `EXPECTED_CONSTRAINTS`, `OUTPUT_CAPS`, `SUBTREE_CAP`,
`TEST_ONLY_FAMILY`, `ARTWORK_RECORD_KEYS`, `MAX_PROJECTION_RESIDUAL_PX`, the
output names). **Errors:** `ExecutionControlError` (an `InputsError`),
`mi.Blocked` for budgets and `mi.UsageError`.

**`validate_output`** (read-only):

- **Content outputs:** rebuilds the three content outputs from the
  authenticated registry inputs and requires **byte equality**.
- **Receipt fields:** checks every deterministic receipt field exactly,
  against the current sources and runtime.
- **Audit counters:** the `reads` record, including `audited_opens`, must
  equal the **exact** counters of the fixed path (`expected_receipt_reads`):

  | Counter | Count | Opens |
  |---|---|---|
  | Each metadata input and the registry | 2 reads | read + closure |
  | Video | 2 reads | seal + closure |
  | GLB | 3 reads | seal + closure + core loader |
  | Assembly outputs | 3 writes, 7 reads | exclusive staging writes; 2 verification reads per output + 1 bundle loader read |

  An erased, emptied, extended or rehashed audit is refused.
- **Historical budget:** the recorded budget is a **historical
  observation**. Its fixed keys, method, caps and planned bytes must match
  and its values must lie within bounds, but the bounds do not authenticate
  it.
- **Current usage:** current read-only usage must stay within the 96 KiB
  subtree, 12 MiB scope and 512 KiB metadata caps, counting retained
  partials and siblings. No free-floor or reserve check applies, because
  nothing is written. Real write checks are unchanged.
- **Final closure:** a fresh input, source and runtime closure, then an
  output reseal. The receipt's own claims are never trusted.

**CLI** (production pin only; no hash, object, frame, ordinal, K, origin,
seed, force or overwrite flags):

```
python -B -S tools/prototype_bootstrap.py tools.workbench_hb_scene3_execution_control inspect
python -B -S tools/prototype_bootstrap.py tools.workbench_hb_scene3_execution_control assemble --out .cache/video-workbench/derived/hb-metric-control-000003-v1/assembly-v1
python -B -S tools/prototype_bootstrap.py tools.workbench_hb_scene3_execution_control validate --out …/assembly-v1
```

Exit codes: 0 ok; 1 error; 2 usage; 3 blocked. A refusal prints a one-line
JSON record to stderr.

## Developer checks (no raw data, no real GLB or video bytes)

- **Real accepted metadata** (pure: 19 records, no GLB, video or raw opens):
  - registry and relations pass, with ordinal 0, the expected seed label and
    raw K;
  - outputs build to bundle 819 B, clip 18.3 KB and sidecar 3.5 KB;
  - the prefix is not attested.
- **Relation refusals** (each refused):
  - observed mask claimed checked, point drift, verdict changed, an altered
    artwork record (hash recomputed);
  - seed ordinal drift, **pose translation ×1000** (caught by the
    frozen-point depth/projection consistency check), binding match count 2,
    binding label drift;
  - missing final closure, independent acceptance not accepted;
  - a guessed origin shift, an origin claimed verified, association ordinal
    set;
  - physical FPS, decimation, a rescaled K;
  - a registry claiming the selection reviewed, and a label mismatch.
- **Stage mechanics** on a temporary copy of the metadata. The GLB and video
  were sized zero placeholders whose seals were stubbed, so no real asset
  was read. Free disk was simulated in the dev script only.
  - `inspect` → ready, with no annotation, camera or payload labels.
  - `assemble` → four files totalling 36 940 B, and `validate` → valid.
  - A tampered sidecar is refused by `validate`. A second assembly into the
    same directory is refused.
  - An input changed after the receipt fails the post-publication closure;
    all four owned outputs are quarantined with `failure.json`. A retry in a
    new attempt then fits the subtree (75 KB ≤ 96 KiB) and validates.
  - On a foreign collision on `clip-proposal.json`, the foreign bytes are
    kept and only the owned bundle is quarantined.
  - A retained 60 KiB staging alias is refused before any write.
  - Source drift is refused before publication.
  - A TEST_ONLY pin on the production registry is refused, and `--out`
    outside `assembly-v1` is refused.
- **Combined repair 1, RAM only** (no fixtures, links, simulated free space
  or file-writing tests, because the machine is below the floor; GLB and
  video not opened):
  - **Real chain:** the real accepted metadata chain still passes (ordinal
    0). The recomputed checker digest equals `d0c29333…`, and the real
    placement passes complete geometry validation. Real video binding
    passes, and production outputs keep family `hb_primesense_640x480`.
  - **Placement refusals:** depth `True`; residual `-1`, `999`, `True` and
    an inconsistent `5e-4`; face `True`, `-1`, `118136`, `82439.0` and a
    different face; barycentric `[2, 0, -1]` and a 1e-9 perturbation.
  - **Artwork and marker refusals:** marker barycentric `[2, 0, -1]`, marker
    missing `normal`, both digests replaced, an extra artwork key, and
    `patch_faces: True`.
  - **Video path refusals:** `derived/other/unsealed-video.mp4`, `../…`,
    `a/../…`, drive-letter, absolute and non-string paths.
  - **Closure and ancestry:** `_closure` reports an escaping path and a
    missing file as refusal values, and `_safe_dir` refuses a directory
    outside the repository.
  - **TEST_ONLY family:** TEST_ONLY outputs carry
    `TEST_ONLY_hb_primesense_640x480` on the bundle, clip provenance,
    sidecar and flags, and pass `check_outputs`.
  - **Exact audit counters:** asserted in `assemble` before the receipt and
    in `validate_output`. They are derived structurally and **not yet
    exercised by a file run**; the parent's file tests confirm them once
    the reserve is restored.

## Limitations

- **Disk space blocks the actual run today:** this machine had **0.33 GiB
  free**, below the registry's 1 GiB free floor. The actual `assemble` is
  refused (`Blocked`) until space is freed; the floor is not relaxed.
- **No end-to-end run on real assets:** the actual GLB and video were not
  read by the implementer, and the actual assembly is not launched here
  (parent-owned).
- **Unresolved:** observed SAM review, selection review, registration,
  serving admission, runtime execution and evaluation, and physical
  principal-point calibration.
- **Portability:** 118 136 triangles exceeds the 20 000 target; nothing is
  decimated.
