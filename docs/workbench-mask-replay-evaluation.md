# Qualified-mask replay anchor evaluation (`tools/workbench_mask_replay_evaluate.py`)

This is a post-run, CPU-only development diagnostic. It compares the **original** cracker-box
run with the **new qualified-mask replay** using the fixed 2D anchor references reused from the
sealed dataset-reference result.

| | Run | Mask set |
| --- | --- | --- |
| Original | `run-20261021T030702-475adc39` | `sam2-workbench-01` |
| New replay | `run-20261021T060541-e8894b79` | `sam2-opening-box-v1-qualified` (manifest **file** digest `055c6071…`) |

- **Same for both runs:** head 27e06, base f7d1, admission 06, metric CAD, artwork, clip
  `dev-ycb-003-cracker-box`, mode `regression`, schedule `s000371..s000480` (110 frames per role).
- **Different:** a new opening-box prompt (`15dce4be…`) and SAM seed. The weights are unchanged.
- **Claims:** none about isolated-head improvement, human accuracy, phone quality, orientation,
  footprint or significance. Objects are not pooled. The original reports are never overwritten.

## What this tool never opens (all phases)

GLB/mesh files, SAM mask PNGs, depth, raw ground truth, `evaluator.json`, reference poses, sample
files, models, torch/SAM, the network and the GPU.

**Geometry and mask closure is parent-attested.** The parent hashes the GLB and every mask PNG and
records those hashes in its terminal record. The tool compares the attestation with:
- the pinned metric-bundle JSON and reviewed-map entry (GLB pin);
- the pinned mask-manifest JSON (inventory digest);
- the mask identities the result rows actually consumed.

It does **not** rehash GLB or PNG bytes itself, so there is no independent before/after rehash of
those files.

**Inference source closure** is the new run's own `report.dependencies` receipt (module hashes and
head identities recorded by the run), plus the launch and admission pins and the parent
observation receipts. This tool's own helper hashes describe the evaluator only and are not proof
of the executed source.

**What it may read:** this tool's source and helpers; the frozen original plan; config, prompt,
registration, clarification and selection files (hashed); the admission `contract.json` and
`parity/parity-receipt.json`; the metric-bundle JSON; the reviewed map; mask-manifest JSON; the
clip JSON (hashed); RGB images; run JSON/JSONL outputs (`run.json`, `results.jsonl`,
`report.json`, `neural-artwork.json`, `stages/*`); and the parent observation receipts (hashed).
Run overlay PNGs are rendered artwork, not observed SAM masks. They are not part of the inventory
and are not opened.

## Pins

| Item | SHA-256 |
| --- | --- |
| Accepted evaluator **source** | `d62d47b8…` (read as a specification; never imported or edited) |
| Frozen original plan `derived/dataset-reference/plan-v1/plan.json` | `3c85e9c0…` |
| Sealed reference `derived/dataset-reference/eval-v1/dataset-reference-projection.json` | `83bccd35…` (parsed only in `evaluate`, after every closure check and the budget preflight) |
| Head / base | `27e06a51…` / `f7d127ab…` |
| Admission `.cache/video-workbench/admission/attempt-06` files | `contract.json` `8e032e82…`, `geometry.json` `4a5b3490…`, `parity/parity-receipt.json` `8a163299…`; semantic contract `83c041f9…`, parity receipt `40746b0d…` |
| Launch `.cache/workbench-cracker-opening-box-neural-evaluation.py` | `cb9fe577…` |
| Registration / clarification v2 / prompt / selection receipt (`derived/validation/…`) | `b0be3906…` / `d55bbad5…` / `15dce4be…` / `bd601c76…` |
| Parent terminal observation `derived/validation/qualified-mask-pair-terminal-observation-v1.json` | `bbca0b3a…` (hashed only) |

The following come from the frozen original plan and are never re-derived from current files:
- the original-run inventory, mask manifest (file and canonical digests), clip path and SHA, and
  sidecar;
- the control directory, candidate GLB, bundle, map entry and reviewed map;
- the anchor, per-frame K, image size, RGB bindings and artwork.

The new mask manifest has a **file** digest (`055c6071…`, the value the launch and registration
pin). Its **canonical** digest (`canonical_sha256(manifest JSON)`, which the run's mask stage
records) is a different value. Both are checked.

## Phases and CLI

```
python -B tools/workbench_mask_replay_evaluate.py prepare --out <derived>/mask-replay/plan-v2
python -B tools/workbench_mask_replay_evaluate.py seal --plan <plan.json> --expected-plan-sha256 <sha> \
    --terminal-record <parent record> --expected-terminal-record-sha256 <sha> --out <derived>/mask-replay/seal-v2   # PARENT ONLY
python -B tools/workbench_mask_replay_evaluate.py evaluate --plan <plan.json> --expected-plan-sha256 <sha> \
    --seal <seal.json> --expected-seal-sha256 <sha> --out <derived>/mask-replay/eval-v2
```

**Rules for every phase:**
- Each phase writes a fresh directory inside `.cache/video-workbench/derived`, published through
  staging and an atomic rename. There is no overwrite and no `--force`.
- The **packet cap** is 4 MiB for plan, seal, evaluation JSON and CSV combined. The **workspace
  budget** is the complete `Workspace.require_budget` accounting: derived, scratch, runs, masks
  and suite against 1 GiB, new inputs against 250 MB, and at least 1 GiB of free disk after the
  write.
- Run the seal and `evaluate` with the **same interpreter** as `prepare`, because the runtime is
  part of the plan.

1. **`prepare`** reads only:
   - the frozen original plan (SHA, tool pin, constants, roles, origin, 110 frames, rotation
     tolerance, thresholds, head/base, run/clip/schedule/mask set);
   - the config and admission file hashes;
   - the admission contract and parity JSON (semantic digests);
   - the metric bundle (GLB pin and origin) and the reviewed-map entry;
   - this tool's sources and runtime.

   The new-run binding stays **deferred**: no new-run output is read, even though the run is now
   terminal. No metrics are computed. `plan.json` (`workbench_mask_replay_plan_v2`) holds every
   expected pin.

   **Plan validation**, on every later load, checks the external SHA **and** that the plan is
   identical to a fresh prospective rebuild. A plan rewritten with changed pins and a matching
   outer hash is therefore rejected.
2. **`seal` (parent-only)**, in order:
   1. Validates the plan and the typed terminal record (schema below).
   2. Takes a closure snapshot, and the expected-closure validator checks every digest against
      the plan **and** the fixed constants.
   3. `authenticate_runs` (below).
   4. Takes a second snapshot plus the plan and record hashes; they must be unchanged.
   5. Writes `seal.json` (`workbench_mask_replay_seal_v2`) with exact keys: `schema`,
      `tool_version`, `plan_sha256`, `tool_sha256`, `terminal_record{path, sha256}`, `closure`,
      `run_authentication`, `sealed_result_sha256_expected`, `labels`, `statement`.
3. **`evaluate`**, in order:
   1. Validates the plan (external SHA plus semantic rebuild) and the seal (exact keys, schema,
      plan, tool and reference pin).
   2. Re-reads and re-validates the terminal record from disk.
   3. Takes a fresh snapshot, runs the expected-closure validator, and checks the snapshot equals
      the seal.
   4. Re-runs `authenticate_runs` from fresh reads; the result must equal the seal.
   5. **Budget preflight:** reserves the remaining 4 MiB packet allowance (at least 1 MiB) and
      checks the workspace budget can absorb it.
   6. Re-hashes the plan, seal and terminal record and takes another snapshot.
   7. Only then: parses the reference (after its SHA check), runs the mandatory original
      control, and computes the new metrics.
   8. Before publishing: another snapshot and expected-closure check, the reference SHA, and
      re-hashes and re-validates the plan, seal and terminal record.
   9. Final size check against the reservation, then the packet and workspace budgets.

   Any failure leaves only `failure.json` (`comparison_emitted: false`, failure metadata only).
   A budget failure happens before the reference is touched.

### `authenticate_runs` (fresh reads; no reference content)

- **Static inputs:**
  - the metric bundle's `glb_sha256` and origin;
  - the reviewed-map entry (canonical SHA, GLB, bundle, `planned_out`);
  - admission `contract_sha256` = `83c041f9…` and parity `receipt_sha256` = `40746b0d…`.
- **Run documents (both runs):**
  - `run.json` is a valid RunSpec: exact run ID, clip, mode `regression`, both roles and the
    schedule;
  - `report.json` has `status: completed`, exact run, clip and mode, `scheduled_frames` 110, and
    each role with `result_rows` 110, `missing_rows` `[]`, and a stage `completed` with
    `frames` 110;
  - the masks stage is `completed`;
  - an exact 220-row FrameResult join.
- **RGB:** each scheduled RGB is rehashed and decoded with the FrameReader convention
  (`cv2.imdecode(IMREAD_COLOR)` → contiguous BGR→RGB → `array_sha256`). It must equal the frozen
  file and pixel SHAs and every row's `image_sha256` in **both** runs.
- **Config:**
  - The original config must have the mask set, the frozen map and sidecar,
    `execution_only_prepared_cad: true` and `independent_correspondence_verified: false`.
  - The new config must pin:
    - the new mask set and manifest file digest;
    - the prompt, registration, clarification and selection receipt;
    - `original_comparison_run_id`;
    - the frozen map and sidecar;
    - `serving_admission_file_hashes` equal to the admission pins.
  - It must also have a hex `surface_control_adapter_sha256`. If the original config recorded an
    adapter digest, the new one must equal it.
  - The development, execution-only and independence flags must be set.
- **Rows:** run, clip, role, frame, source index and decoded-RGB identity.
- **Backend identity (`validate_backend_identities`), per role:**
  - constant across all 110 frames;
  - all required fields present (`IDENTITY_REQUIRED`, plus `export_sha256` for primary and
    `head` for baseline);
  - pinned base, admission contract, origin, `metric_units: metres`, GLB and
    `quality_claim: false`;
  - hex head-state and install digests;
  - effective K equal to the frozen per-frame K;
  - primary installed and export head equal to 27e06; baseline with no installed head and
    `frozen_base_pose_head`;
  - the **complete original and new identity dictionaries exactly equal**. This covers profile,
    units, initialization, seed and target frame, origin, K, head-state digest and
    `install_digest`.
- **Artwork (both runs):**
  - schema `workbench_neural_artwork_v1`;
  - `record_sha256` equal to the canonical body digest and to the frozen record;
  - metres, origin, GLB, face, barycentric coordinates and `mesh_faces` equal to the frozen
    plan;
  - effective K equal to every frozen K;
  - new artwork **identical** to the original. A changed anchor is rejected, because the old 2D
    references cannot be warped.
- **Consumed masks (`validate_mask_consumption`), for both runs:**
  - The manifest JSON has schema `workbench_observed_masks_v1`, the clip, the mask set and
    `evaluator_inputs_read: false`.
  - The original manifest's file and canonical digests equal the frozen plan; the new manifest's
    file digest is `055c6071…`.
  - The run's mask stage is `completed` with `identity.mask_set`, `identity.manifest_sha256`
    equal to the canonical manifest digest, and the matching producer.
  - Every `observed` row needs `mask_frame_id == frame` and `mask_sha256 ==
    manifest.masks[frame].content_sha256`.
  - `unavailable`, `rejected` and `blocked` rows are kept as they are and must carry no mask
    identity.
  - Per-role mask-state counts are reported.
- **Inference sources (`validate_inference_sources`):** `report.dependencies.sources` has hex
  digests for `workbench_cli`, `workbench_contract`, `workbench_geometry` and `workbench_neural`.
  `dependencies.heads` has primary installed and export 27e06 with base f7d1, and baseline base
  f7d1 with no installed head. Original and new module hashes are reported for information only;
  other agents may legitimately have changed sources between the runs.
- **Attestation:** `validate_attestation(record, expected_attestation(...))`.

## Parent terminal record (`workbench_mask_replay_terminal_record_v2`)

The parent creates this record after the implementation is accepted, from the observed
exec 8461 exit-0 proof. The tool validates the record, then checks the actual completed
artifacts. A `true` field alone is never treated as success.

Every object below must have **exactly** these keys:

```json
{
  "schema": "workbench_mask_replay_terminal_record_v2",
  "run_id": "run-20261021T060541-e8894b79",
  "execution_id": "<non-empty, e.g. exec 8461>",
  "command": "<non-empty exact launch command>",
  "exit_code": 0,
  "worker_process_exited": true,
  "observed_by": "<non-empty>",
  "observed_at": "<non-empty>",
  "attested": {
    "geometry": {
      "glb_path": "<frozen control_dir>/model.glb",
      "glb_sha256": "<parent-computed GLB bytes = frozen candidate GLB pin>",
      "physical_bytes_hashed_by_parent": true
    },
    "mask_inventory": {
      "manifest_path": ".cache/video-workbench/masks/dev-ycb-003-cracker-box/sam2-opening-box-v1-qualified/manifest.json",
      "manifest_file_sha256": "055c60711a4b04169a108b9fb4dd0e8dac24756a27b0b7ce6e779ba771004057",
      "manifest_canonical_sha256": "<canonical_sha256(manifest JSON)>",
      "frames": <number of manifest entries>,
      "inventory_sha256": "<see below>",
      "physical_files_hashed_by_parent": true
    },
    "inference_execution": {
      "launch_path": ".cache/workbench-cracker-opening-box-neural-evaluation.py",
      "launch_sha256": "cb9fe577a631682adecda37aa4213bc2029f37adf33e3427dab7263c413b1c63",
      "admission_dir": ".cache/video-workbench/admission/attempt-06",
      "admission_file_hashes": {"contract.json": "8e03…", "geometry.json": "4a5b…", "parity/parity-receipt.json": "8a16…"},
      "report_dependencies_sha256": "<canonical_sha256(new report.json['dependencies'])>",
      "surface_control_adapter_sha256": "<new run config value>",
      "observation_receipts": [
        {"path": ".cache/video-workbench/derived/validation/qualified-mask-pair-terminal-observation-v1.json",
         "sha256": "bbca0b3a02e1751b5d8b4dea633ad0eebce851f53b16a61e3c24eb8692010ef7"}
      ]
    }
  }
}
```

**Typing rules:**
- `exit_code` must be an `int` equal to 0 (`false` and `0.0` are rejected).
- `worker_process_exited` and both `physical_*_hashed_by_parent` flags must be the literal `true`.
- `frames` must be a positive `int`.
- Digests are 64 lowercase hex characters.
- Receipt paths are unique.

**`inventory_sha256`** is `canonical_sha256` (the workbench contract's sorted-key, compact JSON)
of the list `[[frame_id, path, sha256, content_sha256], …]`, sorted by `frame_id`, over **every**
manifest entry. The parent computes it after verifying each physical PNG's byte hash equals its
manifest `sha256`.

**Observation receipts:**
- Must be `.json`, `.jsonl`, `.txt` or `.log` files inside `.cache/video-workbench`.
- Must not be under `masks/`, `models/` or `derived/dataset-reference/`.
- Are rehashed by the tool and must include the pinned terminal observation.

Every other `attested` value must equal what the tool derives from the pins, the pinned manifest
JSON and the run receipts.

## Mandatory original control (unchanged)

All 220 original rows are recomputed from the original FrameResults, the fixed 2D references and
the frozen anchor and K. They must match the sealed rows:
- exact `prediction_status`;
- the null pattern and values of `prediction_uv`, `error_native_px` and `error_px720` (rtol 0,
  atol 1e-8 px);
- exact ≤5 and ≤10 px720 membership.

The legacy summaries must also match: `A_anchor`, `C_anchor_reference_visible`, the status counts,
and legacy `D_availability`. Any mismatch stops the evaluation, and the tolerance is never tuned.

## Semantics (unchanged; independent implementation of the pinned legacy source)

**Prediction:**
- `pose_state == "tracking"` and a pose present, otherwise `no_tracking_pose`;
- metric units, a finite 4×4 with last row exactly `[0, 0, 0, 1]`, and a proper R (tolerance
  1e-4, never repaired), otherwise `invalid_prediction_pose`;
- `u = fx·x/z + s·y/z + cx`, `v = fy·y/z + cy`;
- z ≤ 0 gives `prediction_behind_camera`; outside `[-0.5, W − 0.5) × [-0.5, H − 0.5)` gives
  `prediction_out_of_image`.

**Error:** `e = ‖uv_pred − uv_ref‖` and `px720 = e · 720 / 480`, only for reference-eligible rows
with status `ok`. Eligibility and visibility never depend on a prediction.

**Missing rows:** a missing structural row is an integrity failure.

**Reference extraction:** object 2 only. Missing, duplicate or extra IDs are rejected. Both roles
must agree exactly on `reference_status`, `reference_uv` and `reference_visible`.
`reference_visible` must equal `anchor_visibility.status`, and visible implies eligible.

## Output (`workbench_mask_replay_anchor_evaluation_v2`, deterministic, no wall-clock timestamps)

- **`cells`:** four cells, `original_mask` and `new_qualified_mask` × `primary_neural` and
  `frozen_baseline`, each over all 110 frames. For both `all_reference_eligible` and
  `fixed_visible`:
  - `denominator` and `measured`;
  - conditional median and P95 (numpy linear);
  - failure-aware ≤5 and ≤10 coverage over the full denominator;
  - status counts.

  Each cell also has `availability`, where `validated_tracking_poses` is separate from the legacy
  `legacy_pose_present`. An empty denominator or sample gives null.
- **`original_to_new`:** per role, with any paired conditional comparison carrying its own
  denominator (it never replaces coverage).
- **`run_authentication`:** bindings, backend identity digests, mask consumption and mask-state
  counts, inference sources, and the new mask inventory. This block equals the seal.
- **`parent_attestation`, `rows`, `original_control`, `flags` and `notes`.**
- **`replay-evaluation.csv`:** one row per cell and subset.

Plotting is separate and consumes only the completed JSON.

## Public API (dependency-injectable)

**Numerical (unchanged signatures and behaviour):**
- `validate_schedule(rows, scheduled, roles)`
- `extract_reference(sealed_result, frozen_plan)`
- `project_prediction(anchor_m, K, row, image_size)`
- `score_anchor(reference, prediction, image_height)`
- `summarize_anchor(scored_rows, reference_subset)`
- `summarize_availability(rows)`
- `verify_original_control(original_rows, reference, sealed_result, *, anchor_m, K_by_frame, image_size)`
- `compare_replays(original_scores, new_scores)`
- `decode_pixel_sha256(raw, decode=None)`
- `score_role`, `build_cells`, `render_csv`, `validated_prediction_pose`, `proper_rotation`

**Validation (each raises `ReplayError`; malformed structures never default):**

| Function | Purpose |
| --- | --- |
| `validate_terminal_record(record)` | exact typed parent record |
| `expected_attestation(plan, new_manifest, new_report, new_config)` / `validate_attestation(record, expected)` | parent attestation |
| `mask_inventory_digest(manifest)` | `{frames, inventory_sha256}` |
| `validate_mask_consumption(rows_by_role, manifest, stage, *, mask_set, clip_id)` | the masks the rows consumed |
| `validate_run_documents(run_doc, report, run_id)` | returns the run config |
| `validate_backend_identities(original, new, plan)` | complete backend identity |
| `validate_artwork(art, plan, what)` | artwork record |
| `verify_run_bindings(plan, original_doc, new_doc, original, new, original_art, new_art, pixel_sha)` | config, row, identity and artwork bindings |
| `validate_inference_sources(report)` | the run's dependency receipt |
| `validate_expected_closure(plan, snapshot)` | every digest against the plan and constants |
| `validate_plan(plan, expected)` | semantic plan check |
| `validate_seal(seal, plan, plan_sha)` | exact seal |
| `verify_rgb(plan, decode=None)` | RGB file and pixel identity |

**Filesystem (all file opens are here):** `closure_snapshot(plan, record=None)`,
`hash_receipts(receipts)`, `authenticate_static_inputs(plan)`,
`authenticate_runs(plan, record, decode=None)`, `workspace_budget(extra)`,
`check_budget(planned, related)`, `preflight_eval_budget(related)`, `build_plan()`,
`prepare(out)`, `build_seal(...)`, `evaluate(...)`.

**CPU imports:** numpy at import time. `bench.workbench_contract`, `bench.vision` (cv2),
`bench.workbench_geometry` and `bench.workbench_cli` (numpy and contract only at import) load
lazily, are verified to come from their repository paths, and are part of this tool's own source
closure. There is no torch and no accepted-evaluator import.

## Assumptions to confirm at the parent review

- The original manifest `sam2-workbench-01` has `content_sha256` for every frame whose original
  row is `observed`. If not, the evaluation fails closed.
- Each run report's `stages[role]` has `frames`, as the accepted evaluator already required for
  the original run.
- The backend identity is identical across frames within a role, and between runs (including
  `install_digest` and `seed_target_frame_id`). If either differs, the evaluation fails closed
  and lists the differing keys.
- The new run has no `errors/` receipts.

**Worker verification:** a syntax and import check only. No prepare, seal or evaluate was run, and
no active outputs, reference content, plans, manifests, masks, meshes or models were opened.
