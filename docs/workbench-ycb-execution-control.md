# Execution-only supplemental YCB CAD controls

`tools/workbench_ycb_execution_control.py` turns one **reviewed** supplemental YCB candidate
(objects 2, 4 and 18) into a `workbench_metric_bundle_v1` for execution-only experiments.

It does not do any of the following:
- set the accepted proof's full-UV `serving_render_pass` or independent correspondence to true;
- claim accuracy, mobile readiness or automatic RGB preparation;
- satisfy or alter the 640x480 neural serving admission;
- launch inference;
- touch the original `blocked_unverified_uv` candidates or any proof/GPU record. These are
  read only.

## Contract (accepted prospectively)

Both fixed opening frames (logicalIndex 0 and 9) must PASS the raw-PLY vs loaded-GLTF GPU
reference (`tools/workbench_ycb_gpu_reference.py`). That reference uses unchanged relative
criteria, including informative opposite-UV discrimination. The geometry, PNG, prepared proof,
runtime and tools must all have exact provenance. The parent must also review qualitative
RGB/render placement on both frames.

## Commands (run in the pinned GPU Python environment; all CPU only)

```
# 1. Plan and print every pin. Writes nothing and approves nothing. --planned-out is needed
#    for the planned_out and bundle_sha256 pins.
python -B tools/workbench_ycb_execution_control.py inspect --object-id 2 \
    --prepared .cache/video-workbench/derived/uv-proof/attempt-02/object-000002 \
    --expected-prepared-sha256 <prepared.json sha> \
    --evidence <gpu reference run dir> --expected-evidence-sha256 <gpu-reference.json sha> \
    --planned-out .cache/video-workbench/derived/ycb-exec-controls/object-000002-v1

# 2. After the parent writes the qualitative receipt and the map entry (--out must equal planned_out):
python -B tools/workbench_ycb_execution_control.py emit --object-id 2 \
    --out .cache/video-workbench/derived/ycb-exec-controls/object-000002-v1

# 3. Launch gate, run immediately before loading the bundle (same pinned GPU Python):
python -B tools/workbench_ycb_execution_control.py validate \
    --out .cache/video-workbench/derived/ycb-exec-controls/object-000002-v1
```

**Interpreter:** `inspect` and `emit` call the GPU tool's `preflight`, which runs the frozen
proof validation and the runtime pins. In an interpreter without the pinned pyrender/trimesh
runtime they fail closed. That is what happened in the system Python: "re-derived
source-convention checks differ".

**What `inspect` re-validates:**
- the prepared proof and the runtime closure;
- the exact evidence inventory, which must contain only `gpu-reference.json` plus 14 hashed
  artifacts;
- the record's exact key set, thresholds, tool SHA, proof module, binding, runtime pins,
  source lines, source-reference hashes, texture, opposite hypothesis, scratch removal and
  false claims;
- both frame rows, including clean backend and texture equality and renderer close evidence.

**How a frame pass is checked:** each frame's comparison is **recomputed** from the stored
8-bit RGB PNGs and float32 depth with the GPU tool's own `compare_hypotheses`, at the
re-derived samples (128 to 1024 per frame). It must be `pass` again. The recorded metrics must
also agree with the recomputed ones within 1/255 for RGB and 1e-3 mm for depth. Recorded
booleans are never trusted.

**Geometry and seed checks:**
- The candidate GLB geometry must equal `float32(PLY mm × 0.001)` metres, in source face order.
- The seed is the final onboarding pose (logicalIndex 9), read from the authenticated
  inference manifest. It must equal the proof's authenticated final opening frame.

**Not supported:** `--map`, `--force` or any caller approval.

## Reviewed map (parent-owned; fixed path)

The map lives at `tools/workbench_ycb_execution_controls.reviewed.json`. The worker created no
map and no entries. Every key is mandatory, and there are no optional bypasses.

```json
{
  "schema": "workbench_ycb_execution_controls_reviewed_v1",
  "adapter_sha256": "<sha of tools/workbench_ycb_execution_control.py>",
  "gpu_reference_tool_sha256": "<final repaired GPU reference tool sha>",
  "proof_tool_sha256": "764f58f6ac93d4b85e89fdf41acb6dba944ddc4520a583439c3a3232c9676959",
  "controls_tool_sha256": "<current tools/workbench_ycb_controls.py sha>",
  "contract": "<exact CONTRACT string from the adapter>",
  "entries": {
    "2": {
      "object_id": 2, "canonical_ycb_id": "003_cracker_box",
      "planned_out": ".cache/video-workbench/derived/ycb-exec-controls/object-000002-v1",
      "bundle_sha256": "<from inspect --planned-out>",
      "prepared_dir": ".cache/video-workbench/derived/uv-proof/attempt-02/object-000002",
      "prepared_sha256": "...", "evidence_dir": "<project-relative GPU run dir>",
      "evidence_record_sha256": "...", "candidate_glb_sha256": "...", "candidate_receipt_sha256": "...",
      "png_sha256": "...", "ply_sha256": "...", "manifest_sha256": "...",
      "runtime_pins_sha256": "<from inspect>", "seed_frame_id": "s000370",
      "principal_point_origin": "integer_centers",
      "qualitative_receipt": "<project-relative path>", "qualitative_receipt_sha256": "...",
      "overlays": {"361": {"path": "<rel png>", "sha256": "..."}, "370": {"path": "<rel png>", "sha256": "..."}},
      "reviewed_by": "...", "reviewed_at": "..."
    }
  }
}
```

**Pins come from `inspect --planned-out`:** the entry's pins must equal the `pins` block that
`inspect` prints. That block includes `planned_out` (project-relative, strictly inside
`.cache/video-workbench/derived`) and `bundle_sha256` (the exact bytes of the re-derived bundle,
including the seed pose, label, frame and `glb_path`). The map's four tool SHAs must equal the
current files. `emit` refuses any `--out` other than `planned_out`, and refuses any re-derived
bundle whose hash differs from `bundle_sha256`. A failed emission consumes `planned_out` (only
`failure.json` remains), so a retry needs a new reviewed `planned_out` and `bundle_sha256`.

**Map hash and the adapter:** the map pins the adapter's SHA. The adapter records only the
object's canonical **entry** SHA, not the whole-map SHA and not embedded in the code, so there
is no hash cycle. Freeze the map after this tool's final hash.

**What `validate` (the launch gate) does:**
1. It identifies the object only by the single reviewed entry whose `planned_out` is the given
   directory.
2. It re-runs the full plan from that entry's exact closure:
   - prepared proof and runtime pins;
   - every GPU evidence artifact rehashed, and both frames recomputed;
   - the manifest seed;
   - the qualitative receipt and overlay hashes.
3. It rebuilds all four output files deterministically and requires **byte equality** with what
   is on disk.
4. It requires the bundle to equal the reviewed `bundle_sha256`, and the loaded `MetricBundle`
   (exact seed pose, label and frame, GLB) to equal the plan.
5. It rehashes the outputs once more before returning.

Sidecar and receipt self-consistency is never trusted. A forged pose or flag with consistently
rewritten outer hashes, or any change to the evidence, prepared or qualitative inputs, fails.
Adding well-formed or even malformed entries for *other* objects does not affect this object's
validation.

## Qualitative receipt (parent-owned)

```json
{
  "schema": "workbench_ycb_execution_qualitative_review_v1",
  "object_id": 2, "evidence_record_sha256": "...", "prepared_sha256": "...", "candidate_glb_sha256": "...",
  "reviewer": "...", "reviewed_at": "...",
  "decision": "placement_acceptable_for_execution_only",
  "frames": [
    {"frame_id": 361, "rgb_sha256": "<authenticated opening RGB>", "overlay_sha256": "...", "placement": "acceptable", "notes": "..."},
    {"frame_id": 370, "rgb_sha256": "...", "overlay_sha256": "...", "placement": "acceptable", "notes": "..."}
  ],
  "claims": {"accuracy": false, "independent_correspondence": false, "mobile": false}
}
```

The overlay PNGs are read from the map paths. Each must match its pinned hash and must decode.

## Output (fresh, immutable, under `.cache/video-workbench/derived`)

The output directory contains exactly these files:

- `model.glb`: the original candidate GLB bytes, unchanged.
- `metric-bundle.json`:
  - `family` is `ycb_640x480`;
  - `principal_point_origin` is `integer_centers`, an *assumed* provenance;
  - the seed pose is in metres;
  - the seed label starts with `controlled_init:dataset_onboarding:`;
  - `seed_frame_id` is `s<final opening frame>`;
  - `metric_provenance` is `dataset_metric_cad`.
- `execution-control-sidecar.json`: `execution_only=true`, `independent_correspondence_verified=false`,
  `automatic_rgb_preparation=false`, `independent_tracking_accuracy=false`, `mobile_ready=false`,
  `full_uv_diagnostic_pass=false`, the claims (all false), the contract, the units, the seed,
  the hashes and `map_entry_sha256`.
- `emission-receipt.json`: the evidence closure summary, the qualitative review, the pins,
  `map_entry_sha256`, the tool hashes and the budgets. No new inputs are acquired.

All four files are deterministic functions of the reviewed closure, and `validate` re-derives
them byte for byte.

**Bounds:** the output is capped at 32 MiB, within the 1 GiB derived budget and the 1 GiB
free-disk reserve.

**Atomic writes:** files are written with exclusive creates into a staging directory, then the
directory is renamed into place in one step. On a write failure, staging is removed and
`<out>/failure.json` records the failure honestly. Validation failures write nothing.

## Risks and limits

- **Shared backend:** the GPU reference shares pyrender's rasterizer, sampler, mipmaps and
  shader with the candidate, so common-mode faults are invisible.
- **Not landmark correspondence:** passing is relative evidence only, not printed-landmark
  correspondence.
- **Requires the pinned GPU environment:** the closure re-derives the proof source lines and
  runtime pins, which include absolute package paths. `inspect`, `emit` and the evidence it
  checks therefore require the same pinned GPU Python environment and project location.
- **Renderer identity is recorded only:** `renderer_factory` is not re-derived, because the
  CPU adapter does not import the renderer.
- **Exact record schema:** the GPU record must have *exactly* the current
  `workbench_ycb_gpu_reference_v1` keys. If the GPU tool's schema ever changes, this adapter
  must be updated and re-reviewed.
- **Seed frame must be a selection frame:** using the bundle in the workbench service still
  requires the clip's reviewed selection frame to equal `seed_frame_id`, and still requires
  640x480 serving admission.
