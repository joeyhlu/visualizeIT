# Opening-only box-prompt SAM diagnostic (cracker, state-isolated)

`tools/workbench_sam_opening_box.py` runs ONE frozen prompt through the
accepted forward-first state isolation:

- **Box:** the whole box `[388, 48, 583, 278]` (xyxy, integer pixel centres),
  chosen by the parent from the opening RGB s000370 only.
- **Points:** the ORIGINAL three positive and three negative points. The drill
  negative (520, 165) deliberately lies inside the box.
- **Never used:** no scored RGB, ground truth, pose, depth or mesh, and no
  alternatives, sweeps, retries, registration or neural replay.

This is development-only. It makes no accuracy, phone or automatic-geometry
claim, and it never changes the original isolation output, masks or the six
frozen neural runs.

## Pinned inputs

| Input | SHA-256 |
|---|---|
| Accepted isolation tool `tools/workbench_sam_forward_isolation.py` (reused read-only) | `0aef88a8…` |
| Original control plan `derived/sam-forward-isolation/plans/original-prompt-forward-isolated-v1/plan.json` | `7304cf29…` |
| Prompt `derived/validation/ycb-cracker-opening-box-v1.json` (`cracker-rgb370-whole-box-v1`) | `15dce4be…` |
| RGB s000370 file / decoded | `822ffafa…` / `00090d9f…` |

## Commands (the parent runs them after review and tests)

```powershell
.cache\quality-windows\Scripts\python.exe tools\workbench_sam_opening_box.py prepare `
  --workspace .cache/video-workbench --plan-id cracker-opening-box-isolated-v1 --device cuda
# -> .cache/video-workbench/derived/sam-opening-box/plans/cracker-opening-box-isolated-v1/plan.json

.cache\quality-windows\Scripts\python.exe tools\workbench_sam_opening_box.py run `
  --plan .cache/video-workbench/derived/sam-opening-box/plans/cracker-opening-box-isolated-v1/plan.json `
  --expected-plan-sha256 <sha of plan.json>
```

**Prepare** is CPU only, with no torch or SAM import. It:

1. verifies the accepted isolation source SHA;
2. **rebuilds the original control plan** (full closure: sources, runtime, SAM
   files, checkpoint, config, all 120 RGB file and decoded identities) and
   requires byte equality with `7304cf29…`;
3. validates the prompt;
4. writes a deterministic plan whose `cache_identity` binds the prompt SHA and
   the exact call coordinates.

**Run:** it checks the external plan SHA and rebuilds everything before
loading any model. It rejects a pre-imported `sam2` and an output outside
`derived/sam-opening-box/runs/`, then uses one `gpu_stage`.

## Prompt call (SAM API, local pinned source)

`BoxSamAdapter` is the real `SamAdapter` with **only** `add_prompt`
overridden. It makes this call:

```
add_new_points_or_box(state, frame_idx=9, obj_id=1,
    points=float32[[414.5,110.5],[462.5,213.5],[545.5,88.5],[520.5,165.5],[430.5,290.5],[350.5,210.5]],
    labels=int32[1,1,1,0,0,0], box=float32[388.5,48.5,583.5,278.5],
    clear_old_points=True, normalize_coords=True)
```

- **Inside SAM:** it prepends the box corners with labels 2 and 3, then
  normalises by 640×480 and its image size.
- **No double conversion:** the +0.5 is applied once, and nothing is
  pre-normalised.
- **Recorded:** both actual calls (forward and reverse states) go into the
  ledger, the receipt and any failure record.

## Unchanged from the accepted isolation

- **States:** both have count 120 and released 9.
- **Forward:** 9..119 (111 yields), with each next frame released only after
  the previous mask is written.
- **Teardown:** full, before the reverse state.
- **Reverse:** a separate state, 9..0 (10 yields), never reading above 9.
- **Writes:** 120 unique writes; the duplicate 9 must exactly equal the
  forward seed.
- **Counters:** 2 adds, 2 inits and 2 teardowns.
- **Failure handling:** the causal ledger, cleanup ownership, outer cleanup
  inside the GPU stage, and failure snapshot with a stderr fallback.
- **Model setup:** the source grid, precision, offload and model are
  unchanged.

## Outputs

These go to `derived/sam-opening-box/runs/<plan-id>/`:

- 120 masks;
- `events.jsonl`;
- `preview.png` (the same nine indices: 0, 9, 10, 25, 45, 65, 85, 105, 119);
- `manifest.json` (producer `sam2.1_video_prompted_opening_box_state_isolated_forward_first`,
  mask set `sam2-opening-box-<plan-id>`);
- then `receipt.json`, or `failure.json` on failure.

**Size and budget:** the **8 MiB cap covers all new outputs, the plan
included**. The 1 GiB derived budget and 1 GiB free reserve are checked before
every write.

**Readback:** the exact 123-artifact readback (inventory, per-file SHA, strict
binary decode, content and frame ownership, no hard links) runs after the
post-run closure check and before the receipt.

## New qualifier `qualify_box` (descriptive consistency only)

All of these gates must pass:

- the frozen call and the actual calls agree (two calls);
- the duplicate forward/reverse seed is exact;
- the seed is non-empty;
- the seed's largest 8-connected component fraction is ≥ 0.95;
- the original three positive points are inside the seed and the three
  negative points are outside;
- every one of the 110 scored areas is ≥ 0.5 × the NEW seed area;
- all 109 consecutive scored IoUs are ≥ 0.5, with a zero union counted as 0;
- the preview IDs are the frozen nine.

**Original-seed equality and area 18329 are INAPPLICABLE.** They are marked
`applicable: false`, and the new seed's area and content hash are reported for
description only. The new seed is never compared against itself through the old
qualifier.

The parent's qualitative review of `preview.png` is mandatory and separate. A
failed gate or an unacceptable preview means no registration and no neural
replay.

## Pure API (no torch/SAM import)

| Function | Purpose |
|---|---|
| `validate_prompt(doc)` | Exact keys, schema, IDs, frame, 640×480 size and local 9. Integer, non-bool, in-image coordinates, ordered corners, exact original points and positives inside the box. Negatives may lie inside the box; they are reported in `negatives_inside_box`. |
| `box_call_arguments()` / `expected_call_record()` / `describe_call(kwargs)` | The exact call, and its JSON record. |
| `cache_identity()` | Prompt SHA, call and source pins, with their SHA-256. |
| `qualify_box(masks, *, actual_calls, duplicate_seed_equal, preview_local, …)` | The new gates. |
| `validate_plan_shape(plan)` | Schema, 120-frame accounting, thresholds, flags, preview, prompt/call, cache identity and original/accepted pins. |
| `BoxSamAdapter` | Subclass of `iso.SamAdapter`; only `add_prompt` is overridden, and `calls` records the actual arguments. |

The engine, ledger, writer, codec and readback are used from the accepted
module unchanged: `iso.run_state_isolated`, `iso.IsolationLedger`,
`iso.OutputWriter`, `iso.encode_mask_png`, `iso.decode_mask_png`,
`iso.verify_staged_outputs`.

## Limitations

- Untested until the parent's GPU run, including whether SAM handles the box
  together with an inside-box negative as intended.
- One prompt only, as a conditional diagnostic. A pass is consistency, not
  independent accuracy or a head improvement.
