# SAM 2.1 forward-first state-isolation ablation (original cracker prompt)

`tools/workbench_sam_forward_isolation.py` is a prospective, separately pinned
segmentation experiment. It is not a retry of the closed dynamic-control
packet or of the failed prompt-v3.

**Question:** does same-state reverse-prefix memory influence the original
forward predictions on `dev-ycb-003-cracker-box`?

**Why:** production `segment_prompted` prompts local 9, then propagates in
reverse 9→0 and forward 9→119 in ONE SAM inference state, so the prefix
memories feed the forward pass.

**What changes:** this ablation changes **only** the state history and order.
The prompt, weights, config and settings are fixed. The cause of the original
fragmentation remains **unproven**, and either outcome is conditional evidence
only.

## Frozen inputs (no new prompt selection or tuning)

- **Clip:** `dev-ycb-003-cracker-box` (`afbb1869…`), 120 frames s000361..s000480.
  - Prompt: local 9, s000370.
  - Scored: s000371..s000480 (110 frames).
- **Prompt:** the original selection `4efc716f…` from
  `derived/validation/ycb-reviewed-selections-v1.json` (`bd601c76…`).
  - Positive points: (414,110), (462,213), (545,88).
  - Negative points: (520,165), (430,290), (350,210).
  - Point conversion: the same +0.5 pixel-centre conversion as production.
- **Original mask set:** `sam2-workbench-01` (manifest `f8fc14a5…`).
  - Seed `m00009.png`: file `1c8585a8…`, content `7f6bcc14…`.
  - Seed area 18329; largest 8-connected component 0.980359.
- **SAM:** checkpoint `a2345aed…`, config `configs/sam2.1/sam2.1_hiera_b+.yaml`.
  - Settings unchanged: `apply_postprocessing=False`, `vos_optimized=False`,
    video and state offloaded to CPU, and the deterministic position grid
    from `bench.quality_sam2`.
- **Never read:** no evaluator, ground truth, pose, depth, mesh, scored
  annotation or other mask set is used as a model input.

## State-isolation contract

1. **Same frame count:** both states keep `num_frames = 120` and the original
   indices. A shortened reverse state would change
   `max_obj_ptrs_in_encoder = min(num_frames, …)`, so it is forbidden.
2. **Forward first:**
   - a fresh `_CausalFrames` (count 120, released 9) and a fresh `init_state`.
     `init_state` warms local 0, the opening RGB, and this is recorded;
   - the original prompt on local 9;
   - one forward propagation yielding exactly 9..119;
   - each next RGB released only **after** the previous mask is written;
   - no reverse call on this state.
3. **Teardown, then a separate reverse state:**
   - the forward state, iterator, reader cache and reader are destroyed, and
     CUDA memory is released;
   - then a **new** `_CausalFrames` (count 120, released 9) and a new state get
     the same prompt;
   - one reverse propagation yields 9..0, of which only 0..8 are kept;
   - the duplicate local 9 must binary-equal the forward seed;
   - nothing above 9 is read, and there is never a forward pass on this state.
4. **Combination:** masks are combined by authenticated local index and frame
   ID: exactly 120 unique masks, with the seed and all 110 scored masks from the
   forward state.
   - Expected counters: 111 forward yields, 10 reverse yields, 120 writes,
     2 prompt adds, 2 fresh state initialisations and 2 teardowns.
   - Every read, release, yield and write is recorded with its state label,
     local index and frame ID.

**Enforcement:** an engine-side ledger enforces release-before-read
independently of SAM's own frame guard. State reuse, a teardown that leaves
references, a shortened reader, missing, extra or out-of-order yields, future
reads, reverse reads above the prompt and a duplicate-seed disagreement are all
rejected.

### Cleanup ownership (repair 1)

Cleanup ownership starts at **reader allocation**. Every branch ends with a
teardown that makes independent attempts, each recorded with its result:

1. iterator close;
2. state close (only for a state this branch owns), after which `len(state)`
   must be 0;
3. frames/reader close (`close_frames`, if the adapter provides it);
4. device release (`release_device`, if provided).

A failure in one step never skips a later one. This also covers an
`open_state` failure after the reader was allocated, a wrong reader length, and
a reused or forbidden state, whose new reader is still closed.

**Errors:**

- A primary failure is re-raised unchanged, with the cleanup errors attached as
  `exc.isolation_cleanup_errors` and kept in the ledger.
- A teardown failure without a primary failure raises `IsolationError`.
- A failed forward teardown never starts the reverse state.
- Attempted and completed initialisation and teardown are recorded separately.

**Run level:**

- `PositionEmbeddingRandom.forward` is restored.
- Every reader still registered is closed.
- Adapter and predictor references are dropped.
- A final `_release_device_memory` runs.

All of this happens in an outer `finally` **inside** the `gpu_stage`, and each
step is attempted independently.

### Failure evidence (caller-owned ledger)

`run` owns an `IsolationLedger`, and the engine records each event into it as
it happens:

- reader allocation;
- state initialisation (attempted and completed);
- reads, releases, yields and writes;
- teardown steps;
- outer cleanup.

**Any** failure (initialisation, propagation, write, cleanup or verification)
writes `failure.json` with:

- the stage and the primary error;
- all cleanup errors;
- the ledger snapshot: non-null counters, every read, release, yield and branch
  ID, and the teardown attempts and results;
- the expected, written and missing frame IDs;
- the partial mask identities (hashes only, no arrays).

The 8 MiB cap and the workspace budget/free reserve are checked before every
write, including `failure.json`. If the reserve, cap or a CUDA fault prevents
writing it, the full terminal record goes to **stderr** with
`failure_json_persisted: false`. Partial files are kept, and nothing claims
completion.

### Readback verification before the receipt (repair 1)

`verify_staged_outputs` runs after every write and after the after-run
source/input/runtime closure check:

- **Inventory:** the exact set is `m00000.png`..`m00119.png`, `events.jsonl`,
  `preview.png` and `manifest.json` (the receipt is not written yet). Extra,
  missing, symlinked, hard-linked or non-regular entries are rejected.
- **Hashes:** each file's **actual bytes** are re-hashed against the hash
  recorded at write time. So a changed mask, events file, preview or manifest is
  rejected, and so is the *same pixels in different PNG bytes*.
- **Manifest:** read as strict JSON. Schema, clip and flags must match, the
  frame set must be exact, and each row must have exact keys, path, local index,
  **branch owner** (reverse for 0..8, forward for 9..119) and file hash.
- **Masks:** all 120 are strictly decoded (640×480, binary 0/255), and each
  content SHA and `empty` flag must match its manifest row.
- **Events and preview:** event counts must be 111/10/120, and the preview PNG
  must be 960×810.

Qualification uses only the verified masks. The receipt records the verified
inventory, actual per-file hashes and mask content proof. It is not
self-hashed: the parent hashes `receipt.json` externally after the process.

## Commands (the parent executes them; the implementer ran neither)

```powershell
# CPU only; no torch/SAM import; writes a fresh immutable plan
.cache\quality-windows\Scripts\python.exe tools\workbench_sam_forward_isolation.py prepare `
  --workspace .cache/video-workbench --plan-id cracker-original-prompt-v1 --device cuda
# -> .cache/video-workbench/derived/sam-forward-isolation/plans/cracker-original-prompt-v1/plan.json

# Explicit single-GPU execution with the externally supplied plan SHA
.cache\quality-windows\Scripts\python.exe tools\workbench_sam_forward_isolation.py run `
  --plan .cache/video-workbench/derived/sam-forward-isolation/plans/cracker-original-prompt-v1/plan.json `
  --expected-plan-sha256 <sha of plan.json>
```

**Prepare** is deterministic and has no timestamps. It pins:

- this tool, the bench helper sources and the SAM 2 package closure (all 39
  Python/YAML files);
- the checkpoint;
- the interpreter, package versions and device;
- the clip, the selection, and the original manifest and seed mask;
- all 120 RGB file hashes and decoded identities;
- the thresholds, preview IDs, budgets and flags.

**Run:**

1. Checks the plan SHA and **reconstructs the full plan**, requiring byte
   equality, before any GPU work.
2. Refuses a pre-imported `sam2`, and verifies every imported `sam2.*` module
   lies inside the planned root.
3. Uses one `gpu_stage` for the whole run.
4. Re-checks the full closure afterwards.

There is no `--force` and no automatic approval.

## Outputs (private staging; never registered)

These go to `derived/sam-forward-isolation/runs/<plan-id>/`:

- 120 binary 640×480 masks (`m00000.png`…);
- `manifest.json`, compatible with `workbench_observed_masks_v1`: file and
  content SHAs, `evaluator_inputs_read: false`, `cancelled: false`, and producer
  `sam2.1_video_prompted_state_isolated_forward_first` with state-isolation
  metadata;
- `events.jsonl`;
- `preview.png`: a 960×810, 3×3 grid of local 0, 9, 10, 25, 45, 65, 85, 105
  and 119 (s000361, 370, 371, 386, 406, 426, 446, 466, 480), in the layout of
  `.cache/workbench-ycb-mask-review.py`;
- `receipt.json`, written only after every identity, closure and budget check.

**Failures:** a failure writes `failure.json` with the stage, error,
expected/written/missing frame IDs, counters and partial masks, and keeps the
partial outputs.

**Limits:**

- every write is contained, checked against the 8 MiB cap and checked against
  the workspace budget **before** writing (1 GiB derived limit, 250 MB
  new-input limit, 1 GiB free reserve);
- no full-frame dumps;
- no model-cache growth.

## Frozen qualification

This measures descriptive consistency, **not** independent accuracy:

- **Seed component:** the seed's largest 8-connected component fraction is
  ≥ 0.95.
- **Seed prompts:** all three positive points are inside the seed and all three
  negative points are outside.
- **Seed identity:** the seed equals the original s000370 content exactly, with
  area 18329.
- **Scored areas:** every scored area (s000371..s000480) is ≥ 0.5 × the
  candidate seed area.
- **Consecutive IoU:** all 109 consecutive **scored** IoUs are ≥ 0.5. A zero
  union counts as IoU 0, and the last-prefix/first-scored pair is not counted.

There is no threshold relaxation or prompt change after results. Any failed
gate means no promotion and no neural replay.

**Review:** the receipt never claims a qualitative or human review. The
parent's review of `preview.png` is mandatory and separate.

**Flags:**

- `human_accuracy`, `independent_accuracy`, `phone`, `trained_head_test`,
  `registered_mask_set` and `promotion` are all false;
- `development_only` is true.

**After all gates pass:** only after every gate and the parent's review may the
parent separately pin or register the new mask set. It may then run a paired
cracker diagnostic with the unchanged neural weights and admission, keeping
the original suite. The six original paired runs (head `27e06a51`, base
`f7d127ab`) and their reports are untouched.

## Pure API (SAM-free; for the parent's fake-predictor tests)

| Function | Purpose |
|---|---|
| `run_state_isolated(adapter, *, frame_ids, prompt_index, points, labels, shape, write_mask, events=None, ledger=None)` | The engine. The adapter provides `make_frames(label, count, released, on_read)`, `open_state(frames)`, `add_prompt`, `propagate(state, start, reverse)`, `to_mask(logits)` and `close_state(state, frames)`, which must leave `len(state) == 0`. Optional `close_frames(frames)` and `release_device()` are called as independent teardown steps when present. |
| `IsolationLedger(frame_ids)` | Caller-owned progress ledger. `snapshot()` is JSON-safe: counters, events, reads, releases, per-branch init/yield/teardown records, written identities and cleanup errors. |
| `verify_staged_outputs(directory, *, recorded, frame_ids, prompt_index=9, shape=(480, 640), clip_id=…, preview_size=(960, 810))` | Post-write readback (above). Returns `masks`, `manifest` and `proof`. |
| `mask_record(local, mask, png, state, shape)` | The manifest row for one written mask. |
| `qualify(masks, *, original_seed, …)` | The frozen gates. |
| `iou` | IoU with the zero-union convention (0). |
| `largest_component_fraction` | Largest 8-connected component fraction. |
| `encode_mask_png` / `decode_mask_png` | Strict 0/255 single-channel PNG with exact shape. |
| `OutputWriter(directory, cap_bytes, budget_check)` | Fresh directory; contained names; cap and budget checked before every write. |
| `canonical_bytes` | Canonical JSON bytes. |
| `validate_plan_shape` | Frozen accounting, thresholds, flags and preview. |

Importing the module imports neither torch nor SAM 2.
