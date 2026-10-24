# Qualified-mask replay graph (cracker development diagnostic)

`tools/workbench_mask_replay_graph.py` renders one static, source-bound graph.

**Input:** the completed evaluation
`.cache/video-workbench/derived/mask-replay-evaluation/eval-v1/replay-evaluation.json`
(schema `workbench_mask_replay_anchor_evaluation_v2`, SHA-256 `1a87b22f…`). It
is the only input. The tool runs no model and reads no labels, masks, meshes or
reference files.

## Command (the parent runs it)

```powershell
python tools/workbench_mask_replay_graph.py --out-prefix artifacts/video-workbench/progress/cracker-mask-replay-v1
# --input defaults to the pinned evaluation path
```

**Outputs** (fresh prefix, repository-contained, ≤ 2 MiB in total):

- `<prefix>.png`, about 1600×1200;
- `<prefix>.json`: a deterministic source receipt with the plotted point values,
  input SHA, run IDs, model pins, prompt SHA, original-control proof, flags,
  limitations, and the PNG and CSV SHA-256;
- `<prefix>.csv`: the graph values.

**Publication rules:**

- Each file is published atomically and never overwritten.
- The workbench 1 GiB derived+scratch budget and the 1 GiB free reserve are
  checked before publishing.
- The input is re-hashed just before publishing, and it is never copied or
  rewritten.
- There is no wall-clock timestamp, so JSON and CSV are byte-deterministic.

## Validation (before plotting)

**Document-level checks:**

- The SHA-256 is verified **before** the JSON is parsed.
- Schema v2 and an exact flag set: development-only, prompt and SAM seed
  changed, weights unchanged, and no human, phone, orientation, footprint,
  isolated-head, significance or pooling claim.
- Head and base pins (`27e06a51…`, `f7d127ab…`) and the prompt pin `15dce4be…`.
- `original_control`: passed, 220 rows, atol 1e-8 px, rtol 0.
- Exactly four cells (original/qualified mask × new head/frozen baseline),
  with the recorded mask sets `sam2-workbench-01` and
  `sam2-opening-box-v1-qualified`.

**Per cell:**

- 110 frames, all reference-eligible.
- Denominator 110.
- Status counts account for all 110, and `ok` equals the measured count.
- The 110 recorded rows re-derive the measured count, the conditional
  median/P95 (NumPy linear) and the ≤ 5 / ≤ 10 coverage counts over the
  **full** denominator.
- Pose and render counts are consistent.
- A missing metric must be null and is shown as "unmeasured", never as zero.

The plotted numbers are the recorded values. The rows are used only to
validate them.

**Run IDs:** these are not recorded in the evaluation JSON. They are pinned in
the tool (original `run-20261021T030702-475adc39`, new
`run-20261021T060541-e8894b79`) and labelled that way in the receipt.

## Panels

1. Conditional anchor median (solid) and P95 (light), in px720, with the number
   measured per cell (109/110). Dashed lines mark the provisional targets:
   median 5 and P95 10.
2. Validated poses vs artwork rendered, as counts of 110. These are
   availability, **not** accuracy.
3. Failure-aware coverage at ≤ 10 and ≤ 5 px720, as count/110 and percent. The
   failed frame stays in the denominator.
4. Limitations, original control, run IDs, mask sets, and the prompt and source
   SHAs.

## Interpretation limits

- This is a development dataset-reference 2-D anchor diagnostic on one clip,
  and adjacent frames are not independent replications.
- The weights are unchanged. Only the opening prompt (RGB s000370 whole box +
  original points) and the SAM seed changed. This is not new training.
- The median still misses the < 5 px720 target in every cell. A P95 under 10 is
  not the whole attachment gate.
- No statistical test was performed and nothing is pooled. Head-vs-baseline
  differences are not presented as superiority. Independent human accuracy,
  phone quality, orientation and footprint are unverified.

## Pure API (for parent tests)

| Function | Purpose |
|---|---|
| `load_source(path)` | Bounded regular file inside the repo; pinned SHA checked before parse; strict JSON. |
| `validate(doc, *, source_sha256=None)` | Schema, flags, pins, control, cells, rows and mask sets. |
| `extract_points(doc, *, source_sha256=None)` | Four validated point dicts in fixed order. |
| `provenance(doc, sha, points)` / `json_bytes` / `csv_bytes` | Deterministic exports. |
| `render_png(points, sha)` | Lazy matplotlib (Agg); refuses clipped text. |
| `build(input, out_prefix)` | Validate → render → budget → publish. |

Importing the module imports neither matplotlib, torch nor any evaluator.
