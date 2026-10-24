# YCB dataset-reference diagnostic graph

`tools/workbench_dataset_reference_graph.py` is a CPU-only reporting tool. Its
**only** input is the sealed post-run diagnostic:

`.cache/video-workbench/derived/dataset-reference/eval-v1/dataset-reference-projection.json`
(SHA-256 `83bccd35…`, schema `workbench_ycb_dataset_reference_projection_v1`)

It runs no model or inference. It never reads `evaluator.json`, raw
labels, masks, depth or frames, and it does not change any pipeline, cache or
earlier tool.

This is a development **dataset-reference** diagnostic. It is **not**
independent human attachment evidence:

- the initialisation uses known dataset poses (the onboarding seed);
- the CAD geometry and calibration (K, assumed `integer_centers` origin) are
  shared with the reference;
- the public YCB-Video dataset may have been seen in training;
- the phone is unverified;
- wrong-surface and recovery correctness remain unverified.

## Command

```powershell
python tools/workbench_dataset_reference_graph.py `
  --input .cache/video-workbench/derived/dataset-reference/eval-v1/dataset-reference-projection.json `
  --expected-input-sha256 83bccd35f98f70a46730d99bb15ec33883e6e6f3d421f13dc5c9410a451250b1 `
  --out-prefix artifacts/video-workbench/progress/dataset-reference-ycb-v1
```

This writes `<prefix>.png`, `<prefix>.json` and `<prefix>.csv` (about 0.3 MB in
total). Each file is published atomically through a hard link and never
overwrites an existing one. The input is re-hashed just before publication.

## Validation (before any plotting or publishing)

**Document-level checks:**

- **Input SHA-256:** the input must hash to `--expected-input-sha256`, and that
  value must equal the pinned `83bccd35…` for this graph version.
- **Plan and evaluator tool:** plan `3c85e9c0…` and evaluator tool `d62d47b8…`.
- **Pins:** head `27e06a51…`, base `f7d127ab…`, evaluator labels `c54534d6…`,
  inference `9b936de3…` and reviewed map `c5ae3b00…`.
- **Flags:** must be exact, with no human, phone or accuracy promotion.
- **Hash closure:** equal, with the terminal seal matching.

**Objects and runs:** objects exactly 2, 4 and 18, with the **original** runs:

- `…475adc39` (cracker box);
- `…0986ef46` (soup can);
- `…539343a7` (marker).

**Per object and role:** each must have exactly the two roles and 110 unique
anchor rows, all reference-eligible.

**Summaries recomputed from the sealed per-frame rows:**

- anchor median/P95 (NumPy linear percentiles);
- the failure-aware ≤5 and ≤10 px720 coverages over all 110 frames;
- measured and eligible counts;
- prediction-status counts;
- tracking and artwork-render counts;
- the SAM vs dataset IoU median.

**Missing metrics:** these must be null and are never accepted as zero.
Human metrics must stay `unverified` and null.

**Pure helpers for unit tests:** `validate_document`, `validate_object`,
`validate_role`, `build_rows` and `csv_bytes`.

## Panels

The new head and the frozen baseline are shown separately for each clip, and
nothing is pooled.

| Panel | Content | Scope |
|---|---|---|
| A | anchor error px720, median (bar) / P95 (tick) | conditional on the measured frames (109/110) |
| B | failure-aware anchor coverage ≤5 px720 (hatched) and ≤10 px720 (solid) | all 110 reference-eligible frames |
| C | observed SAM mask vs dataset `mask_visib`, median IoU | shared mask input; not head-specific |
| D | tracking-pose and actual artwork-render counts | availability only; **not** accuracy |

- **Provisional lines:** the 5 and 10 px720 lines are provisional targets, not
  a passing gate. All three objects miss them: P95 is above 10 px720 for both
  roles, and the medians are above 5.
- **Head vs baseline:** no reliable head improvement is established, and no
  statistical test was run. The subtitle says so, and no better/worse claim is
  made.
- **Tracking vs attachment:** high tracking availability is not accurate
  attachment. For example, cracker box tracking is 109/110 but artwork renders
  in only 54/110 frames.

## Values (validated against the sealed rows)

| Clip | New head median / P95 | Baseline median / P95 |
|---|---|---|
| YCB cracker box | 6.96 / 10.31 | 6.76 / 10.32 |
| YCB soup can | 8.69 / 11.07 | 8.67 / 11.03 |
| YCB marker | 9.50 / 12.40 | 9.42 / 12.40 |

## Outputs

- **JSON and CSV:** they record the exact underlying values, numerators and
  denominators, actual render counts and each metric's scope. They also carry
  the source SHA, the run IDs and the model pins.
- **Figure footer:** it carries the source SHA, plan, evaluator tool, model
  pins, run IDs and generation time (UTC).
- **No private data:** no image is embedded and no absolute path is written.
