# Saved-mask continuity audit

This tool describes registered observed masks without rerunning SAM or GoTrack.
It reads clip metadata and selected mask PNGs, authenticates their identities,
and writes a new JSON/CSV report. It never changes masks or tracking state.

```text
python -B -m tools.workbench_mask_diagnostics --workspace PATH --clip-id CLIP --mask-set SET --output NEW_DERIVED_DIRECTORY --area-ratio-min 0.5 --split scoring
```

The output must be a new direct child of the workspace's `derived` directory.
The default split is the clip's existing scoring window. `onboarding` and `all`
are available for separate diagnostic reports; no scoring window is rewritten.
Outputs are `report.json` and `report.csv`.

## Measurements

Each selected frame remains in the report, in source order. Observed masks have
area, fraction of image area, exclusive-maximum bounding box and pixel-center
centroid. Empty masks are observed masks with measured zero area and null box
and centroid. Missing entries are unavailable, with null measurements.

Only neighboring source indices with both masks available produce pair metrics:
spatial mask overlap IoU, area ratio and centroid displacement. Displacement is
reported in native pixels and at 720-pixel image-height scale. These are changes
between masks, not comparisons with independent segmentation labels.

IoU is null when both masks are empty. The area ratio is null when the previous
area is zero. A missing mask or source gap breaks the comparison. `pair_status`
explains these cases, rather than turning undefined values into zero or one.

The baseline is the first observed nonempty mask encountered in the selected
schedule. Earlier rows keep a null baseline; later observations do not backfill
them. A mask's area relative to that baseline can expose persistent changes.

`area_drop` and `low_baseline_area` use a strict comparison with
`area_ratio_min`, a finite value strictly between zero and one. The default 0.5
is an unvalidated diagnostic setting. Low-baseline episodes group flagged frames
only while source indices are consecutive. Missing masks and gaps close an
episode. The report includes exact frame IDs and counts; a timestamp span exists
only when genuine source timestamps are declared. Frame-index clocks retain
null elapsed time.

## Interpretation

An area decline can result from distance, viewpoint, occlusion, object exit or a
bad mask. Low temporal overlap can also reflect normal object motion. These
signals locate frames for inspection; they do not establish undercoverage,
object visibility, tracking loss, improvement or correctness.

Reports explicitly keep `quality_claim` false and `independent_accuracy` null.
The threshold must not be promoted into a tracking decision without a separate
validated experiment. These descriptive values do not replace independent mask,
attachment, identity or recovery annotations.

The audit does not read RGB pixels, evaluator masks, reference poses, depth,
meshes or model weights. Its clip binding authenticates registered metadata,
not the underlying RGB bytes. The mask producer is retained as recorded
provenance; the audit does not certify that producer's model execution.

## Integrity and resource bounds

The tool reuses the observed-mask registry and its file verifier. It validates
the bounded PNG header before decoding and checks file hashes, decoded content
hashes and any declared dimensions/empty flags. Registered corrupt or missing
files refuse; a legitimately absent manifest entry remains unavailable.
Evaluator-derived mask manifests refuse.

Limits are 1000 selected frames, eight million pixels per frame and eight MiB
per PNG. Only current and previous masks are retained. Clip, manifest, selected
files, configuration and source identities are rechecked before publication.
Existing workspace storage/free-space checks and exclusive output publication
remain active. Reports are capped at sixteen MiB combined, and a failed attempt
cleans only its own staging directory. No inference cache or model residency is
created.

```text
python -B -m unittest bench.test_workbench_mask_diagnostics tools.test_workbench_mask_diagnostics_independent
```

Tests manufacture rectangles with analytically known areas, centroids and
overlaps, then exercise missing/empty inputs, source gaps, clocks, integrity,
budget/publication failures and the actual CLI. Historical development masks may
be audited locally when their inputs are verified. Their availability does not
establish that another dataset, including the separate HB300 case, is present.
