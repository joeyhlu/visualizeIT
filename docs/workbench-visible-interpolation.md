# Visibility-filtered surface interpolation

The CPU mesh rasterizer computes original-triangle, perspective-correct surface
barycentrics only for pixels that pass its existing coverage, positive-depth and
strict-nearer tests. Pixels outside the triangle or hidden behind previously
written geometry no longer receive the discarded surface-interpolation product,
reduction and division.

This preserves the existing per-retained-pixel arithmetic order and float64
precision. Projection, fixed-point edge ownership, near-plane clipping, guard
bands, depth tests and ties, face order, ROI, raster statistics, masks and artwork
sampling are unchanged. It introduces no persistent cache, mesh simplification,
texture filtering change or additional public option. Unwritten barycentrics
keep their previous values. Existing source-hash compatibility checks still
apply; an old artifact does not become compatible merely because pixels match.

The motivation is a saved historical CPU mug-render profile: 2.667 seconds total,
with rasterization at 2.650 seconds cumulative and triangle rasterization at
2.190 seconds cumulative over 14,386 calls. These nested times overlap. That
profile used older source and does not establish current latency or the gain
from this change. Narrowing the interpolation also adds visibility-indexing
work; a speed improvement must be measured on representative inputs.

## Software verification

The tests load the actual published geometry source from commit
`c02fcb89074a283f63f4967fd39d9580e55ff230`, verify its SHA-256 and compare it with
the candidate. Preserve that commit in the test checkout's Git history.

```sh
python -B -m unittest bench.test_workbench_visible_interpolation tools.test_workbench_visible_interpolation_independent bench.test_workbench_geometry tools.test_workbench_cylinder_oracle tools.test_workbench_mesh_export_independent
```

Controls cover perspective interpolation, clipping, winding and shared edges,
depth ties and occlusion, crop windows, deterministic randomized scenes and exact
RGB/RGBA output. Independent raster comparisons check bytes, including signed
zeros, rather than increasing a numerical tolerance. Test-only tracing inspects
the actual interpolation-array sizes to demonstrate reduced work; this is not
a timing benchmark or a tracking-quality result.

The cylinder oracle's previously documented single-texel expected failure remains
visible. Exact agreement with the old renderer does not resolve that independent
accuracy discrepancy or establish curved-surface attachment quality.

## Performance admission

Current-source real neural parity, representative renderer speed, memory and
whole-flow throughput remain unmeasured. This change does not establish the
offline target of 300 genuinely processed scored frames in at most 20 seconds,
including encoder drain, with current-frame SAM2, the pinned trained GoTrack head
and supported real artwork. Cold whole-job time must also be reported separately.

Performance evaluation requires the existing process-ownership, exact-source,
runtime, authenticated-assets, device and resource admission. Do not treat fast
failure frames, cached outputs or software-test duration as processed-video FPS.
The separate experimental preparation-reuse option remains off by default.
