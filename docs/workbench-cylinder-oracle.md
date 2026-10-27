# Independent faceted-cylinder rendering checks

`tools/test_workbench_cylinder_oracle.py` compares the actual RGBA renderer with
an independent manufactured surface reference. These checks exercise artwork
footprint, axis orientation, straight-alpha sampling, current-mask clipping,
partial geometry and foreground depth occlusion. They do not use videos, model
weights or evaluator labels.

The surface is a vertical faceted cylinder of radius 0.25 metres. Its front
cross-section uses known circular endpoints joined by straight chords. The
artwork has a fixed 0.32-metre tangent-chart extent and an asymmetric texture
whose colours identify both texture axes. Geometry, support and marker inputs
must remain unchanged across camera poses.

The reference intersects camera rays with the extruded chord planes using
two-dimensional linear systems and explicit height bounds. It does not call
the production raycaster, triangle rasterizer, surface-point reconstruction,
chart traversal or renderer to calculate expected pixels. The tested surface is
the actual faceted mesh rather than an ideal smooth cylinder.

Known source geometry defines the supported connected component. Removing a
facet creates a real gap and disconnects unsupported regions from the anchor.
A separate foreground rectangle has no artwork support but must still occlude
paint behind it. Its expected depth comes from an independent ray-plane
intersection. An observed half-frame mask separately verifies pixel clipping.

The chart remains a tangent projection, not a geodesic cylinder unwrap.
Physical footprint comparisons apply to that declared chart. They do not prove
distortion-free wrapping or attachment to a real curved object.

Discontinuous texel choices within `1e-7` texel and geometric boundaries within
`1e-7` metre are counted explicitly. The tests do not assume a universal
one-pixel allowance. A manufactured rotated/translated pose exposed an additional
strict mismatch outside these exclusions. That check is retained as an explicit
expected failure pending resolution, rather than widening the tolerance or
changing production rendering in this verification slice.

Run:

```text
python -B -m unittest tools.test_workbench_cylinder_oracle -v
```

Read expected failures separately from passing checks. A successful software
suite containing an expected failure does not close the corresponding numerical
correctness requirement. Existing RGB sampling and default export behaviour
remain unchanged by this test-only addition. Real-video attachment accuracy,
automatic geometry preparation and phone performance remain separate gates.
