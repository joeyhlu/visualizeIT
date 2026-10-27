# Selected PNG artwork on a saved metric mesh

The offline exporter can place the selected project PNG on the registered target
mesh, using authenticated saved metric poses and current observed masks. This
is an optional output path. It does not run segmentation or a pose model,
prepare a new mesh, recover tracking or generate a product model.

The existing default is still `--surface-mode planar`. Select the new path with
`--surface-mode mesh`. The remaining required workspace, run, project revision,
bundle, admission, output and playback-FPS arguments are unchanged. For example:

```text
python -B -m tools.workbench_offline_export --help
```

Mesh mode also accepts `--max-chart-stretch NUMBER`, using the existing optional
local chart-distortion guard. There is no calibrated default threshold. The
limit must be finite and at least 1; it is refused in planar mode rather than
silently ignored. Mesh mode currently refuses `--experimental-tracking`, whose
reviewed causal admission remains specific to planar charts.

## Placement and texture

Mesh mode retains the saved face/barycentric anchor. Scale, rotation and opacity
use the existing placement record. Nonzero tangent or bitangent offsets refuse:
moving a persistent mesh anchor needs a separately admitted placement path.
Planar-mode offset controls remain available.

The selected PNG is straight RGBA. Its pixels are centered in a transparent
square without resizing, preserving aspect ratio. Odd extra padding goes to the
bottom or right. The square has at least eight texels per edge, up to 1024. When
an input is smaller than eight pixels, the physical canvas is enlarged so that
the selected PNG's longest dimension still has the requested physical size.
The report records selected dimensions, exact padding and rectangle size.

The sampler uses floor/clamp nearest texel cells, with PNG top toward the
marker's positive bitangent. This is a discrete texture convention; it does not
promise continuous subtexel centering. The derived draw marker rotates the
tangent frame and scales its canvas, while keeping the original surface anchor
and geometry intact. Its digest binds the padded texture identity. Original
saved artwork records and `FrameResult` evidence are not rewritten.

## Visibility and composition

`render_marker_rgba` samples the actual mesh with the existing depth,
front-facing, observed-mask, support, fold and overlap gates. Color and alpha
use two CPU texture passes sharing one geometry raster. Fully transparent
texels contribute no artwork. The compositor applies placement opacity with
straight alpha and round-half-up RGB output.

The RGBA path uses conservative triangle bounds during patch traversal, so a
large triangle crossing the artwork region is retained even if all its vertices
lie outside the reach square. This does not join disconnected faces or add
geometry. The legacy RGB path retains its existing default traversal. Explicit
stretch limits and current support cannot be waived by a supplied old patch.

Geometry statistics distinguish visible geometric pixels from nonzero-alpha
artwork pixels and pixels actually composited after opacity. Recovering, lost,
blocked or unavailable poses retain source RGB. Empty masks, transparent artwork,
unsupported patches and invisible mesh regions likewise produce no artwork.

Depth checks establish visibility only against the registered mesh. The current
mask is still required for other occlusion. Neither source is an independent
visibility or surface-identity label, and a wrong mask can still yield a wrong
result. This feature does not establish real-video curved attachment accuracy.

## Export identity and limits

Mesh outputs use `offline_saved_pose_mesh_artwork`. Their manifest includes the
derived marker, PNG and padded texture hashes, mesh/support hashes, placement,
chart limit, texture convention and per-frame geometry statistics. Original
model, pose, calibration, source, project and mask identities remain required.
All scheduled frames are accounted for, including failures.

The existing bounded FFV1 export, complete decoded-pixel verification, exclusive
new output directory, source reauthentication and storage checks remain active.
Playback uses explicit constant FPS and contains no copied audio. Inputs stream
one frame at a time; no full-frame dump is written. Two texture passes add work;
no speed, memory, phone or realtime claim is made.

```text
python -B -m unittest bench.test_workbench_rgba_marker tools.test_workbench_mesh_export tools.test_workbench_mesh_export_independent
```

These tests manufacture metric planes and a faceted cylinder, RGBA quadrants,
masks and saved evidence. Independent checks intersect camera rays with a known
plane, evaluate the declared texel rule and alpha equation, then compare decoded
video pixels. No private footage, model weights or neural execution is required.
