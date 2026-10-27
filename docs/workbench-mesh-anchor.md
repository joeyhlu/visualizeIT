# Reviewed mesh placement for saved-video artwork

The offline exporter can place the selected PNG at a new point on a prepared
metric mesh. Supply `--surface-mode mesh --mesh-anchor PATH` alongside the
existing required export arguments. The default export still uses the original
saved anchor; planar mode does not accept this option.

Create a small JSON request after inspecting the registered initialization
frame. Coordinates use source-image pixel centers, before any viewer resizing
or cropping. Replace the example identities and coordinates with the actual
clip and its authorized seed frame:

```json
{
  "schema": "workbench_mesh_anchor_v1",
  "clip_id": "your-registered-clip",
  "frame_id": "your-seed-frame",
  "pixel": [320.0, 240.0],
  "reviewed": true,
  "reviewed_by": "operator"
}
```

The request must have exactly these fields, fit within 64 KiB, and name the
same seed frame as the original reviewed selection and metric bundle. A click
on a later scoring frame is refused. Coordinates must be finite numbers within
the image, inside the registered observed object mask, and intersect the mesh.
The operator review declaration is recorded provenance; the exporter cannot
independently establish that a human inspected the frame.

The exporter authenticates the original saved run and artwork record first.
It casts the click through the registered calibration and authorized seed pose
onto the frozen mesh, producing a face and barycentric coordinates. That point
remains fixed in object coordinates while the saved per-frame poses move it.
The original physical artwork size is retained; existing scale, rotation and
opacity controls apply to the resulting placement. Nonzero tangent-plane
offsets remain unsupported for mesh export.

The output adds a `mesh_anchor` placement revision. It records the request,
original artwork identity, derived marker, metric position and seed-input
provenance. The mesh composition identity includes the revision. Original run
files, poses, masks and artwork records are not rewritten. This output represents
a new placement over saved tracking evidence rather than reproduction of the
original artwork rendering.

Seed media/file and decoded image identities, the request, masks, bundle and
source dependencies are rechecked before publication to a new derived directory.
The seed is checked even when it falls outside the exported scoring window.
No reference evaluation poses, scoring annotations, model weights or new
neural inference are used to choose the anchor. Existing current-mask clipping,
mesh depth/chart checks and loss/recovery suppression remain in force.

Use `python -B -m tools.workbench_offline_export --help` for the full command.
For model-free software checks:

```text
python -B -m unittest tools.test_workbench_mesh_anchor tools.test_workbench_mesh_anchor_independent
```

Manufactured plane and video checks test geometry, provenance and composition.
They do not establish attachment accuracy on real curved objects, automatic
surface preparation, tracking recovery or physical-phone performance. Mesh
support still comes from the supplied metric geometry; it is not evidence that
all its regions have been observed. The existing nearest-texel sampler can differ
at exact floating-point texel boundaries, as documented for mesh export.
