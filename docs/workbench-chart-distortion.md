# Experimental tangent-chart stretch limit

The existing mesh artwork renderer accepts an optional `max_chart_stretch`
argument. It limits local stretching when artwork coordinates are projected onto
a prepared surface. It does not unwrap a cylinder, reconstruct geometry or
establish real-video attachment accuracy.

The option is available on:

- `bench.workbench_geometry.marker_patch`
- `bench.workbench_geometry.render_marker`
- `bench.workbench_neural.NeuralArtwork`

The default is `None`, preserving the existing renderer and artwork record.
Callers may supply a finite number greater than or equal to 1; booleans, NaN,
infinity and smaller values refuse. No default limit has been calibrated on
footage. This slice adds no workbench CLI option or phone implementation.

For example, `NeuralArtwork(..., max_chart_stretch=1.25)` bounds the local inverse
projection stretch to 1.25. That number is an illustrative caller choice, not a
recommended or validated acceptance threshold.

## What the bound measures

For an orthonormal anchor tangent frame and a nondegenerate mesh triangle,
orthogonal projection from the face plane into the chart has singular values
`1` and `abs(n_face dot n_chart)`. Its maximum inverse local stretch is therefore
`1 / abs(n_face dot n_chart)`. A parallel plane has stretch 1; a face turned 60
degrees has stretch 2. The value is dimensionless and does not change under a
rigid coordinate transformation or uniform change of units.

This is a per-triangle differential bound. It is not a bound on global geodesic
distortion, a physical size measurement, proof of surface identity or a complete
artwork footprint accuracy metric. Faceted cylinders use their actual triangle
planes, rather than an inferred smooth surface.

`face_chart_stretches(vertices, faces, marker)` exposes these values for numerical
checks. It validates finite indexed triangles and a right-handed orthonormal
chart frame. Face edges are rescaled before computing normals. A normalized
cross-product length at most `1e-12`, or projection cosine at most `1e-12`, makes
the inverse unavailable. The helper represents unavailable values as infinity;
published render statistics use counts and null, never JSON infinity or a false
zero. Cosines within eight float64 epsilons of 1 are treated as parallel.

## Rendering and identity

Faces exceeding the explicit limit are excluded during connected patch
traversal. Traversal cannot pass through those faces to reach another region.
Existing support, connectivity, fold, overlap, front-facing, depth and current
observed-mask constraints remain active. Unsupported or disconnected geometry
does not become a prepared surface. The input mesh and persistent anchor remain
unchanged.

An explicit render limit causes the patch to be recomputed from the supplied
geometry and current support, even if a precomputed patch is supplied. This
prevents an older patch from bypassing the limit. It adds preparation work to an
opt-in render call; no latency or memory improvement is claimed.

Opt-in patch and render statistics include:

| Field | Meaning |
|---|---|
| `chart_stretch_limit` | The caller's explicit limit |
| `max_accepted_chart_stretch` | Maximum across the final accepted patch; null when empty |
| `chart_stretch_rejected_faces` | Unique supported faces encountered and rejected by this limit |
| `chart_stretch_unavailable_faces` | Rejected faces with unavailable inverse projection |

These are geometry accounting values. They do not cover disconnected or
unvisited faces and are not independent quality metrics. Artwork may be
suppressed entirely when the anchor itself cannot satisfy the limit.

`NeuralArtwork` binds an enabled limit into `chart_stretch_policy` and its record
digest. Its public limit property is read-only, and rendering checks the bound
policy. Omitted limits add no field to historical artwork records. Existing
source/admission checks stay strict: changes to producer source can require new
matching admissions, rather than rewriting old runs.

## Model-free verification

```text
python -B -m unittest bench.test_workbench_geometry bench.test_workbench_neural bench.test_workbench_chart_distortion tools.test_workbench_chart_distortion
```

The tests use manufactured planes, faceted cylinders and partial supported
meshes. Independent controls calculate cylinder stretch from 3D edge lengths
and their chart projections, exercise actual depth/mask-clipped rendering and
load a manufactured metric GLB through `NeuralArtwork`. The existing Packet A
suite remains a separate compatibility gate. Passing these checks does not
establish physical attachment accuracy, tracking quality or phone performance.
