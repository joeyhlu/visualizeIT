# Surface-topology and marker-patch diagnostics

Module: `bench/workbench_surface_topology.py` (algorithm `workbench_surface_topology_v1.1`,
repairs 1 and 2).
Status: additive, pure diagnostics. There is **no runtime wiring**. Nothing here changes the current
mesh, anchor, marker patch, render, cache or admission. Nothing here changes any stored result. The
module performs no filesystem, model, renderer, GPU or persistent-cache operations. Its only
internal import is `workbench_contract`, which it needs to validate `ClipSpec`/`RunSpec`.

## Why it exists

The saved keyboard run has 238 primary tracking frames. In 80 of them the artwork rendered, and in
the other 158 no patch was rendered (unavailable). In those 158 frames the patch was far-side or
back-facing, so the mesh fully occluded it under the SAVED predicted poses. Independent CPU ray tests
confirm that this suppression is geometrically consistent with the saved predictions. That is
consistency with saved poses only: it is not physical ground truth and it is not evidence of pose
accuracy.

Separately, the frozen anchor face 2539 has no shared-index edge neighbours. Matching exactly
coincident edge positions instead finds 10 geometric faces. The current patch is a single triangle of
108.3 mm². The nominal footprint is a 107.26 mm square, 11 504.7 mm², so the patch is a ratio of
0.00941, which is **0.941 percent** of the nominal square. This module only measures that
representation-dependent fragmentation:

- The exact-coordinate 10-face set is **not** an approved replacement.
- Back-facing artwork must not be shown.
- The frozen anchor must not be retuned on scored frames.

## Units and floating-point policy

`units` is one of `"metres"`, `"millimetres"` or `"relative"`. It is a declaration only, and no
conversion is ever performed. Area outputs are labelled `<units>^2`, or `relative_units^2` for
relative units. For relative units no metric claim is made (`claims.metric_scale_claimed: false`).
For declared metric units the claim is `"declared_not_verified"`.

- **Errors:** overflow, invalid operations and division by zero raise `ValueError`.
- **Underflow:** it is ignored only inside normalized computations whose dominant terms are O(1).
  These are the power-of-two-rescaled face edges and the chart coordinates in half-size units. A
  flushed value is below 2⁻¹⁰²² relative to the result.
- **Physical squared-unit outputs:** coverage is computed dimensionlessly first and converted to
  declared units² only when the result is a normal finite float64. Otherwise the physical fields are
  `null` and the traversal status is `unavailable_numerical_range`. There is no zero area, no
  infinity and no arithmetic crash. For example, a quad and marker scaled by 1e±200 give
  `unavailable_numerical_range`, while 1e±150 is measured.
- **Marker half-size:** a marker whose `size_object_units / 2` is not a normal float64 is refused
  with `ValueError`.

## Policy and limits records (all fields are in every digest)

`TopologyPolicyV1` (schema `workbench_topology_policy_v1`, frozen), factory `default_policy()`:

| field | default | meaning |
|---|---|---|
| `name` | `"strict_attributes_v1"` | label (≤ 128 characters) |
| `max_neighbor_normal_angle_deg` | `75.0` | crease threshold for geometric candidates, in (0, 180] |
| `require_region_labels` | `True` | recorded only. Approval **always** needs declared equal region labels; `False` cannot relax this |
| `require_uv_continuity` | `True` | recorded only. Approval **always** needs exactly equal corner UVs; `False` cannot relax this |
| `degenerate_shape_tol` | `1e-12` | a face is degenerate when its shape quality (below) is ≤ this |
| `angle_uncertainty_deg` | `1e-9` | angles within this of a threshold are ambiguous and block |
| `overlap_relative_area_tol` | `1e-12` | an uncertified chart intersection with area > `tol · 4` (normalized nominal area) is a positive overlap |

**Validity ranges** (`POLICY_TOLERANCE_RANGES`; the constructor raises `ValueError` outside them):

| field | allowed range |
|---|---|
| `degenerate_shape_tol` | [1e-14, 1e-3] |
| `angle_uncertainty_deg` | [1e-12, 1.0] |
| `overlap_relative_area_tol` | [1e-14, 1e-3] |

Lower bounds keep each tolerance above the float64 rounding of the quantity it measures. Upper
bounds keep it an uncertainty allowance rather than a relaxation of validity.

**Degeneracy definition** (recorded as `degeneracy_definition`):
`quality = |e01 × e02| / max_i |e_i|²`, which equals 2·area / (longest edge)². It is computed
per face after an exact power-of-two rescaling of that face's edges, so the largest component lies
in [0.5, 1). It is invariant under rotation, translation and uniform scaling, apart from rounding.
There is no global bounding-box or diagonal scale. A small well-shaped face, for example legs of
√1.5e-12 next to a unit triangle, is valid in every rotation, while a true sliver is degenerate.

`TopologyLimitsV1` (schema `workbench_topology_limits_v1`, frozen). Positional fields come first,
then keyword fields with defaults:

| field | `portable_20k_limits()` | `execution_200k_limits()` |
|---|---|---|
| `name` | `portable_20k` | `execution_200k` |
| `max_vertices` | 60 000 | 600 000 |
| `max_faces` | 20 000 | 200 000 |
| `max_half_edges` (= 3 × faces) | 60 000 | 600 000 |
| `max_traversal_faces` | 20 000 | 200 000 |
| `max_overlap_candidates` (pair-generation work) | 200 000 | 2 000 000 |
| `max_overlap_operations` | 4 000 000 | 40 000 000 |
| `max_output_bytes` (actual returned `to_dict` JSON) | 4 MiB | 32 MiB |
| `max_prefix_frames` | 24 | 24 |
| `max_prefix_seconds` | 10.0 | 10.0 |
| `orthonormal_tol` (dimensionless) | 1e-6 | 1e-6 |
| `barycentric_tol` (dimensionless) | 1e-9 | 1e-9 |
| `max_overlap_memberships` (default 1 000 000) | 400 000 | 4 000 000 |
| `max_prefix_view_pixels` (default 3840×2160) | 8 294 400 | 8 294 400 |
| `max_prefix_string_chars` (default 512) | 512 | 512 |

A limits record can never relax the prefix bounds above 24 frames or 10 seconds; trying to raises
`ValueError`.

**Validity ranges** (`LIMITS_TOLERANCE_RANGES`; the constructor raises `ValueError` outside them):

| field | allowed range |
|---|---|
| `orthonormal_tol` | [1e-12, 1e-3] |
| `barycentric_tol` | [1e-15, 1e-6] |

These are uncertainty allowances only. Independently of them, the following validity checks cannot
be relaxed:

- A marker's barycentric coordinates must each lie in [−tol, 1 + tol] and sum to 1.
- The chart frame determinant `det[t; b; n]` must exceed `MIN_FRAME_DETERMINANT = 0.5`.
- A prefix pose rotation must have `det R > 0.5` before its orthonormality is tested.

So zero, singular, reflected and scaled rotations, and far-outside anchors, are refused under every
accepted limits record.

Mesh size limits are checked before any allocation that scales with the mesh. The
coordinate tolerance is exactly `0`; there is no metric distance epsilon anywhere.

## `build_topology(vertices, faces, *, units, policy, limits, face_region_ids=None, corner_uv=None) -> TopologyV1`

Inputs (all must be `numpy.ndarray`; lists are refused):

- `vertices` `(N, 3)`, float16/32/64 or an integer dtype whose values are within ±2⁵³. These convert
  losslessly to float64. bool, object, complex, long-double and non-finite values are refused.
- `faces` `(M, 3)`, integer dtype only (bool/float/object refused), with indices in `[0, N)`.
- `face_region_ids` `(M,)`, integer dtype (bool refused), optional.
- `corner_uv` `(M, 3, 2)`, finite real (same dtype rules as vertices), optional.

Semantics:

- The original arrays are never welded, reordered or mutated. Face IDs and corners are the original
  ones.
- Coordinates are compared for exact float64 equality after `+0.0` canonicalises `-0.0`. There is no
  rounding, nearest-neighbour matching or distance tolerance.
- **Blocked faces** (deterministic sorted IDs per reason): `zero_length_edge`, `degenerate_face`,
  `duplicate_geometric_triangle` (either winding, so every copy is blocked), `winding_conflict` (a
  two-owner coordinate edge traversed in the same direction blocks both faces) and
  `nonmanifold_edge` (an edge with more than two owners blocks every owner; no convenient pair is
  chosen).
- A point contact is not an edge. Offset layers stay separate because their coordinates differ.
- **Relations**, reported separately:
  - `index`: faces sharing an original vertex-index edge with exactly two owners. This is the raw
    relation the current runtime patch uses.
  - `geometric_candidate`: a coordinate edge with exactly two distinct, unblocked owners, opposite
    traversal directions, and a neighbour-normal angle no greater than the policy threshold and not
    ambiguous. Larger angles count as `crease`.
  - `approved`: a candidate whose **declared** region labels are equal **and** whose **declared**
    corner UVs are exactly equal at both shared corners. This is the `approval_rule`. If either
    attribute is missing, every candidate stays unverified, whatever the policy switches say. A
    region mismatch or a UV seam is a barrier.
- `attribute_status` reports one of `no_boundary_attributes`, `partial_boundary_attributes` or
  `boundary_attributes_checked`. The current neural GLB helper drops these attributes. Supplying them
  is a real later integration dependency.
- **Output cap:** the ACTUAL `to_dict()` canonical JSON, including the digest and envelope fields,
  must fit `max_output_bytes`. Otherwise `ValueError` is raised.

`TopologyV1` fields: `units`, `policy`, `limits`, `input_digests` (`vertices`, `faces`,
`face_region_ids`, `corner_uv`; each is `array_digest` or `null`), `counts` (`vertices`, `faces`,
`exact_coordinates`, `boundary_edges`, `nonmanifold_edges`, `winding_conflict_edges`,
`index_adjacent_pairs`, `geometric_candidate_pairs`, `approved_pairs`, `unverified_candidate_pairs`,
`crease_edges`, `angle_ambiguous_edges`), `blocked_faces`, `barrier_counts` (`region_barrier`,
`uv_seam`, `crease`, `angle_ambiguous`), `attribute_status` and `digest`. `to_dict()` also contains
`approval_rule`, `degeneracy_definition`, `coordinate_tolerance: 0.0` and `mesh_mutated: false`. The
method `neighbors(relation, face_id) -> tuple[int, ...]` returns sorted neighbour IDs.

**Immutability and identity.** Mappings are read-only proxies. Every internal array, including the
retained original region and UV arrays, is backed by an immutable `bytes` object, so its WRITEABLE
flag cannot be re-enabled. `diagnose_patch` never trusts a supplied record. It re-runs
`build_topology` from the caller's (digest-checked) vertices and faces and the record's retained
attribute arrays, policy, limits and units. It then requires that the digest, the `to_dict()`
output and every internal array and adjacency structure (CSR) are byte-identical. A record with
substituted adjacency, blocked mask, normals, coordinates, attributes or counts is refused with
`ValueError`, even under the original digest, and only the re-derived state is used.

## `diagnose_patch(vertices, faces, marker, face_support, topology, *, units, min_support=3, max_anchor_normal_angle_deg=75.0, max_neighbor_normal_angle_deg=75.0, limits) -> PatchDiagnosisV1`

- **Bounds before hashing:** before any hashing, `vertices` and `faces` must be `(N, 3)` / `(M, 3)`
  ndarrays. N, M and 3M must fit both the supplied diagnosis `limits` and the topology's own limits,
  and (N, M) must equal the topology counts. All of this is checked from shape metadata only, so an
  oversized or broadcast array is refused without being hashed or materialized. Only then must
  `vertices` and `faces` match the topology's input digests. `units` must equal `topology.units`.
  The topology is re-derived as described above, using its own recorded limits so that its identity
  is preserved.
- `marker` must contain exactly the `make_marker` keys: `face_id` (int), `barycentric` (3 reals, each
  ≥ −`barycentric_tol`, summing to 1 within the tolerance), `tangent`, `bitangent`, `normal` (unit
  length, orthogonal and right-handed `t × b = n` within `orthonormal_tol`), `size_object_units`
  (finite, positive, half-size a normal float) and `artwork_sha256` (lowercase 64-hex).
- `face_support`: an integer `(M,)` ndarray of non-negative counts. Bool is refused.
- The chart is the marker tangent chart centred on the barycentric anchor and normalized by
  `half = size/2`, so the nominal square is `[-1, 1]²`. It is computed elementwise, not through BLAS,
  so identical 3D coordinates give bit-identical chart coordinates.
- One bounded breadth-first traversal runs per relation. Each enforces, independently of the build
  policy:
  - the face is not blocked and has support ≥ `min_support`;
  - the anchor-normal angle is ≤ the maximum;
  - the neighbour-normal angle is ≤ the maximum, and angles within `angle_uncertainty_deg` of a
    threshold are ambiguous and stop;
  - chart reach (the vertex Chebyshev distance must be ≤ `1.5·√2` half-widths, the runtime rule);
  - the certified chart orientation sign equals the anchor face's, with chart shape quality above
    `degenerate_shape_tol`. A certified opposite sign is `folded`; an uncertain sign or degenerate
    shape is `chart_degenerate_or_uncertain`. Both stop the traversal and are listed in
    `folded_faces`.

  The anchor face is checked too (giving `anchor_rejected`). Exceeding `max_traversal_faces` gives
  `unavailable_budget`.
- **Coverage** is computed in dimensionless half-size units; the nominal area is 4. The
  `dimensionless` block holds `patch_surface_area_over_nominal`, `surface_area_inside_over_nominal`,
  `summed_clipped_chart_area_ratio` and `union_coverage_fraction`. The physical fields hold the same
  areas in declared units², or `null` if they are not representable:
  - `patch_surface_area`;
  - `surface_area_inside_nominal_footprint`: each face's 3D area weighted by its exactly clipped
    chart-area fraction inside the square, from Sutherland–Hodgman clipping. No raster is used.
  - `summed_clipped_chart_area_counting_multiplicity` and `nominal_square_area`.

  `summed_clipped_chart_area_ratio` counts multiplicity, is never clamped, can exceed 1, and is not a
  fraction, a visibility or an accuracy. `physical_units_representable` is reported; when it is
  false, the traversal status is `unavailable_numerical_range`.
- **Overlap** (`_overlap_analysis`). Each stage is charged before it runs:
  1. Grid memberships are counted arithmetically from each polygon's bounding box and charged to
     `max_overlap_memberships` and `max_overlap_operations` before any bucket is populated.
  2. The total pair-generation work, Σ k(k−1)/2 over buckets, is charged to
     `max_overlap_candidates` and the operations budget before any pair is generated.
  3. Each pair's worst-case separating-axis cost, and then its worst-case clip cost, is pre-charged
     before it runs.

  Exceeding any of these gives `unavailable_budget` with null overlap counts. **Disjoint interiors are
  proven only by a certified separating axis** on the unclipped chart triangles. That test needs a
  certified counter-clockwise orientation and error-bounded orientation signs; exactly shared
  vertices give exact zeros, so shared edges certify. A pair without a certificate is counted as
  follows:
  - `positive_overlap` if its clipped intersection area exceeds `overlap_relative_area_tol · 4`;
  - otherwise uncertain (`unavailable_numerical_uncertainty`), even when the computed area is
    exactly 0. Shoelace cancellation can hide a true overlap of δ²/2.

  `union_coverage_fraction` is reported only for `disjoint_interiors_proven`; otherwise it is `null`.
  As a sum of clipped areas it can deviate from 1 at rounding level, for example 1.0000000000000042
  on a 1 800-face grid, and it is not clamped.
- Output keys: `schema`, `algorithm_version`, `units`, `topology_digest` (of the re-derived record),
  `limits`, `parameters`, `marker` (normalized), `face_support_digest`, `anchor_neighbors`,
  `traversals` (`status`, `reason`, `faces`, `folded_faces`, `stops`, `anchor_chart_sign`,
  `coverage`; `geometric_candidate` also carries an UNVERIFIED label), `claims` and `digest`.
- **Output cap:** if the ACTUAL `to_dict()` JSON exceeds `max_output_bytes`, the face lists are
  replaced by `face_lists: "omitted: output budget"`. If it is still too large, `ValueError` is
  raised.

## `validate_prefix_binding(clip_spec, run_spec, prefix_manifest, supplied_views, *, limits) -> PrefixBindingDiagnosisV1`

This validates supplied evidence **in memory only**. It cannot authenticate files, producers or
physical observation. `authenticated` is always `false`, and no argument or manifest field can set
it; unknown keys raise. Before any user-facing authenticated claim, a future loader integration must
verify the registered media, masks, surface package, bundle and source closure.

**Record validation (raises `ValueError` before any prefix use).** `clip_spec` and `run_spec` must be
exactly `ClipSpec` and `RunSpec`. Their ACTUAL `schema` attributes must be the contract schemas, and
`clip_spec.evaluator_inputs_read` must be `False`. This is checked on the attribute itself because
`to_dict()` hard-codes it. Both must re-validate through `from_dict(to_dict())` and compare equal to
the supplied record, so attributes hidden by serialization are caught.

**Bounds before hashing.** These checks run in this order before the manifest or any view is hashed
or serialized:

1. The manifest has exact keys, and the frame count is at most `max_prefix_frames`.
2. Every frame has exact keys. Every exact-key check compares `len()` before iterating.
3. A plain-JSON walk enforces a node bound (16 + 96 per frame). Every list or mapping's `len()` is
   compared with the REMAINING node budget before the container is iterated or extended, so an
   oversized container (for example a 100 000-element `K`) is refused without being traversed. The
   walk also enforces at most `max_prefix_string_chars` per string, finite floats and integers within
   ±2⁵³.
4. Then: the schema and units; at most `max_prefix_frames` supplied views (checked by `len()` before
   the keys are scanned), each with exact keys; and clip `width × height` within
   `max_prefix_view_pixels`.

`prefix_manifest` (exact keys):

```json
{"schema": "workbench_prefix_manifest_v1", "clip_id": "...", "run_id": "...",
 "units": "metres|millimetres|relative",
 "frames": [{
   "frame_id": "f0", "clip_position": 0, "source_frame_index": 1148,
   "timestamp_s": 0.0, "clock_kind": "timestamp_seconds",
   "width": 640, "height": 480, "K": [9 numbers],
   "image": {"array_sha256": "<array_digest of the uint8 HxWx3 RGB>"} | null,
   "observed_mask": {"array_sha256": "<mask_digest>", "kind": "observed", "producer": "..."} | null,
   "pose": {"camera_from_object": [[4 x 4 non-bool reals]], "units": "<manifest units>",
            "provenance": "controlled_init:...|onboarding:...|reconstruction:...",
            "producer_receipt_sha256": "<64 hex>" | null} | null,
   "producer_receipt_sha256": "<64 hex>" | null}]}
```

`supplied_views`: `{frame_id: {"rgb": uint8 (H, W, 3) ndarray | None, "mask": bool/uint (H, W) ndarray | None}}`.

`array_digest(a)` is SHA-256 over `dtype.str|shape|` followed by the C-order bytes. This is the same
convention as `workbench_geometry.array_sha256`, a semantic array identity rather than a file hash.
`mask_digest(m)` is `array_digest((m > 0).astype(uint8))`.

Rules:

- **Structural errors** raise `ValueError`: unknown or missing keys, oversize or non-plain manifest
  content, more than 24 frames or views, non-clip frame IDs, and invalid records.
- **Refusals** (`status: refused`, with reasons):
  - frame identity problems: clip/run identity mismatch; a run that does not validate against the
    clip; `clip_position`, `source_frame_index`, timestamp (bool/int/str refused), clock kind,
    width/height (crop) or `K` differing from the clip; positions that are not strictly unique and
    increasing;
  - prefix scope: a frame outside the run onboarding keyframes; a scheduled or clip-scoring frame,
    or any frame at or after the first scored position; supplied views for frames not in the
    manifest;
  - evidence content: RGB or mask bytes that differ from the manifest identity; a mask `kind` other
    than `observed`; producers or provenance containing evaluator, ground-truth, scored or
    annotation words; pose provenance outside the allowed prefixes; pose units differing from the
    manifest units;
  - a pose that is not an exact 4×4 list of non-bool finite reals forming a proper rigid
    transform. Bool matrices and numeric strings are refused, never coerced.
  - a non-null receipt that is not a 64-hex digest;
  - a timestamp clock span over 10 s.
- **Missing evidence** gives `unavailable`. That covers a missing image, observed mask, pose, frame
  receipt, nested pose receipt (`null`) or clip intrinsics. It never counts as zero support.
- **Duration** is `attested_within_10s` only when `clock_kind == "timestamp_seconds"`. A
  `frame_index` clock gives `unattested_frame_index_clock` with `duration_s: null`.
- Post-prefix tracking rows are never accepted or consumed (`post_prefix_rows_consumed: false`).
- **Output cap:** the ACTUAL `to_dict()` JSON, including the digest, must fit `max_output_bytes`;
  otherwise `ValueError` is raised.

## Records, digests and immutability

Every record is a frozen dataclass. `PatchDiagnosisV1` and `PrefixBindingDiagnosisV1` store their
canonical JSON bytes (`canonical_json`); `body` and `to_dict()` return fresh copies. `digest` is
SHA-256 over the canonical JSON (sorted keys, compact separators, ASCII, no NaN).

The topology digest binds the original array dtype/shape/bytes digests, units, policy, limits,
approval rule, degeneracy definition and algorithm version. The derived relations are re-derived
from exactly those inputs before every use. The patch digest additionally binds the re-derived
topology digest, normalized marker, support digest, parameters and limits. Mutating any input after
a call does not change a returned record.

## Known limits

- These are diagnostics, not runtime adjacency. The `approved` relation needs real region and UV
  attributes that the current GLB loading path does not supply.
- `diagnose_patch` rebuilds the topology on every call, roughly doubling the topology cost; numerical
  honesty comes before speed. Traversal, clipping and overlap testing are bounded pure-Python loops.
  For example, an 1 800-face grid is diagnosed in about 0.2 s, but patches near 200k faces will be
  slow.
- Separation certification uses a forward error bound for 2D orientation. Near-touching pairs
  without exactly shared vertices, such as T-junctions or sub-ulp gaps, can end as
  `unavailable_numerical_uncertainty`. That is honest, but it withholds the union fraction.
- A `positive_overlap` count relies on the clipped area exceeding `overlap_relative_area_tol · 4`,
  far above the float64 rounding error of the chart coordinates. It never unlocks a union fraction.
- `index` adjacency counts only edges with exactly two index owners, and all traversals apply the
  same face-level gates. Its results can therefore differ from the runtime `marker_patch` on blocked
  or degenerate faces. The runtime also does not angle-check its anchor face.
- Python objects can always be reconstructed by a determined caller. The guarantee is that
  `diagnose_patch` uses only state it has re-derived from the digest-bound inputs.
- No claim is made that a larger diagnosed area improves tracking, visibility or accuracy.
