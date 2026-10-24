"""Pure surface-topology and marker-patch diagnostics (additive; no runtime wiring).

This module explains representation-dependent marker-patch fragmentation. Nothing here changes a
mesh, anchor, patch, render, cache or admission. It does NO filesystem, model, renderer, GPU or
persistent-cache work. Every record is a frozen, detached snapshot with a deterministic
``to_dict()`` and ``digest``.

* ``build_topology``: relations on the ORIGINAL face IDs and corners. Coordinates are compared for
  EXACT equality after lossless float64 conversion, with signed zero canonicalised; there is no
  rounding, welding or distance tolerance. Three relations are reported separately:
  - ``index``: faces sharing an original vertex-index edge;
  - ``geometric_candidate``: faces sharing an exact coordinate edge with exactly two owners,
    opposite traversal directions, valid (unblocked) faces and a neighbour normal angle within
    policy;
  - ``approved``: candidates whose DECLARED face-region labels AND corner UVs both prove
    continuity. Missing attributes always leave an edge an UNVERIFIED candidate; no policy switch
    can relax this.
* ``diagnose_patch``: re-derives the topology from its bound inputs, then runs bounded traversals
  around the marker anchor, one per relation. Exact coverage comes from clipping chart polygons
  against the nominal square, in dimensionless half-size units. The union fraction is reported only
  when disjoint chart interiors are CERTIFIED.
* ``validate_prefix_binding``: IN-MEMORY consistency of supplied prefix evidence. It cannot
  authenticate files, producers or physical observation, and never claims to.

Coordinate tolerance is exactly 0. Every other tolerance is dimensionless, named, versioned and
recorded in the policy or limits record, and is part of every digest.

Floating-point policy: overflow, invalid operations and division by zero raise (and become
``ValueError``). Underflow is ignored only inside normalized computations whose dominant terms are
O(1) (power-of-two-rescaled face edges, chart coordinates in half-size units); a flushed value is then
below 2**-1022 relative to the result. Physical squared-unit outputs are checked explicitly and are
reported as ``unavailable_numerical_range`` when they are not normal finite float64 values.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

ALGORITHM_VERSION = "workbench_surface_topology_v1.1"
SCHEMA_POLICY = "workbench_topology_policy_v1"
SCHEMA_LIMITS = "workbench_topology_limits_v1"
SCHEMA_TOPOLOGY = "workbench_surface_topology_v1"
SCHEMA_PATCH = "workbench_patch_diagnosis_v1"
SCHEMA_PREFIX = "workbench_prefix_binding_diagnosis_v1"
SCHEMA_PREFIX_MANIFEST = "workbench_prefix_manifest_v1"
UNITS = ("metres", "millimetres", "relative")
RELATIONS = ("index", "geometric_candidate", "approved")
MARKER_KEYS = frozenset({"face_id", "barycentric", "tangent", "bitangent", "normal", "size_object_units", "artwork_sha256"})
REACH_FACTOR = math.sqrt(2.0) * 1.5          # same chart reach rule as the current runtime marker_patch
INT_EXACT = 2 ** 53
FLOAT_MIN_NORMAL = sys.float_info.min
ALLOWED_POSE_PROVENANCE = ("controlled_init:", "onboarding:", "reconstruction:")
FORBIDDEN_PROVENANCE_WORDS = ("evaluator", "ground_truth", "groundtruth", "gt_", "scored", "annotation")
APPROVAL_RULE = ("approved requires BOTH declared equal face-region labels AND exactly equal corner UVs on both "
                 "shared corners; require_region_labels/require_uv_continuity are recorded and cannot relax this")
DEGENERACY_DEFINITION = ("shape quality |e01 x e02| / max_i |e_i|^2 (= 2 * area / longest edge^2) after exact "
                         "power-of-two rescaling; degenerate when <= degenerate_shape_tol; rotation, translation "
                         "and scale invariant")
# Validity ranges for every tolerance. Upper bounds keep a tolerance an UNCERTAINTY allowance and never
# a relaxation of geometric validity; lower bounds keep it above float64 rounding of the measured quantity.
POLICY_TOLERANCE_RANGES = {"degenerate_shape_tol": (1e-14, 1e-3), "angle_uncertainty_deg": (1e-12, 1.0),
                           "overlap_relative_area_tol": (1e-14, 1e-3)}
LIMITS_TOLERANCE_RANGES = {"orthonormal_tol": (1e-12, 1e-3), "barycentric_tol": (1e-15, 1e-6)}
MIN_FRAME_DETERMINANT = 0.5          # non-relaxable: chart frame / pose rotation determinant must exceed this
_HEX = set("0123456789abcdef")
_NUMERIC_ERR = dict(over="raise", invalid="raise", divide="raise", under="ignore")


# --------------------------------------------------------------------------
# policy / limits


@dataclass(frozen=True)
class TopologyPolicyV1:
    """Adjacency policy. Every tolerance is dimensionless and recorded in every digest."""
    name: str = "strict_attributes_v1"
    max_neighbor_normal_angle_deg: float = 75.0
    require_region_labels: bool = True        # recorded; approval ALWAYS needs declared equal region labels
    require_uv_continuity: bool = True        # recorded; approval ALWAYS needs exactly equal corner UVs
    degenerate_shape_tol: float = 1e-12       # see DEGENERACY_DEFINITION
    angle_uncertainty_deg: float = 1e-9       # an angle this close to a threshold is ambiguous -> blocked
    overlap_relative_area_tol: float = 1e-12  # uncertified chart overlap area <= tol * nominal area is uncertain
    schema: str = SCHEMA_POLICY

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise ValueError("policy name must be a non-empty string of at most 128 characters")
        for key in ("max_neighbor_normal_angle_deg", "degenerate_shape_tol", "angle_uncertainty_deg",
                    "overlap_relative_area_tol"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"policy {key} must be a finite non-negative number")
        if not 0 < self.max_neighbor_normal_angle_deg <= 180:
            raise ValueError("policy max_neighbor_normal_angle_deg must be in (0, 180]")
        for key, (lo, hi) in POLICY_TOLERANCE_RANGES.items():
            if not lo <= getattr(self, key) <= hi:
                raise ValueError(f"policy {key} must be within the validity range [{lo}, {hi}]")
        for key in ("require_region_labels", "require_uv_continuity"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"policy {key} must be a bool")
        if self.schema != SCHEMA_POLICY:
            raise ValueError("unsupported policy schema")

    def to_dict(self) -> dict:
        return {"schema": self.schema, "name": self.name,
                "max_neighbor_normal_angle_deg": float(self.max_neighbor_normal_angle_deg),
                "require_region_labels": self.require_region_labels, "require_uv_continuity": self.require_uv_continuity,
                "degenerate_shape_tol": float(self.degenerate_shape_tol),
                "angle_uncertainty_deg": float(self.angle_uncertainty_deg),
                "overlap_relative_area_tol": float(self.overlap_relative_area_tol)}


@dataclass(frozen=True)
class TopologyLimitsV1:
    """Bounds charged BEFORE the corresponding work or allocation; exceeding one is an explicit
    refusal or ``unavailable_budget``, never a fabricated result."""
    name: str
    max_vertices: int
    max_faces: int
    max_half_edges: int
    max_traversal_faces: int
    max_overlap_candidates: int
    max_overlap_operations: int
    max_output_bytes: int
    max_prefix_frames: int = 24
    max_prefix_seconds: float = 10.0
    orthonormal_tol: float = 1e-6        # dimensionless: unit length / orthogonality / handedness of the chart
    barycentric_tol: float = 1e-9        # dimensionless: barycentric sum and non-negativity
    max_overlap_memberships: int = 1_000_000      # spatial-grid cell memberships (also charged as operations)
    max_prefix_view_pixels: int = 3840 * 2160     # per supplied view, checked before any hashing
    max_prefix_string_chars: int = 512            # every manifest string, checked before hashing
    schema: str = SCHEMA_LIMITS

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise ValueError("limits name must be a non-empty string of at most 128 characters")
        for key in ("max_vertices", "max_faces", "max_half_edges", "max_traversal_faces", "max_overlap_candidates",
                    "max_overlap_operations", "max_output_bytes", "max_prefix_frames", "max_overlap_memberships",
                    "max_prefix_view_pixels", "max_prefix_string_chars"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"limits {key} must be a positive int")
        for key in ("max_prefix_seconds", "orthonormal_tol", "barycentric_tol"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"limits {key} must be a finite positive number")
        for key, (lo, hi) in LIMITS_TOLERANCE_RANGES.items():
            if not lo <= getattr(self, key) <= hi:
                raise ValueError(f"limits {key} must be within the non-relaxable validity range [{lo}, {hi}]")
        if self.max_prefix_frames > 24 or self.max_prefix_seconds > 10.0:
            raise ValueError("limits may not relax the 24-frame / 10-second onboarding prefix bounds")
        if self.schema != SCHEMA_LIMITS:
            raise ValueError("unsupported limits schema")

    def to_dict(self) -> dict:
        return {"schema": self.schema, "name": self.name, "max_vertices": self.max_vertices, "max_faces": self.max_faces,
                "max_half_edges": self.max_half_edges, "max_traversal_faces": self.max_traversal_faces,
                "max_overlap_candidates": self.max_overlap_candidates,
                "max_overlap_operations": self.max_overlap_operations,
                "max_overlap_memberships": self.max_overlap_memberships, "max_output_bytes": self.max_output_bytes,
                "max_prefix_frames": self.max_prefix_frames, "max_prefix_seconds": float(self.max_prefix_seconds),
                "max_prefix_view_pixels": self.max_prefix_view_pixels,
                "max_prefix_string_chars": self.max_prefix_string_chars,
                "orthonormal_tol": float(self.orthonormal_tol), "barycentric_tol": float(self.barycentric_tol)}


def default_policy() -> TopologyPolicyV1:
    return TopologyPolicyV1()


def portable_20k_limits() -> TopologyLimitsV1:
    return TopologyLimitsV1(name="portable_20k", max_vertices=60_000, max_faces=20_000, max_half_edges=60_000,
                            max_traversal_faces=20_000, max_overlap_candidates=200_000,
                            max_overlap_operations=4_000_000, max_output_bytes=4 << 20,
                            max_overlap_memberships=400_000)


def execution_200k_limits() -> TopologyLimitsV1:
    return TopologyLimitsV1(name="execution_200k", max_vertices=600_000, max_faces=200_000, max_half_edges=600_000,
                            max_traversal_faces=200_000, max_overlap_candidates=2_000_000,
                            max_overlap_operations=40_000_000, max_output_bytes=32 << 20,
                            max_overlap_memberships=4_000_000)


# --------------------------------------------------------------------------
# helpers


def array_digest(array: np.ndarray) -> str:
    """Semantic array identity: dtype string, shape and C-order bytes (same convention as
    ``workbench_geometry.array_sha256``). This is NOT a source-file hash."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def mask_digest(mask: np.ndarray) -> str:
    return array_digest(np.ascontiguousarray(np.asarray(mask) > 0).astype(np.uint8))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _frozen(array: np.ndarray) -> np.ndarray:
    """Immutable copy backed by a ``bytes`` object: its WRITEABLE flag cannot be re-enabled."""
    a = np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _same_array(a: Any, b: Any) -> bool:
    return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def _check_units(units: Any) -> str:
    if units not in UNITS:
        raise ValueError(f"units must be one of {UNITS} (no conversion is performed)")
    return units


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _coords_float64(array: Any, name: str, *, shape_tail: tuple[int, ...]) -> np.ndarray:
    """Lossless float64 view of a numeric array; bool/object/complex/long-double inputs are rejected."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    kind = array.dtype.kind
    if kind == "f":
        if array.dtype.itemsize > 8:
            raise ValueError(f"{name} dtype {array.dtype} cannot be converted to float64 losslessly")
        out = array.astype(np.float64)
    elif kind in "iu":
        if array.size and (int(np.max(array)) > INT_EXACT or int(np.min(array)) < -INT_EXACT):
            raise ValueError(f"{name} integer values exceed exact float64 range")
        out = array.astype(np.float64)
    else:
        raise ValueError(f"{name} dtype {array.dtype} is not an explicit real numeric dtype")
    if out.ndim != 1 + len(shape_tail) or tuple(out.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out + 0.0          # canonical signed zero: -0.0 + 0.0 == +0.0


def _index_array(array: Any, name: str, n_rows: int | None, width: int | None) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must use an integer dtype (bool/float/object refused)")
    if (width is None and array.ndim != 1) or (width is not None and (array.ndim != 2 or array.shape[1] != width)):
        raise ValueError(f"{name} has a malformed shape")
    if n_rows is not None and array.shape[0] != n_rows:
        raise ValueError(f"{name} must have one row per face")
    if array.dtype.kind == "u" and array.size and int(np.max(array)) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} values exceed int64")
    return array.astype(np.int64)


# --------------------------------------------------------------------------
# topology


@dataclass(frozen=True)
class TopologyV1:
    """Topology record. Internal arrays are immutable; ``diagnose_patch`` never trusts them and
    re-derives the whole record from the bound inputs before use."""
    units: str
    policy: TopologyPolicyV1
    limits: TopologyLimitsV1
    input_digests: Mapping[str, Any]
    counts: Mapping[str, int]
    blocked_faces: Mapping[str, tuple]
    barrier_counts: Mapping[str, int]
    attribute_status: str
    digest: str
    _v: np.ndarray = field(repr=False, compare=False)
    _f: np.ndarray = field(repr=False, compare=False)
    _unit_normals: np.ndarray = field(repr=False, compare=False)
    _blocked: np.ndarray = field(repr=False, compare=False)
    _csr: Mapping[str, tuple] = field(repr=False, compare=False)
    _regions_raw: np.ndarray | None = field(repr=False, compare=False)
    _uv_raw: np.ndarray | None = field(repr=False, compare=False)
    schema: str = SCHEMA_TOPOLOGY

    def neighbors(self, relation: str, face_id: int) -> tuple[int, ...]:
        if relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}")
        if type(face_id) is not int or not 0 <= face_id < len(self._f):
            raise ValueError("face_id out of range")
        offsets, values = self._csr[relation]
        return tuple(int(x) for x in values[offsets[face_id]:offsets[face_id + 1]])

    def to_dict(self) -> dict:
        return {"schema": self.schema, "algorithm_version": ALGORITHM_VERSION, "units": self.units,
                "policy": self.policy.to_dict(), "limits": self.limits.to_dict(),
                "input_digests": dict(self.input_digests), "counts": dict(self.counts),
                "blocked_faces": {k: list(v) for k, v in self.blocked_faces.items()},
                "barrier_counts": dict(self.barrier_counts), "attribute_status": self.attribute_status,
                "approval_rule": APPROVAL_RULE, "degeneracy_definition": DEGENERACY_DEFINITION,
                "coordinate_tolerance": 0.0, "mesh_mutated": False, "digest": self.digest}


def _csr(n: int, a: np.ndarray, b: np.ndarray) -> tuple:
    if len(a) == 0:
        return _frozen(np.zeros(n + 1, np.int64)), _frozen(np.zeros(0, np.int64))
    pairs = np.unique(np.stack([np.concatenate([a, b]), np.concatenate([b, a])], axis=1), axis=0)
    src, dst = pairs[:, 0], pairs[:, 1]
    offsets = np.zeros(n + 1, np.int64)
    np.add.at(offsets, src + 1, 1)
    return _frozen(np.cumsum(offsets)), _frozen(dst)


def _pairs_by_key(key: np.ndarray):
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if len(ks) else np.zeros(0, np.int64)
    sizes = np.diff(np.r_[starts, len(ks)]) if len(ks) else np.zeros(0, np.int64)
    return order, starts, sizes


def _face_shape(v: np.ndarray, f: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-face rotation/translation/scale-invariant shape quality and unit normals.

    Each face's three edge vectors are rescaled by an exact power of two so the largest component
    lies in [0.5, 1); the quality is |e01 x e02| / max_i |e_i|^2 on the rescaled edges."""
    with np.errstate(**_NUMERIC_ERR):
        e = np.stack([v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 1]]], axis=1)
        mag = np.max(np.abs(e), axis=(1, 2))
        _mant, expo = np.frexp(mag)
        es = np.ldexp(e, -expo[:, None, None])
        len2 = np.max(np.einsum("mij,mij->mi", es, es), axis=1)
        cross = np.cross(es[:, 0], es[:, 1])
        cnorm = np.sqrt(np.einsum("ij,ij->i", cross, cross))
        quality = np.where(mag > 0, cnorm / np.where(len2 > 0, len2, 1.0), 0.0)
        good = (mag > 0) & (quality > tol)
        normals = np.zeros_like(cross)
        normals[good] = cross[good] / cnorm[good, None]
    return good, normals


def build_topology(vertices: np.ndarray, faces: np.ndarray, *, units: str, policy: TopologyPolicyV1,
                   limits: TopologyLimitsV1, face_region_ids: np.ndarray | None = None,
                   corner_uv: np.ndarray | None = None) -> TopologyV1:
    """Exact-coordinate topology on the original arrays (never welded, reordered or mutated)."""
    _check_units(units)
    if type(policy) is not TopologyPolicyV1 or type(limits) is not TopologyLimitsV1:
        raise ValueError("policy and limits must be TopologyPolicyV1 / TopologyLimitsV1 records")
    if not isinstance(vertices, np.ndarray) or not isinstance(faces, np.ndarray):
        raise ValueError("vertices and faces must be numpy ndarrays")
    if vertices.ndim != 2 or faces.ndim != 2 or faces.shape[1:] != (3,) or vertices.shape[1:] != (3,):
        raise ValueError("vertices must be (N, 3) and faces (M, 3)")
    if vertices.shape[0] > limits.max_vertices or faces.shape[0] > limits.max_faces \
            or 3 * faces.shape[0] > limits.max_half_edges:
        raise ValueError(f"mesh exceeds the {limits.name} vertex/face/half-edge limits (checked before allocation)")
    if vertices.shape[0] < 3 or faces.shape[0] < 1:
        raise ValueError("mesh needs at least 3 vertices and 1 face")
    v = _coords_float64(vertices, "vertices", shape_tail=(3,))
    f = _index_array(faces, "faces", None, 3)
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("face index out of range")
    regions = None if face_region_ids is None else _index_array(face_region_ids, "face_region_ids", len(f), None)
    uv = None
    if corner_uv is not None:
        if not isinstance(corner_uv, np.ndarray) or corner_uv.shape != (len(f), 3, 2):
            raise ValueError("corner_uv must be an (M, 3, 2) ndarray")
        uv = _coords_float64(corner_uv.reshape(len(f), 6), "corner_uv", shape_tail=(6,)).reshape(len(f), 3, 2)
    digests = {"vertices": array_digest(vertices), "faces": array_digest(faces),
               "face_region_ids": None if face_region_ids is None else array_digest(face_region_ids),
               "corner_uv": None if corner_uv is None else array_digest(corner_uv)}
    m = len(f)
    try:
        good, unit_normals = _face_shape(v, f, float(policy.degenerate_shape_tol))
    except FloatingPointError as exc:
        raise ValueError(f"numerical range error while building topology: {exc}") from exc
    # exact coordinate identities (no tolerance)
    _uniq, coord_id = np.unique(v, axis=0, return_inverse=True)
    coord_id = coord_id.reshape(-1).astype(np.int64)
    c = coord_id[f]
    blocked: dict[str, set] = {"zero_length_edge": set(), "degenerate_face": set(), "duplicate_geometric_triangle": set(),
                               "winding_conflict": set(), "nonmanifold_edge": set()}
    zero_len = (c[:, 0] == c[:, 1]) | (c[:, 1] == c[:, 2]) | (c[:, 0] == c[:, 2])
    blocked["zero_length_edge"].update(np.flatnonzero(zero_len).tolist())
    blocked["degenerate_face"].update(np.flatnonzero(~zero_len & ~good).tolist())
    tri_key = np.sort(c, axis=1)
    _u, tri_inv, tri_cnt = np.unique(tri_key, axis=0, return_inverse=True, return_counts=True)
    blocked["duplicate_geometric_triangle"].update(np.flatnonzero(tri_cnt[tri_inv.reshape(-1)] > 1).tolist())
    # directed half-edges on coordinate ids
    a = c[:, [0, 1, 2]].reshape(-1)
    b = c[:, [1, 2, 0]].reshape(-1)
    face_of = np.repeat(np.arange(m, dtype=np.int64), 3)
    local = np.tile(np.arange(3, dtype=np.int64), m)
    valid = a != b
    ncoord = int(coord_id.max()) + 1
    key = np.where(valid, np.minimum(a, b) * ncoord + np.maximum(a, b), -1 - np.arange(len(a)))
    order, starts, sizes = _pairs_by_key(key)
    starts_valid = starts[key[order[starts]] >= 0]
    sizes_valid = sizes[key[order[starts]] >= 0]
    boundary = int(np.sum(sizes_valid == 1))
    many = starts_valid[sizes_valid > 2]
    nonmanifold_edges = len(many)
    for s, size in zip(many.tolist(), sizes_valid[sizes_valid > 2].tolist()):
        blocked["nonmanifold_edge"].update(face_of[order[s:s + size]].tolist())
    two = starts_valid[sizes_valid == 2]
    r1, r2 = order[two], order[two + 1]
    f1, f2 = face_of[r1], face_of[r2]
    distinct = f1 != f2
    r1, r2, f1, f2 = r1[distinct], r2[distinct], f1[distinct], f2[distinct]
    opposite = (a[r1] == b[r2]) & (b[r1] == a[r2])
    conflict = ~opposite
    blocked["winding_conflict"].update(f1[conflict].tolist())
    blocked["winding_conflict"].update(f2[conflict].tolist())
    blocked_mask = np.zeros(m, bool)
    for faces_set in blocked.values():
        if faces_set:
            blocked_mask[np.fromiter(faces_set, np.int64)] = True
    keep = opposite & ~blocked_mask[f1] & ~blocked_mask[f2]
    r1, r2, f1, f2 = r1[keep], r2[keep], f1[keep], f2[keep]
    cosang = np.clip(np.einsum("ij,ij->i", unit_normals[f1], unit_normals[f2]), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    thr, unc = float(policy.max_neighbor_normal_angle_deg), float(policy.angle_uncertainty_deg)
    ambiguous = np.abs(angle - thr) <= unc
    crease = (angle > thr) & ~ambiguous
    candidate = ~crease & ~ambiguous
    cf1, cf2, cr1, cr2 = f1[candidate], f2[candidate], r1[candidate], r2[candidate]
    # attribute evidence on candidate edges: approval needs BOTH declared attributes, whatever the policy says
    region_barrier = np.zeros(len(cf1), bool)
    uv_seam = np.zeros(len(cf1), bool)
    if regions is not None:
        region_barrier = regions[cf1] != regions[cf2]
    if uv is not None:
        l1, l2 = local[cr1], local[cr2]
        # corner (l, l+1) of face 1 meets corner (l2+1, l2) of face 2 (opposite traversal)
        uv1a, uv1b = uv[cf1, l1], uv[cf1, (l1 + 1) % 3]
        uv2a, uv2b = uv[cf2, (l2 + 1) % 3], uv[cf2, l2]
        uv_seam = ~(np.all(uv1a == uv2a, axis=1) & np.all(uv1b == uv2b, axis=1))
    both_declared = regions is not None and uv is not None
    barrier = region_barrier | uv_seam
    approved = ~barrier & both_declared
    unverified = ~barrier & (not both_declared)
    # index adjacency (original vertex indices; raw relation as the current runtime uses)
    ia, ib = f[:, [0, 1, 2]].reshape(-1), f[:, [1, 2, 0]].reshape(-1)
    nvert = len(v)
    ikey = np.where(ia != ib, np.minimum(ia, ib) * nvert + np.maximum(ia, ib), -1 - np.arange(len(ia)))
    iorder, istarts, isizes = _pairs_by_key(ikey)
    itwo = istarts[(isizes == 2) & (ikey[iorder[istarts]] >= 0)]
    if1, if2 = face_of[iorder[itwo]], face_of[iorder[itwo + 1]]
    idist = if1 != if2
    csr = {"index": _csr(m, if1[idist], if2[idist]), "geometric_candidate": _csr(m, cf1, cf2),
           "approved": _csr(m, cf1[approved], cf2[approved])}
    if regions is None and uv is None:
        attribute_status = "no_boundary_attributes: geometric candidates have UNVERIFIED attribute boundaries"
    elif not both_declared:
        attribute_status = ("partial_boundary_attributes: approval needs BOTH region labels and corner UVs; "
                            "candidates UNVERIFIED")
    else:
        attribute_status = "boundary_attributes_checked"
    counts = {"vertices": int(len(v)), "faces": int(m), "exact_coordinates": ncoord, "boundary_edges": boundary,
              "nonmanifold_edges": int(nonmanifold_edges), "winding_conflict_edges": int(np.sum(conflict)),
              "index_adjacent_pairs": int(np.sum(idist)), "geometric_candidate_pairs": int(len(cf1)),
              "approved_pairs": int(np.sum(approved)), "unverified_candidate_pairs": int(np.sum(unverified)),
              "crease_edges": int(np.sum(crease)), "angle_ambiguous_edges": int(np.sum(ambiguous))}
    barrier_counts = {"region_barrier": int(np.sum(region_barrier)), "uv_seam": int(np.sum(uv_seam)),
                      "crease": int(np.sum(crease)), "angle_ambiguous": int(np.sum(ambiguous))}
    blocked_out = {k: tuple(sorted(int(x) for x in s)) for k, s in sorted(blocked.items())}
    body = {"schema": SCHEMA_TOPOLOGY, "algorithm_version": ALGORITHM_VERSION, "units": units,
            "policy": policy.to_dict(), "limits": limits.to_dict(), "input_digests": digests, "counts": counts,
            "blocked_faces": {k: list(t) for k, t in blocked_out.items()}, "barrier_counts": barrier_counts,
            "attribute_status": attribute_status, "approval_rule": APPROVAL_RULE,
            "degeneracy_definition": DEGENERACY_DEFINITION}
    record = TopologyV1(units=units, policy=policy, limits=limits, input_digests=MappingProxyType(dict(digests)),
                        counts=MappingProxyType(dict(counts)), blocked_faces=MappingProxyType(blocked_out),
                        barrier_counts=MappingProxyType(dict(barrier_counts)), attribute_status=attribute_status,
                        digest=_digest(body), _v=_frozen(v), _f=_frozen(f), _unit_normals=_frozen(unit_normals),
                        _blocked=_frozen(blocked_mask), _csr=MappingProxyType(csr),
                        _regions_raw=None if face_region_ids is None else _frozen(face_region_ids),
                        _uv_raw=None if corner_uv is None else _frozen(corner_uv))
    if len(_canonical(record.to_dict())) > limits.max_output_bytes:
        raise ValueError(f"topology record exceeds the {limits.name} output budget (actual to_dict representation)")
    return record


def _same_topology_state(derived: TopologyV1, claimed: Any) -> bool:
    try:
        if claimed.digest != derived.digest or claimed.to_dict() != derived.to_dict():
            return False
        for key in ("_v", "_f", "_unit_normals", "_blocked"):
            if not _same_array(getattr(derived, key), getattr(claimed, key)):
                return False
        for key in ("_regions_raw", "_uv_raw"):
            x, y = getattr(derived, key), getattr(claimed, key)
            if (x is None) != (y is None) or (x is not None and not _same_array(x, y)):
                return False
        if set(claimed._csr) != set(RELATIONS):
            return False
        for relation in RELATIONS:
            (do, dv), (co, cv) = derived._csr[relation], claimed._csr[relation]
            if not (_same_array(do, co) and _same_array(dv, cv)):
                return False
        return True
    except (AttributeError, TypeError, ValueError, KeyError):
        return False


def _rederive(topology: Any, vertices: np.ndarray, faces: np.ndarray) -> TopologyV1:
    """Never trust a supplied record: rebuild from the bound inputs and require identical state."""
    if type(topology) is not TopologyV1:
        raise ValueError("topology must be a TopologyV1 produced by build_topology")
    try:
        derived = build_topology(vertices, faces, units=topology.units, policy=topology.policy, limits=topology.limits,
                                 face_region_ids=topology._regions_raw, corner_uv=topology._uv_raw)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"topology cannot be re-derived from its bound inputs: {exc}") from exc
    if not _same_topology_state(derived, topology):
        raise ValueError("topology record or internal state differs from the state re-derived from its inputs")
    return derived


# --------------------------------------------------------------------------
# patch diagnosis


def _clip_to_square(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a convex polygon against the normalized square [-1, 1]^2."""
    for axis, bound, keep_le in ((0, 1.0, True), (0, -1.0, False), (1, 1.0, True), (1, -1.0, False)):
        out: list[tuple[float, float]] = []
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            pin = p[axis] <= bound if keep_le else p[axis] >= bound
            qin = q[axis] <= bound if keep_le else q[axis] >= bound
            if pin:
                out.append(p)
            if pin != qin:
                t = (bound - p[axis]) / (q[axis] - p[axis])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
        if not poly:
            return []
    return poly


def _area(poly: Sequence[tuple[float, float]]) -> float:
    s = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def _clip_convex(subject: list, clip: list) -> tuple[list, int]:
    """Intersection of convex polygons (``clip`` CCW); returns (polygon, edge operations)."""
    ops, out = 0, list(subject)
    for i in range(len(clip)):
        if not out:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % len(clip)]
        inp, out = out, []
        for j in range(len(inp)):
            ops += 1
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
            sq = (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out, ops


_ORIENT_ERR = 3.3306690738754716e-16 * 1.0000001     # Shewchuk ccwerrboundA-style bound for 2D orientation


def _orient_sign(a, b, c) -> int | None:
    """Sign of orient2d(a, b, c), or None when floating-point evaluation cannot certify it."""
    left = (a[0] - c[0]) * (b[1] - c[1])
    right = (a[1] - c[1]) * (b[0] - c[0])
    det = left - right
    bound = _ORIENT_ERR * (abs(left) + abs(right))
    if not (math.isfinite(det) and math.isfinite(bound)):
        return None
    if det > bound:
        return 1
    if det < -bound:
        return -1
    return 0 if det == 0.0 and bound == 0.0 else None


def _certainly_separated(p: list, q: list) -> tuple[bool, int]:
    """Separating-axis test for convex polygons with certified orientation signs: True only if
    some edge line of a certified-CCW p or q has every vertex of the other polygon certainly on its
    closed outer side (so the interiors are disjoint). Exactly shared vertices give exact zero."""
    ops = 0
    for s, o in ((p, q), (q, p)):
        ops += 1
        if len(s) < 3 or _orient_sign(s[0], s[1], s[2]) != 1:
            continue                      # orientation not certified: its edges cannot certify separation
        for i in range(len(s)):
            a, b = s[i], s[(i + 1) % len(s)]
            sep = True
            for c in o:
                ops += 1
                sign = _orient_sign(a, b, c)
                if sign is None or sign > 0:
                    sep = False
                    break
            if sep:
                return True, ops
    return False, ops


def _budget(reason: str, candidates: int, operations: int, memberships: int) -> dict:
    return {"status": "unavailable_budget", "reason": reason, "candidates": candidates, "operations": operations,
            "memberships": memberships, "positive_overlaps": None, "uncertain_pairs": None}


def _overlap_analysis(polys: list[list], tris: list[list], limits: TopologyLimitsV1, tol_area: float) -> dict:
    """Bounded overlap analysis of clipped chart polygons.

    Every stage is charged against the limits BEFORE it runs: grid memberships (counted
    arithmetically from each bounding box), candidate-pair generation work, and per-pair
    certification/intersection work. Disjoint interiors are proven ONLY by a certified
    separating axis on the unclipped chart triangles (subsets of disjoint sets are disjoint).
    An uncertified pair whose clipped intersection area is <= ``tol_area`` (including a computed 0)
    is numerically uncertain, never disjoint. Never a dense all-pairs matrix."""
    n = len(polys)
    if n != len(tris):
        raise ValueError("polygons and triangles must correspond")
    if n <= 1:
        return {"status": "disjoint_interiors_proven", "candidates": 0, "operations": 0, "memberships": 0,
                "positive_overlaps": 0, "uncertain_pairs": 0}
    cells = max(1, min(256, math.isqrt(n) + 1))
    width = 2.0 / cells
    ops = memberships = 0
    boxes = []
    for poly in polys:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        x0, x1 = max(0, int((min(xs) + 1.0) // width)), min(cells - 1, int((max(xs) + 1.0) // width))
        y0, y1 = max(0, int((min(ys) + 1.0) // width)), min(cells - 1, int((max(ys) + 1.0) // width))
        count = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
        memberships += count
        ops += count
        if memberships > limits.max_overlap_memberships or ops > limits.max_overlap_operations:
            return _budget("grid membership budget exceeded before population", 0, ops, memberships)
        boxes.append((x0, x1, y0, y1))
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (x0, x1, y0, y1) in enumerate(boxes):
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                buckets.setdefault((cx, cy), []).append(idx)
    pair_work = sum(len(m) * (len(m) - 1) // 2 for m in buckets.values())
    if pair_work > limits.max_overlap_candidates or ops + pair_work > limits.max_overlap_operations:
        return _budget("overlap candidate-generation budget exceeded before generation", 0, ops, memberships)
    ops += pair_work
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    positive = uncertain = 0
    for i, j in sorted(pairs):
        cert_bound = 2 + 2 * len(tris[i]) * len(tris[j])
        if ops + cert_bound > limits.max_overlap_operations:
            return _budget("overlap operation budget exceeded", len(pairs), ops, memberships)
        separated, used = _certainly_separated(tris[i], tris[j])
        ops += used
        if separated:
            continue
        clip_bound = len(polys[j]) * (len(polys[i]) + len(polys[j]))
        if ops + clip_bound > limits.max_overlap_operations:
            return _budget("overlap operation budget exceeded", len(pairs), ops, memberships)
        inter, used = _clip_convex(polys[i], polys[j])
        ops += used
        area = abs(_area(inter)) if len(inter) >= 3 else 0.0
        if area > tol_area:
            positive += 1
        else:
            uncertain += 1                 # no separation certificate: a small or zero area is NOT proof
    status = ("positive_overlap" if positive else "unavailable_numerical_uncertainty" if uncertain
              else "disjoint_interiors_proven")
    return {"status": status, "candidates": len(pairs), "operations": ops, "memberships": memberships,
            "positive_overlaps": positive, "uncertain_pairs": uncertain}


def _validate_marker(marker: Any, n_faces: int, limits: TopologyLimitsV1) -> dict:
    if not isinstance(marker, Mapping) or set(marker) != MARKER_KEYS:
        raise ValueError(f"marker must have exactly the keys {sorted(MARKER_KEYS)}")
    fid = marker["face_id"]
    if type(fid) is not int or not 0 <= fid < n_faces:
        raise ValueError("marker face_id must be an int face index")

    def vec(key, n):
        value = marker[key]
        if not isinstance(value, (list, tuple)) or len(value) != n or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                                           for x in value):
            raise ValueError(f"marker {key} must be {n} real numbers")
        arr = np.asarray(value, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"marker {key} must be finite")
        return arr
    bary, t, b, n = vec("barycentric", 3), vec("tangent", 3), vec("bitangent", 3), vec("normal", 3)
    tol = limits.barycentric_tol
    if bary.min() < -tol or bary.max() > 1.0 + tol or abs(bary.sum() - 1.0) > tol:
        raise ValueError("marker barycentric must lie in [0, 1] and sum to 1 (anchor inside its triangle)")
    otol = limits.orthonormal_tol
    for name, x in (("tangent", t), ("bitangent", b), ("normal", n)):
        if abs(float(np.linalg.norm(x)) - 1.0) > otol:
            raise ValueError(f"marker {name} must be unit length")
    if max(abs(float(t @ b)), abs(float(t @ n)), abs(float(b @ n))) > otol or float(np.cross(t, b) @ n) < 1.0 - otol:
        raise ValueError("marker chart must be orthonormal and right-handed (t x b = n)")
    if not float(np.linalg.det(np.stack([t, b, n]))) > MIN_FRAME_DETERMINANT:
        raise ValueError("marker chart frame is singular or reflected (non-relaxable validity check)")
    size = marker["size_object_units"]
    if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0:
        raise ValueError("marker size_object_units must be finite and positive")
    if not float(size) / 2.0 >= FLOAT_MIN_NORMAL:
        raise ValueError("marker size_object_units / 2 is not a normal float64 value (numerical range refusal)")
    if not _is_hex64(marker["artwork_sha256"]):
        raise ValueError("marker artwork_sha256 must be an explicit lowercase SHA-256 digest")
    return {"face_id": fid, "barycentric": bary, "t": t, "b": b, "n": n, "size": float(size),
            "artwork_sha256": marker["artwork_sha256"]}


def _chart_shape(chart: np.ndarray, face: np.ndarray, tol: float) -> tuple[int | None, float]:
    """Certified chart orientation sign and dimensionless chart shape quality of one face."""
    a, b, c = (tuple(float(x) for x in chart[i]) for i in face)
    sign = _orient_sign(a, b, c)
    det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    l2 = max((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]),
             (c[0] - a[0]) * (c[0] - a[0]) + (c[1] - a[1]) * (c[1] - a[1]),
             (c[0] - b[0]) * (c[0] - b[0]) + (c[1] - b[1]) * (c[1] - b[1]))
    if not (math.isfinite(det) and math.isfinite(l2)) or l2 <= 0:
        return None, 0.0
    quality = abs(det) / l2
    return (sign if quality > tol else None), quality


def _traverse(topology: TopologyV1, relation: str, mk: dict, support: np.ndarray, *, min_support: int,
              max_anchor: float, max_neighbor: float, limits: TopologyLimitsV1, chart: np.ndarray) -> dict:
    f, normals, blocked = topology._f, topology._unit_normals, topology._blocked
    unc, shape_tol = topology.policy.angle_uncertainty_deg, topology.policy.degenerate_shape_tol
    f0 = mk["face_id"]
    sign = 1.0 if float(normals[f0] @ mk["n"]) >= 0 else -1.0

    def angle(u, w):
        return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ w)))))
    stops = {"unsupported": 0, "anchor_angle": 0, "neighbor_angle": 0, "angle_ambiguous": 0, "reach": 0, "blocked": 0,
             "folded": 0, "chart_degenerate_or_uncertain": 0}
    folded: set[int] = set()
    if blocked[f0]:
        return {"status": "anchor_rejected", "reason": "anchor face is blocked (degenerate/duplicate/conflict/nonmanifold)"}
    if support[f0] < min_support:
        return {"status": "anchor_rejected", "reason": "anchor face unsupported"}
    a0 = angle(sign * normals[f0], mk["n"])
    if a0 > max_anchor or abs(a0 - max_anchor) <= unc:
        return {"status": "anchor_rejected", "reason": f"anchor face normal angle {a0:.6f} deg not within {max_anchor}"}
    anchor_sign, _q = _chart_shape(chart, f[f0], shape_tol)
    if anchor_sign not in (1, -1):
        return {"status": "anchor_rejected", "reason": "anchor face is degenerate or orientation-uncertain in the chart"}
    chosen, queue = {f0}, deque([f0])
    offsets, values = topology._csr[relation]
    while queue:
        i = queue.popleft()
        for j in values[offsets[i]:offsets[i + 1]].tolist():
            if j in chosen or j in folded:
                continue
            if blocked[j]:
                stops["blocked"] += 1; continue
            if support[j] < min_support:
                stops["unsupported"] += 1; continue
            aa, na = angle(sign * normals[j], mk["n"]), angle(normals[i], normals[j])
            if abs(aa - max_anchor) <= unc or abs(na - max_neighbor) <= unc:
                stops["angle_ambiguous"] += 1; continue
            if aa > max_anchor:
                stops["anchor_angle"] += 1; continue
            if na > max_neighbor:
                stops["neighbor_angle"] += 1; continue
            if float(np.min(np.max(np.abs(chart[f[j]]), axis=1))) > REACH_FACTOR:
                stops["reach"] += 1; continue
            s_j, _q = _chart_shape(chart, f[j], shape_tol)
            if s_j != anchor_sign:
                folded.add(j)
                stops["folded" if s_j in (1, -1) else "chart_degenerate_or_uncertain"] += 1
                continue
            if len(chosen) >= limits.max_traversal_faces:
                return {"status": "unavailable_budget", "reason": "traversal face budget exceeded"}
            chosen.add(j)
            queue.append(j)
    return {"status": "measured", "faces": sorted(chosen), "folded_faces": sorted(folded), "stops": stops,
            "anchor_chart_sign": anchor_sign}


def _physical(ratio_half_units: float, h2: float) -> float | None:
    """Convert an area in half-size^2 units to declared units^2, or None if not representable."""
    if not (math.isfinite(h2) and h2 >= FLOAT_MIN_NORMAL):
        return None
    value = ratio_half_units * h2
    if not math.isfinite(value) or (ratio_half_units != 0.0 and value < FLOAT_MIN_NORMAL):
        return None
    return value


def _coverage(topology: TopologyV1, faces: list[int], chart: np.ndarray, rel: np.ndarray, half: float,
              anchor_sign: int, limits: TopologyLimitsV1) -> dict:
    """Coverage computed dimensionlessly (half-size units; nominal square area 4) and converted to
    declared squared units only when representable."""
    f = topology._f
    with np.errstate(**_NUMERIC_ERR):
        tri = rel[f[faces]]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        surface_half = 0.5 * np.sqrt(np.einsum("ij,ij->i", cr, cr))
    polys, tris, clipped_half, inside_half = [], [], 0.0, 0.0
    for k, face in enumerate(faces):
        pts = [tuple(float(x) for x in chart[idx]) for idx in f[face]]
        if anchor_sign < 0:
            pts = pts[::-1]                 # certified orientation (traversal) -> CCW
        full = abs(_area(pts))
        clip = _clip_to_square(pts)
        a = abs(_area(clip)) if len(clip) >= 3 else 0.0
        clipped_half += a
        if full > 0:
            inside_half += float(surface_half[k]) * (a / full)
        if a > 0:
            polys.append(clip)
            tris.append(pts)
    nominal_half = 4.0
    tol = topology.policy.overlap_relative_area_tol * nominal_half
    overlap = _overlap_analysis(polys, tris, limits, tol)
    surface_total = float(np.sum(surface_half))
    ratio = clipped_half / nominal_half
    union = ratio if overlap["status"] == "disjoint_interiors_proven" else None
    h2 = half * half
    physical = {"patch_surface_area": _physical(surface_total, h2),
                "surface_area_inside_nominal_footprint": _physical(inside_half, h2),
                "summed_clipped_chart_area_counting_multiplicity": _physical(clipped_half, h2),
                "nominal_square_area": _physical(nominal_half, h2)}
    values = [surface_total, inside_half, clipped_half, ratio] + ([] if union is None else [union])
    if not all(math.isfinite(x) for x in values):
        raise FloatingPointError("non-finite dimensionless coverage value")
    return {**physical,
            "summed_clipped_chart_area_ratio": ratio, "union_coverage_fraction": union,
            "dimensionless": {"patch_surface_area_over_nominal": surface_total / nominal_half,
                              "surface_area_inside_over_nominal": inside_half / nominal_half,
                              "summed_clipped_chart_area_ratio": ratio, "union_coverage_fraction": union},
            "physical_units_representable": all(x is not None for x in physical.values()),
            "overlap": overlap,
            "area_units": f"{topology.units}^2" if topology.units != "relative" else "relative_units^2",
            "notes": "summed ratio counts multiplicity and is never clamped; it is not a fraction, visibility or accuracy"}


@dataclass(frozen=True)
class PatchDiagnosisV1:
    """Detached snapshot: the canonical JSON bytes are the record; ``body`` returns a fresh copy."""
    canonical_json: bytes
    digest: str
    schema: str = SCHEMA_PATCH

    @property
    def body(self) -> dict:
        return json.loads(self.canonical_json)

    def to_dict(self) -> dict:
        return {**self.body, "digest": self.digest}


def _seal(body: dict, max_bytes: int, what: str) -> tuple[bytes, str]:
    """Canonical bytes and digest, refusing when the ACTUAL ``to_dict`` representation exceeds the cap."""
    raw = _canonical(body)
    digest = hashlib.sha256(raw).hexdigest()
    if len(_canonical({**body, "digest": digest})) > max_bytes:
        raise ValueError(f"{what} exceeds the output budget ({max_bytes} bytes, actual returned representation)")
    return raw, digest


def _check_mesh_bounds(vertices: np.ndarray, faces: np.ndarray, limit_records: Sequence[TopologyLimitsV1],
                       topology: Any) -> None:
    """Shape/count/half-edge bounds from metadata only, BEFORE any hashing or materialization. Every
    supplied limits record applies (the diagnosis limits and the topology's own)."""
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("vertices must be (N, 3) and faces (M, 3)")
    n, m = int(vertices.shape[0]), int(faces.shape[0])
    for lim in limit_records:
        if n > lim.max_vertices or m > lim.max_faces or 3 * m > lim.max_half_edges:
            raise ValueError(f"mesh exceeds the {lim.name} vertex/face/half-edge limits (checked before hashing)")
    try:
        expected = (int(topology.counts["vertices"]), int(topology.counts["faces"]))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("topology record lacks valid counts") from exc
    if (n, m) != expected:
        raise ValueError("vertices/faces shapes differ from the topology record")


def diagnose_patch(vertices: np.ndarray, faces: np.ndarray, marker: Mapping[str, Any], face_support: np.ndarray,
                   topology: TopologyV1, *, units: str, min_support: int = 3, max_anchor_normal_angle_deg: float = 75.0,
                   max_neighbor_normal_angle_deg: float = 75.0, limits: TopologyLimitsV1) -> PatchDiagnosisV1:
    """Independent bounded traversals (index / geometric_candidate / approved) with exact coverage."""
    if type(topology) is not TopologyV1 or type(limits) is not TopologyLimitsV1:
        raise ValueError("topology must be a TopologyV1 and limits a TopologyLimitsV1")
    _check_units(units)
    if units != topology.units:
        raise ValueError("units differ from the topology record")
    if not isinstance(vertices, np.ndarray) or not isinstance(faces, np.ndarray):
        raise ValueError("vertices and faces must be numpy ndarrays")
    _check_mesh_bounds(vertices, faces, (limits, topology.limits), topology)
    if array_digest(vertices) != topology.input_digests["vertices"] or array_digest(faces) != topology.input_digests["faces"]:
        raise ValueError("vertices/faces differ from the arrays the topology was built from")
    topology = _rederive(topology, vertices, faces)
    if type(min_support) is not int or min_support < 0:
        raise ValueError("min_support must be a non-negative int")
    for name, value in (("max_anchor_normal_angle_deg", max_anchor_normal_angle_deg),
                        ("max_neighbor_normal_angle_deg", max_neighbor_normal_angle_deg)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 180:
            raise ValueError(f"{name} must be in (0, 180]")
    if not isinstance(face_support, np.ndarray) or face_support.ndim != 1:
        raise ValueError("face_support must be an integer (M,) ndarray")
    support = _index_array(face_support, "face_support", len(topology._f), None)
    if support.size and support.min() < 0:
        raise ValueError("face_support counts must be non-negative")
    mk = _validate_marker(marker, len(topology._f), limits)
    v, f = topology._v, topology._f
    half = mk["size"] / 2.0
    t, b = mk["t"], mk["b"]
    try:
        with np.errstate(**_NUMERIC_ERR):
            p = (mk["barycentric"][:, None] * v[f[mk["face_id"]]]).sum(axis=0)
            rel = (v - p) / half                                   # dimensionless; nominal square is [-1, 1]^2
            # elementwise (not BLAS) so identical coordinates give bit-identical chart coordinates
            cx = rel[:, 0] * t[0] + rel[:, 1] * t[1] + rel[:, 2] * t[2]
            cy = rel[:, 0] * b[0] + rel[:, 1] * b[1] + rel[:, 2] * b[2]
            chart = np.stack([cx, cy], axis=1)
    except FloatingPointError as exc:
        raise ValueError(f"numerical range error in the marker chart: {exc}") from exc
    if not (np.isfinite(rel).all() and np.isfinite(chart).all()):
        raise ValueError("numerical range error in the marker chart")
    traversals = {}
    for relation in RELATIONS:
        tr = _traverse(topology, relation, mk, support, min_support=min_support, max_anchor=float(max_anchor_normal_angle_deg),
                       max_neighbor=float(max_neighbor_normal_angle_deg), limits=limits, chart=chart)
        if tr["status"] == "measured":
            try:
                cov = _coverage(topology, tr["faces"], chart, rel, half, tr["anchor_chart_sign"], limits)
                tr["coverage"] = cov
                if not cov["physical_units_representable"]:
                    tr["status"] = "unavailable_numerical_range"
                    tr["reason"] = "declared squared units are not representable as normal finite float64 values"
            except FloatingPointError as exc:
                tr = {"status": "unavailable_numerical", "reason": f"numerical range error in coverage: {exc}"}
        traversals[relation] = tr
    traversals["geometric_candidate"]["label"] = ("geometric candidates with UNVERIFIED attribute boundaries; "
                                                  "not approved runtime adjacency")
    anchor_neighbors = {rel_name: list(topology.neighbors(rel_name, mk["face_id"])) for rel_name in RELATIONS}
    body = {"schema": SCHEMA_PATCH, "algorithm_version": ALGORITHM_VERSION, "units": units,
            "topology_digest": topology.digest, "limits": limits.to_dict(),
            "parameters": {"min_support": min_support, "max_anchor_normal_angle_deg": float(max_anchor_normal_angle_deg),
                           "max_neighbor_normal_angle_deg": float(max_neighbor_normal_angle_deg),
                           "chart_reach_half_multiple": REACH_FACTOR},
            "marker": {"face_id": mk["face_id"], "barycentric": mk["barycentric"].tolist(), "tangent": mk["t"].tolist(),
                       "bitangent": mk["b"].tolist(), "normal": mk["n"].tolist(), "size_object_units": mk["size"],
                       "artwork_sha256": mk["artwork_sha256"]},
            "face_support_digest": array_digest(face_support), "anchor_neighbors": anchor_neighbors,
            "traversals": traversals,
            "claims": {"runtime_patch_replaced": False, "physical_visibility": False, "tracking_improvement": False,
                       "declared_units_only_no_conversion": True,
                       "metric_scale_claimed": False if units == "relative" else "declared_not_verified"}}
    try:
        raw, digest = _seal(body, limits.max_output_bytes, "patch diagnosis")
    except ValueError:
        for tr in traversals.values():
            if "faces" in tr:
                tr["faces"], tr["folded_faces"], tr["face_lists"] = None, None, "omitted: output budget"
        raw, digest = _seal(body, limits.max_output_bytes, "patch diagnosis")
    return PatchDiagnosisV1(canonical_json=raw, digest=digest)


# --------------------------------------------------------------------------
# prefix binding (in-memory evidence only)

MANIFEST_KEYS = {"schema", "clip_id", "run_id", "units", "frames"}
FRAME_KEYS = {"frame_id", "clip_position", "source_frame_index", "timestamp_s", "clock_kind", "width", "height", "K",
              "image", "observed_mask", "pose", "producer_receipt_sha256"}
IMAGE_KEYS = {"array_sha256"}
MASK_KEYS = {"array_sha256", "kind", "producer"}
POSE_KEYS = {"camera_from_object", "units", "provenance", "producer_receipt_sha256"}
VIEW_KEYS = {"rgb", "mask"}
_MANIFEST_NODES_PER_FRAME = 96


@dataclass(frozen=True)
class PrefixBindingDiagnosisV1:
    """Detached snapshot: the canonical JSON bytes are the record; ``body`` returns a fresh copy."""
    canonical_json: bytes
    digest: str
    schema: str = SCHEMA_PREFIX

    @property
    def body(self) -> dict:
        return json.loads(self.canonical_json)

    def to_dict(self) -> dict:
        return {**self.body, "digest": self.digest}


def _exact_keys(value: Any, keys: set, what: str) -> Mapping:
    if not isinstance(value, Mapping) or len(value) != len(keys) or set(value) != keys:
        raise ValueError(f"{what} must have exactly the keys {sorted(keys)}")
    return value


def _bounded_plain(value: Any, *, max_nodes: int, max_chars: int) -> None:
    """Refuse non-plain or oversized manifest content BEFORE it is serialized or hashed."""
    stack, nodes = [value], 0
    while stack:
        item = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("prefix_manifest exceeds its structural size bound")
        if isinstance(item, (Mapping, list)):
            # length is checked against the remaining node budget BEFORE any iteration or extension
            if len(item) > max_nodes - nodes - len(stack):
                raise ValueError("prefix_manifest container exceeds its structural size bound")
        if isinstance(item, Mapping):
            for k in item:
                if not isinstance(k, str) or len(k) > max_chars:
                    raise ValueError("prefix_manifest keys must be bounded strings")
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            if len(item) > max_chars:
                raise ValueError(f"prefix_manifest string exceeds {max_chars} characters")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("prefix_manifest numbers must be finite")
        elif item is None or isinstance(item, (bool, int)):
            if isinstance(item, int) and not isinstance(item, bool) and abs(item) > INT_EXACT:
                raise ValueError("prefix_manifest integers must be within +/-2**53")
        else:
            raise ValueError(f"prefix_manifest must be plain JSON data (found {type(item).__name__})")


def _validated_records(clip_spec: Any, run_spec: Any, wc: Any) -> None:
    """Structural validation of the ACTUAL record attributes (``to_dict`` hard-codes some fields, so it
    alone could hide a violation)."""
    if type(clip_spec) is not wc.ClipSpec or type(run_spec) is not wc.RunSpec:
        raise ValueError("clip_spec/run_spec must be ClipSpec/RunSpec records")
    if clip_spec.schema != wc.CLIP_SCHEMA or run_spec.schema != wc.RUN_SCHEMA:
        raise ValueError("clip_spec/run_spec carry an unsupported schema")
    if clip_spec.evaluator_inputs_read is not False:
        raise ValueError("clip_spec records evaluator inputs; prefix evidence refused")
    try:
        clip_again = wc.ClipSpec.from_dict(clip_spec.to_dict())
        run_again = wc.RunSpec.from_dict(run_spec.to_dict())
    except (wc.ContractError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"clip_spec/run_spec do not re-validate: {exc}") from exc
    if clip_again != clip_spec or run_again != run_spec:
        raise ValueError("clip_spec/run_spec attributes differ from their validated serialization")


def _rigid_pose(value: Any, tol: float) -> bool:
    """Exact 4x4 list of non-bool real numbers forming a finite proper rigid transform."""
    if not isinstance(value, list) or len(value) != 4:
        return False
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return False
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return False
    try:
        T = np.array([[float(x) for x in row] for row in value], dtype=np.float64)
    except (OverflowError, ValueError):
        return False
    if not np.isfinite(T).all() or not np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0]):
        return False
    R = T[:3, :3]
    if not float(np.linalg.det(R)) > MIN_FRAME_DETERMINANT:
        return False                      # non-relaxable: singular or reflected rotation
    return float(np.max(np.abs(R.T @ R - np.eye(3)))) <= tol and abs(float(np.linalg.det(R)) - 1.0) <= tol


def validate_prefix_binding(clip_spec: Any, run_spec: Any, prefix_manifest: Mapping[str, Any],
                            supplied_views: Mapping[str, Mapping[str, Any]], *, limits: TopologyLimitsV1) -> PrefixBindingDiagnosisV1:
    """In-memory consistency of supplied prefix evidence. NEVER authenticates files, producers or physical
    observation (``authenticated`` is always False; no argument can set it). Structural errors raise
    ValueError; semantic violations give ``status: refused`` with reasons; missing evidence gives
    ``unavailable`` (never zero support). Nothing here consumes post-prefix tracking rows."""
    from . import workbench_contract as wc
    if type(limits) is not TopologyLimitsV1:
        raise ValueError("limits must be a TopologyLimitsV1")
    _validated_records(clip_spec, run_spec, wc)
    # ---- structural bounds BEFORE any serialization or hashing
    m = _exact_keys(prefix_manifest, MANIFEST_KEYS, "prefix_manifest")
    frames = m["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("prefix_manifest.frames must be a non-empty list")
    if len(frames) > limits.max_prefix_frames:
        raise ValueError(f"prefix exceeds {limits.max_prefix_frames} frames")
    for raw in frames:
        _exact_keys(raw, FRAME_KEYS, "prefix frame")
    _bounded_plain(m, max_nodes=16 + _MANIFEST_NODES_PER_FRAME * len(frames), max_chars=limits.max_prefix_string_chars)
    if m["schema"] != SCHEMA_PREFIX_MANIFEST:
        raise ValueError("unsupported prefix_manifest schema")
    _check_units(m["units"])
    if not isinstance(supplied_views, Mapping):
        raise ValueError("supplied_views must be a mapping frame_id -> {rgb, mask}")
    if len(supplied_views) > limits.max_prefix_frames:
        raise ValueError(f"supplied_views exceeds {limits.max_prefix_frames} views")
    if not all(isinstance(k, str) for k in supplied_views):
        raise ValueError("supplied_views must be a mapping frame_id -> {rgb, mask}")
    if any(len(k) > limits.max_prefix_string_chars for k in supplied_views):
        raise ValueError("supplied_views keys exceed the string bound")
    for view in supplied_views.values():
        _exact_keys(view, VIEW_KEYS, "supplied view")
    if clip_spec.width * clip_spec.height > limits.max_prefix_view_pixels:
        raise ValueError(f"clip views exceed {limits.max_prefix_view_pixels} pixels (checked before hashing)")
    manifest_sha = _digest(m)
    reasons: list[str] = []
    try:
        run_spec.validate_against(clip_spec)
    except (wc.ContractError, KeyError) as exc:
        reasons.append(f"run does not validate against the clip: {exc}")
    if m["clip_id"] != clip_spec.clip_id or m["run_id"] != run_spec.run_id or run_spec.clip_id != clip_spec.clip_id:
        reasons.append("manifest clip/run identity differs from the supplied ClipSpec/RunSpec")
    position = clip_spec.frame_position()
    onboarding = set(run_spec.onboarding.keyframe_ids)
    scored = set(run_spec.scheduled_frame_ids) | set(clip_spec.scoring_frame_ids)
    first_scored = min((position[f] for f in scored if f in position), default=None)
    clip_K = None if clip_spec.intrinsics.K is None else [float(x) for x in clip_spec.intrinsics.K]
    manifest_ids = {fr["frame_id"] for fr in frames if isinstance(fr["frame_id"], str)}
    extra = sorted(set(supplied_views) - manifest_ids)
    if extra:
        reasons.append(f"supplied views outside the prefix manifest refused: {extra}")
    shape_rgb, shape_mask = (clip_spec.height, clip_spec.width, 3), (clip_spec.height, clip_spec.width)
    per_frame, last_pos, view_digests = [], None, {}
    for fr in frames:
        fid, refuse, missing = fr["frame_id"], [], []
        if not isinstance(fid, str) or fid not in position:
            raise ValueError(f"prefix frame {fid!r} is not a clip frame")
        spec = clip_spec.frame(fid)
        pos = fr["clip_position"]
        if type(pos) is not int or pos != position[fid]:
            refuse.append("clip_position differs from the clip manifest")
        if last_pos is not None and position[fid] <= last_pos:
            refuse.append("clip positions are not strictly unique and increasing")
        last_pos = position[fid]
        if fid not in onboarding:
            refuse.append("frame is not in the reviewed run onboarding prefix")
        if fid in scored or (first_scored is not None and position[fid] >= first_scored):
            refuse.append("scored or future frame refused")
        if type(fr["source_frame_index"]) is not int or fr["source_frame_index"] != spec.source_frame_index:
            refuse.append("source_frame_index differs from the clip manifest")
        if fr["clock_kind"] != clip_spec.clock_kind:
            refuse.append("clock_kind differs from the clip")
        ts = fr["timestamp_s"]
        if (ts is not None and type(ts) is not float) or ts != spec.timestamp_s:
            refuse.append("timestamp differs from the clip manifest (bool/int/str refused)")
        if type(fr["width"]) is not int or type(fr["height"]) is not int \
                or fr["width"] != clip_spec.width or fr["height"] != clip_spec.height:
            refuse.append("frame dimensions/crop differ from the clip")
        if clip_K is None:
            missing.append("clip intrinsics unavailable")
        elif (not isinstance(fr["K"], list) or len(fr["K"]) != 9
              or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in fr["K"])
              or [float(x) for x in fr["K"]] != clip_K):
            refuse.append("K differs from the clip intrinsics")
        if fr["producer_receipt_sha256"] is None:
            missing.append("producer receipt missing")
        elif not _is_hex64(fr["producer_receipt_sha256"]):
            refuse.append("producer receipt is not a SHA-256 digest")
        view = supplied_views.get(fid)
        # image (required)
        img = fr["image"]
        if img is None or view is None or view["rgb"] is None:
            missing.append("image evidence missing")
        else:
            _exact_keys(img, IMAGE_KEYS, "image")
            rgb = view["rgb"]
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != shape_rgb:
                refuse.append("supplied RGB is not a uint8 array of the clip dimensions")
            elif not _is_hex64(img["array_sha256"]) or array_digest(rgb) != img["array_sha256"]:
                refuse.append("supplied RGB bytes differ from the manifest image identity")
            else:
                view_digests[f"{fid}:rgb"] = img["array_sha256"]
        # observed mask (never a silhouette, never zero support when absent)
        om = fr["observed_mask"]
        mask = None if view is None else view["mask"]
        if om is None or mask is None:
            missing.append("observed mask missing (unavailable, not zero support)")
        else:
            _exact_keys(om, MASK_KEYS, "observed_mask")
            if om["kind"] != "observed":
                refuse.append("mask provenance is not an observed mask (silhouettes/projections refused)")
            if not isinstance(om["producer"], str) or not om["producer"] \
                    or any(w in om["producer"].lower() for w in FORBIDDEN_PROVENANCE_WORDS):
                refuse.append("mask producer is empty or evaluator/scored-derived")
            if not isinstance(mask, np.ndarray) or mask.shape != shape_mask or mask.dtype.kind not in "bu":
                refuse.append("supplied observed mask is malformed or has the wrong dimensions")
            elif not _is_hex64(om["array_sha256"]) or mask_digest(mask) != om["array_sha256"]:
                refuse.append("supplied mask differs from the manifest observed-mask identity")
            else:
                view_digests[f"{fid}:mask"] = om["array_sha256"]
        # pose
        pose = fr["pose"]
        if pose is None:
            missing.append("pose evidence missing")
        else:
            _exact_keys(pose, POSE_KEYS, "pose")
            prov = pose["provenance"]
            if (not isinstance(prov, str) or not prov.startswith(ALLOWED_POSE_PROVENANCE)
                    or any(w in prov.lower() for w in FORBIDDEN_PROVENANCE_WORDS)):
                refuse.append("pose provenance is not controlled_init:/onboarding:/reconstruction: (evaluator/scored refused)")
            if pose["units"] != m["units"]:
                refuse.append("pose units differ from the manifest units")
            if pose["producer_receipt_sha256"] is None:
                missing.append("pose producer receipt missing")
            elif not _is_hex64(pose["producer_receipt_sha256"]):
                refuse.append("pose producer receipt is not a SHA-256 digest")
            if not _rigid_pose(pose["camera_from_object"], limits.orthonormal_tol):
                refuse.append("pose is not an exact 4x4 list of non-bool finite reals forming a proper rigid transform")
        status = "refused" if refuse else "unavailable" if missing else "bound"
        per_frame.append({"frame_id": fid, "clip_position": position[fid], "status": status,
                          "refusals": refuse, "missing": missing})
    refused = bool(reasons) or any(p["status"] == "refused" for p in per_frame)
    unavailable = any(p["status"] == "unavailable" for p in per_frame)
    if clip_spec.clock_kind == "timestamp_seconds":
        stamps = [clip_spec.frame(p["frame_id"]).timestamp_s for p in per_frame]
        duration = float(max(stamps) - min(stamps))
        duration_status = "attested_within_10s" if duration <= limits.max_prefix_seconds else "exceeds_10s"
        if duration_status == "exceeds_10s":
            refused = True
            reasons.append("prefix spans more than 10 seconds")
    else:
        duration, duration_status = None, "unattested_frame_index_clock"
    status = "refused" if refused else "unavailable" if unavailable else "bound_in_memory_unauthenticated"
    body = {"schema": SCHEMA_PREFIX, "algorithm_version": ALGORITHM_VERSION, "status": status,
            "authenticated": False,
            "authentication_note": ("supplied in-memory evidence only; files, producers and physical observation are NOT "
                                    "authenticated. Future loader integration must verify registered media, masks, package, "
                                    "bundle and source closure before any authenticated claim."),
            "clip_id": clip_spec.clip_id, "run_id": run_spec.run_id, "clip_sha256": _digest(clip_spec.to_dict()),
            "run_sha256": _digest(run_spec.to_dict()), "manifest_sha256": manifest_sha,
            "supplied_view_digests": dict(sorted(view_digests.items())), "frames": per_frame,
            "duration_s": duration, "duration_status": duration_status, "reasons": reasons, "limits": limits.to_dict(),
            "post_prefix_rows_consumed": False}
    raw, digest = _seal(body, limits.max_output_bytes, "prefix binding diagnosis")
    return PrefixBindingDiagnosisV1(canonical_json=raw, digest=digest)
