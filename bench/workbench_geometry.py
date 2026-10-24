"""Pure geometry for the workbench: normalization, projection, rasterization,
anchoring, curved marker rendering and the relative RGB tracker.

Conventions follow ``workbench_contract``: OpenCV camera axes, integer pixel
centres, camera-from-object poses acting on column vectors. Rasterization uses
nearest positive depth, perspective-correct barycentrics
``a = sum(l_i a_i / z_i) / sum(l_i / z_i)``, ``z = 1 / sum(l_i / z_i)``,
near-plane clipping (triangles crossing the plane are clipped, not discarded)
and exact integer edge functions with a deterministic ownership rule so a pixel
centre on a shared edge belongs to exactly one triangle.

A projected silhouette is never treated as an observed mask: every renderer or
tracker entry point that needs a mask takes the observed mask explicitly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping

import numpy as np

SUBPIXEL_BITS = 8
SUBPIXEL = 1 << SUBPIXEL_BITS
GUARD_BAND_PX = float(1 << 22)


class GeometryError(ValueError):
    """Invalid geometry input."""


# --------------------------------------------------------------------------
# identities

def array_sha256(array: np.ndarray) -> str:
    """Content identity of an image/mask array (dtype, shape and bytes)."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    return array_sha256(np.ascontiguousarray(np.asarray(mask) > 0).astype(np.uint8))


# --------------------------------------------------------------------------
# mesh validation and normalization

def validate_mesh(vertices: np.ndarray, faces: np.ndarray, *, max_faces: int = 20_000,
                  require_oriented: bool = True) -> dict:
    v = np.asarray(vertices)
    f = np.asarray(faces)
    if v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
        raise GeometryError("vertices must be float32 Nx3 with N >= 3")
    if f.dtype != np.int32 or f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 1:
        raise GeometryError("faces must be int32 Mx3 with M >= 1")
    if f.shape[0] > max_faces:
        raise GeometryError(f"mesh exceeds {max_faces} triangles")
    if not np.isfinite(v).all():
        raise GeometryError("vertices must be finite")
    if f.min() < 0 or f.max() >= v.shape[0]:
        raise GeometryError("face index out of range")
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])):
        raise GeometryError("face repeats a vertex")
    areas = np.linalg.norm(face_cross(v, f), axis=1)
    scale = float(np.max(np.ptp(v.astype(np.float64), axis=0))) or 1.0
    if np.any(areas <= 1e-12 * scale * scale):
        raise GeometryError("mesh contains degenerate triangles")
    oriented, manifold = orientation_consistent(f)
    if require_oriented and not oriented:
        raise GeometryError("mesh faces are not consistently oriented")
    return {"vertex_count": int(v.shape[0]), "face_count": int(f.shape[0]),
            "consistently_oriented": oriented, "edge_manifold": manifold}


def face_cross(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return np.cross(b - a, c - a)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = face_cross(vertices, faces)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(length, 1e-300)


def orientation_consistent(faces: np.ndarray) -> tuple[bool, bool]:
    """(consistently oriented, every edge used by <= 2 faces)."""
    f = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    # A repeated directed edge means two neighbours traverse it the same way.
    keys = directed[:, 0] * (int(f.max()) + 1) + directed[:, 1]
    _, dcounts = np.unique(keys, return_counts=True)
    undirected = np.sort(directed, axis=1)
    ukeys = undirected[:, 0] * (int(f.max()) + 1) + undirected[:, 1]
    _, ucounts = np.unique(ukeys, return_counts=True)
    return bool(np.all(dcounts == 1)), bool(np.all(ucounts <= 2))


def orient_faces(faces: np.ndarray) -> np.ndarray:
    """Flip faces by BFS so shared edges are traversed oppositely (when possible)."""
    f = np.array(faces, dtype=np.int32, copy=True)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for i, (a, b, c) in enumerate(f.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(u, v), max(u, v)), []).append(i)
    seen = np.zeros(len(f), dtype=bool)
    for seed in range(len(f)):
        if seen[seed]:
            continue
        seen[seed] = True
        queue = deque([seed])
        while queue:
            i = queue.popleft()
            a, b, c = f[i].tolist()
            for u, v in ((a, b), (b, c), (c, a)):
                for j in edge_faces[(min(u, v), max(u, v))]:
                    if j == i or seen[j]:
                        continue
                    ja, jb, jc = f[j].tolist()
                    same = (u, v) in ((ja, jb), (jb, jc), (jc, ja))
                    if same:
                        f[j] = f[j][[0, 2, 1]]
                    seen[j] = True
                    queue.append(j)
    return f


def edge_adjacency(faces: np.ndarray) -> list[list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for i, (a, b, c) in enumerate(np.asarray(faces).tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(u, v), max(u, v)), []).append(i)
    adjacency: list[list[int]] = [[] for _ in range(len(faces))]
    for owners in edge_faces.values():
        for i in owners:
            adjacency[i].extend(j for j in owners if j != i)
    return adjacency


@dataclass(frozen=True)
class Normalization:
    center: np.ndarray  # (3,)
    scale: float
    method: str = "coordinatewise_median_center_p95_minus_p5_max_span_v1"

    def to_dict(self) -> dict:
        return {"center": [float(x) for x in self.center], "scale": float(self.scale), "method": self.method}


def normalize_geometry(vertices: np.ndarray) -> tuple[np.ndarray, Normalization]:
    """X' = (X - c) / s with c = coordinatewise median, s = max(p95 - p5)."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3 or not np.isfinite(v).all():
        raise GeometryError("vertices must be finite Nx3")
    center = np.median(v, axis=0)
    span = np.percentile(v, 95, axis=0) - np.percentile(v, 5, axis=0)
    scale = float(np.max(span))
    if not math.isfinite(scale) or scale <= 0:
        raise GeometryError("normalization scale must be positive and finite")
    out = ((v - center) / scale).astype(np.float32)
    return out, Normalization(center, scale)


def normalize_pose(pose: np.ndarray, norm: Normalization) -> np.ndarray:
    """Pose acting on normalized points with identical projections: t' = (R c + t) / s."""
    p = np.asarray(pose, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = p[:3, :3]
    out[:3, 3] = (p[:3, :3] @ norm.center + p[:3, 3]) / norm.scale
    return out


def normalize_points(points: np.ndarray, norm: Normalization) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - norm.center) / norm.scale


# --------------------------------------------------------------------------
# projection

def check_K(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise GeometryError("K must be a finite 3x3 pinhole matrix")
    if K[1, 0] != 0 or K[0, 1] != 0 or K[2, 0] != 0 or K[2, 1] != 0 or K[2, 2] != 1:
        raise GeometryError("K must be a zero-skew pinhole matrix with bottom row [0,0,1]")
    return K


def check_pose(pose: np.ndarray) -> np.ndarray:
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (4, 4) or not np.isfinite(p).all():
        raise GeometryError("pose must be a finite 4x4 matrix")
    if not np.array_equal(p[3], [0.0, 0.0, 0.0, 1.0]):
        raise GeometryError("pose bottom row must be exactly [0,0,0,1]")
    R = p[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or abs(np.linalg.det(R) - 1) > 1e-5:
        raise GeometryError("pose rotation must be proper orthonormal")
    return p


def to_camera(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    p = check_pose(pose)
    return np.asarray(points, dtype=np.float64) @ p[:3, :3].T + p[:3, 3]


def project(points: np.ndarray, K: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project object points; returns (uv Nx2 in integer-centre pixels, z N)."""
    K = check_K(K)
    cam = to_camera(points, pose)
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    uv[z <= 0] = np.nan
    return uv, z


# --------------------------------------------------------------------------
# rasterization

@dataclass
class RasterResult:
    depth: np.ndarray        # HxW float64, +inf where empty
    face_id: np.ndarray      # HxW int32, -1 where empty
    barycentric: np.ndarray  # HxWx3 float64 (original-triangle, perspective-correct)
    stats: dict = field(default_factory=dict)

    @property
    def coverage(self) -> np.ndarray:
        return self.face_id >= 0


def _clip_near(cam: np.ndarray, near: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sutherland-Hodgman clip of one triangle against z >= near.

    Returns polygon vertices as (camera point, original barycentric) pairs.
    """
    poly = [(cam[i], np.eye(3)[i]) for i in range(3)]
    out = []
    for k in range(3):
        p, bp = poly[k]
        q, bq = poly[(k + 1) % 3]
        pin, qin = p[2] >= near, q[2] >= near
        if pin:
            out.append((p, bp))
        if pin != qin:
            t = (near - p[2]) / (q[2] - p[2])
            out.append((p + t * (q - p), bp + t * (bq - bp)))
    return out


def rasterize(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, pose: np.ndarray,
              width: int, height: int, *, near: float = 1e-3,
              roi: tuple[int, int, int, int] | None = None) -> RasterResult:
    """Rasterize; ``roi=(x0, y0, x1, y1)`` (inclusive pixel bounds) restricts work
    to a window without changing results inside it."""
    K = check_K(K)
    if near <= 0 or not math.isfinite(near):
        raise GeometryError("near plane must be positive")
    if K[0, 1] != 0:
        raise GeometryError("rasterizer requires zero skew")
    cam = to_camera(vertices, pose)
    f = np.asarray(faces, dtype=np.int64)
    depth = np.full((height, width), np.inf)
    face_id = np.full((height, width), -1, dtype=np.int32)
    bary = np.zeros((height, width, 3))
    stats = {"faces": int(len(f)), "clipped": 0, "culled_behind": 0, "guard_band_skipped": 0,
             "degenerate_screen": 0, "culled_outside": 0}
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rx0, ry0, rx1, ry1 = (0, 0, width - 1, height - 1) if roi is None else (
        max(0, int(roi[0])), max(0, int(roi[1])), min(width - 1, int(roi[2])), min(height - 1, int(roi[3])))
    if rx0 > rx1 or ry0 > ry1:
        return RasterResult(depth, face_id, bary, stats)
    # Vectorized conservative cull of faces fully in front whose screen bbox misses the window.
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = fx * cam[:, 0] / z + cx
        sy = fy * cam[:, 1] / z + cy
    front = (z[f] >= near).all(axis=1)
    fsx, fsy = sx[f], sy[f]
    outside = front & ((fsx.max(axis=1) < rx0 - 1) | (fsx.min(axis=1) > rx1 + 1) |
                       (fsy.max(axis=1) < ry0 - 1) | (fsy.min(axis=1) > ry1 + 1))
    stats["culled_outside"] = int(outside.sum())
    window = (rx0, ry0, rx1, ry1)
    for fid in np.flatnonzero(~outside):
        tri = cam[f[fid]]
        if np.all(tri[:, 2] < near):
            stats["culled_behind"] += 1
            continue
        if np.all(tri[:, 2] >= near):
            poly = [(tri[i], np.eye(3)[i]) for i in range(3)]
        else:
            poly = _clip_near(tri, near)
            stats["clipped"] += 1
            if len(poly) < 3:
                continue
        for k in range(1, len(poly) - 1):
            _raster_triangle(int(fid), (poly[0], poly[k], poly[k + 1]), fx, fy, cx, cy,
                             width, height, depth, face_id, bary, stats, window)
    return RasterResult(depth, face_id, bary, stats)


def _raster_triangle(fid, verts, fx, fy, cx, cy, width, height, depth, face_id, bary, stats, window=None):
    P = np.array([v[0] for v in verts])
    B = np.array([v[1] for v in verts])
    z = P[:, 2]
    sx = fx * P[:, 0] / z + cx
    sy = fy * P[:, 1] / z + cy
    if np.any(np.abs(sx) > GUARD_BAND_PX) or np.any(np.abs(sy) > GUARD_BAND_PX):
        stats["guard_band_skipped"] += 1
        return
    X = np.round(sx * SUBPIXEL).astype(np.int64)
    Y = np.round(sy * SUBPIXEL).astype(np.int64)
    area = (X[1] - X[0]) * (Y[2] - Y[0]) - (Y[1] - Y[0]) * (X[2] - X[0])
    if area == 0:
        stats["degenerate_screen"] += 1
        return
    if area < 0:
        order = [0, 2, 1]
        X, Y, z, B = X[order], Y[order], z[order], B[order]
        area = -area
    wx0, wy0, wx1, wy1 = window if window is not None else (0, 0, width - 1, height - 1)
    x0 = max(int(math.ceil(X.min() / SUBPIXEL)), wx0)
    x1 = min(int(math.floor(X.max() / SUBPIXEL)), wx1)
    y0 = max(int(math.ceil(Y.min() / SUBPIXEL)), wy0)
    y1 = min(int(math.floor(Y.max() / SUBPIXEL)), wy1)
    if x0 > x1 or y0 > y1:
        return
    px = (np.arange(x0, x1 + 1, dtype=np.int64) * SUBPIXEL)[None, :]
    py = (np.arange(y0, y1 + 1, dtype=np.int64) * SUBPIXEL)[:, None]
    weights = []
    inside = np.ones((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    for a, b in ((1, 2), (2, 0), (0, 1)):  # edge opposite vertex (3 - a - b)
        dx, dy = X[b] - X[a], Y[b] - Y[a]
        e = dx * (py - Y[a]) - dy * (px - X[a])
        owned = dy > 0 or (dy == 0 and dx < 0)
        inside &= (e > 0) | ((e == 0) if owned else False)
        weights.append(e)
    if not inside.any():
        return
    lam = np.stack(weights, axis=-1).astype(np.float64) / float(area)  # lam[...,i] for vertex i
    w = lam / z  # perspective weights
    inv_z = w.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = 1.0 / inv_z
        orig = (w[..., :, None] * B[None, None, :, :]).sum(axis=-2) / inv_z[..., None]
    region_depth = depth[y0:y1 + 1, x0:x1 + 1]
    closer = inside & (zz > 0) & (zz < region_depth)
    if not closer.any():
        return
    region_depth[closer] = zz[closer]
    face_id[y0:y1 + 1, x0:x1 + 1][closer] = fid
    bary[y0:y1 + 1, x0:x1 + 1][closer] = orig[closer]


def surface_points(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray,
                   barycentric: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    tri = v[np.asarray(faces)[face_ids]]  # (...,3,3)
    return (np.asarray(barycentric)[..., :, None] * tri).sum(axis=-2)


# --------------------------------------------------------------------------
# anchoring

def ray_mesh(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, pose: np.ndarray,
             u: float, v: float, *, near: float = 1e-6) -> dict | None:
    """Nearest positive-depth hit of the pixel ray (Moller-Trumbore)."""
    K = check_K(K)
    cam = to_camera(vertices, pose)
    f = np.asarray(faces, dtype=np.int64)
    d = np.linalg.solve(K, np.array([u, v, 1.0]))
    a, b, c = cam[f[:, 0]], cam[f[:, 1]], cam[f[:, 2]]
    e1, e2 = b - a, c - a
    pvec = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-15
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = -a
    b1 = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    b2 = (qvec @ d) * inv
    t = np.einsum("ij,ij->i", e2, qvec) * inv  # depth along d (d has z = 1)
    hit = ok & (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1) & (t > near)
    if not hit.any():
        return None
    idx = np.flatnonzero(hit)
    best = idx[np.lexsort((idx, t[idx]))[0]]  # nearest, ties -> lowest face id
    bc = np.array([1 - b1[best] - b2[best], b1[best], b2[best]])
    bc = np.clip(bc, 0, None)
    bc /= bc.sum()
    point = (bc[:, None] * np.asarray(vertices, dtype=np.float64)[f[best]]).sum(axis=0)
    return {"face_id": int(best), "barycentric": bc, "point": point, "depth": float(t[best])}


def anchor_from_pixel(vertices, faces, K, pose, u: float, v: float, *, width: int, height: int,
                      observed_mask: np.ndarray | None = None) -> dict:
    """Anchor a user tap (source pixels) on the frozen surface."""
    if not (-0.5 <= u < width - 0.5 and -0.5 <= v < height - 0.5):
        raise GeometryError("anchor pixel is outside the frame")
    if observed_mask is not None:
        iu, iv = int(round(u)), int(round(v))
        if not np.asarray(observed_mask)[iv, iu]:
            raise GeometryError("anchor pixel is outside the observed object mask")
    hit = ray_mesh(vertices, faces, K, pose, u, v)
    if hit is None:
        raise GeometryError("anchor ray does not hit the surface")
    return hit


def tangent_chart(vertices, faces, face_id: int, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal right-handed (t, b, n) at a face, n facing the anchor camera,
    t aligned with the camera x axis projected on the tangent plane."""
    n = face_normals(vertices, np.asarray(faces)[[face_id]])[0]
    p = check_pose(pose)
    R = p[:3, :3]
    centroid = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)[face_id]].mean(axis=0)
    cam_center = -R.T @ p[:3, 3]
    if np.dot(n, cam_center - centroid) < 0:
        n = -n
    x_axis = R.T @ np.array([1.0, 0.0, 0.0])
    t = x_axis - np.dot(x_axis, n) * n
    if np.linalg.norm(t) < 1e-6:
        alt = R.T @ np.array([0.0, 1.0, 0.0])
        t = alt - np.dot(alt, n) * n
    t /= np.linalg.norm(t)
    b = np.cross(n, t)
    return t, b, n


def make_marker(vertices, faces, hit: dict, pose: np.ndarray, size: float, artwork_sha256: str) -> dict:
    t, b, n = tangent_chart(vertices, faces, hit["face_id"], pose)
    return {"face_id": int(hit["face_id"]), "barycentric": [float(x) for x in hit["barycentric"]],
            "tangent": t.tolist(), "bitangent": b.tolist(), "normal": n.tolist(),
            "size_object_units": float(size), "artwork_sha256": artwork_sha256}


# --------------------------------------------------------------------------
# artwork

def builtin_artwork(kind: str, edge: int = 256) -> np.ndarray:
    """Fixed checkerboard or directional (asymmetric) test artwork, RGB uint8."""
    if not 8 <= edge <= 1024:
        raise GeometryError("artwork edge must be within 8..1024")
    yy, xx = np.mgrid[0:edge, 0:edge]
    if kind == "checkerboard":
        cell = max(edge // 8, 1)
        on = ((xx // cell) + (yy // cell)) % 2 == 0
        img = np.where(on[..., None], np.array([240, 240, 240]), np.array([30, 30, 30]))
        img[: max(edge // 16, 1), :] = (220, 40, 40)  # top border marks orientation
        return img.astype(np.uint8)
    if kind == "directional":
        img = np.zeros((edge, edge, 3), dtype=np.uint8)
        img[:] = (40, 40, 160)
        cx = edge / 2
        shaft = (np.abs(xx - cx) < edge * 0.08) & (yy > edge * 0.35) & (yy < edge * 0.9)
        head = (yy >= edge * 0.1) & (yy <= edge * 0.4) & (np.abs(xx - cx) <= (yy - edge * 0.1) * 0.9)
        img[shaft | head] = (250, 220, 40)
        img[:, : max(edge // 12, 1)] = (40, 200, 60)  # left bar breaks mirror symmetry
        return img
    raise GeometryError("artwork kind must be checkerboard or directional")


def check_artwork(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise GeometryError("artwork must be RGB uint8")
    if max(a.shape[:2]) > 1024 or min(a.shape[:2]) < 8:
        raise GeometryError("artwork edge must be within 8..1024 pixels")
    return a


# --------------------------------------------------------------------------
# marker rendering on the curved mesh

def _chart_overlap_cells(chart_uv: np.ndarray, faces_local: np.ndarray, half: float, grid: int) -> np.ndarray:
    """Count chart coverage per cell over [-half, half]^2; >1 means overlap."""
    counts = np.zeros((grid, grid), dtype=np.int32)
    scale = grid / (2 * half)
    for tri in faces_local:
        pts = (chart_uv[tri] + half) * scale - 0.5  # cell centres at integers
        X = np.round(pts[:, 0] * SUBPIXEL).astype(np.int64)
        Y = np.round(pts[:, 1] * SUBPIXEL).astype(np.int64)
        area = (X[1] - X[0]) * (Y[2] - Y[0]) - (Y[1] - Y[0]) * (X[2] - X[0])
        if area == 0:
            continue
        if area < 0:
            X, Y = X[[0, 2, 1]], Y[[0, 2, 1]]
        x0 = max(int(math.ceil(X.min() / SUBPIXEL)), 0)
        x1 = min(int(math.floor(X.max() / SUBPIXEL)), grid - 1)
        y0 = max(int(math.ceil(Y.min() / SUBPIXEL)), 0)
        y1 = min(int(math.floor(Y.max() / SUBPIXEL)), grid - 1)
        if x0 > x1 or y0 > y1:
            continue
        px = (np.arange(x0, x1 + 1, dtype=np.int64) * SUBPIXEL)[None, :]
        py = (np.arange(y0, y1 + 1, dtype=np.int64) * SUBPIXEL)[:, None]
        inside = np.ones((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
        for a, b in ((1, 2), (2, 0), (0, 1)):
            dx, dy = X[b] - X[a], Y[b] - Y[a]
            e = dx * (py - Y[a]) - dy * (px - X[a])
            owned = dy > 0 or (dy == 0 and dx < 0)
            inside &= (e > 0) | ((e == 0) if owned else False)
        counts[y0:y1 + 1, x0:x1 + 1] += inside
    return counts


def marker_patch(vertices, faces, face_support: np.ndarray, marker: Mapping[str, Any], *,
                 min_support: int = 3, max_normal_angle_deg: float = 75.0, overlap_grid: int = 128) -> dict:
    """Connected supported faces around the anchor with valid (unfolded) chart coordinates."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    support = np.asarray(face_support)
    fid0 = int(marker["face_id"])
    t, b, n = (np.asarray(marker[k], dtype=np.float64) for k in ("tangent", "bitangent", "normal"))
    p = (np.asarray(marker["barycentric"])[:, None] * v[f[fid0]]).sum(axis=0)
    half = float(marker["size_object_units"]) / 2
    if support[fid0] < min_support:
        return {"faces": np.zeros(0, dtype=np.int64), "reason": "anchor_face_unsupported",
                "anchor_point": p, "half": half}
    normals = face_normals(v, f)
    # Orient patch normals with the marker normal sign of the anchor face.
    sign = 1.0 if np.dot(normals[fid0], n) >= 0 else -1.0
    cos_lim = math.cos(math.radians(max_normal_angle_deg))
    adjacency = edge_adjacency(f)
    reach = half * math.sqrt(2) * 1.5
    chosen = {fid0}
    queue = deque([fid0])
    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if j in chosen or support[j] < min_support:
                continue
            if sign * np.dot(normals[j], n) < cos_lim:
                continue
            rel = v[f[j]] - p
            chart = np.stack([rel @ t, rel @ b], axis=1)
            if np.min(np.max(np.abs(chart), axis=1)) > reach:
                continue
            chosen.add(j)
            queue.append(j)
    patch = np.array(sorted(chosen), dtype=np.int64)
    rel = v - p
    chart_all = np.stack([rel @ t, rel @ b], axis=1)
    # Folded faces flip orientation in chart space relative to the anchor face.
    def chart_area(face):
        a, bb, c = chart_all[face]
        return (bb[0] - a[0]) * (c[1] - a[1]) - (bb[1] - a[1]) * (c[0] - a[0])
    ref = chart_area(f[fid0])
    areas = np.array([chart_area(f[i]) for i in patch])
    keep = np.sign(areas) == np.sign(ref)
    keep &= np.abs(areas) > 1e-12 * max(half * half, 1e-300)
    folded = patch[~keep]
    patch = patch[keep]
    counts = _chart_overlap_cells(chart_all, f[patch], half, overlap_grid) if len(patch) else None
    return {"faces": patch, "folded_faces": folded, "anchor_point": p, "half": half,
            "chart_t": t, "chart_b": b, "overlap_counts": counts, "overlap_grid": overlap_grid,
            "reason": None}


def render_marker(vertices, faces, face_support, marker: Mapping[str, Any], artwork: np.ndarray,
                  K, pose, width: int, height: int, observed_mask: np.ndarray, *,
                  raster: RasterResult | None = None, patch: dict | None = None,
                  min_support: int = 3) -> dict:
    """Render the anchored artwork over the actual curved surface.

    Pixels are kept only when the depth-tested visible face belongs to the
    supported unfolded patch, the chart cell is not overlapped, the face is
    front-facing, and the *observed* mask covers the pixel.
    """
    if observed_mask is None:
        raise GeometryError("an observed mask is required; projected silhouettes are not masks")
    mask = np.asarray(observed_mask) > 0
    if mask.shape != (height, width):
        raise GeometryError("observed mask dimensions differ from the frame")
    art = check_artwork(artwork)
    if patch is None:
        patch = marker_patch(vertices, faces, face_support, marker, min_support=min_support)
    if raster is None and len(patch["faces"]):
        roi = None
        pv = np.unique(np.asarray(faces)[patch["faces"]])
        uv, z = project(np.asarray(vertices, dtype=np.float64)[pv], K, pose)
        if len(pv) and np.all(z > 0) and np.isfinite(uv).all():
            roi = (int(math.floor(uv[:, 0].min())) - 1, int(math.floor(uv[:, 1].min())) - 1,
                   int(math.ceil(uv[:, 0].max())) + 1, int(math.ceil(uv[:, 1].max())) + 1)
        # The window covers every patch pixel; occluders inside it are still depth-tested.
        raster = rasterize(vertices, faces, K, pose, width, height, roi=roi)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    stats = {"rendered_pixels": 0, "mask_rejected_pixels": 0, "overlap_rejected_pixels": 0,
             "backfacing_rejected_pixels": 0, "patch_faces": int(len(patch["faces"])),
             "folded_faces": int(len(patch.get("folded_faces", ()))), "reason": patch.get("reason")}
    if not len(patch["faces"]):
        return {"rgba": rgba, "stats": stats}
    in_patch = np.isin(raster.face_id, patch["faces"])
    ys, xs = np.nonzero(in_patch)
    fids = raster.face_id[ys, xs]
    pts = surface_points(vertices, faces, fids, raster.barycentric[ys, xs])
    rel = pts - patch["anchor_point"]
    s = rel @ patch["chart_t"]
    r = rel @ patch["chart_b"]
    half = patch["half"]
    inside = (np.abs(s) <= half) & (np.abs(r) <= half)
    grid = patch["overlap_grid"]
    ci = np.clip(((s + half) / (2 * half) * grid).astype(int), 0, grid - 1)
    cj = np.clip(((r + half) / (2 * half) * grid).astype(int), 0, grid - 1)
    overlap = patch["overlap_counts"][cj, ci] > 1
    p = check_pose(pose)
    normals = face_normals(vertices, np.asarray(faces)[fids]) @ p[:3, :3].T
    cam_pts = pts @ p[:3, :3].T + p[:3, 3]
    sign = 1.0 if np.dot(face_normals(vertices, np.asarray(faces)[[int(marker["face_id"])]])[0],
                         np.asarray(marker["normal"])) >= 0 else -1.0
    front = np.einsum("ij,ij->i", sign * normals, cam_pts) < 0
    in_mask = mask[ys, xs]
    ok = inside & ~overlap & front & in_mask
    stats["mask_rejected_pixels"] = int(np.sum(inside & ~overlap & front & ~in_mask))
    stats["overlap_rejected_pixels"] = int(np.sum(inside & overlap))
    stats["backfacing_rejected_pixels"] = int(np.sum(inside & ~overlap & ~front))
    edge_h, edge_w = art.shape[:2]
    col = np.clip(((s / (2 * half)) + 0.5) * edge_w, 0, edge_w - 1).astype(int)
    row = np.clip((0.5 - (r / (2 * half))) * edge_h, 0, edge_h - 1).astype(int)
    rgba[ys[ok], xs[ok], :3] = art[row[ok], col[ok]]
    rgba[ys[ok], xs[ok], 3] = 255
    stats["rendered_pixels"] = int(ok.sum())
    return {"rgba": rgba, "stats": stats}


# --------------------------------------------------------------------------
# descriptor bank from registered prefix views

def build_descriptor_bank(views: list[dict], vertices, faces, *, max_per_view: int = 600,
                          depth_tolerance: float = 0.02) -> dict:
    """ORB descriptors on observed masks lifted to frozen 3D surface points.

    ``views`` items: {frame_id, gray (HxW uint8), mask (HxW bool), K, pose}.
    The pose acts on the (normalized) mesh vertices supplied here.
    """
    from .vision import cv2
    orb = cv2.ORB_create(nfeatures=max_per_view)
    points, descs, owners = [], [], []
    f = np.asarray(faces, dtype=np.int64)
    v64 = np.asarray(vertices, dtype=np.float64)
    for view in views:
        gray = view["gray"]
        h, w = gray.shape
        mask = (np.asarray(view["mask"]) > 0).astype(np.uint8) * 255
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        kps, des = orb.detectAndCompute(gray, mask)
        if des is None or not kps:
            continue
        ras = rasterize(vertices, faces, view["K"], view["pose"], w, h)
        K = check_K(view["K"])
        pose = check_pose(view["pose"])
        for kp, d in zip(kps, des):
            u, vv = kp.pt
            iu, iv = int(round(u)), int(round(vv))
            if not (0 <= iu < w and 0 <= iv < h):
                continue
            fid = int(ras.face_id[iv, iu])
            if fid < 0:
                continue
            tri_cam = to_camera(v64[f[fid]], pose)
            ray = np.linalg.solve(K, np.array([u, vv, 1.0]))
            nrm = np.cross(tri_cam[1] - tri_cam[0], tri_cam[2] - tri_cam[0])
            denom = float(nrm @ ray)
            if abs(denom) < 1e-15:
                continue
            depth = float(nrm @ tri_cam[0]) / denom
            if depth <= 0 or abs(depth - ras.depth[iv, iu]) > depth_tolerance * ras.depth[iv, iu] + 1e-9:
                continue
            cam_pt = ray * depth
            obj_pt = pose[:3, :3].T @ (cam_pt - pose[:3, 3])
            points.append(obj_pt)
            descs.append(d)
            owners.append(view["frame_id"])
    if not points:
        return {"points": np.zeros((0, 3), np.float32), "descriptors": np.zeros((0, 32), np.uint8), "owners": []}
    return {"points": np.asarray(points, dtype=np.float32), "descriptors": np.asarray(descs, dtype=np.uint8),
            "owners": owners}


# --------------------------------------------------------------------------
# relative tracker

@dataclass
class TrackOutput:
    frame_id: str
    mask_state: str
    pose_state: str
    pose: np.ndarray | None
    failure_reason: str | None
    metrics: dict


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


class RelativeTracker:
    """Masked ORB + forward/backward LK correspondences, absolute PnP each frame.

    Public tracking after initialization or any loss needs two consecutive
    checked, mutually consistent PnP solutions; the first recovering pose is
    withheld. A missing/empty observed mask suppresses pose immediately.
    """

    def __init__(self, *, min_inliers: int = 15, reprojection_px: float = 4.0, ratio: float = 0.8,
                 fb_px: float = 1.0, min_inlier_ratio: float = 0.25, consistency_deg: float = 15.0,
                 consistency_translation: float = 0.25, orb_features: int = 1500):
        self.cfg = dict(min_inliers=min_inliers, reprojection_px=reprojection_px, ratio=ratio, fb_px=fb_px,
                        min_inlier_ratio=min_inlier_ratio, consistency_deg=consistency_deg,
                        consistency_translation=consistency_translation, orb_features=orb_features)
        self._package_id = None
        self._closed = False
        self._points = None
        self._desc = None
        self.reset()

    # lifecycle -----------------------------------------------------------
    def initialize(self, package: Mapping[str, Any]) -> None:
        """``package``: {package_id, frozen, status, points (Nx3), descriptors (Nx32 uint8)}."""
        self._require_open()
        if package.get("status") != "available" or package.get("frozen") is not True:
            raise GeometryError("tracker requires an available frozen surface package")
        pts = np.asarray(package["points"], dtype=np.float64)
        des = np.asarray(package["descriptors"])
        if pts.ndim != 2 or pts.shape[1] != 3 or des.dtype != np.uint8 or des.shape != (len(pts), 32):
            raise GeometryError("descriptor bank must be Nx3 points with Nx32 uint8 ORB descriptors")
        if len(pts) < self.cfg["min_inliers"] or not np.isfinite(pts).all():
            raise GeometryError("descriptor bank is too small or non-finite")
        self._points, self._desc = pts, des
        self._package_id = str(package["package_id"])
        self.reset()

    def reset(self) -> None:
        self._state = "lost"
        self._candidate = None
        self._prev_gray = None
        self._prev_obs = None  # (Nx2 float32 points, N int ids)
        self._last_index = None
        self._last_image_sha = None

    def close(self) -> None:
        self._closed = True
        self._points = self._desc = None
        self.reset()

    def _require_open(self):
        if self._closed:
            raise GeometryError("tracker is closed")

    # processing ----------------------------------------------------------
    def process(self, rgb: np.ndarray, observed_mask: np.ndarray | None, frame_identity: Mapping[str, Any],
                K: np.ndarray) -> TrackOutput:
        self._require_open()
        if self._points is None:
            raise GeometryError("tracker is not initialized")
        from .vision import cv2
        fid = str(frame_identity["frame_id"])
        index = int(frame_identity["source_frame_index"])
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise GeometryError("rgb must be HxWx3 uint8")
        h, w = image.shape[:2]
        K = check_K(K)
        if self._last_index is not None and index <= self._last_index:
            return self._fail(fid, "stale_image: source_frame_index did not advance", "rejected")
        if frame_identity.get("image_sha256") != array_sha256(image):
            return self._fail(fid, "image_identity_mismatch: frame bytes differ from declared identity", "rejected")
        # Freshness is by advancing source index plus exact identity; identical
        # pixels in two different frames (a static scene) are legitimate.
        self._last_index = index
        if observed_mask is None:
            return self._fail(fid, "observed_mask_unavailable", "unavailable")
        mask = np.asarray(observed_mask) > 0
        if mask.shape != (h, w):
            return self._fail(fid, "mask_dimension_mismatch", "rejected")
        if frame_identity.get("mask_frame_id") != fid:
            return self._fail(fid, "stale_mask: mask belongs to another frame", "rejected")
        if frame_identity.get("mask_sha256") != mask_sha256(mask):
            return self._fail(fid, "mask_identity_mismatch", "rejected")
        if not mask.any():
            return self._fail(fid, "observed_mask_empty", "observed")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        obs_pts, obs_ids, lk_count = self._lk(gray, mask, cv2)
        orb_pts, orb_ids, orb_count = self._orb(gray, mask, cv2)
        pts, ids = _merge(obs_pts, obs_ids, orb_pts, orb_ids)
        metrics = {"lk_correspondences": float(lk_count), "orb_correspondences": float(orb_count),
                   "correspondences": float(len(ids)), "inliers": None, "reprojection_median_px": None}
        self._prev_gray = gray
        self._prev_obs = None
        if len(ids) < self.cfg["min_inliers"]:
            return self._fail(fid, "insufficient_correspondences", "observed", metrics, keep_gray=True)
        solved = self._pnp(pts, ids, K, mask, cv2)
        if solved["pose"] is None:
            metrics.update(solved["metrics"])
            return self._fail(fid, solved["reason"], "observed", metrics, keep_gray=True)
        pose = solved["pose"]
        metrics.update(solved["metrics"])
        self._prev_obs = (pts[solved["inliers"]].astype(np.float32), ids[solved["inliers"]])
        if self._state == "tracking":
            return TrackOutput(fid, "observed", "tracking", pose, None, metrics)
        if self._state == "recovering" and self._candidate is not None and self._consistent(self._candidate, pose):
            self._state = "tracking"
            self._candidate = None
            return TrackOutput(fid, "observed", "tracking", pose, None, metrics)
        self._state = "recovering"
        self._candidate = pose
        return TrackOutput(fid, "observed", "recovering", None,
                           "recovering: first checked pose after loss is withheld", metrics)

    def _fail(self, fid, reason, mask_state, metrics=None, keep_gray=False) -> TrackOutput:
        self._state = "lost"
        self._candidate = None
        self._prev_obs = None
        if not keep_gray:
            self._prev_gray = None
        pose_state = "unavailable" if mask_state in ("unavailable", "rejected") else "lost"
        return TrackOutput(fid, mask_state, pose_state, None, reason, dict(metrics or {}))

    def _lk(self, gray, mask, cv2):
        if self._prev_gray is None or self._prev_obs is None or not len(self._prev_obs[1]):
            return np.zeros((0, 2), np.float32), np.zeros(0, np.int64), 0
        if self._prev_gray.shape != gray.shape:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.int64), 0
        p0 = self._prev_obs[0].reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
        if p1 is None:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.int64), 0
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb <= self.cfg["fb_px"])
        pts = p1.reshape(-1, 2)
        h, w = gray.shape
        iu = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        iv = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        inb = (pts[:, 0] >= -0.5) & (pts[:, 0] < w - 0.5) & (pts[:, 1] >= -0.5) & (pts[:, 1] < h - 0.5)
        good &= inb & mask[iv, iu]
        return pts[good].astype(np.float32), self._prev_obs[1][good], int(good.sum())

    def _orb(self, gray, mask, cv2):
        orb = cv2.ORB_create(nfeatures=self.cfg["orb_features"])
        kps, des = orb.detectAndCompute(gray, mask.astype(np.uint8) * 255)
        if des is None or len(kps) < 2:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.int64), 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(des, self._desc, k=2)
        best: dict[int, tuple[float, tuple[float, float]]] = {}
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.cfg["ratio"] * n.distance:
                continue
            prev = best.get(m.trainIdx)
            if prev is None or m.distance < prev[0]:
                best[m.trainIdx] = (m.distance, kps[m.queryIdx].pt)
        if not best:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.int64), 0
        ids = np.array(sorted(best), dtype=np.int64)
        pts = np.array([best[i][1] for i in ids], dtype=np.float32)
        return pts, ids, len(ids)

    def _pnp(self, pts, ids, K, mask, cv2) -> dict:
        obj = self._points[ids].astype(np.float64)
        img = pts.astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=300,
                                                 reprojectionError=self.cfg["reprojection_px"],
                                                 confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None:
            return {"pose": None, "reason": "pnp_failed", "metrics": {}}
        inl = inl.ravel()
        if len(inl) < self.cfg["min_inliers"]:
            return {"pose": None, "reason": "insufficient_inliers", "metrics": {"inliers": float(len(inl))}}
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.ravel()
        if not np.isfinite(pose).all():
            return {"pose": None, "reason": "pnp_nonfinite", "metrics": {}}
        uv, z = project(obj, K, pose)
        err = np.linalg.norm(uv - img, axis=1)
        inliers = np.flatnonzero(np.isfinite(err) & (err <= self.cfg["reprojection_px"]) & (z > 0))
        metrics = {"inliers": float(len(inliers)),
                   "reprojection_median_px": float(np.median(err[inliers])) if len(inliers) else None}
        if len(inliers) < self.cfg["min_inliers"]:
            return {"pose": None, "reason": "insufficient_inliers_after_refinement", "metrics": metrics}
        if len(inliers) / len(ids) < self.cfg["min_inlier_ratio"]:
            return {"pose": None, "reason": "low_inlier_ratio", "metrics": metrics}
        if metrics["reprojection_median_px"] > self.cfg["reprojection_px"] / 2:
            return {"pose": None, "reason": "high_reprojection_error", "metrics": metrics}
        spread = np.cov(img[inliers].T)
        if np.min(np.linalg.eigvalsh(spread)) < 4.0:
            return {"pose": None, "reason": "degenerate_inlier_layout", "metrics": metrics}
        return {"pose": pose, "inliers": inliers, "reason": None, "metrics": metrics}

    def _consistent(self, a: np.ndarray, b: np.ndarray) -> bool:
        ang = rotation_angle_deg(a[:3, :3], b[:3, :3])
        dt = np.linalg.norm(a[:3, 3] - b[:3, 3]) / max(np.linalg.norm(a[:3, 3]), 1e-9)
        return ang <= self.cfg["consistency_deg"] and dt <= self.cfg["consistency_translation"]


def _merge(p1, i1, p2, i2):
    """LK observations keep their original 3D ids; ORB adds only new ids."""
    if not len(i1):
        return p2, i2
    if not len(i2):
        return p1, i1
    known = set(int(i) for i in i1)
    keep = np.array([int(i) not in known for i in i2], dtype=bool)
    p2, i2 = p2[keep], i2[keep]
    if len(i2):
        d = np.linalg.norm(p2[:, None, :] - p1[None, :, :], axis=2).min(axis=1)
        far = d > 0.5
        p2, i2 = p2[far], i2[far]
    return np.concatenate([p1, p2]).astype(np.float32), np.concatenate([i1, i2]).astype(np.int64)
