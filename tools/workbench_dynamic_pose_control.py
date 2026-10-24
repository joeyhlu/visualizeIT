"""Synthetic dynamic pose control through the actual WorkbenchIntegerRefiner (bounded, two modes).

Question: does an exact geometric flow recover nonzero pose changes through the actual production
``WorkbenchIntegerRefiner.trace`` (crop, template render, correspondence gates, RANSAC/LM PnP and
crop inverse)? And does the unchanged new head (27e06, base f7d1) stay stable on the same synthetic
input?

Synthetic truth is exact by construction. This makes no claim about physical onboarding
alignment, real appearance, segmentation or initialisation. Labels: ``human_accuracy=false``,
``phone_ready=false``, ``development_only=true``, ``automatic_rgb_preparation=false``. No scored
inputs, ``evaluator.json``, dataset depth or masks are read, and no reference error is fitted.

The authenticated source seeds are slightly non-orthogonal (about 1e-6). For this *synthetic*
experiment only, the target is the deterministic SVD SO(3) projection of each seed (see
``synthetic_target_pose``). The raw seed, the projection delta and the orthogonality are
recorded. The source bundles are unchanged, and no physical pose correction is claimed.

``prepare`` (CPU; no torch/model import) writes one fresh ``plan.json``.
``run`` (parent-scheduled; requires ``--expected-plan-sha256``; one GPU stage) first rebuilds the
whole plan from the authenticated inputs and requires exact equality. It then runs the ORACLE
branch (diagnostic exact-flow network, strict per-iteration gate, all objects). Only if every
oracle case and every cleanup succeeds does it run the LEARNED branch (unchanged installed 27e06
head; descriptive metrics only).

Oracle construction (independent NumPy, millimetres; no production lifting/projection helper)::

    A = C @ P0_mm,  B = C @ P*_mm
    Xc = d * inv(Kc) @ [u, v, 1]
    Xo = R_A^T (Xc - t_A)
    q = proj(Kc, R_B Xo + t_B)
    flow = q - [u, v]

C is the captured ``crop_from_orig``, P0 is the current native pose and ``d`` is the captured
template z-depth in mm at integer crop pixel centres (x = column, y = row).
``flow[row, col] = (dx, dy)``, from the template (source) to the query crop.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_PLAN = "workbench_dynamic_pose_control_plan_v2"
SCHEMA_RESULT = "workbench_dynamic_pose_control_result_v2"
SCHEMA_FAILURE = "workbench_dynamic_pose_control_failure_v2"
TOOL_VERSION = "2"

UPSTREAM_ROOT_REL = ".cache/model-quality/sources/gotrack"
PRODUCTION_PINS = {
    "bench/workbench_neural.py": "86d0caf6d3f32607d28ddb9cfc7c9b063e6767ded8710fa5d8aa0384968d6078",
    "bench/quality_gotrack.py": "0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29",
    "bench/workbench_render.py": "da9d79157289d093399cc055ab98e8c8534f3aa2f0cbc52ac288c327d2fdf57a",
    "tools/workbench_ycb_execution_controls.reviewed.json": "c5ae3b009007ac1600af345c56f1b719e8881591baf37187a917d361eb99ca20",
    "bench/quality_gotrack_inference.py": "d1cd703ffe389ea72947d86a37ddbd42a065323af35c9e73547362027d3486b3",
    "bench/quality_adaptation_export_parity.py": "910370f26131e31055a8b9fa7952670dee026f71408b5e9a0fcb429f669197e3",
    "bench/runtime/gotrack-flow-terminal-20261021.json": "05fe3012a18c2ed2669bcf2690e0b48650d9472fc1bffdf0980b1a253e6fc770",
    f"{UPSTREAM_ROOT_REL}/utils/crop_generation.py": "3d38667f4ce732b66fe472937b83bb31c1962d494db3c7ff13748755e9774342",
    f"{UPSTREAM_ROOT_REL}/utils/poser_util.py": "0a8648917b713c86d8ba1343fd350deef33be7c43bf1517963560a809caf2b7f",
    f"{UPSTREAM_ROOT_REL}/utils/im_util.py": "97610c926bdf50f045ad0ff11bc05aa33e2f7acd497ee0f9369b92ca6af87e95",
    f"{UPSTREAM_ROOT_REL}/utils/transform3d.py": "cbb0d47b14a4b60764e21b8cadc11c5653863af1bec08271d9ddb1a70326256c",
}
# Further code on the trace/renderer/loader path: recorded in the plan and rehashed before/after.
CLOSURE_SOURCES = ("bench/__init__.py", "bench/vision.py", "bench/quality_capture.py", "bench/quality_render_stability.py",
                   "bench/workbench_telemetry.py", "bench/workbench_contract.py", "bench/workbench_geometry.py",
                   f"{UPSTREAM_ROOT_REL}/utils/data_util.py", f"{UPSTREAM_ROOT_REL}/utils/misc.py",
                   f"{UPSTREAM_ROOT_REL}/utils/structs.py")
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
BASE_BYTES = 1_608_850_339
BASE_PATH = ROOT / ".cache/model-quality/checkpoints/gotrack_checkpoint.pt"
RECORD_SHA256 = PRODUCTION_PINS["bench/runtime/gotrack-flow-terminal-20261021.json"]
SMALL_RECORDS_PATH = ROOT / ".cache/cloud-prepare/flow-terminal-small-records-20261021.json"
SMALL_RECORDS_SHA256 = "09226dda6f08c739b05c42540156784b4783be5bf3d65b34cf0549b9959104aa"
HEAD_PATH = ROOT / ".cache/video-workbench/models/flow-terminal-20261021/head.pt"
ADMISSION_DIR = ROOT / ".cache/video-workbench/admission/attempt-06"
ADMISSION_CONTRACT_SHA256 = "83c041f9198161ff3ba3d5f6c477a3e7955eb24c20e451c014e62e01dced6cf0"
INFERENCE_PATH = ROOT / ".cache/prototype-sprint/dataset-pilot-v1/inference.json"
INFERENCE_SHA256 = "9b936de31c5511e3644e065538dedf47d0cdcd22f2ba17cee083e75a25403945"
MAP_PATH = ROOT / "tools/workbench_ycb_execution_controls.reviewed.json"
DERIVED_ROOT = ROOT / ".cache/video-workbench/derived"
OBJECTS = (2, 4, 18)
IMAGE_SIZE = (640, 480)
CROP = 280
ITERATIONS = 5
MIN_INLIERS = 24
CONFIDENCE_THRESHOLD = 0.3
PERTURBATION = {"rotation_deg": {"z": 2.0, "y": -2.0, "x": 1.0}, "translation_m": [0.005, -0.004, 0.010],
                "composition": "R0 = Rz(2) Ry(-2) Rx(1) R*; t0 = t* + (0.005, -0.004, 0.010) m (camera frame)"}
ORACLE_THRESHOLDS = {"translation_m": 1e-5, "rotation_deg": 1e-3, "checkpoint_p95_px": 1e-3, "min_inliers": MIN_INLIERS}
SEED_LIMITS = {"raw_orthogonality_max_abs": 1e-4, "projection_max_abs_delta": 1e-4}
SUPPORT_RATIO_MIN = 1e-6          # normalised s2/s1 for 3D object points and 2D image points (prospective)
CHECKPOINT_COUNT = 64
CHECKPOINT_SEED_BASE = 20261022
MAX_RENDERS = 63
MAX_PRIMARY_FORWARDS = 30
MAX_OUTPUT_BYTES = 8 << 20
DERIVED_BUDGET_BYTES = 1 << 30
FREE_RESERVE_BYTES = 1 << 30
FLAGS = {"synthetic_only": True, "human_accuracy": False, "phone_ready": False, "development_only": True,
         "automatic_rgb_preparation": False, "scored_inputs_read": False,
         "oracle_network_is_diagnostic_not_trained_head": True, "physical_pose_correction_claimed": False}
INTERPRETATION = [
    "oracle failure identifies a coordinate, solver or instrumentation issue in the tested path",
    "oracle success with learned drift localises the synthetic failure to learned correspondence/confidence and selection",
    "success of both leaves real appearance, segmentation and initialisation unresolved",
    "this experiment alone does not establish a cause and must not alter production",
]


class ControlError(Exception):
    """The control cannot be prepared or run safely."""


# --------------------------------------------------------------------------
# hashing / IO


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path) -> str:
    return Path(os.path.abspath(path)).relative_to(ROOT.resolve()).as_posix()


def _loads(raw: bytes, what: str) -> Any:
    from bench import workbench_contract as wc
    try:
        return wc.loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ControlError(f"{what}: invalid strict JSON ({exc})") from exc


def _dump(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


# --------------------------------------------------------------------------
# pure math (independently testable; no production lifting/projection helpers)


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def synthetic_target_pose(seed_pose_m: Any) -> dict:
    """Deterministic SVD SO(3) projection R = U diag(1, 1, det(U Vt)) Vt of an authenticated seed
    (synthetic experiment only). Rejects a raw seed with max|R^T R - I| > 1e-4 or a projection
    delta max|R_proper - R_raw| > 1e-4. The translation is kept exactly."""
    raw = np.asarray(seed_pose_m, dtype=np.float64)
    if raw.shape != (4, 4) or not np.isfinite(raw).all() or not np.array_equal(raw[3], [0.0, 0.0, 0.0, 1.0]):
        raise ControlError("seed pose must be a finite 4x4 rigid matrix")
    R = raw[:3, :3]
    raw_orth = float(np.abs(R.T @ R - np.eye(3)).max())
    U, _s, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    Rp = U @ D @ Vt
    delta = float(np.abs(Rp - R).max())
    if raw_orth > SEED_LIMITS["raw_orthogonality_max_abs"] or delta > SEED_LIMITS["projection_max_abs_delta"]:
        raise ControlError(f"seed rotation too far from SO(3) (orthogonality {raw_orth:.3g}, projection delta {delta:.3g})")
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = Rp, raw[:3, 3]
    return {"pose_m": target, "raw_seed_pose_m": raw, "projection_max_abs_delta": delta,
            "raw_orthogonality_max_abs": raw_orth, "raw_determinant": float(np.linalg.det(R)),
            "target_orthogonality_max_abs": float(np.abs(Rp.T @ Rp - np.eye(3)).max()),
            "target_determinant": float(np.linalg.det(Rp)),
            "method": "SVD SO(3) projection U diag(1,1,det(U Vt)) Vt; translation unchanged"}


def perturbed_pose(pose_m: np.ndarray) -> np.ndarray:
    """Fixed camera-frame perturbation (frozen before any GPU result)."""
    P = np.asarray(pose_m, dtype=np.float64)
    a = PERTURBATION["rotation_deg"]
    out = np.eye(4)
    out[:3, :3] = rot_z(a["z"]) @ rot_y(a["y"]) @ rot_x(a["x"]) @ P[:3, :3]
    out[:3, 3] = P[:3, 3] + np.asarray(PERTURBATION["translation_m"], dtype=np.float64)
    return out


def metres_to_mm(pose_m: np.ndarray) -> np.ndarray:
    out = np.asarray(pose_m, dtype=np.float64).copy()
    out[:3, 3] *= 1000.0
    return out


def proper_pose(T: Any, tol: float = 1e-6) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0]):
        return False
    R = T[:3, :3]
    return float(np.abs(R.T @ R - np.eye(3)).max()) <= tol and abs(float(np.linalg.det(R)) - 1.0) <= tol


def pose_errors(T: np.ndarray, T_ref: np.ndarray) -> tuple[float, float]:
    """Translation distance and rotation angle (degrees), via a numerically stable atan2 form."""
    T, T_ref = np.asarray(T, dtype=np.float64), np.asarray(T_ref, dtype=np.float64)
    dt = float(np.linalg.norm(T[:3, 3] - T_ref[:3, 3]))
    Rr = T_ref[:3, :3].T @ T[:3, :3]
    vee = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / 2.0
    return dt, float(math.degrees(math.atan2(float(np.linalg.norm(vee)), (float(np.trace(Rr)) - 1.0) / 2.0)))


def lift_zdepth_mm(depth_mm: np.ndarray, Kc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer pixel centres (x = col, y = row) lifted by z-depth: X = d * inv(Kc) [x, y, 1]."""
    d = np.asarray(depth_mm, dtype=np.float64)
    h, w = d.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(np.asarray(Kc, dtype=np.float64)).T
    support = np.isfinite(d) & (d > 0)
    xyz = np.full((h, w, 3), np.nan)
    xyz[support] = rays[support] * (d[support] / rays[support][:, 2])[:, None]
    return np.stack([xx, yy], axis=-1), xyz


def independent_object_points(depth_mm: np.ndarray, Kc: np.ndarray, C: np.ndarray, P0_mm: np.ndarray) -> np.ndarray:
    """HxWx3 object points (mm) from z-depth, crop K, crop transform and current native pose."""
    A = np.asarray(C, dtype=np.float64) @ np.asarray(P0_mm, dtype=np.float64)
    _pixels, Xc = lift_zdepth_mm(depth_mm, Kc)
    return (Xc - A[:3, 3]) @ A[:3, :3]


def project_points(K: np.ndarray, T: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection of Nx3 points with a 4x4 camera-from-object pose (same units)."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    cam = X @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
    z = cam[:, 2]
    uv = np.full((len(X), 2), np.nan)
    ok = z > 0
    h = cam[ok] @ np.asarray(K, dtype=np.float64).T
    uv[ok] = h[:, :2] / h[:, 2:3]
    return uv, z


def oracle_flow(depth_mm: np.ndarray, Kc: np.ndarray, C: np.ndarray, P0_mm: np.ndarray,
                Pstar_mm: np.ndarray) -> dict:
    """Exact template(source)->query flow on the crop grid, in millimetres throughout. ``support``
    marks pixels with a lifted point in front of the query camera; elsewhere the flow is 0 and
    must not be used as a correspondence."""
    C = np.asarray(C, dtype=np.float64)
    A = C @ np.asarray(P0_mm, dtype=np.float64)
    B = C @ np.asarray(Pstar_mm, dtype=np.float64)
    pixels, Xc = lift_zdepth_mm(depth_mm, Kc)
    lifted = np.isfinite(Xc).all(axis=-1)
    flow = np.zeros(pixels.shape, dtype=np.float64)
    Xo = (Xc[lifted] - A[:3, 3]) @ A[:3, :3]                      # R_A^T (Xc - t_A)
    q, z = project_points(Kc, B, Xo)
    front = (z > 0) & np.isfinite(q).all(axis=1)
    idx = np.flatnonzero(lifted.ravel())
    flat = flow.reshape(-1, 2)
    flat[idx[front]] = q[front] - pixels.reshape(-1, 2)[idx[front]]
    support = np.zeros(lifted.shape, dtype=bool)
    support.reshape(-1)[idx[front]] = True
    h, w = support.shape
    target = pixels + flow
    inside = support & (target[..., 0] >= 0) & (target[..., 0] < w) & (target[..., 1] >= 0) & (target[..., 1] < h)
    return {"flow": flow, "support": support, "support_pixels": int(support.sum()),
            "behind_query_camera": int((~front).sum()), "targets_outside_crop": int((support & ~inside).sum()),
            "max_abs_flow_px": float(np.abs(flow[support]).max()) if support.any() else None}


def crop_to_native(refined_crop_mm: np.ndarray, C: np.ndarray, mode: str = "production") -> np.ndarray:
    """Final pose conversion. ``production`` = inv(C) @ refined, translation / 1000.
    DIAGNOSTIC modes for negative controls only: ``crop_not_inverted`` (C @ refined) and
    ``omit_mm_to_m`` (no / 1000). Never used to change production."""
    refined = np.asarray(refined_crop_mm, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if mode == "production":
        native = np.linalg.inv(C) @ refined
        native[:3, 3] /= 1000.0
    elif mode == "crop_not_inverted":
        native = C @ refined
        native[:3, 3] /= 1000.0
    elif mode == "omit_mm_to_m":
        native = np.linalg.inv(C) @ refined
    else:
        raise ValueError(f"unknown conversion mode {mode}")
    return native


def support_diagnostics(points: Any, min_count: int = MIN_INLIERS, ratio_min: float = SUPPORT_RATIO_MIN) -> dict:
    """Scale-normalised rank diagnostics of an Nx3 (object) or Nx2 (image) point set.
    Rejects non-finite, insufficient (< min_count) or effectively collinear (s2/s1 < ratio_min)
    support. Planarity (3D s3/s1) is recorded but never rejected."""
    P = np.asarray(points, dtype=np.float64)
    out = {"count": int(P.shape[0]) if P.ndim == 2 else 0, "dimension": int(P.shape[1]) if P.ndim == 2 else None,
           "finite": bool(P.ndim == 2 and P.shape[1] in (2, 3) and np.isfinite(P).all()),
           "singular_values_normalised": None, "ratio_2_1": None, "ratio_3_1": None,
           "planar": None, "collinear": None, "sufficient": False, "accepted": False}
    if not out["finite"] or out["count"] == 0:
        return out
    centred = P - P.mean(axis=0)
    s = np.linalg.svd(centred, compute_uv=False)
    s1 = float(s[0])
    out["sufficient"] = out["count"] >= min_count
    if s1 <= 0 or not math.isfinite(s1):
        out["collinear"] = True
        return out
    normalised = (s / s1).tolist()
    out["singular_values_normalised"] = normalised
    out["ratio_2_1"] = normalised[1] if len(normalised) > 1 else 0.0
    out["collinear"] = out["ratio_2_1"] < ratio_min
    if out["dimension"] == 3:
        out["ratio_3_1"] = normalised[2] if len(normalised) > 2 else 0.0
        out["planar"] = out["ratio_3_1"] < ratio_min
    out["accepted"] = bool(out["sufficient"] and not out["collinear"])
    return out


def checkpoint_errors(K: np.ndarray, T: np.ndarray, T_ref: np.ndarray, X_m: np.ndarray) -> dict:
    uv, z = project_points(K, T, X_m)
    uv_r, z_r = project_points(K, T_ref, X_m)
    ok = (z > 0) & (z_r > 0)
    err = np.linalg.norm(uv[ok] - uv_r[ok], axis=1)
    return {"points": int(ok.sum()), "of": int(len(X_m)),
            "median_px": float(np.median(err)) if err.size else None,
            "p95_px": float(np.percentile(err, 95)) if err.size else None}


def oracle_iteration_check(solver_state: int, inliers: int, solved_m: Any, pstar_m: np.ndarray, K: np.ndarray,
                           checkpoints_m: np.ndarray, thresholds: Mapping = ORACLE_THRESHOLDS) -> dict:
    out = {"solver_ok": int(solver_state) == 1, "inliers_ok": int(inliers) >= thresholds["min_inliers"],
           "proper_pose": proper_pose(solved_m) if solved_m is not None else False,
           "translation_error_m": None, "rotation_error_deg": None, "checkpoints": None, "passed": False}
    if out["solver_ok"] and out["proper_pose"]:
        dt, dr = pose_errors(solved_m, pstar_m)
        cp = checkpoint_errors(K, solved_m, pstar_m, checkpoints_m)
        out.update(translation_error_m=dt, rotation_error_deg=dr, checkpoints=cp)
        out["passed"] = bool(out["inliers_ok"] and dt <= thresholds["translation_m"] and dr <= thresholds["rotation_deg"]
                             and cp["p95_px"] is not None and cp["p95_px"] <= thresholds["checkpoint_p95_px"]
                             and cp["points"] == cp["of"])
    return out


def area_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    V, F = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    tri = V[F]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cdf = np.cumsum(areas) / areas.sum()
    rng = np.random.Generator(np.random.PCG64(seed))
    face = np.minimum(np.searchsorted(cdf, rng.random(count), side="right"), len(F) - 1)
    r = rng.random((count, 2))
    s = np.sqrt(r[:, 0])
    bary = np.stack([1.0 - s, s * (1.0 - r[:, 1]), s * r[:, 1]], axis=1)
    return np.einsum("ij,ijk->ik", bary, tri[face])


def validate_head_provenance(provenance: Any) -> dict:
    """Strict-installer provenance (HeadContext schema): nested requested/installed SHA, installation
    status, the record manifest and a tensor digest. It carries NO base field; base identity is
    verified separately from the actual checkpoint bytes and is never defaulted."""
    if not isinstance(provenance, Mapping):
        raise ControlError("installer provenance is not a mapping")
    head = provenance.get("head")
    if (not isinstance(head, Mapping) or head.get("requested_sha256") != HEAD_SHA256
            or head.get("installed_sha256") != HEAD_SHA256):
        raise ControlError("installed head requested/installed SHA-256 differ from 27e06")
    if provenance.get("head_installation") != {"requested": True, "installed": True}:
        raise ControlError("installer did not report a completed requested installation")
    if provenance.get("artifact_manifest_sha256") != RECORD_SHA256:
        raise ControlError("installer provenance names another primary record")
    state = provenance.get("installed_head_state_sha256")
    if not isinstance(state, str) or len(state) != 64:
        raise ControlError("installed head tensor digest missing")
    return {"requested_sha256": head["requested_sha256"], "installed_sha256": head["installed_sha256"],
            "head_installation": dict(provenance["head_installation"]),
            "artifact_manifest_sha256": provenance["artifact_manifest_sha256"],
            "installed_head_state_sha256": state, "installer_sha256": provenance.get("installer_sha256"),
            "base_field_in_provenance": "absent by schema; verified from checkpoint bytes"}


def verify_base_checkpoint(path: Path = BASE_PATH) -> dict:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise ControlError("base checkpoint missing")
    size = Path(path).stat().st_size
    digest = sha256_file(path)
    if size != BASE_BYTES or digest != BASE_SHA256:
        raise ControlError(f"base checkpoint differs (bytes {size}, sha256 {digest})")
    return {"path": _rel(path), "bytes": size, "sha256": digest, "verified": True}


# --------------------------------------------------------------------------
# manufactured CPU preflight (no model, no torch)


def _sphere_depth(Kc: np.ndarray, crop_pose_mm: np.ndarray, radius_mm: float, size: int = CROP) -> np.ndarray:
    centre = crop_pose_mm[:3, 3]
    yy, xx = np.indices((size, size), dtype=np.float64)
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(Kc).T     # z = 1 rays
    b = rays @ centre
    a = (rays * rays).sum(-1)
    disc = b * b - a * (centre @ centre - radius_mm ** 2)
    depth = np.full((size, size), 0.0)
    hit = disc > 0
    depth[hit] = (b[hit] - np.sqrt(disc[hit])) / a[hit]                              # z-depth (ray z = 1)
    return depth


def production_like_solve(depth_mm: np.ndarray, flow: np.ndarray, Kc: np.ndarray, C: np.ndarray, P0_mm: np.ndarray,
                          seed: int = 0, conversion: str = "production", support: np.ndarray | None = None) -> tuple:
    """Preflight-only replica of the consumer contract: object points from (C @ P0) and z-depth,
    target = pixel + flow, RANSAC + LM PnP on crop K, then ``crop_to_native(conversion)``."""
    from bench.vision import cv2
    A = np.asarray(C) @ P0_mm
    pixels, Xc = lift_zdepth_mm(depth_mm, Kc)
    usable = np.isfinite(Xc).all(-1) if support is None else (np.isfinite(Xc).all(-1) & support)
    obj = (Xc[usable] - A[:3, 3]) @ A[:3, :3]
    tgt = (pixels + flow)[usable]
    if len(obj) < MIN_INLIERS:
        return 0, 0, None
    ids = np.linspace(0, len(obj) - 1, min(len(obj), 10000), dtype=int)
    cv2.setRNGSeed(seed)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj[ids], tgt[ids], Kc, None, cv2.Rodrigues(A[:3, :3])[0],
                                             A[:3, 3].copy(), useExtrinsicGuess=True, iterationsCount=100,
                                             reprojectionError=3.0, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        return -1, 0, None
    chosen = ids[inl.ravel()]
    rvec, tvec = cv2.solvePnPRefineLM(obj[chosen], tgt[chosen], Kc, None, rvec, tvec)
    refined = np.eye(4)
    refined[:3, :3], refined[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return 1, len(chosen), crop_to_native(refined, C, conversion)


def manufactured_preflight(K: np.ndarray | None = None) -> dict:
    """The correct oracle must pass. Each negative control must fail the same unchanged gate:
    ``reversed_flow`` and ``half_pixel_displacement`` corrupt the correspondences;
    ``wrong_crop_inverse`` and ``omitted_mm_to_m`` keep correct correspondences and corrupt only the
    FINAL pose conversion. Two retained bad-flow variants are labelled as such."""
    K = np.asarray(K if K is not None else [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]], float)
    Kc = np.array([[620.0, 0, 139.5], [0, 620.0, 139.5], [0, 0, 1]])
    C = np.eye(4)
    C[:3, :3] = rot_z(1.0) @ rot_y(4.0) @ rot_x(-3.0)
    pstar_m = np.eye(4)
    pstar_m[:3, :3] = rot_x(20.0) @ rot_y(-30.0)
    pstar_m[:3, 3] = [0.01, -0.02, 0.60]
    p0_m = perturbed_pose(pstar_m)
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(CHECKPOINT_COUNT, 3))
    checkpoints = 0.04 * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    P0_mm, Ps_mm = metres_to_mm(p0_m), metres_to_mm(pstar_m)
    depth = _sphere_depth(Kc, C @ P0_mm, 40.0)
    good = oracle_flow(depth, Kc, C, P0_mm, Ps_mm)
    variants = {
        "correct": (good["flow"], "production"),
        "reversed_flow": (-good["flow"], "production"),
        "half_pixel_displacement": (good["flow"] + 0.5, "production"),
        "wrong_crop_inverse": (good["flow"], "crop_not_inverted"),
        "omitted_mm_to_m": (good["flow"], "omit_mm_to_m"),
        "bad_flow_oracle_uses_inverse_crop": (oracle_flow(depth, Kc, np.linalg.inv(C), P0_mm, Ps_mm)["flow"], "production"),
        "bad_flow_oracle_target_in_metres": (oracle_flow(depth, Kc, C, P0_mm, pstar_m)["flow"], "production"),
    }
    out = {}
    for name, (flow, conversion) in variants.items():
        state, inliers, native = production_like_solve(depth, flow, Kc, C, P0_mm, conversion=conversion,
                                                       support=good["support"])
        check = oracle_iteration_check(state, inliers, native, pstar_m, K, checkpoints)
        out[name] = {"conversion": conversion, "passed_oracle_check": check["passed"],
                     "translation_error_m": check["translation_error_m"],
                     "rotation_error_deg": check["rotation_error_deg"],
                     "checkpoint_p95_px": (check["checkpoints"] or {}).get("p95_px")}
    obj = independent_object_points(depth, Kc, C, P0_mm)[good["support"]]
    support = support_diagnostics(obj)
    ok = (out["correct"]["passed_oracle_check"] and support["accepted"]
          and not any(v["passed_oracle_check"] for k, v in out.items() if k != "correct"))
    return {"passed": bool(ok), "variants": _jsonable(out), "support": _jsonable(support),
            "scene": "analytic sphere (r = 40 mm) z-depth on a 280x280 crop grid; crop rotation C; fixed perturbation"}


# --------------------------------------------------------------------------
# closure


def runtime_identity() -> dict:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    from bench.vision import cv2
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "opencv": cv2.__version__,
            "opencv_module": str(Path(cv2.__file__).resolve()),
            **{name: version(name) for name in ("torch", "pyrender", "trimesh", "PyOpenGL", "Pillow")}}


def source_closure() -> dict[str, str]:
    files = {}
    for rel in (*PRODUCTION_PINS, *CLOSURE_SOURCES):
        path = ROOT / rel
        if not path.is_file():
            raise ControlError(f"closure source {rel} is missing")
        files[rel] = sha256_file(path)
    files[_rel(Path(__file__))] = sha256_file(Path(__file__))
    return dict(sorted(files.items()))


def admission_source_closure() -> dict:
    """The broader current closure that serving admission verifies (workbench_neural.source_closure)."""
    from bench import workbench_neural as wn
    closure = wn.source_closure(ROOT)
    return {"sha256": _canonical_sha(closure),
            "keys": sorted(closure) if isinstance(closure, Mapping) else None}


def input_closure(objects: Mapping) -> dict[str, str]:
    files = {_rel(INFERENCE_PATH): sha256_file(INFERENCE_PATH), _rel(HEAD_PATH): sha256_file(HEAD_PATH),
             _rel(BASE_PATH): sha256_file(BASE_PATH), _rel(SMALL_RECORDS_PATH): sha256_file(SMALL_RECORDS_PATH)}
    for path in sorted(p for p in ADMISSION_DIR.rglob("*") if p.is_file()):
        files[_rel(path)] = sha256_file(path)
    for obj in objects.values():
        for rel in (obj["glb_path"], obj["bundle_path"]):
            files[rel] = sha256_file(ROOT / rel)
    return dict(sorted(files.items()))


def _check_pins() -> None:
    bad = [rel for rel, sha in PRODUCTION_PINS.items() if not (ROOT / rel).is_file() or sha256_file(ROOT / rel) != sha]
    if bad:
        raise ControlError(f"production files differ from their pins: {bad}")
    if (sha256_file(INFERENCE_PATH) != INFERENCE_SHA256 or sha256_file(HEAD_PATH) != HEAD_SHA256
            or sha256_file(SMALL_RECORDS_PATH) != SMALL_RECORDS_SHA256):
        raise ControlError("inference manifest, head.pt or small records differ from their pins")


# --------------------------------------------------------------------------
# prepare


def build_plan() -> dict:
    """Deterministic plan from authenticated inputs (also rebuilt and compared by ``run``)."""
    before_torch = "torch" in sys.modules
    _check_pins()
    base = verify_base_checkpoint()
    from bench import workbench_neural as wn
    reviewed = _loads(MAP_PATH.read_bytes(), "reviewed map")
    inference = _loads(INFERENCE_PATH.read_bytes(), "inference.json")
    objects = {}
    for object_id in OBJECTS:
        entry = reviewed["entries"][str(object_id)]
        control = ROOT / entry["planned_out"]
        glb, bundle_path = control / "model.glb", control / "metric-bundle.json"
        if sha256_file(glb) != entry["candidate_glb_sha256"] or sha256_file(bundle_path) != entry["bundle_sha256"]:
            raise ControlError(f"object {object_id}: control GLB/bundle differ from the reviewed map")
        bundle = _loads(bundle_path.read_bytes(), "metric bundle")
        if (bundle["obj_id"] != object_id or bundle["glb_sha256"] != entry["candidate_glb_sha256"]
                or bundle["principal_point_origin"] != "integer_centers"
                or bundle["seed_frame_id"] != entry["seed_frame_id"]):
            raise ControlError(f"object {object_id}: bundle seed/geometry/origin is not the reviewed control")
        target = synthetic_target_pose(bundle["seed_pose_m"])
        frame = int(bundle["seed_frame_id"][1:])
        inf_obj = [o for o in inference["objects"] if o.get("objectId") == object_id][0]
        rows = [r for r in inf_obj["onboarding"] if r.get("frameId") == frame]
        if len(rows) != 1 or rows[0]["logicalIndex"] != max(r["logicalIndex"] for r in inf_obj["onboarding"]):
            raise ControlError(f"object {object_id}: seed frame is not the final onboarding row")
        K = wn.native_K(rows[0]["calibration"]["cam_K"], "integer_centers")
        V, F = wn.glb_mesh(glb, max_faces=200_000)
        checkpoints = area_samples(V, F, CHECKPOINT_COUNT, CHECKPOINT_SEED_BASE + object_id)
        pstar = target["pose_m"]
        objects[str(object_id)] = {
            "object_id": object_id, "glb_path": _rel(glb), "glb_sha256": entry["candidate_glb_sha256"],
            "bundle_path": _rel(bundle_path), "bundle_sha256": entry["bundle_sha256"],
            "seed_frame_id": bundle["seed_frame_id"], "native_K": K.tolist(),
            "raw_seed_pose_m": target["raw_seed_pose_m"].tolist(),
            "synthetic_target": {k: v for k, v in target.items() if k not in ("pose_m", "raw_seed_pose_m")},
            "pstar_m": pstar.tolist(),
            "initialisations": {"exact": pstar.tolist(), "perturbed": perturbed_pose(pstar).tolist()},
            "checkpoints_m": checkpoints.tolist(),
            "checkpoints_sha256": sha256_bytes(np.ascontiguousarray(checkpoints, "<f8").tobytes()),
        }
    if set(objects) != {str(o) for o in OBJECTS}:
        raise ControlError("object set differs from {2, 4, 18}")
    preflight = manufactured_preflight(np.asarray(objects["2"]["native_K"]))
    if not preflight["passed"]:
        raise ControlError(f"manufactured preflight failed: {preflight['variants']}")
    admission_closure = admission_source_closure()
    if "torch" in sys.modules and not before_torch:
        raise ControlError("prepare unexpectedly imported torch")
    return _jsonable({
        "schema": SCHEMA_PLAN, "tool_version": TOOL_VERSION, "flags": FLAGS, "interpretation": INTERPRETATION,
        "production_pins": PRODUCTION_PINS, "sources": source_closure(), "runtime": runtime_identity(),
        "inputs": input_closure(objects), "admission_source_closure": admission_closure, "base_checkpoint": base,
        "pins": {"head_sha256": HEAD_SHA256, "base_sha256": BASE_SHA256, "base_bytes": BASE_BYTES,
                 "record_sha256": RECORD_SHA256, "small_records_sha256": SMALL_RECORDS_SHA256,
                 "admission_contract_sha256": ADMISSION_CONTRACT_SHA256, "inference_sha256": INFERENCE_SHA256},
        "experiment": {"iterations": ITERATIONS, "min_inliers": MIN_INLIERS, "perturbation": PERTURBATION,
                       "oracle_thresholds": ORACLE_THRESHOLDS, "seed_limits": SEED_LIMITS,
                       "support_ratio_min": SUPPORT_RATIO_MIN, "checkpoint_count": CHECKPOINT_COUNT,
                       "checkpoint_seed_base": CHECKPOINT_SEED_BASE, "max_renders": MAX_RENDERS,
                       "max_primary_forwards": MAX_PRIMARY_FORWARDS,
                       "query": "one synthetic 640x480 render at the proper synthetic target per object (admitted "
                                "renderer, gray background); mask = rendered depth > 0"},
        "preflight": preflight, "objects": objects})


def check_out(out: Path) -> Path:
    out = Path(os.path.abspath(out))
    if DERIVED_ROOT.resolve() not in out.parents:
        raise ControlError("output must be a new directory inside .cache/video-workbench/derived")
    for part in [out, *out.parents]:
        if part.exists() and part.is_symlink():
            raise ControlError("output path traverses a symlink")
    if os.path.lexists(out):
        raise ControlError(f"output {out} already exists; outputs are immutable")
    return out


def _budget(planned: int, out: Path) -> None:
    root = DERIVED_ROOT.resolve()
    usage = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()) if root.exists() else 0
    if planned > MAX_OUTPUT_BYTES or usage + planned > DERIVED_BUDGET_BYTES:
        raise ControlError("8 MiB output cap or 1 GiB derived budget would be exceeded")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free - planned < FREE_RESERVE_BYTES:
        raise ControlError("1 GiB free-disk reserve would be violated")


def publish(out: Path, files: Mapping[str, bytes]) -> None:
    staging = out.parent / f".{out.name}.staging-{secrets.token_hex(6)}"
    staging.mkdir()
    try:
        for name, data in files.items():
            with os.fdopen(os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                                   0o644), "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        if os.path.lexists(out):
            raise ControlError("output appeared during publication")
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare(out: Path) -> dict:
    out = check_out(out)
    blob = _dump(build_plan())
    _budget(len(blob), out)
    publish(out, {"plan.json": blob})
    return {"directory": out.as_posix(), "plan_sha256": sha256_bytes(blob),
            "note": "pin plan_sha256 externally; the parent schedules the single GPU run"}


# --------------------------------------------------------------------------
# run (GPU; parent-scheduled)


class Counters:
    def __init__(self):
        self.query_renders = 0
        self.template_renders = 0
        self.primary_forwards = 0
        self.oracle_forwards = 0

    @property
    def renders(self) -> int:
        return self.query_renders + self.template_renders

    def take_render(self, kind: str) -> None:
        if self.renders >= MAX_RENDERS:
            raise ControlError(f"render bound {MAX_RENDERS} reached")
        setattr(self, kind, getattr(self, kind) + 1)

    def as_dict(self) -> dict:
        return {"query_renders": self.query_renders, "template_renders": self.template_renders,
                "renders": self.renders, "primary_forwards": self.primary_forwards,
                "oracle_forwards": self.oracle_forwards, "bounds": {"renders": MAX_RENDERS,
                                                                    "primary_forwards": MAX_PRIMARY_FORWARDS}}


def _squeeze2(value: Any) -> np.ndarray:
    array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array.astype(np.float64)


class CropObserver:
    """Scoped observer of the unchanged upstream crop function (restored on exit, verified)."""

    def __init__(self, counters: Counters, misc: Any):
        self.counters, self.misc = counters, misc
        self.latest: dict | None = None

    @contextlib.contextmanager
    def attached(self, data_util: Any):
        original = data_util.compute_gotrack_inputs_from_init_poses

        def observed(*args, **kwargs):
            self.counters.take_render("template_renders")
            result = original(*args, **kwargs)
            data, cameras, crop_from_orig = result
            self.latest = {"depth_mm": _squeeze2(data["templates"].depths),
                           "crop_k": np.asarray(self.misc.get_intrinsic_matrix(cameras[0]), dtype=np.float64),
                           "crop_from_orig": np.asarray(crop_from_orig[0], dtype=np.float64),
                           "init_pose_mm": (_squeeze2(kwargs["init_poses_cam_from_model"])
                                            if kwargs.get("init_poses_cam_from_model") is not None else None)}
            return result

        data_util.compute_gotrack_inputs_from_init_poses = observed
        try:
            yield self
        finally:
            data_util.compute_gotrack_inputs_from_init_poses = original
            self.latest = None
        if data_util.compute_gotrack_inputs_from_init_poses is not original:
            raise ControlError("upstream crop function was not restored")


class OracleNetwork:
    """Diagnostic stand-in for the trained network: exact geometric flow, unit confidence ONLY on
    template_mask & oracle support (outputs multiplied like the production network). Unsupported
    pixels get zero flow AND zero confidence, so the unchanged gate excludes them."""

    def __init__(self, observer: CropObserver, pstar_mm: np.ndarray, counters: Counters):
        self.observer, self.pstar_mm, self.counters = observer, pstar_mm, counters
        self.current_pose_mm: np.ndarray | None = None
        self.last: dict | None = None

    def __call__(self, query, template, mask):
        import torch
        self.counters.oracle_forwards += 1
        cap = self.observer.latest
        if cap is None or self.current_pose_mm is None:
            raise ControlError("oracle forward without a captured crop/current pose")
        if cap["depth_mm"].shape != (CROP, CROP):
            raise ControlError(f"captured template depth has shape {cap['depth_mm'].shape}")
        pose_gap = (None if cap["init_pose_mm"] is None
                    else float(np.abs(cap["init_pose_mm"] - self.current_pose_mm).max()))
        if pose_gap is None or pose_gap > 1e-2:
            raise ControlError(f"captured crop pose differs from the oracle's current pose ({pose_gap})")
        result = oracle_flow(cap["depth_mm"], cap["crop_k"], cap["crop_from_orig"], self.current_pose_mm, self.pstar_mm)
        template_mask = _squeeze2(mask) > 0.5
        usable = template_mask & result["support"]
        flow = result["flow"] * usable[..., None]
        conf = usable.astype(np.float32)
        self.last = {"support_pixels": result["support_pixels"], "template_mask_pixels": int(template_mask.sum()),
                     "confident_pixels": int(usable.sum()),
                     "excluded_template_mask_without_support": int((template_mask & ~result["support"]).sum()),
                     "behind_query_camera": result["behind_query_camera"],
                     "targets_outside_crop": result["targets_outside_crop"],
                     "max_abs_flow_px": result["max_abs_flow_px"], "captured_pose_gap_mm": pose_gap,
                     "crop_k": cap["crop_k"].tolist(), "crop_from_orig": cap["crop_from_orig"].tolist()}
        device = query.device
        return (torch.from_numpy(np.ascontiguousarray(flow.transpose(2, 0, 1), np.float32))[None].to(device),
                torch.from_numpy(conf)[None].to(device))


class CountingNetwork:
    def __init__(self, network: Any, counters: Counters):
        self.network, self.counters = network, counters

    def __call__(self, *args):
        if self.counters.primary_forwards >= MAX_PRIMARY_FORWARDS:
            raise ControlError(f"primary forward bound {MAX_PRIMARY_FORWARDS} reached")
        self.counters.primary_forwards += 1
        return self.network(*args)


def render_query(renderer: Any, structs: Any, obj_id: int, K: np.ndarray, pstar_m: np.ndarray,
                 counters: Counters) -> dict:
    from bench import workbench_geometry as wg
    counters.take_render("query_renders")
    W, H = IMAGE_SIZE
    camera = structs.PinholePlaneCameraModel(width=W, height=H, f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]),
                                             T_world_from_eye=np.linalg.inv(metres_to_mm(pstar_m)))
    outmap = renderer.render_object_model(obj_id, camera)
    color = depth = None
    for key, value in outmap.items():
        name = getattr(key, "name", str(key)).upper()
        if name.endswith("COLOR"):
            color = np.asarray(value, dtype=np.float64)
        elif name.endswith("DEPTH"):
            depth = np.asarray(value, dtype=np.float64)
    if color is None or depth is None or color.shape[:2] != (H, W) or depth.shape != (H, W):
        raise ControlError("query render returned no colour/depth of the native size")
    rgb = (np.clip(color[:, :, :3], 0, 1) * 255).round().astype(np.uint8)
    mask = depth > 0
    if mask.sum() < 100:
        raise ControlError("synthetic query silhouette is too small")
    return {"rgb": rgb, "mask": mask, "rgb_sha256": wg.array_sha256(rgb), "mask_sha256": wg.mask_sha256(mask),
            "mask_pixels": int(mask.sum())}


def correspondence_diagnostics(capture: Mapping, pose_m: np.ndarray, trace: Mapping) -> dict:
    """Independent reconstruction (captured template depth/K/C + current pose) of the retained,
    production-sampled and inlier correspondence sets; 3D object and 2D crop-target rank checks."""
    obj_all = independent_object_points(capture["depth_mm"], capture["crop_k"], capture["crop_from_orig"],
                                        metres_to_mm(pose_m)).reshape(-1, 3)
    h, w = capture["depth_mm"].shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    targets = (np.stack([xx, yy], axis=-1) + np.asarray(trace["flow"], dtype=np.float64)).reshape(-1, 2)
    retained = np.asarray(trace["retained_indices"], dtype=np.int64)
    sets = {"retained": np.arange(len(retained))}
    sets["production_sampled"] = (np.linspace(0, len(retained) - 1, min(len(retained), 10000), dtype=int)
                                  if len(retained) else np.zeros(0, dtype=int))
    sets["inliers"] = np.asarray(trace["ransac_inliers"], dtype=np.int64)
    out = {}
    for name, idx in sets.items():
        sel = retained[idx] if len(idx) else np.zeros(0, dtype=np.int64)
        out[name] = {"object_points_3d": support_diagnostics(obj_all[sel].reshape(-1, 3)),
                     "image_points_2d": support_diagnostics(targets[sel].reshape(-1, 2))}
    out["accepted"] = all(v["object_points_3d"]["accepted"] and v["image_points_2d"]["accepted"]
                          for k, v in out.items() if isinstance(v, dict) and "object_points_3d" in v)
    return out


def run_branch(refiner: Any, observer: CropObserver, query: Mapping, obj: Mapping, init_name: str,
               oracle: OracleNetwork | None) -> dict:
    K = np.asarray(obj["native_K"], dtype=np.float64)
    pstar = np.asarray(obj["pstar_m"], dtype=np.float64)
    checkpoints = np.asarray(obj["checkpoints_m"], dtype=np.float64)
    pose = np.asarray(obj["initialisations"][init_name], dtype=np.float64)
    rows, status = [], "completed"
    for iteration in range(ITERATIONS):
        if oracle is not None:
            oracle.current_pose_mm = metres_to_mm(pose)
        trace = refiner.trace({"rgb": query["rgb"], "mask": query["mask"], "native_K": K, "initial_pose_m": pose,
                               "glb_path": str(ROOT / obj["glb_path"]), "glb_sha256": obj["glb_sha256"],
                               "obj_id": obj["object_id"], "rng_seed": iteration})
        capture = observer.latest
        if capture is None:
            raise ControlError("no captured crop for this trace")
        state = int(np.asarray(trace["solver_state"]))
        inliers = int(np.asarray(trace["ransac_inliers"]).size)
        solved = np.asarray(trace["solved_native_pose_m"], dtype=np.float64)
        conf = np.asarray(trace["confidence"], dtype=np.float64)
        tmask = np.asarray(trace["template_masks"]).reshape(-1)[:conf.size].reshape(conf.shape).astype(bool)
        init_dt, init_dr = pose_errors(pose, pstar)
        support = correspondence_diagnostics(capture, pose, trace)
        check = oracle_iteration_check(state, inliers, solved if state == 1 else None, pstar, K, checkpoints)
        row = {"iteration": iteration, "rng_seed": iteration, "initial_pose_m": pose,
               "initial_translation_error_m": init_dt, "initial_rotation_error_deg": init_dr,
               "solver_state": state, "inliers": inliers, "retained": int(np.asarray(trace["retained_indices"]).size),
               "confidence_above_threshold": int((conf > CONFIDENCE_THRESHOLD).sum()),
               "confidence_mean_in_template": float(conf[tmask].mean()) if tmask.any() else None,
               "crop_k_trace": np.asarray(trace["crop_k"]).tolist(), "support": support,
               "solved_pose_m": solved if state == 1 else None, "proper_pose": check["proper_pose"],
               "translation_error_m": check["translation_error_m"], "rotation_error_deg": check["rotation_error_deg"],
               "checkpoints": check["checkpoints"]}
        if oracle is not None:
            row["oracle"] = dict(oracle.last or {})
            row["oracle_crop_k_matches_trace"] = bool(np.array_equal(np.asarray(row["oracle"].get("crop_k")),
                                                                     np.asarray(trace["crop_k"], dtype=np.float64)))
            row["oracle_passed"] = bool(check["passed"] and row["oracle_crop_k_matches_trace"] and support["accepted"])
            row["oracle_failure"] = (None if row["oracle_passed"] else
                                     "support_rejected" if not support["accepted"] else
                                     "crop_k_mismatch" if not row["oracle_crop_k_matches_trace"] else "gate_failed")
        rows.append(row)
        if state != 1 or inliers < MIN_INLIERS:
            status = f"solver_unavailable_at_iteration_{iteration}"
            break
        if oracle is not None and not row["oracle_passed"]:
            status = f"oracle_failed_at_iteration_{iteration}"
            break
        pose = solved
    return {"initialisation": init_name, "status": status, "iterations": rows,
            "oracle_passed": ((status == "completed" and len(rows) == ITERATIONS and all(r["oracle_passed"] for r in rows))
                              if oracle is not None else None)}


def _closure_now(plan_objects: Mapping) -> dict:
    return {"sources": source_closure(), "inputs": input_closure(plan_objects), "runtime": runtime_identity(),
            "admission_source_closure": admission_source_closure()}


def _release_cuda() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(plan_path: Path, expected_plan_sha256: str, out: Path, device: str) -> dict:
    out = check_out(out)
    raw = Path(plan_path).read_bytes()
    if sha256_bytes(raw) != expected_plan_sha256:
        raise ControlError("plan.json differs from the externally pinned SHA-256")
    plan = _loads(raw, "plan.json")
    if plan.get("schema") != SCHEMA_PLAN:
        raise ControlError("not a dynamic pose control plan")
    rebuilt = build_plan()     # full reconstruction from authenticated inputs (CPU, no torch)
    if rebuilt != plan:
        differing = sorted(k for k in set(rebuilt) | set(plan) if rebuilt.get(k) != plan.get(k))
        raise ControlError(f"plan does not match its reconstruction from authenticated inputs: {differing}")
    before = {k: plan[k] for k in ("sources", "inputs", "runtime", "admission_source_closure")}
    _budget(MAX_OUTPUT_BYTES, out)
    from bench import workbench_neural as wn
    admission = wn.load_admission(ADMISSION_DIR)
    if not admission.admitted or admission.body.get("contract_sha256") != ADMISSION_CONTRACT_SHA256:
        raise ControlError("serving admission 06 is not admitted for the pinned contract")
    counters = Counters()
    results: dict[str, Any] = {"oracle": {}, "learned": {}}
    evidence: dict[str, Any] = {"head_provenance": None, "base_before_load": None}
    cleanup_errors: list[str] = []
    status, stage, after = "running", "setup", None
    try:
        from bench import quality_gotrack as qg
        from bench import workbench_render as wr
        from bench import workbench_telemetry as wt
        qg.upstream_path()
        from utils import data_util, misc, structs  # type: ignore
        observer = CropObserver(counters, misc)
        queries: dict[str, dict] = {}
        with wt.gpu_stage("dynamic_pose_control"):
            stage = "oracle"
            oracle_ok = True
            for key, obj in plan["objects"].items():
                renderer = wr.WorkbenchZeroStorageRenderer(str(ROOT / obj["glb_path"]), obj["object_id"])
                refiner = None
                try:
                    queries[key] = render_query(renderer, structs, obj["object_id"], np.asarray(obj["native_K"]),
                                                np.asarray(obj["pstar_m"]), counters)
                    results["oracle"][key] = {"query": {k: v for k, v in queries[key].items() if k not in ("rgb", "mask")},
                                              "branches": []}
                    oracle = OracleNetwork(observer, metres_to_mm(np.asarray(obj["pstar_m"])), counters)
                    refiner = wn.WorkbenchIntegerRefiner(oracle, device, renderer_factory=lambda _case, r=renderer: r)
                    with observer.attached(data_util):
                        for init in ("exact", "perturbed"):
                            branch = run_branch(refiner, observer, queries[key], obj, init, oracle)
                            results["oracle"][key]["branches"].append(branch)
                            oracle_ok &= bool(branch["oracle_passed"])
                            if not branch["oracle_passed"]:
                                break
                finally:
                    try:
                        (refiner.close() if refiner is not None else renderer.close())
                    except Exception as exc:  # noqa: BLE001
                        cleanup_errors.append(f"oracle {key}: {type(exc).__name__}: {exc}")
                if not oracle_ok or cleanup_errors:
                    break
            if cleanup_errors:
                status = "incomplete_cleanup_failed"          # stops BEFORE any learned load
            elif not oracle_ok:
                status = "oracle_failed"
            else:
                stage = "learned_load"
                evidence["base_before_load"] = verify_base_checkpoint()
                network = counting = None
                try:
                    network, provenance = wn.load_primary_network(HEAD_PATH, device)
                    evidence["head_provenance_raw"] = provenance
                    evidence["head_provenance"] = validate_head_provenance(provenance)
                    counting = CountingNetwork(network, counters)
                    stage = "learned"
                    for key, obj in plan["objects"].items():
                        renderer = wr.WorkbenchZeroStorageRenderer(str(ROOT / obj["glb_path"]), obj["object_id"])
                        refiner = None
                        results["learned"][key] = {"query_rgb_sha256": queries[key]["rgb_sha256"], "branches": [],
                                                   "note": "descriptive; the oracle pass does not certify the trained head"}
                        try:
                            refiner = wn.WorkbenchIntegerRefiner(counting, device,
                                                                 renderer_factory=lambda _case, r=renderer: r)
                            with observer.attached(data_util):
                                for init in ("exact", "perturbed"):
                                    results["learned"][key]["branches"].append(
                                        run_branch(refiner, observer, queries[key], obj, init, None))
                        finally:
                            try:
                                (refiner.close() if refiner is not None else renderer.close())
                            except Exception as exc:  # noqa: BLE001
                                cleanup_errors.append(f"learned {key}: {type(exc).__name__}: {exc}")
                        if cleanup_errors:
                            break
                finally:
                    if counting is not None:
                        counting.network = None
                    network = None
                    _release_cuda()
                status = "incomplete_cleanup_failed" if cleanup_errors else "completed"
        stage = "post_rehash"
        after = _closure_now(plan["objects"])
        if _jsonable(after) != before:
            status = "incomplete_closure_changed"
    except BaseException as exc:
        _release_cuda()
        failure = {"schema": SCHEMA_FAILURE, "status": "failed", "stage": stage,
                   "error": f"{type(exc).__name__}: {exc}"[:2000], "flags": FLAGS, "plan_sha256": expected_plan_sha256,
                   "counters": counters.as_dict(), "cleanup_errors": cleanup_errors, "results_so_far": results,
                   "evidence": evidence, "closure_before": before, "production_pins": PRODUCTION_PINS,
                   "pins": plan.get("pins")}
        try:
            publish(out, {"failure.json": _dump(_jsonable(failure))})
        except Exception:  # noqa: BLE001
            pass
        raise
    document = {"schema": SCHEMA_RESULT, "status": status, "flags": FLAGS, "interpretation": INTERPRETATION,
                "plan_sha256": expected_plan_sha256, "counters": counters.as_dict(), "cleanup_errors": cleanup_errors,
                "closure": {"before": before, "after": after, "equal": _jsonable(after) == before},
                "admission": {"admitted": True, "contract_sha256": ADMISSION_CONTRACT_SHA256},
                "evidence": evidence, "results": results}
    blob = _dump(_jsonable(document))
    _budget(len(blob), out)
    publish(out, {"result.json": blob})
    return {"directory": out.as_posix(), "status": status, "sha256": sha256_bytes(blob), "counters": counters.as_dict()}


# --------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic dynamic pose control (prepare on CPU; run is parent-scheduled).")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="CPU plan + manufactured preflight; no model import")
    s.add_argument("--out", required=True)
    s = sub.add_parser("run", help="single GPU stage; requires the external plan pin")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda")
    sub.add_parser("preflight", help="manufactured CPU preflight only; writes nothing")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(Path(args.out))
        elif args.command == "run":
            result = run(Path(args.plan), args.expected_plan_sha256, Path(args.out), args.device)
        else:
            result = manufactured_preflight()
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0 if result.get("passed", True) else 2
    except (ControlError, OSError, KeyError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
