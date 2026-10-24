"""Post-run dataset-reference projection evaluator for the supplemental YCB runs (CPU only).

Scope: a dataset-reference *diagnostic* of the three terminal supplemental YCB runs (new head
27e06 vs frozen base f7d1, 110 frames per role). BOP YCB-Video reference poses and masks are
dataset labels, not independent human truth. ``independent_human_accuracy``,
``phone_attachment_verified`` and the human gates stay false or unverified. Existing run
reports, masks, poses, caches, controls, maps and preparation are never written.

Two mandatory phases, each in its own process:

``prepare`` (no ground-truth access)
    Authenticates the terminal runs, the inference manifest, the reviewed control map, the
    controls, the candidate geometry and the SAM mask manifests. It hashes the
    ``evaluator.json`` *bytes* against the known pin without parsing them, and opens no
    scored mask, depth or pose. It freezes the anchor and 256 deterministic CAD surface
    samples per object, plus the metric specification, into a fresh immutable plan
    directory. It approves nothing and seals nothing.

``evaluate``
    Requires ``--plan``, the externally pinned ``--expected-plan-sha256`` and a separate
    parent ``--terminal-seal``. Before any reference label is read it validates the seal,
    rehashes every input against the plan, and checks the output path and budgets. Only then
    does it read ``evaluator.json`` and stream the cached scored masks and depth. It rehashes
    every input again after reading, and writes
    ``workbench_ycb_dataset_reference_projection_v1``.

Conventions: BOP K and R are row-major, model-to-camera; translations and depth are in mm
(``depth_scale`` multiplies raw depth to mm). Converting mm to metres happens exactly once.
Workbench ``pose_units='metric'`` means metres. The camera origin is fixed to
``integer_centers``, an assumption that is not optimised.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import workbench_contract as wc  # noqa: E402  (pure contract/JSON helpers)

TOOL_VERSION = "1"
SCHEMA_PLAN = "workbench_ycb_dataset_reference_plan_v1"
SCHEMA_SAMPLES = "workbench_ycb_dataset_reference_samples_v1"
SCHEMA_SEAL = "workbench_ycb_dataset_reference_terminal_seal_v1"
SCHEMA_OUT = "workbench_ycb_dataset_reference_projection_v1"
SCHEMA_FAILURE = "workbench_ycb_dataset_reference_failure_v1"

HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
INFERENCE_SHA256 = "9b936de31c5511e3644e065538dedf47d0cdcd22f2ba17cee083e75a25403945"
EVALUATOR_SHA256 = "c54534d6835653eb367f8d6fee34bd56cd3417056d48f5468d4f6e7d565b1870"
DATASET_ROOT = ROOT / ".cache/prototype-sprint/dataset-pilot-v1"
INFERENCE_PATH = DATASET_ROOT / "inference.json"
EVALUATOR_PATH = DATASET_ROOT / "evaluator.json"
MAP_PATH = ROOT / "tools/workbench_ycb_execution_controls.reviewed.json"
WORKSPACE = ROOT / ".cache/video-workbench"
DERIVED_ROOT = WORKSPACE / "derived"
RUNS = {2: "run-20261021T030702-475adc39", 4: "run-20261021T031729-0986ef46", 18: "run-20261021T032932-539343a7"}
CLIPS = {2: "dev-ycb-003-cracker-box", 4: "dev-ycb-005-tomato-soup-can", 18: "dev-ycb-040-large-marker"}
ROLES = ("primary_neural", "frozen_baseline")
FRAMES_PER_RUN = 110
MASK_SET = "sam2-workbench-01"
ORIGIN = "integer_centers"

SAMPLE_COUNT = 256
SAMPLE_SEED_BASE = 20261021
PX720_HEIGHT = 720.0
COVERAGE_THRESHOLDS_PX720 = (5.0, 10.0)
DEPTH_TOL_MIN_MM = 5.0
DEPTH_TOL_FRACTION = 0.01
ROTATION_TOL = 1e-4
OCCLUSION_TOL_M = 1e-5
DERIVED_BUDGET_BYTES = 1 << 30
FREE_RESERVE_BYTES = 1 << 30
MAX_OUTPUT_BYTES = 16 << 20
MAX_JSON_BYTES = 64 << 20
MAX_LABEL_BYTES = 16 << 20
FLAGS = {"dataset_reference_diagnostic": True, "independent_human_accuracy": False,
         "phone_attachment_verified": False, "development_only": True,
         "full_printed_uv_correspondence": "inconclusive", "report_accuracy_overwritten": False,
         "human_gates": "unverified"}
LIMITATIONS = [
    "dataset reference poses/masks/depth are BOP YCB-Video labels, not independent human truth",
    "shared CAD geometry and shared calibration (K, assumed integer_centers origin) for prediction and reference",
    "development selection: these are development objects with reference poses used for onboarding seeds; "
    "public YCB-Video training exposure of the underlying models is possible",
    "fixed surface identity: no symmetry minimisation, registration or alignment of any kind",
    "a single anchor cannot certify orientation, footprint or occlusion correctness",
    "CAD surface samples are not independent landmarks and may include hidden surfaces; not BOP MSPD",
    "pose projections do not prove that artwork was rendered correctly",
    "wrong-surface events and recovery correctness remain unverified human metrics",
]
METRIC_SPEC = {
    "A_anchor": "X = sum_i bary_i * V[face][i] from neural-artwork.json marker on the candidate GLB (metres); "
                "project with exact frame K under T_pred and T_BOP; native Euclidean px error, px720 = native * 720 / H; "
                "reference eligibility (valid R, z > 0, inside image) is independent of prediction; failures keep rows "
                "with null error; coverage <= 5 / 10 px720 over all reference-eligible frames",
    "B_cad_samples": "256 area-weighted CAD surface samples per object, PCG64(20261021 + object_id); draw order: "
                     "u_face = rng.random(256) float64, face = searchsorted(area CDF, u_face, 'right'), then "
                     "r = rng.random((256, 2)) float64, s = sqrt(r[:, 0]), bary = (1 - s, s * (1 - r[:, 1]), s * r[:, 1]); "
                     "same point identities under both poses; reference-eligible points (z > 0 under T_BOP) are fixed "
                     "first and form each frame's denominator; paired points also have z > 0 under T_pred; per frame: "
                     "paired fraction, failure-aware coverage <= 5/10 px720 over reference-eligible points (missing, "
                     "invalid or behind-camera predictions count as not covered) and conditional median/P95 over "
                     "paired points; status complete/partial/missing_prediction/no_paired_points; "
                     "frames summarised separately (no pooling across points/frames/objects); hidden surfaces included",
    "C_anchor_reference_visible": "anchor visible under the REFERENCE only: valid R, z > 0, in bounds, no nearer source-mesh "
                                  "intersection (tolerance 1e-5 m), BOP mask_visib positive at the rounded pixel, measured "
                                  "depth within max(5 mm, 1% of measured depth); missing/hash-mismatched mask or depth, or "
                                  "zero depth, means visibility unknown (never a prediction or projected-silhouette fallback)",
    "D_availability_display": "all 110 frames per role: tracking pose present, render_state rendered and render_stats "
                              "rendered_pixels counted separately; optional dataset mask_visib IoU and SAM outside-reference "
                              "leakage from the sealed SAM mask set (null when a label is missing)",
    "percentiles": "numpy linear interpolation", "infinity": "never emitted; failures are null with a status",
}


class EvalError(Exception):
    """The inputs cannot support this diagnostic; nothing is evaluated."""


# Project modules this evaluator actually imports (directly or transitively); each must load from
# its repository path, and its file hash is part of the frozen plan and the pre/post closure.
HELPER_MODULES = {"bench": "bench/__init__.py", "bench.vision": "bench/vision.py",
                  "bench.workbench_contract": "bench/workbench_contract.py",
                  "bench.workbench_geometry": "bench/workbench_geometry.py",
                  "bench.workbench_neural": "bench/workbench_neural.py"}
LABEL_DECODE_ERRORS: tuple[type[BaseException], ...] = (OSError, ValueError, SyntaxError)


def load_helpers() -> dict:
    """Import every helper used by either phase and return the modules (no model/GPU imports)."""
    import importlib
    from PIL import Image, UnidentifiedImageError
    global LABEL_DECODE_ERRORS
    LABEL_DECODE_ERRORS = (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError)
    modules = {name: importlib.import_module(name) for name in HELPER_MODULES}
    for name, rel in HELPER_MODULES.items():
        if Path(modules[name].__file__).resolve() != (ROOT / rel).resolve():
            raise EvalError(f"helper {name} was imported from {modules[name].__file__}, not {rel}")
    return modules


def helper_sources() -> dict[str, str]:
    load_helpers()
    sources = {rel: sha256_file(ROOT / rel) for rel in HELPER_MODULES.values()}
    sources[_rel(Path(__file__))] = sha256_file(Path(__file__))
    return dict(sorted(sources.items()))


def runtime_identity() -> dict:
    import platform
    import PIL
    load_helpers()
    from bench.vision import cv2
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "numpy": np.__version__, "pillow": PIL.__version__, "opencv": cv2.__version__,
            "opencv_module": str(Path(cv2.__file__).resolve()), "platform": platform.platform()}


# --------------------------------------------------------------------------
# hashing / IO


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path, limit: int, what: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise EvalError(f"{what} is missing or not a regular file: {path}")
    if path.stat().st_size > limit:
        raise EvalError(f"{what} exceeds {limit} bytes")
    return path.read_bytes()


def _loads(raw: bytes, what: str) -> Any:
    try:
        return wc.loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise EvalError(f"{what}: invalid strict JSON ({exc})") from exc


def _dump(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _rel(path: Path) -> str:
    return Path(os.path.abspath(path)).relative_to(ROOT.resolve()).as_posix()


def _safe_member(base: Path, rel: Any, what: str) -> Path:
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        raise EvalError(f"{what}: unsafe relative path")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or any(p in ("", ".", "..") for p in pure.parts):
        raise EvalError(f"{what}: unsafe relative path")
    return base.joinpath(*pure.parts)


def _hex(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise EvalError(f"{what} must be a lowercase SHA-256")
    return value


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


# --------------------------------------------------------------------------
# pure geometry


def proper_rotation(R: Any, tol: float = ROTATION_TOL) -> bool:
    R = np.asarray(R, dtype=np.float64)
    return (R.shape == (3, 3) and bool(np.isfinite(R).all()) and float(np.abs(R.T @ R - np.eye(3)).max()) <= tol
            and abs(float(np.linalg.det(R)) - 1.0) <= tol)


def bop_pose_m(cam_R_m2c: Sequence[float], cam_t_m2c_mm: Sequence[float]) -> np.ndarray | None:
    """Row-major BOP model-to-camera R and t in mm -> 4x4 metres (converted exactly once).
    Returns None for an improper/non-finite pose (never repaired)."""
    R = np.asarray(cam_R_m2c, dtype=np.float64)
    t = np.asarray(cam_t_m2c_mm, dtype=np.float64)
    if R.size != 9 or t.size != 3 or not np.isfinite(t).all():
        return None
    R = R.reshape(3, 3)
    if not proper_rotation(R):
        return None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t * 0.001
    return T


def predicted_pose_m(row: Mapping) -> tuple[np.ndarray | None, str]:
    if row.get("pose_state") != "tracking" or row.get("pose") is None:
        return None, "no_tracking_pose"
    if row.get("pose_units") != "metric":
        return None, "invalid_prediction_pose"
    T = np.asarray(row["pose"], dtype=np.float64)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0])
            or not proper_rotation(T[:3, :3])):
        return None, "invalid_prediction_pose"
    return T, "ok"


def project(points_m: np.ndarray, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer-centre pixel projection; uv is NaN where z <= 0."""
    P = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    cam = P @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]
    uv = np.full((len(P), 2), np.nan)
    ok = z > 0
    uv[ok, 0] = K[0, 0] * cam[ok, 0] / z[ok] + K[0, 1] * cam[ok, 1] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]
    return uv, z


def in_image(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(uv).all(axis=1)
    return finite & (uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5)


def px720(native_px: Any, height: int) -> Any:
    return np.asarray(native_px, dtype=np.float64) * (PX720_HEIGHT / float(height))


def anchor_point(vertices: np.ndarray, faces: np.ndarray, face_id: int, barycentric: Sequence[float]) -> np.ndarray:
    b = np.asarray(barycentric, dtype=np.float64)
    if b.shape != (3,) or not np.isfinite(b).all() or (b < -1e-9).any() or abs(b.sum() - 1.0) > 1e-6:
        raise EvalError("artwork barycentric coordinates are invalid")
    if not 0 <= int(face_id) < len(faces):
        raise EvalError("artwork face id outside the mesh")
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)[int(face_id)]]
    return b @ tri


def cad_samples(vertices: np.ndarray, faces: np.ndarray, object_id: int, count: int = SAMPLE_COUNT) -> dict:
    """Deterministic area-weighted surface samples (frozen before any reference read)."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    tri = V[F]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(areas.sum())
    if not total > 0 or not np.isfinite(areas).all():
        raise EvalError("mesh has no positive finite area")
    cdf = np.cumsum(areas) / total
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED_BASE + int(object_id)))
    u_face = rng.random(count)
    face_ids = np.minimum(np.searchsorted(cdf, u_face, side="right"), len(F) - 1).astype(np.int64)
    r = rng.random((count, 2))
    s = np.sqrt(r[:, 0])
    bary = np.stack([1.0 - s, s * (1.0 - r[:, 1]), s * r[:, 1]], axis=1)
    positions = np.einsum("ij,ijk->ik", bary, tri[face_ids])
    return {"positions_m": positions, "face_ids": face_ids, "barycentric": bary,
            "hashes": {"positions_m_float64": sha256_bytes(np.ascontiguousarray(positions, "<f8").tobytes()),
                       "face_ids_int64": sha256_bytes(np.ascontiguousarray(face_ids, "<i8").tobytes()),
                       "barycentric_float64": sha256_bytes(np.ascontiguousarray(bary, "<f8").tobytes())}}


def first_hit_occludes(vertices: np.ndarray, faces: np.ndarray, T: np.ndarray, point_m: np.ndarray,
                       own_face: int | None = None, tol_m: float = OCCLUSION_TOL_M) -> bool:
    """True when the camera ray to ``point_m`` hits the mesh nearer than the point (self-occlusion)."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    R, t = T[:3, :3], T[:3, 3]
    origin = -R.T @ t
    direction = np.asarray(point_m, dtype=np.float64) - origin
    length = float(np.linalg.norm(direction))
    if length <= 0:
        return True
    d = direction / length
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-15
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = origin - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    dist = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 0) & (dist < length - tol_m)
    if own_face is not None:
        hit[int(own_face)] = False
    return bool(hit.any())


def depth_consistent(measured_mm: float, predicted_mm: float) -> bool:
    return abs(measured_mm - predicted_mm) <= max(DEPTH_TOL_MIN_MM, DEPTH_TOL_FRACTION * measured_mm)


def percentile(values: Sequence[float], q: float) -> float | None:
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.percentile(data, q)) if data else None


def error_summary(errors: Sequence[float | None], eligible: int) -> dict:
    measured = [float(e) for e in errors if e is not None]
    return {"reference_eligible": eligible, "errors_measured": len(measured),
            "median_px720": percentile(measured, 50), "p95_px720": percentile(measured, 95),
            **{f"coverage_le_{int(t)}px720": (sum(e <= t for e in measured) / eligible if eligible else None)
               for t in COVERAGE_THRESHOLDS_PX720},
            "denominator": "all reference-eligible frames (failures and missing predictions count as not covered)"}


def anchor_frame(X: np.ndarray, K: np.ndarray, T_ref: np.ndarray | None, row: Mapping | None,
                 width: int, height: int) -> dict:
    """One role/frame row of metric A; reference eligibility never depends on the prediction."""
    out: dict[str, Any] = {"reference_status": "eligible", "prediction_status": None, "error_native_px": None,
                           "error_px720": None, "reference_uv": None, "prediction_uv": None}
    if T_ref is None:
        out["reference_status"] = "invalid_reference_pose"
    else:
        uv_r, z_r = project(X, K, T_ref)
        if not z_r[0] > 0:
            out["reference_status"] = "reference_behind_camera"
        elif not in_image(uv_r, width, height)[0]:
            out["reference_status"] = "reference_out_of_image"
        out["reference_uv"] = [_finite(v) for v in uv_r[0]] if z_r[0] > 0 else None
    if row is None:
        out["prediction_status"] = "missing_row"
        return out
    T_pred, status = predicted_pose_m(row)
    if T_pred is not None:
        uv_p, z_p = project(X, K, T_pred)
        if not z_p[0] > 0:
            status = "prediction_behind_camera"
        else:
            out["prediction_uv"] = [_finite(v) for v in uv_p[0]]
            if not in_image(uv_p, width, height)[0]:
                status = "prediction_out_of_image"
    out["prediction_status"] = status
    if out["reference_status"] == "eligible" and status == "ok":
        native = float(np.linalg.norm(np.asarray(out["prediction_uv"]) - np.asarray(out["reference_uv"])))
        out["error_native_px"] = native
        out["error_px720"] = float(px720(native, height))
    return out


def samples_frame(P: np.ndarray, K: np.ndarray, T_ref: np.ndarray | None, row: Mapping | None, height: int) -> dict:
    """Metric B for one role/frame. Reference eligibility (z > 0 under T_BOP) is fixed first and
    is never reduced by the prediction: missing, invalid or behind-camera predictions stay in the
    reference-defined denominator. Conditional quantiles cover paired points only."""
    out: dict[str, Any] = {"status": None, "prediction_status": None, "reference_eligible_points": 0,
                           "paired_points": 0, "prediction_behind_camera_points": 0, "prediction_invalid_points": 0,
                           "paired_fraction": None, "coverage_le_5px720": None, "coverage_le_10px720": None,
                           "conditional_median_px720": None, "conditional_p95_px720": None}
    if T_ref is None:
        out["status"] = "invalid_reference_pose"
        return out
    _uv_r, z_r = project(P, K, T_ref)
    ref_ok = z_r > 0
    n_ref = int(ref_ok.sum())
    out["reference_eligible_points"] = n_ref
    T_pred, pred_status = predicted_pose_m(row) if row is not None else (None, "missing_row")
    out["prediction_status"] = pred_status
    if n_ref == 0:
        out["status"] = "no_reference_points"
        return out
    errors = np.zeros(0)
    if T_pred is None:
        out["prediction_invalid_points"] = n_ref
    else:
        uv_r, _ = project(P, K, T_ref)
        uv_p, z_p = project(P, K, T_pred)
        paired = ref_ok & (z_p > 0) & np.isfinite(uv_p).all(axis=1)
        out["prediction_behind_camera_points"] = int((ref_ok & ~(z_p > 0)).sum())
        out["prediction_invalid_points"] = int((ref_ok & (z_p > 0) & ~np.isfinite(uv_p).all(axis=1)).sum())
        errors = px720(np.linalg.norm(uv_p[paired] - uv_r[paired], axis=1), height)
    n_paired = int(errors.size)
    out["paired_points"] = n_paired
    out["paired_fraction"] = n_paired / n_ref
    for t in COVERAGE_THRESHOLDS_PX720:
        out[f"coverage_le_{int(t)}px720"] = int((errors <= t).sum()) / n_ref
    out["conditional_median_px720"] = percentile(errors, 50)
    out["conditional_p95_px720"] = percentile(errors, 95)
    out["status"] = ("complete" if n_paired == n_ref else "partial" if n_paired > 0
                     else "missing_prediction" if T_pred is None else "no_paired_points")
    return out


def mask_overlap(sam: np.ndarray | None, reference: np.ndarray | None) -> dict:
    if sam is None or reference is None or sam.shape != reference.shape:
        return {"status": "label_or_mask_missing", "iou": None, "sam_outside_reference_fraction": None}
    a, b = np.asarray(sam) > 0, np.asarray(reference) > 0
    union = int((a | b).sum())
    return {"status": "ok", "iou": (int((a & b).sum()) / union) if union else None,
            "sam_outside_reference_fraction": (int((a & ~b).sum()) / int(a.sum())) if a.any() else None}


def join_role_rows(rows: Sequence[Mapping], scheduled: Sequence[str], roles: Sequence[str] = ROLES) -> dict:
    """Exact (role, frame) join; duplicates, extras or missing rows are rejected (no silent drop/reorder)."""
    table: dict[str, dict[str, Mapping]] = {role: {} for role in roles}
    for row in rows:
        role, frame = row.get("backend_role"), row.get("frame_id")
        if role not in table or frame not in set(scheduled):
            raise EvalError(f"unscheduled row ({role}, {frame})")
        if frame in table[role]:
            raise EvalError(f"duplicate row ({role}, {frame})")
        table[role][frame] = row
    for role in roles:
        if list(table[role]) and sorted(table[role], key=list(scheduled).index) != list(scheduled):
            raise EvalError(f"{role}: rows do not cover every scheduled frame")
        if len(table[role]) != len(scheduled):
            raise EvalError(f"{role}: {len(table[role])} rows for {len(scheduled)} scheduled frames")
    return table


# --------------------------------------------------------------------------
# prepare (no ground-truth access)


def _load_mesh(glb: Path) -> tuple[np.ndarray, np.ndarray]:
    from bench import workbench_neural as wn  # numpy-only at import; same loader the runs used
    v, f = wn.glb_mesh(glb, max_faces=200_000)
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def _run_files(run_dir: Path) -> dict[str, str]:
    names = ["run.json", "results.jsonl", "report.json", "neural-artwork.json"]
    stages = sorted(p.name for p in (run_dir / "stages").iterdir()) if (run_dir / "stages").is_dir() else []
    files = {}
    for name in names + [f"stages/{s}" for s in stages]:
        path = run_dir.joinpath(*name.split("/"))
        if path.is_symlink() or not path.is_file():
            raise EvalError(f"{run_dir.name}: {name} missing")
        files[name] = sha256_file(path)
    return files


def rgb_binding(rgb_ref: Mapping) -> dict:
    """Authenticate one scored RGB file against the inference pin and compute the workbench
    decoded-pixel identity with the pipeline's exact convention (bench.workbench_cli.FrameReader.read):
    cv2.imdecode(raw, IMREAD_COLOR) -> contiguous BGR->RGB -> workbench_geometry.array_sha256."""
    load_helpers()
    from bench.vision import cv2
    from bench import workbench_geometry as wg
    if rgb_ref.get("available") is not True:
        raise EvalError("scored RGB is unavailable in the inference manifest")
    path = _safe_member(DATASET_ROOT, rgb_ref.get("path"), "scored RGB")
    raw = _read(path, 16 << 20, "scored RGB")
    if sha256_bytes(raw) != _hex(rgb_ref.get("sha256"), "rgb.sha256"):
        raise EvalError(f"scored RGB {rgb_ref.get('path')} differs from the inference pin")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise EvalError(f"scored RGB {rgb_ref.get('path')} cannot be decoded")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    if rgb.shape != (480, 640, 3):
        raise EvalError(f"scored RGB {rgb_ref.get('path')} is not 640x480")
    return {"path": _rel(path), "file_sha256": rgb_ref["sha256"], "pixel_sha256": wg.array_sha256(rgb)}


def authenticate_run(object_id: int, inf_obj: Mapping, entry: Mapping, map_sha: str) -> dict:
    run_id, clip_id = RUNS[object_id], CLIPS[object_id]
    run_dir = WORKSPACE / "runs" / run_id
    files = _run_files(run_dir)
    doc = _loads(_read(run_dir / "run.json", MAX_JSON_BYTES, "run.json"), "run.json")
    run = wc.RunSpec.from_dict(doc["run"])
    frames = inf_obj["frames"]
    scheduled = [f"s{int(f['frameId']):06d}" for f in frames]
    if (run.run_id != run_id or run.clip_id != clip_id or doc.get("clip_id") != clip_id or run.mode != "regression"
            or tuple(run.roles) != ROLES or list(run.scheduled_frame_ids) != scheduled or len(scheduled) != FRAMES_PER_RUN):
        raise EvalError(f"{run_id}: run identity/roles/schedule differ from the frozen scored window")
    config = run.config
    control_dir = ROOT / entry["planned_out"]
    sidecar_sha = sha256_file(control_dir / "execution-control-sidecar.json")
    if (config.get("mask_set") != MASK_SET or config.get("surface_control_map_sha256") != map_sha
            or config.get("surface_control_sidecar_sha256") != sidecar_sha
            or config.get("execution_only_prepared_cad") is not True
            or config.get("independent_correspondence_verified") is not False):
        raise EvalError(f"{run_id}: mask set/control map/sidecar configuration differs")
    report = _loads(_read(run_dir / "report.json", MAX_JSON_BYTES, "report.json"), "report.json")
    stages = report.get("stages", {})
    if (report.get("status") != "completed" or report.get("run_id") != run_id
            or any(stages.get(r, {}).get("status") != "completed" or stages.get(r, {}).get("frames") != FRAMES_PER_RUN
                   for r in ROLES)):
        raise EvalError(f"{run_id}: report is not terminal-completed for both roles")
    raw_rows = [_loads(line, "results row") for line in
                _read(run_dir / "results.jsonl", MAX_JSON_BYTES, "results.jsonl").splitlines() if line.strip()]
    rows = [wc.FrameResult.from_dict(r).to_dict() for r in raw_rows]
    table = join_role_rows(rows, scheduled)
    rgb = {f"s{int(f['frameId']):06d}": f["rgb"]["sha256"] for f in frames}
    clip_path = WORKSPACE / "clips" / clip_id / "clip.json"
    clip = wc.ClipSpec.from_dict(_loads(_read(clip_path, MAX_JSON_BYTES, "clip.json"), "clip.json"))
    for fid in scheduled:
        frame = clip.frame(fid)
        if frame.image_sha256 != rgb[fid] or frame.source_frame_index != int(fid[1:]):
            raise EvalError(f"{run_id}: clip frame {fid} file hash/source index differs from the inference RGB")
    rgb_bindings = {}
    for f in frames:
        fid = f"s{int(f['frameId']):06d}"
        binding = rgb_binding(f["rgb"])
        pixel_hashes = {table[role][fid]["image_sha256"] for role in ROLES}
        if pixel_hashes != {binding["pixel_sha256"]}:
            raise EvalError(f"{run_id}: result rows for {fid} do not carry this frame's decoded RGB identity")
        rgb_bindings[fid] = binding
    glb_sha = entry["candidate_glb_sha256"]
    for role, by_frame in table.items():
        for fid, row in by_frame.items():
            ident = row["backend_identity"]
            if (row["run_id"] != run_id or row["clip_id"] != clip_id or row["source_frame_index"] != int(fid[1:])
                    or row["image_sha256"] is None or row["image_sha256"] != table[ROLES[0]][fid]["image_sha256"]
                    or ident.get("base_checkpoint_sha256") != BASE_SHA256
                    or ident.get("glb_sha256") != glb_sha or ident.get("principal_point_origin") != ORIGIN
                    or (role == "primary_neural" and (ident.get("installed_head_sha256") != HEAD_SHA256
                                                      or ident.get("export_sha256") != HEAD_SHA256))
                    or (role == "frozen_baseline" and (ident.get("installed_head_sha256") is not None
                                                       or ident.get("head") != "frozen_base_pose_head"))):
                raise EvalError(f"{run_id}: row ({role}, {fid}) identity/RGB/geometry differs")
    art_raw = _read(run_dir / "neural-artwork.json", MAX_JSON_BYTES, "neural-artwork.json")
    art = _loads(art_raw, "neural-artwork.json")
    body = {k: v for k, v in art.items() if k != "record_sha256"}
    K_run = np.asarray(art.get("effective_K"), dtype=np.float64).reshape(3, 3)
    if (art.get("schema") != "workbench_neural_artwork_v1" or art.get("units") != "metres"
            or art.get("glb_sha256") != glb_sha or art.get("principal_point_origin") != ORIGIN
            or art.get("record_sha256") != wc.canonical_sha256(body)):
        raise EvalError(f"{run_id}: neural-artwork.json does not bind the candidate geometry")
    for f in frames:
        if not np.array_equal(np.asarray(f["calibration"]["cam_K"], dtype=np.float64).reshape(3, 3), K_run):
            raise EvalError(f"{run_id}: frame {f['frameId']} K differs from the run's effective K")
    masks = stages_mask_identity(run_id, report, clip_id)
    return {"run_id": run_id, "clip_id": clip_id, "files": files, "scheduled": scheduled,
            "rgb": rgb_bindings,
            "clip": {"path": _rel(clip_path), "sha256": sha256_file(clip_path),
                     "rgb_binding": "clip frame image_sha256 == inference rgb.sha256 (file); both result roles carry "
                                    "the frame's decoded pixel identity (see run.rgb)"},
            "artwork": {"face_id": int(art["marker"]["face_id"]), "barycentric": [float(b) for b in art["marker"]["barycentric"]],
                        "mesh_faces": int(art["mesh_faces"]), "record_sha256": art["record_sha256"]},
            "mask_set": masks, "control_dir": entry["planned_out"], "sidecar_sha256": sidecar_sha}


def stages_mask_identity(run_id: str, report: Mapping, clip_id: str) -> dict:
    stage = report.get("stages", {}).get("masks", {})
    manifest_path = WORKSPACE / "masks" / clip_id / MASK_SET / "manifest.json"
    manifest = _loads(_read(manifest_path, MAX_JSON_BYTES, "mask manifest"), "mask manifest")
    if (stage.get("status") != "completed" or stage.get("identity", {}).get("mask_set") != MASK_SET
            or stage.get("identity", {}).get("manifest_sha256") != wc.canonical_sha256(manifest)
            or manifest.get("evaluator_inputs_read") is not False or manifest.get("clip_id") != clip_id):
        raise EvalError(f"{run_id}: sealed SAM mask manifest differs from the run's mask stage")
    return {"manifest": _rel(manifest_path), "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_canonical_sha256": wc.canonical_sha256(manifest)}


def build_plan() -> tuple[dict, dict[str, bytes]]:
    before_torch = "torch" in sys.modules
    if sha256_file(EVALUATOR_PATH) != EVALUATOR_SHA256:   # bytes hashed only; never parsed here
        raise EvalError("evaluator.json bytes differ from the known pin")
    inference_raw = _read(INFERENCE_PATH, MAX_JSON_BYTES, "inference.json")
    if sha256_bytes(inference_raw) != INFERENCE_SHA256:
        raise EvalError("inference.json differs from its pin")
    inference = _loads(inference_raw, "inference.json")
    if inference.get("evaluatorManifestSHA256") != EVALUATOR_SHA256:
        raise EvalError("inference manifest does not bind the pinned evaluator")
    map_raw = _read(MAP_PATH, MAX_JSON_BYTES, "reviewed map")
    map_sha = sha256_bytes(map_raw)
    reviewed = _loads(map_raw, "reviewed map")
    objects, samples_files = {}, {}
    for object_id in sorted(RUNS):
        inf_obj = [o for o in inference["objects"] if o.get("objectId") == object_id]
        entry = reviewed.get("entries", {}).get(str(object_id))
        if len(inf_obj) != 1 or not isinstance(entry, dict):
            raise EvalError(f"object {object_id}: inference object or reviewed map entry missing")
        inf_obj = inf_obj[0]
        if inf_obj["model"]["sha256"] != entry["ply_sha256"]:
            raise EvalError(f"object {object_id}: PLY pin differs between inference and the reviewed map")
        control_dir = ROOT / entry["planned_out"]
        glb = control_dir / "model.glb"
        if (sha256_file(glb) != entry["candidate_glb_sha256"]
                or sha256_file(control_dir / "metric-bundle.json") != entry["bundle_sha256"]):
            raise EvalError(f"object {object_id}: execution-control GLB/bundle differ from the reviewed map")
        run = authenticate_run(object_id, inf_obj, entry, map_sha)
        V, F = _load_mesh(glb)
        if len(F) != run["artwork"]["mesh_faces"]:
            raise EvalError(f"object {object_id}: GLB face count differs from the frozen artwork record")
        X = anchor_point(V, F, run["artwork"]["face_id"], run["artwork"]["barycentric"])
        samples = cad_samples(V, F, object_id)
        name = f"samples-object-{object_id:06d}.json"
        sample_doc = {"schema": SCHEMA_SAMPLES, "object_id": object_id, "count": SAMPLE_COUNT,
                      "seed": SAMPLE_SEED_BASE + object_id, "glb_sha256": entry["candidate_glb_sha256"],
                      "positions_m": samples["positions_m"].tolist(), "face_ids": samples["face_ids"].tolist(),
                      "barycentric": samples["barycentric"].tolist(), "hashes": samples["hashes"],
                      "label": "CAD surface samples; NOT independent landmarks; may include hidden surfaces"}
        samples_files[name] = _dump(sample_doc)
        frames = [{"frame_id": f"s{int(f['frameId']):06d}", "frameId": int(f["frameId"]),
                   "logicalIndex": int(f["logicalIndex"]), "rgb_sha256": f["rgb"]["sha256"],
                   "cam_K": [float(v) for v in f["calibration"]["cam_K"]],
                   "depth_scale": float(f["calibration"]["depth_scale"])} for f in inf_obj["frames"]]
        objects[str(object_id)] = {
            "object_id": object_id, "canonical_ycb_id": inf_obj["canonicalYcbId"], "scene_id": inf_obj["sceneId"],
            "run": run, "map_entry_sha256": sha256_bytes(wc.canonical_json(entry)),
            "candidate_glb_sha256": entry["candidate_glb_sha256"], "ply_sha256": entry["ply_sha256"],
            "bundle_sha256": entry["bundle_sha256"], "mesh": {"vertices": int(len(V)), "faces": int(len(F)),
                                                              "units": "metres (candidate GLB = float32(PLY mm x 0.001))"},
            "anchor_m": X.tolist(), "samples_file": name, "samples_file_sha256": sha256_bytes(samples_files[name]),
            "sample_hashes": samples["hashes"], "frames": frames, "image_size": [640, 480],
        }
    if "torch" in sys.modules and not before_torch:
        raise EvalError("prepare unexpectedly imported torch")
    plan = {"schema": SCHEMA_PLAN, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__)),
            "sources": helper_sources(), "runtime": runtime_identity(),
            "flags": FLAGS, "limitations": LIMITATIONS, "metric_spec": METRIC_SPEC,
            "constants": {"sample_count": SAMPLE_COUNT, "sample_seed_base": SAMPLE_SEED_BASE,
                          "coverage_thresholds_px720": list(COVERAGE_THRESHOLDS_PX720),
                          "depth_tolerance": {"min_mm": DEPTH_TOL_MIN_MM, "fraction": DEPTH_TOL_FRACTION},
                          "rotation_tolerance": ROTATION_TOL, "occlusion_tolerance_m": OCCLUSION_TOL_M,
                          "principal_point_origin": ORIGIN, "roles": list(ROLES), "frames_per_run": FRAMES_PER_RUN},
            "pins": {"head_sha256": HEAD_SHA256, "base_sha256": BASE_SHA256, "inference_sha256": INFERENCE_SHA256,
                     "evaluator_sha256": EVALUATOR_SHA256, "reviewed_map_sha256": map_sha},
            "evaluator_json_decoded": False, "scored_labels_read": False,
            "objects": objects}
    return plan, samples_files


# --------------------------------------------------------------------------
# output helpers


def check_out(out: Path) -> Path:
    out = Path(os.path.abspath(out))
    if DERIVED_ROOT.resolve() not in out.parents:
        raise EvalError("output must be a new directory inside .cache/video-workbench/derived")
    for part in [out, *out.parents]:
        if part.exists() and part.is_symlink():
            raise EvalError("output path traverses a symlink")
    if os.path.lexists(out):
        raise EvalError(f"output {out} already exists; outputs are immutable")
    return out


def _budget(planned: int, out: Path) -> None:
    root = DERIVED_ROOT.resolve()
    usage = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()) if root.exists() else 0
    if planned > MAX_OUTPUT_BYTES or usage + planned > DERIVED_BUDGET_BYTES:
        raise EvalError("derived output cap or the 1 GiB derived budget would be exceeded")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free - planned < FREE_RESERVE_BYTES:
        raise EvalError("1 GiB free-disk reserve would be violated")


def publish(out: Path, blobs: Mapping[str, bytes]) -> None:
    staging = out.parent / f".{out.name}.staging-{secrets.token_hex(6)}"
    staging.mkdir()
    try:
        for name, data in blobs.items():
            with os.fdopen(os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                                   0o644), "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if sha256_file(staging / name) != sha256_bytes(data):
                raise EvalError(f"staged {name} does not verify")
        if os.path.lexists(out):
            raise EvalError("output appeared during publication")
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare(out: Path) -> dict:
    out = check_out(out)
    plan, samples = build_plan()
    blobs = {"plan.json": _dump(plan), **samples}
    _budget(sum(len(b) for b in blobs.values()), out)
    publish(out, blobs)
    return {"directory": out.as_posix(), "plan_sha256": sha256_bytes(blobs["plan.json"]),
            "files": {k: sha256_bytes(v) for k, v in blobs.items()},
            "note": "pin plan_sha256 externally; the parent creates the terminal seal; no approval was generated"}


# --------------------------------------------------------------------------
# evaluate (ground truth only after the seal and a full pre-read rehash)

SEAL_KEYS = {"schema", "plan_sha256", "tool_sha256", "inference_sha256", "evaluator_sha256", "reviewed_map_sha256",
             "runs", "gpu_worker_terminal_observed", "observed_by", "observed_at", "statement"}


def load_plan(plan_path: Path, expected_sha256: str) -> tuple[dict, Path]:
    raw = _read(plan_path, MAX_JSON_BYTES, "plan.json")
    if sha256_bytes(raw) != _hex(expected_sha256, "expected plan SHA-256"):
        raise EvalError("plan.json differs from the externally pinned SHA-256")
    plan = _loads(raw, "plan.json")
    if plan.get("schema") != SCHEMA_PLAN or plan.get("tool_sha256") != sha256_file(Path(__file__)):
        raise EvalError("plan schema or tool version differs from this evaluator")
    plan_dir = Path(plan_path).parent
    expected = {"plan.json"} | {o["samples_file"] for o in plan["objects"].values()}
    if set(os.listdir(plan_dir)) != expected:
        raise EvalError("plan directory inventory differs")
    return plan, plan_dir


def validate_seal(seal_path: Path, plan: Mapping, plan_sha: str) -> tuple[dict, str]:
    raw = _read(seal_path, MAX_JSON_BYTES, "terminal seal")
    seal = _loads(raw, "terminal seal")
    expected_runs = {k: {"run_id": o["run"]["run_id"], "files": o["run"]["files"]} for k, o in plan["objects"].items()}
    if (not isinstance(seal, dict) or set(seal) != SEAL_KEYS or seal["schema"] != SCHEMA_SEAL
            or seal["plan_sha256"] != plan_sha or seal["tool_sha256"] != sha256_file(Path(__file__))
            or seal["inference_sha256"] != INFERENCE_SHA256 or seal["evaluator_sha256"] != EVALUATOR_SHA256
            or seal["reviewed_map_sha256"] != plan["pins"]["reviewed_map_sha256"]
            or seal["runs"] != expected_runs or seal["gpu_worker_terminal_observed"] is not True
            or not isinstance(seal["observed_by"], str) or not seal["observed_by"].strip()
            or not isinstance(seal["observed_at"], str) or not seal["observed_at"].strip()):
        raise EvalError("terminal seal is missing, stale, partial or does not pin this plan/tool/runs")
    return seal, sha256_bytes(raw)


def input_closure(plan: Mapping, plan_dir: Path) -> dict[str, str]:
    """Every non-label input the evaluation depends on, freshly hashed."""
    hashes = {_rel(plan_dir / "plan.json"): sha256_file(plan_dir / "plan.json"),
              _rel(INFERENCE_PATH): sha256_file(INFERENCE_PATH), _rel(MAP_PATH): sha256_file(MAP_PATH),
              _rel(EVALUATOR_PATH): sha256_file(EVALUATOR_PATH)}
    for obj in plan["objects"].values():
        run = obj["run"]
        run_dir = WORKSPACE / "runs" / run["run_id"]
        for name in run["files"]:
            hashes[_rel(run_dir.joinpath(*name.split("/")))] = sha256_file(run_dir.joinpath(*name.split("/")))
        control = ROOT / run["control_dir"]
        for name in ("model.glb", "metric-bundle.json", "execution-control-sidecar.json"):
            hashes[_rel(control / name)] = sha256_file(control / name)
        hashes[run["mask_set"]["manifest"]] = sha256_file(ROOT / run["mask_set"]["manifest"])
        hashes[run["clip"]["path"]] = sha256_file(ROOT / run["clip"]["path"])
        hashes[_rel(plan_dir / obj["samples_file"])] = sha256_file(plan_dir / obj["samples_file"])
        for binding in run["rgb"].values():
            hashes[binding["path"]] = sha256_file(ROOT / binding["path"])
    for rel in plan["sources"]:
        hashes[rel] = sha256_file(ROOT / rel)
    return hashes


def expected_closure(plan: Mapping, plan_dir: Path, plan_sha: str) -> dict[str, str]:
    expected = {_rel(plan_dir / "plan.json"): plan_sha, _rel(INFERENCE_PATH): INFERENCE_SHA256,
                _rel(MAP_PATH): plan["pins"]["reviewed_map_sha256"], _rel(EVALUATOR_PATH): EVALUATOR_SHA256}
    for obj in plan["objects"].values():
        run = obj["run"]
        run_dir = WORKSPACE / "runs" / run["run_id"]
        for name, digest in run["files"].items():
            expected[_rel(run_dir.joinpath(*name.split("/")))] = digest
        control = ROOT / run["control_dir"]
        expected[_rel(control / "model.glb")] = obj["candidate_glb_sha256"]
        expected[_rel(control / "metric-bundle.json")] = obj["bundle_sha256"]
        expected[_rel(control / "execution-control-sidecar.json")] = run["sidecar_sha256"]
        expected[run["mask_set"]["manifest"]] = run["mask_set"]["manifest_file_sha256"]
        expected[run["clip"]["path"]] = run["clip"]["sha256"]
        expected[_rel(plan_dir / obj["samples_file"])] = obj["samples_file_sha256"]
        for binding in run["rgb"].values():
            expected[binding["path"]] = binding["file_sha256"]
    current_sources = helper_sources()
    if set(current_sources) != set(plan["sources"]):
        raise EvalError("the imported helper set differs from the frozen plan")
    expected.update(plan["sources"])
    return expected


def _decode_label(path: Path, expected_sha: str | None, kind: str, read_hashes: dict) -> tuple[np.ndarray | None, str]:
    """Hash-authenticated label decode. Expected Pillow format/truncation/decode errors become an
    explicit per-label unavailable status; unrelated programming errors still propagate."""
    from PIL import Image
    if expected_sha is None:
        return None, "label_unavailable_in_evaluator"
    if not path.is_file() or path.is_symlink():
        return None, "label_file_missing"
    data = _read(path, MAX_LABEL_BYTES, kind)
    digest = sha256_bytes(data)
    read_hashes[_rel(path)] = digest
    if digest != expected_sha:
        return None, "label_hash_mismatch"
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            array = np.asarray(image)
    except LABEL_DECODE_ERRORS as exc:
        return None, f"label_decode_failed:{type(exc).__name__}"
    if array.ndim != 2:
        return None, "label_not_single_channel"
    return array, "ok"


def anchor_visibility(X: np.ndarray, V: np.ndarray, F: np.ndarray, face_id: int, K: np.ndarray,
                      T_ref: np.ndarray | None, mask: np.ndarray | None, mask_status: str,
                      depth: np.ndarray | None, depth_status: str, depth_scale: float, width: int, height: int) -> dict:
    """Reference-only visibility of the anchor (computed before any prediction is considered)."""
    row: dict[str, Any] = {"status": "not_visible", "reason": None, "pixel": None, "anchor_depth_mm": None,
                           "measured_depth_mm": None, "mask_status": mask_status, "depth_status": depth_status}
    if T_ref is None:
        row["reason"] = "invalid_reference_pose"
        return row
    uv, z = project(X, K, T_ref)
    if not z[0] > 0:
        row["reason"] = "reference_behind_camera"
        return row
    if not in_image(uv, width, height)[0]:
        row["reason"] = "reference_out_of_image"
        return row
    px, py = int(np.floor(uv[0, 0] + 0.5)), int(np.floor(uv[0, 1] + 0.5))
    row["pixel"], row["anchor_depth_mm"] = [px, py], float(z[0] * 1000.0)
    if first_hit_occludes(V, F, T_ref, X, own_face=face_id):
        row["reason"] = "self_occluded_on_source_mesh"
        return row
    if mask is None:
        row.update(status="unknown", reason=f"mask_{mask_status}")
        return row
    if mask.shape != (height, width):
        row.update(status="unknown", reason="mask_shape_mismatch")
        return row
    if not mask[py, px] > 0:
        row["reason"] = "outside_bop_visible_mask"
        return row
    if depth is None or depth.shape != (height, width):
        row.update(status="unknown", reason=f"depth_{depth_status if depth is None else 'shape_mismatch'}")
        return row
    raw = float(depth[py, px])
    if raw <= 0:
        row.update(status="unknown", reason="depth_zero")
        return row
    measured = raw * float(depth_scale)
    row["measured_depth_mm"] = measured
    if not depth_consistent(measured, row["anchor_depth_mm"]):
        row["reason"] = "depth_inconsistent"
        return row
    row.update(status="visible", reason=None)
    return row


def evaluate(plan_path: Path, expected_plan_sha256: str, seal_path: Path, out: Path) -> dict:
    out = check_out(out)
    plan, plan_dir = load_plan(Path(plan_path), expected_plan_sha256)
    plan_sha = expected_plan_sha256
    seal, seal_sha = validate_seal(Path(seal_path), plan, plan_sha)
    before = input_closure(plan, plan_dir)
    expected = expected_closure(plan, plan_dir, plan_sha)
    drift = sorted(k for k in expected if before.get(k) != expected[k])
    if drift:
        raise EvalError(f"inputs changed since the plan; refusing to read reference labels: {drift[:6]}")
    runtime = runtime_identity()
    if runtime != plan["runtime"]:
        raise EvalError(f"runtime differs from the frozen plan; refusing to read reference labels: "
                        f"{sorted(k for k in runtime if runtime.get(k) != plan['runtime'].get(k))}")
    _budget(MAX_OUTPUT_BYTES, out)
    stage = "read_evaluator"
    read_hashes: dict[str, str] = {}
    try:
        result = _evaluate_after_seal(plan, plan_dir, read_hashes)
        stage = "post_read_rehash"
        after = input_closure(plan, plan_dir)
        relabel = {k: sha256_file(ROOT / k) for k in read_hashes}
        seal_after = sha256_file(Path(seal_path))
        if after != before or relabel != read_hashes or seal_after != seal_sha or runtime_identity() != runtime:
            raise EvalError("an input, reference label, the terminal seal or the runtime changed during evaluation")
        document = {"schema": SCHEMA_OUT, "tool_sha256": plan["tool_sha256"], "plan_sha256": plan_sha,
                    "terminal_seal_sha256": seal_sha, "flags": FLAGS, "limitations": LIMITATIONS,
                    "metric_spec": METRIC_SPEC, "pins": plan["pins"], "runtime": runtime, "sources": plan["sources"],
                    "hash_closure": {"before_read": before, "after_read": after, "labels_read": read_hashes,
                                     "terminal_seal": {"before": seal_sha, "after": seal_after},
                                     "equal": True},
                    "objects": result}
        blob = _dump(document)
        stage = "publish"
        _budget(len(blob), out)
        publish(out, {"dataset-reference-projection.json": blob})
        return {"directory": out.as_posix(), "sha256": sha256_bytes(blob)}
    except BaseException as exc:
        failure = {"schema": SCHEMA_FAILURE, "status": "failed", "stage": stage, "plan_sha256": plan_sha,
                   "error": f"{type(exc).__name__}: {exc}"[:2000], "labels_read": sorted(read_hashes),
                   "note": "no diagnostic emitted; nothing else was written"}
        try:
            if not os.path.lexists(out):
                publish(out, {"failure.json": _dump(failure)})
        except Exception:  # noqa: BLE001 - never hide the original failure
            pass
        raise


def join_eval_frames(eval_frames: Sequence[Mapping], plan_frames: Sequence[Mapping], object_id: int) -> list:
    """Key-join evaluator frames to the frozen plan order; reordering is tolerated, duplicate,
    missing or unscheduled frame identities are rejected."""
    by_id: dict[int, Mapping] = {}
    for f in eval_frames:
        fid = int(f["frameId"])
        if fid in by_id:
            raise EvalError(f"object {object_id}: evaluator frame {fid} is duplicated")
        by_id[fid] = f
    if set(by_id) != {int(f["frameId"]) for f in plan_frames}:
        raise EvalError(f"object {object_id}: evaluator frames are missing or unscheduled")
    return [by_id[int(f["frameId"])] for f in plan_frames]


def _evaluate_after_seal(plan: Mapping, plan_dir: Path, read_hashes: dict) -> dict:
    raw = _read(EVALUATOR_PATH, MAX_JSON_BYTES, "evaluator.json")
    if sha256_bytes(raw) != EVALUATOR_SHA256:
        raise EvalError("evaluator.json changed")
    read_hashes[_rel(EVALUATOR_PATH)] = EVALUATOR_SHA256
    evaluator = _loads(raw, "evaluator.json")
    if "not independent human truth" not in str(evaluator.get("labelKind")):
        raise EvalError("evaluator label kind is not the expected dataset-reference kind")
    results = {}
    for key, obj in plan["objects"].items():
        object_id = obj["object_id"]
        eval_objs = [o for o in evaluator.get("objects", []) if o.get("objectId") == object_id]
        if len(eval_objs) != 1 or eval_objs[0]["model"]["sha256"] != obj["ply_sha256"]:
            raise EvalError(f"object {object_id}: evaluator object/model binding differs")
        eval_frames = join_eval_frames(eval_objs[0]["frames"], obj["frames"], object_id)
        run = obj["run"]
        run_dir = WORKSPACE / "runs" / run["run_id"]
        rows = [wc.FrameResult.from_dict(wc.loads_strict(line)).to_dict()
                for line in (run_dir / "results.jsonl").read_bytes().splitlines() if line.strip()]
        table = join_role_rows(rows, run["scheduled"])
        V, F = _load_mesh(ROOT / run["control_dir"] / "model.glb")
        X = np.asarray(obj["anchor_m"], dtype=np.float64)
        samples = _loads((plan_dir / obj["samples_file"]).read_bytes(), "samples")
        P = np.asarray(samples["positions_m"], dtype=np.float64)
        if sha256_bytes(np.ascontiguousarray(P, "<f8").tobytes()) != obj["sample_hashes"]["positions_m_float64"]:
            raise EvalError(f"object {object_id}: frozen sample positions differ")
        manifest = _loads((ROOT / run["mask_set"]["manifest"]).read_bytes(), "mask manifest")
        W, H = obj["image_size"]
        per_role = {role: {"A": [], "B": [], "D": []} for role in ROLES}
        visibility, overlaps = [], []
        for plan_frame, ef in zip(obj["frames"], eval_frames):
            fid = plan_frame["frame_id"]
            if (ef["rgb"]["sha256"] != plan_frame["rgb_sha256"]
                    or [float(v) for v in ef["calibration"]["cam_K"]] != plan_frame["cam_K"]
                    or float(ef["calibration"]["depth_scale"]) != plan_frame["depth_scale"]):
                raise EvalError(f"object {object_id} {fid}: evaluator RGB/calibration differs from inference")
            K = np.asarray(plan_frame["cam_K"], dtype=np.float64).reshape(3, 3)
            pose = ef.get("evaluationPose") or {}
            T_ref = bop_pose_m(pose.get("cam_R_m2c", []), pose.get("cam_t_m2c", [])) \
                if pose.get("obj_id") == object_id else None
            mref, dref = ef.get("evaluationMask") or {}, ef.get("depth") or {}
            mask, mask_status = _decode_label(_safe_member(DATASET_ROOT, mref.get("path") or "missing", "mask"),
                                              mref.get("sha256") if mref.get("available") else None, "mask", read_hashes)
            depth, depth_status = _decode_label(_safe_member(DATASET_ROOT, dref.get("path") or "missing", "depth"),
                                                dref.get("sha256") if dref.get("available") else None, "depth",
                                                read_hashes)
            vis = anchor_visibility(X, V, F, run["artwork"]["face_id"], K, T_ref, mask, mask_status, depth,
                                    depth_status, plan_frame["depth_scale"], W, H)
            visibility.append({"frame_id": fid, **vis})
            entry = manifest.get("masks", {}).get(fid)
            sam, sam_status = None, "sam_mask_missing_from_manifest"
            if isinstance(entry, dict):
                sam, sam_status = _decode_label(WORKSPACE / "masks" / run["clip_id"] / MASK_SET / entry["path"],
                                                entry.get("sha256"), "SAM mask", read_hashes)
            overlaps.append({"frame_id": fid, **mask_overlap(sam, mask), "reference_mask_status": mask_status,
                             "sam_mask_status": sam_status})
            for role in ROLES:
                row = table[role].get(fid)
                a = anchor_frame(X, K, T_ref, row, W, H)
                per_role[role]["A"].append({"frame_id": fid, **a, "reference_visible": vis["status"]})
                per_role[role]["B"].append({"frame_id": fid, **samples_frame(P, K, T_ref, row, H)})
                stats = (row.get("evidence") or {}).get("render_stats") or {}
                per_role[role]["D"].append({"frame_id": fid, "pose_state": row["pose_state"],
                                            "tracking_pose": row["pose"] is not None,
                                            "render_state": row["render_state"],
                                            "rendered_pixels": stats.get("rendered_pixels")})
        results[key] = {"object_id": object_id, "run_id": run["run_id"], "frames": len(obj["frames"]),
                        "anchor_visibility": visibility,
                        "visibility_counts": _counts(v["status"] for v in visibility),
                        "visibility_reasons": _counts(v["reason"] for v in visibility if v["reason"]),
                        "mask_overlap_sam_vs_dataset": overlaps,
                        "mask_overlap_summary": {"iou_median": percentile([o["iou"] for o in overlaps], 50),
                                                 "leakage_median": percentile(
                                                     [o["sam_outside_reference_fraction"] for o in overlaps], 50),
                                                 "frames_with_labels": sum(o["status"] == "ok" for o in overlaps)},
                        "roles": {role: _role_summary(per_role[role], visibility) for role in ROLES},
                        "paired_anchor": _paired(per_role)}
    return results


def _counts(items) -> dict:
    out: dict[str, int] = {}
    for item in items:
        out[str(item)] = out.get(str(item), 0) + 1
    return dict(sorted(out.items()))


def _role_summary(rows: Mapping[str, list], visibility: Sequence[Mapping]) -> dict:
    A = rows["A"]
    eligible = [r for r in A if r["reference_status"] == "eligible"]
    visible_ids = {v["frame_id"] for v in visibility if v["status"] == "visible"}
    visible = [r for r in eligible if r["frame_id"] in visible_ids]
    B_rows = rows["B"]
    B_ref = [r for r in B_rows if r["reference_eligible_points"] > 0]
    D = rows["D"]
    rendered_px = [r["rendered_pixels"] for r in D if isinstance(r["rendered_pixels"], (int, float))]
    return {
        "A_anchor_rows": A,
        "A_anchor": error_summary([r["error_px720"] for r in eligible], len(eligible)),
        "A_reference_status_counts": _counts(r["reference_status"] for r in A),
        "A_prediction_status_counts": _counts(r["prediction_status"] for r in A),
        "C_anchor_reference_visible": error_summary([r["error_px720"] for r in visible], len(visible)),
        "B_cad_sample_rows": B_rows,
        "B_cad_samples": {
            "frames": len(B_rows), "frames_with_reference_points": len(B_ref),
            "status_counts": _counts(r["status"] for r in B_rows),
            "frames_complete": sum(r["status"] == "complete" for r in B_ref),
            "frames_partial": sum(r["status"] == "partial" for r in B_ref),
            "frames_without_paired_points": sum(r["status"] in ("missing_prediction", "no_paired_points") for r in B_ref),
            "median_paired_fraction": percentile([r["paired_fraction"] for r in B_ref], 50),
            **{f"failure_aware_coverage_le_{int(t)}px720_median_over_frames":
               percentile([r[f"coverage_le_{int(t)}px720"] for r in B_ref], 50) for t in COVERAGE_THRESHOLDS_PX720},
            **{f"failure_aware_coverage_le_{int(t)}px720_min_over_frames":
               (min(r[f"coverage_le_{int(t)}px720"] for r in B_ref) if B_ref else None) for t in COVERAGE_THRESHOLDS_PX720},
            "conditional_median_of_frame_medians_px720": percentile([r["conditional_median_px720"] for r in B_ref], 50),
            "conditional_p95_of_frame_medians_px720": percentile([r["conditional_median_px720"] for r in B_ref], 95),
            "conditional_median_of_frame_p95_px720": percentile([r["conditional_p95_px720"] for r in B_ref], 50),
            "note": ("per-frame statistics, not pooled across points/frames/objects. Coverage uses each frame's "
                     "reference-eligible points as the denominator (missing/invalid/behind-camera predictions count as "
                     "not covered). 'conditional_*' quantiles cover paired points only and are not success on their "
                     "own. Hidden surfaces are included.")},
        "D_availability": {"frames": len(D), "tracking_pose": sum(r["tracking_pose"] for r in D),
                           "render_state_rendered": sum(r["render_state"] == "rendered" for r in D),
                           "rendered_pixels_positive": sum(1 for v in rendered_px if v > 0),
                           "rendered_pixels_median_when_positive": percentile([v for v in rendered_px if v > 0], 50),
                           "rows": D,
                           "note": "availability/display only; pose projections do not prove artwork rendered correctly"},
        "human_metrics": {"wrong_surface_events": None, "recovery_correctness": None, "status": "unverified"},
    }


def _paired(per_role: Mapping[str, Mapping[str, list]]) -> dict:
    a, b = per_role["primary_neural"]["A"], per_role["frozen_baseline"]["A"]
    both = [(x["error_px720"], y["error_px720"]) for x, y in zip(a, b)
            if x["reference_status"] == "eligible" and x["error_px720"] is not None and y["error_px720"] is not None]
    return {"paired_frames": len(both),
            "primary_median_px720": percentile([p for p, _ in both], 50),
            "baseline_median_px720": percentile([q for _, q in both], 50),
            "primary_p95_px720": percentile([p for p, _ in both], 95),
            "baseline_p95_px720": percentile([q for _, q in both], 95),
            "note": "paired subset only: reference-eligible frames with a measured error for both roles"}


# --------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset-reference projection diagnostic (CPU only; two phases).")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="freeze the plan and samples; never reads reference labels")
    s.add_argument("--out", required=True)
    s = sub.add_parser("evaluate", help="requires an external plan pin and the parent's terminal seal")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--terminal-seal", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(Path(args.out))
        else:
            result = evaluate(Path(args.plan), args.expected_plan_sha256, Path(args.terminal_seal), Path(args.out))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (EvalError, wc.ContractError, OSError, KeyError, ValueError, TypeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
