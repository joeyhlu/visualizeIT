"""Qualified-mask replay anchor evaluator (post-run, CPU only, three phases).

This compares the ORIGINAL cracker-box run (``run-20261021T030702-475adc39``, mask set
``sam2-workbench-01``) with the NEW replay (``run-20261021T060541-e8894b79``, mask set
``sam2-opening-box-v1-qualified``). Both use the same head 27e06, the same base f7d1, admission
06, the metric CAD and the artwork. The comparison uses the fixed 2D anchor references *reused*
from the sealed dataset-reference result of the accepted evaluator.

This tool never opens a GLB/mesh, a SAM mask PNG, depth, raw ground truth, sample files or models,
in any phase. Geometry and mask bytes are a PARENT-ATTESTED physical closure: the parent hashes
them and records the hashes in its terminal record. The tool compares those attestations with the
pinned bundle, reviewed map and mask-manifest JSON, and with the mask identities the result rows
actually consumed.

The new replay uses a different (opening-box) prompt and SAM seed, with unchanged weights. This
is a development diagnostic. It makes no isolated-head-improvement, human-accuracy,
phone-quality, orientation, footprint or significance claim, does not pool objects, and never
overwrites the original reports.

Phases:

``prepare``
    Reads only the pinned frozen original plan, pinned config/admission/map/bundle JSON and this
    tool's source/runtime identity. Freezes every expected pin. Reads no new-run outputs and no
    reference content, and computes no metrics.
``seal`` (parent-only helper)
    Requires the parent's typed terminal record (path and SHA). Validates the expected closure,
    the terminal-completed runs, the consumed masks, the run identities and the parent
    attestations, and then binds them. It never claims termination by itself.
``evaluate``
    Re-authenticates the plan, seal, terminal record, closure and runs, and preflights the output
    budget BEFORE parsing the sealed reference. Reproduces all 220 original rows and summaries
    exactly (the mandatory control) before any new metric. Rechecks everything before publishing.

Projection semantics follow the pinned accepted evaluator source (``d62d47b8…``) but are
implemented independently here. That module is never imported, edited or monkeypatched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import secrets
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_PLAN = "workbench_mask_replay_plan_v2"
SCHEMA_SEAL = "workbench_mask_replay_seal_v2"
SCHEMA_TERMINAL = "workbench_mask_replay_terminal_record_v2"
SCHEMA_RESULT = "workbench_mask_replay_anchor_evaluation_v2"
SCHEMA_FAILURE = "workbench_mask_replay_failure_v2"
TOOL_VERSION = "2"

WORKSPACE = ROOT / ".cache/video-workbench"
DERIVED_ROOT = WORKSPACE / "derived"
RUNS_ROOT = WORKSPACE / "runs"
MASKS_ROOT = WORKSPACE / "masks"
ACCEPTED_EVALUATOR = ROOT / "tools/workbench_ycb_dataset_evaluate.py"
ACCEPTED_EVALUATOR_SHA256 = "d62d47b81ead461567699ff41be89a913634a9e717cf7a44ed1a0ca2ca31c538"
FROZEN_PLAN = DERIVED_ROOT / "dataset-reference/plan-v1/plan.json"
FROZEN_PLAN_SHA256 = "3c85e9c0e6f4b7b7414fec200a00a6071c4406f64e2e9bfa76f5f4e4b6de92f6"
SEALED_RESULT = DERIVED_ROOT / "dataset-reference/eval-v1/dataset-reference-projection.json"
SEALED_RESULT_SHA256 = "83bccd35f98f70a46730d99bb15ec33883e6e6f3d421f13dc5c9410a451250b1"
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
ADMISSION_CONTRACT_SHA256 = "83c041f9198161ff3ba3d5f6c477a3e7955eb24c20e451c014e62e01dced6cf0"
ADMISSION_PARITY_RECEIPT_SHA256 = "40746b0d017b545747fa45c269a45fa175270ffb74983b8dfb4ad9a1d1a7a040"
ADMISSION_DIR = WORKSPACE / "admission/attempt-06"
ADMISSION_FILE_PINS = {"contract.json": "8e032e8211c9d7c57e40868ae904948cd153c9966e352bce754dcc43f90dba96",
                       "geometry.json": "4a5b34901e0c2153ef5849adadfd5bf3d358e047ef7d8c2df45e02b65bad03f0",
                       "parity/parity-receipt.json": "8a163299975f7032827162ea391a2f4d1ab3639a780a4990ea583ddcbf10b325"}
ORIGINAL_RUN = "run-20261021T030702-475adc39"
NEW_RUN = "run-20261021T060541-e8894b79"
CLIP = "dev-ycb-003-cracker-box"
RUN_MODE = "regression"
OBJECT_KEY = "2"
SCHEDULE = tuple(f"s{i:06d}" for i in range(371, 481))
ROLES = ("primary_neural", "frozen_baseline")
ORIGIN = "integer_centers"
ORIGINAL_MASK_SET = "sam2-workbench-01"
NEW_MASK_SET = "sam2-opening-box-v1-qualified"
MASK_SCHEMA = "workbench_observed_masks_v1"
NEW_MASK_MANIFEST = MASKS_ROOT / CLIP / NEW_MASK_SET / "manifest.json"
NEW_MASK_MANIFEST_SHA256 = "055c60711a4b04169a108b9fb4dd0e8dac24756a27b0b7ce6e779ba771004057"  # FILE digest
REVIEWED_MAP = ROOT / "tools/workbench_ycb_execution_controls.reviewed.json"
ARTWORK_SCHEMA = "workbench_neural_artwork_v1"
CONFIG_FILES = {
    "launch": (ROOT / ".cache/workbench-cracker-opening-box-neural-evaluation.py",
               "cb9fe577a631682adecda37aa4213bc2029f37adf33e3427dab7263c413b1c63"),
    "registration": (DERIVED_ROOT / "validation/ycb-cracker-opening-box-registration-v1.json",
                     "b0be3906ad8cbd735824546f03f94ec1ab246c86406393a7e7eb65f919577efb"),
    "seed_rule_clarification": (DERIVED_ROOT / "validation/ycb-cracker-opening-box-seed-rule-clarification-v2.json",
                                "d55bbad5de4643e531a464442f44d2b0ae3c5456b4c04fa3787acc07a7baf7ce"),
    "prompt": (DERIVED_ROOT / "validation/ycb-cracker-opening-box-v1.json",
               "15dce4be95eb362c9044f808c872abd810f98f072cc8f9bd96b79439ddd308a5"),
    "selection_receipt": (DERIVED_ROOT / "validation/ycb-reviewed-selections-v1.json",
                          "bd601c76a329452d4ce5f9ac71cfd4f0fa81c8030b028cba74516cf00df36623"),
}
TERMINAL_OBSERVATION = (DERIVED_ROOT / "validation/qualified-mask-pair-terminal-observation-v1.json",
                        "bbca0b3a02e1751b5d8b4dea633ad0eebce851f53b16a61e3c24eb8692010ef7")
HELPER_MODULES = {"bench": "bench/__init__.py", "bench.vision": "bench/vision.py",
                  "bench.workbench_contract": "bench/workbench_contract.py",
                  "bench.workbench_geometry": "bench/workbench_geometry.py",
                  "bench.workbench_cli": "bench/workbench_cli.py"}
INFERENCE_REQUIRED_SOURCES = ("workbench_cli", "workbench_contract", "workbench_geometry", "workbench_neural")
IDENTITY_REQUIRED = ("profile", "base_checkpoint_sha256", "installed_head_sha256", "installed_head_state_sha256",
                     "install_digest", "admission_contract_sha256", "principal_point_origin", "initialization",
                     "seed_frame_id", "seed_target_frame_id", "glb_sha256", "metric_units", "effective_K",
                     "quality_claim")
RECEIPT_SUFFIXES = (".json", ".jsonl", ".txt", ".log")
PX720_HEIGHT = 720.0
COVERAGE_THRESHOLDS_PX720 = (5.0, 10.0)
ROTATION_TOL = 1e-4
CONTROL_ATOL_PX = 1e-8                 # frozen before any comparison; rtol = 0
MAX_COMBINED_BYTES = 4 << 20           # plan + seal + evaluation JSON + CSV (separate from workspace budgets)
EVAL_MIN_RESERVE_BYTES = 1 << 20       # reserved before the sealed reference is read
MAX_JSON_BYTES = 64 << 20
FLAGS = {"development_diagnostic_only": True, "independent_human_accuracy": False, "phone_quality": False,
         "isolated_head_improvement_claimed": False, "orientation_or_footprint_claimed": False,
         "pooled_objects": False, "significance_tested": False, "original_reports_overwritten": False,
         "prompt_and_sam_seed_changed": True, "weights_changed": False,
         "geometry_and_mask_closure": ("parent-attested physical closure: the parent hashed the GLB and every SAM mask "
                                       "PNG; this tool compares those attestations with the pinned bundle/map/manifest "
                                       "JSON and the rows' consumed mask identities, and never opens GLB or mask PNG "
                                       "files (no independent before/after rehash of those bytes)"),
         "inference_source_closure": ("the new run's own report.dependencies module hashes, the launch/admission pins "
                                      "and parent observation receipts, attested by the parent; this tool's helper "
                                      "hashes describe the evaluator only and are not proof of executed source")}


class ReplayError(Exception):
    """Fail-closed: nothing is compared or published beyond a failure receipt."""


# --------------------------------------------------------------------------
# hashing / IO (outside pure math)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return Path(os.path.abspath(path)).relative_to(ROOT.resolve()).as_posix()


def _read(path: Path, limit: int = MAX_JSON_BYTES) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"missing or non-regular file: {path}")
    if path.stat().st_size > limit:
        raise ReplayError(f"{path} exceeds {limit} bytes")
    return path.read_bytes()


def _loads(raw: bytes, what: str) -> Any:
    from bench import workbench_contract as wc
    try:
        return wc.loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReplayError(f"{what}: invalid strict JSON ({exc})") from exc


def _read_pinned_json(path: Path, sha: str, what: str) -> Any:
    raw = _read(path)
    if sha256_bytes(raw) != sha:
        raise ReplayError(f"{what} differs from its pinned SHA-256")
    return _loads(raw, what)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _dump(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


# --------------------------------------------------------------------------
# pure math (independent; semantics of the pinned accepted evaluator source)


def _finite_or_none(value: Any) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def proper_rotation(R: Any, tol: float = ROTATION_TOL) -> bool:
    R = np.asarray(R, dtype=np.float64)
    return (R.shape == (3, 3) and bool(np.isfinite(R).all()) and float(np.abs(R.T @ R - np.eye(3)).max()) <= tol
            and abs(float(np.linalg.det(R)) - 1.0) <= tol)


def validated_prediction_pose(row: Mapping) -> tuple[np.ndarray | None, str]:
    """Legacy contract: tracking + pose present -> metric units -> finite 4x4 with exact homogeneous
    last row and a proper R (tol 1e-4), never repaired."""
    if row.get("pose_state") != "tracking" or row.get("pose") is None:
        return None, "no_tracking_pose"
    if row.get("pose_units") != "metric":
        return None, "invalid_prediction_pose"
    try:
        T = np.asarray(row["pose"], dtype=np.float64)
    except (TypeError, ValueError):
        return None, "invalid_prediction_pose"
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0])
            or not proper_rotation(T[:3, :3])):
        return None, "invalid_prediction_pose"
    return T, "ok"


def _project_one(X: np.ndarray, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, float]:
    cam = T[:3, :3] @ X + T[:3, 3]
    z = float(cam[2])
    if not z > 0:
        return np.array([np.nan, np.nan]), z
    u = K[0, 0] * cam[0] / z + K[0, 1] * cam[1] / z + K[0, 2]
    v = K[1, 1] * cam[1] / z + K[1, 2]
    return np.array([u, v], dtype=np.float64), z


def _in_image(uv: np.ndarray, width: int, height: int) -> bool:
    return bool(np.isfinite(uv).all() and -0.5 <= uv[0] < width - 0.5 and -0.5 <= uv[1] < height - 0.5)


def project_prediction(anchor_m: Sequence[float], K: Any, row: Mapping, image_size: Sequence[int]) -> dict:
    """Prediction status and UV for one FrameResult row (status order as the legacy source)."""
    X = np.asarray(anchor_m, dtype=np.float64).reshape(3)
    Km = np.asarray(K, dtype=np.float64).reshape(3, 3)
    width, height = int(image_size[0]), int(image_size[1])
    T, status = validated_prediction_pose(row)
    uv_out = None
    if T is not None:
        uv, z = _project_one(X, Km, T)
        if not z > 0:
            status = "prediction_behind_camera"
        else:
            uv_out = [_finite_or_none(uv[0]), _finite_or_none(uv[1])]
            if not _in_image(uv, width, height):
                status = "prediction_out_of_image"
    return {"prediction_status": status, "prediction_uv": uv_out}


def score_anchor(reference: Mapping, prediction: Mapping, image_height: int) -> dict:
    """One scored row. Error only for reference-eligible + ok; px720 = native * 720 / H."""
    out = {"reference_status": reference["reference_status"], "reference_uv": reference["reference_uv"],
           "reference_visible": reference["visibility"], "prediction_status": prediction["prediction_status"],
           "prediction_uv": prediction["prediction_uv"], "error_native_px": None, "error_px720": None}
    if out["reference_status"] == "eligible" and out["prediction_status"] == "ok":
        native = float(np.linalg.norm(np.asarray(out["prediction_uv"], dtype=np.float64)
                                      - np.asarray(out["reference_uv"], dtype=np.float64)))
        out["error_native_px"] = native
        out["error_px720"] = native * (PX720_HEIGHT / float(image_height))
    return out


def _percentile(values: Iterable[float], q: float) -> float | None:
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.percentile(data, q)) if data else None


def _counts(items: Iterable[Any]) -> dict:
    out: dict[str, int] = {}
    for item in items:
        out[str(item)] = out.get(str(item), 0) + 1
    return dict(sorted(out.items()))


def summarize_anchor(scored_rows: Sequence[Mapping], reference_subset: Iterable[str]) -> dict:
    """Denominator = reference-eligible frames in ``reference_subset`` (failed predictions stay in it)."""
    subset = set(reference_subset)
    rows = [r for r in scored_rows if r["frame_id"] in subset and r["reference_status"] == "eligible"]
    errors = [r["error_px720"] for r in rows if r["error_px720"] is not None]
    denom = len(rows)
    return {"denominator": denom, "measured": len(errors),
            "conditional_median_px720": _percentile(errors, 50), "conditional_p95_px720": _percentile(errors, 95),
            **{f"coverage_le_{int(t)}px720": (sum(e <= t for e in errors) / denom if denom else None)
               for t in COVERAGE_THRESHOLDS_PX720},
            "prediction_status_counts": _counts(r["prediction_status"] for r in rows),
            "note": "failure-aware coverage over the full denominator; conditional quantiles over measured errors only"}


def summarize_availability(rows: Sequence[Mapping]) -> dict:
    """All scheduled rows of one role. ``validated_tracking_poses`` passes the legacy pose contract;
    ``legacy_pose_present`` is the old D ``tracking_pose`` (pose is not None), kept for the control only."""
    rendered_px = []
    for r in rows:
        stats = (r.get("evidence") or {}).get("render_stats") or {}
        value = stats.get("rendered_pixels")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            rendered_px.append(float(value))
    positive = [v for v in rendered_px if v > 0]
    return {"frames": len(rows), "pose_state_counts": _counts(r.get("pose_state") for r in rows),
            "render_state_counts": _counts(r.get("render_state") for r in rows),
            "validated_tracking_poses": sum(validated_prediction_pose(r)[0] is not None for r in rows),
            "legacy_pose_present": sum(r.get("pose") is not None for r in rows),
            "render_state_rendered": sum(r.get("render_state") == "rendered" for r in rows),
            "rendered_pixels_positive": len(positive),
            "rendered_pixels_median_when_positive": _percentile(positive, 50)}


def validate_schedule(rows: Sequence[Mapping], scheduled: Sequence[str], roles: Sequence[str] = ROLES) -> dict:
    """Exact (role, frame) join. A missing row is a structural integrity failure, never a lost prediction."""
    table: dict[str, dict[str, Mapping]] = {role: {} for role in roles}
    allowed = set(scheduled)
    for row in rows:
        role, fid = row.get("backend_role"), row.get("frame_id")
        if role not in table or fid not in allowed:
            raise ReplayError(f"unscheduled row ({role}, {fid})")
        if fid in table[role]:
            raise ReplayError(f"duplicate row ({role}, {fid})")
        table[role][fid] = row
    for role in roles:
        missing = [f for f in scheduled if f not in table[role]]
        if missing:
            raise ReplayError(f"{role}: missing scheduled rows {missing[:5]}")
    return table


def extract_reference(sealed_result: Mapping, frozen_plan: Mapping) -> dict:
    """Fixed object-2 anchor references and visibility, identical across both original roles."""
    try:
        obj = sealed_result["objects"][OBJECT_KEY]
        scheduled = [f["frame_id"] for f in frozen_plan["objects"][OBJECT_KEY]["frames"]]
    except (KeyError, TypeError) as exc:
        raise ReplayError(f"sealed result or frozen plan lacks object {OBJECT_KEY}") from exc
    if scheduled != list(SCHEDULE):
        raise ReplayError("frozen plan schedule is not s000371..s000480")
    vis_rows = obj.get("anchor_visibility")
    vis = {}
    for v in vis_rows if isinstance(vis_rows, list) else []:
        if v.get("frame_id") in vis:
            raise ReplayError(f"duplicate visibility row {v.get('frame_id')}")
        vis[v.get("frame_id")] = v.get("status")
    if set(vis) != set(scheduled):
        raise ReplayError("visibility rows are missing, extra or unscheduled")
    per_role = {}
    for role in ROLES:
        rows = (obj.get("roles", {}).get(role) or {}).get("A_anchor_rows")
        if not isinstance(rows, list):
            raise ReplayError(f"{role}: A_anchor_rows missing")
        by_id = {}
        for r in rows:
            if r.get("frame_id") in by_id:
                raise ReplayError(f"{role}: duplicate reference row {r.get('frame_id')}")
            by_id[r.get("frame_id")] = r
        if set(by_id) != set(scheduled):
            raise ReplayError(f"{role}: reference rows are missing, extra or unscheduled")
        per_role[role] = by_id
    reference = {}
    for fid in scheduled:
        a, b = per_role[ROLES[0]][fid], per_role[ROLES[1]][fid]
        fields = ("reference_status", "reference_uv", "reference_visible")
        if any(a.get(k) != b.get(k) for k in fields):
            raise ReplayError(f"{fid}: original roles disagree on reference status/UV/visibility")
        if a.get("reference_visible") != vis[fid]:
            raise ReplayError(f"{fid}: reference_visible differs from anchor_visibility")
        if vis[fid] == "visible" and a.get("reference_status") != "eligible":
            raise ReplayError(f"{fid}: visible reference is not eligible")
        uv = a.get("reference_uv")
        if a.get("reference_status") == "eligible" and (not isinstance(uv, list) or len(uv) != 2
                                                        or any(_finite_or_none(x) is None for x in uv)):
            raise ReplayError(f"{fid}: eligible reference has no finite UV")
        reference[fid] = {"reference_status": a["reference_status"], "reference_uv": uv, "visibility": vis[fid]}
    return reference


def score_role(rows_by_frame: Mapping[str, Mapping], reference: Mapping, anchor_m: Sequence[float],
               K_by_frame: Mapping[str, Any], image_size: Sequence[int]) -> list[dict]:
    out = []
    for fid in SCHEDULE:
        pred = project_prediction(anchor_m, K_by_frame[fid], rows_by_frame[fid], image_size)
        out.append({"frame_id": fid, **score_anchor(reference[fid], pred, int(image_size[1]))})
    return out


def _close(a: Any, b: Any, atol: float = CONTROL_ATOL_PX) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) or isinstance(b, list):
        return (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
                and all(_close(x, y, atol) for x, y in zip(a, b)))
    return abs(float(a) - float(b)) <= atol


def verify_original_control(original_rows: Mapping[str, Mapping[str, Mapping]], reference: Mapping,
                            sealed_result: Mapping, *, anchor_m: Sequence[float], K_by_frame: Mapping[str, Any],
                            image_size: Sequence[int]) -> dict:
    """Reproduce all 220 original rows and the legacy summaries (rtol 0, atol 1e-8 px; exact statuses,
    null patterns, counts and <=5/<=10 membership). Any mismatch means no comparison."""
    mismatches: list[str] = []
    scores = {}
    obj = sealed_result["objects"][OBJECT_KEY]
    for role in ROLES:
        scored = score_role(original_rows[role], reference, anchor_m, K_by_frame, image_size)
        scores[role] = scored
        sealed = {r["frame_id"]: r for r in obj["roles"][role]["A_anchor_rows"]}
        for row in scored:
            s = sealed[row["frame_id"]]
            if row["prediction_status"] != s.get("prediction_status"):
                mismatches.append(f"{role} {row['frame_id']} prediction_status {row['prediction_status']} != {s.get('prediction_status')}")
            for key in ("prediction_uv", "error_native_px", "error_px720"):
                if not _close(row[key], s.get(key)):
                    mismatches.append(f"{role} {row['frame_id']} {key}")
            for t in COVERAGE_THRESHOLDS_PX720:
                mine = row["error_px720"] is not None and row["error_px720"] <= t
                theirs = s.get("error_px720") is not None and s["error_px720"] <= t
                if mine != theirs:
                    mismatches.append(f"{role} {row['frame_id']} <= {t} membership")
        summary = obj["roles"][role]
        eligible = [r["frame_id"] for r in scored if r["reference_status"] == "eligible"]
        visible = [r["frame_id"] for r in scored if r["reference_status"] == "eligible" and r["reference_visible"] == "visible"]
        for key, ids in (("A_anchor", eligible), ("C_anchor_reference_visible", visible)):
            mine = summarize_anchor(scored, ids)
            theirs = summary.get(key) or {}
            if (mine["denominator"] != theirs.get("reference_eligible") or mine["measured"] != theirs.get("errors_measured")
                    or not _close(mine["conditional_median_px720"], theirs.get("median_px720"))
                    or not _close(mine["conditional_p95_px720"], theirs.get("p95_px720"))
                    or any(mine[f"coverage_le_{int(t)}px720"] != theirs.get(f"coverage_le_{int(t)}px720")
                           for t in COVERAGE_THRESHOLDS_PX720)):
                mismatches.append(f"{role} {key} summary")
        if _counts(r["prediction_status"] for r in scored) != summary.get("A_prediction_status_counts"):
            mismatches.append(f"{role} A_prediction_status_counts")
        if _counts(r["reference_status"] for r in scored) != summary.get("A_reference_status_counts"):
            mismatches.append(f"{role} A_reference_status_counts")
        avail = summarize_availability([original_rows[role][f] for f in SCHEDULE])
        legacy = summary.get("D_availability") or {}
        if (avail["legacy_pose_present"] != legacy.get("tracking_pose")
                or avail["render_state_rendered"] != legacy.get("render_state_rendered")
                or avail["rendered_pixels_positive"] != legacy.get("rendered_pixels_positive")
                or not _close(avail["rendered_pixels_median_when_positive"], legacy.get("rendered_pixels_median_when_positive"))
                or avail["frames"] != legacy.get("frames")):
            mismatches.append(f"{role} legacy D_availability counts")
    return {"passed": not mismatches, "mismatches": mismatches[:200], "mismatch_count": len(mismatches),
            "scores": scores, "tolerance": {"rtol": 0.0, "atol_px": CONTROL_ATOL_PX}}


def compare_replays(original_scores: Mapping[str, Sequence[Mapping]], new_scores: Mapping[str, Sequence[Mapping]]) -> dict:
    """Original -> new per role (no pooling). Paired conditional rows expose their own denominator and
    never replace the failure-aware coverage."""
    out = {}
    for role in ROLES:
        old = {r["frame_id"]: r for r in original_scores[role]}
        new = {r["frame_id"]: r for r in new_scores[role]}
        eligible = [f for f in SCHEDULE if old[f]["reference_status"] == "eligible"]
        visible = [f for f in eligible if old[f]["reference_visible"] == "visible"]
        cells = {}
        for name, ids in (("all_reference_eligible", eligible), ("fixed_visible", visible)):
            a, b = summarize_anchor(original_scores[role], ids), summarize_anchor(new_scores[role], ids)
            paired = [f for f in ids if old[f]["error_px720"] is not None and new[f]["error_px720"] is not None]
            diffs = [new[f]["error_px720"] - old[f]["error_px720"] for f in paired]
            cells[name] = {
                "denominator": a["denominator"],
                **{f"coverage_le_{int(t)}px720_change": (None if a[f"coverage_le_{int(t)}px720"] is None
                                                         else b[f"coverage_le_{int(t)}px720"] - a[f"coverage_le_{int(t)}px720"])
                   for t in COVERAGE_THRESHOLDS_PX720},
                "measured_original": a["measured"], "measured_new": b["measured"],
                "paired_conditional": {"paired_frames": len(paired),
                                       "median_original_px720": _percentile([old[f]["error_px720"] for f in paired], 50),
                                       "median_new_px720": _percentile([new[f]["error_px720"] for f in paired], 50),
                                       "median_change_px720": _percentile(diffs, 50),
                                       "note": "paired-measured subset only; does not replace failure-aware coverage"},
                "status_transitions": _counts(f"{old[f]['prediction_status']}->{new[f]['prediction_status']}" for f in ids)}
        out[role] = cells
    return out


def decode_pixel_sha256(raw: bytes, decode: Callable[[bytes], np.ndarray] | None = None) -> str:
    """Workbench FrameReader convention: cv2.imdecode(IMREAD_COLOR) -> contiguous BGR->RGB ->
    workbench_geometry.array_sha256. ``decode`` may be injected for tests."""
    from bench import workbench_geometry as wg
    if decode is None:
        from bench.vision import cv2
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    else:
        bgr = decode(raw)
    if bgr is None:
        raise ReplayError("RGB cannot be decoded")
    return wg.array_sha256(np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1]))


# --------------------------------------------------------------------------
# small typed helpers


_HEX64 = re.compile(r"[0-9a-f]{64}")


def _hex(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_keys(value: Any, keys: set, what: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReplayError(f"{what} must be an object with exactly the keys {sorted(keys)}")
    return value


def _safe_rel(rel: Any, what: str) -> Path:
    """A repository-relative POSIX path without traversal or symlinks."""
    if not isinstance(rel, str) or not rel or "\\" in rel or rel.startswith("/") or ":" in rel:
        raise ReplayError(f"{what}: unsafe path {rel!r}")
    parts = PurePosixPath(rel).parts
    if any(p in ("", ".", "..") for p in parts):
        raise ReplayError(f"{what}: unsafe path {rel!r}")
    current = ROOT
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ReplayError(f"{what}: path traverses a symlink")
    return current


def _guard(fn: Callable[[], Any], what: str) -> Any:
    """Malformed structures (missing keys / wrong types) are integrity failures, never defaults."""
    try:
        return fn()
    except ReplayError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise ReplayError(f"{what}: malformed ({type(exc).__name__}: {exc})") from exc


# --------------------------------------------------------------------------
# closure helpers (file hashes only; GLB and mask PNG files are never opened)


def load_helpers() -> dict:
    import importlib
    modules = {name: importlib.import_module(name) for name in HELPER_MODULES}
    for name, rel in HELPER_MODULES.items():
        if Path(modules[name].__file__).resolve() != (ROOT / rel).resolve():
            raise ReplayError(f"helper {name} imported from {modules[name].__file__}, not {rel}")
    return modules


def source_closure() -> dict[str, str]:
    """This evaluator's own source closure (not the inference source)."""
    load_helpers()
    files = {rel: sha256_file(ROOT / rel) for rel in HELPER_MODULES.values()}
    files[_rel(Path(__file__))] = sha256_file(Path(__file__))
    files[_rel(ACCEPTED_EVALUATOR)] = sha256_file(ACCEPTED_EVALUATOR)
    return dict(sorted(files.items()))


def runtime_identity() -> dict:
    load_helpers()
    from bench.vision import cv2
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "opencv": cv2.__version__,
            "opencv_module": str(Path(cv2.__file__).resolve())}


def _run_files(run_id: str) -> dict[str, str]:
    """JSON run inventory (same file set as the accepted evaluator); overlay PNGs are not opened."""
    run_dir = RUNS_ROOT / run_id
    names = ["run.json", "results.jsonl", "report.json", "neural-artwork.json"]
    stages = sorted(p.name for p in (run_dir / "stages").iterdir()) if (run_dir / "stages").is_dir() else []
    files = {}
    for name in names + [f"stages/{s}" for s in stages]:
        path = run_dir.joinpath(*name.split("/"))
        if path.is_symlink() or not path.is_file():
            raise ReplayError(f"{run_id}: {name} missing")
        files[name] = sha256_file(path)
    return files


def _error_receipts(run_id: str) -> list[str]:
    errors = RUNS_ROOT / run_id / "errors"
    return sorted(p.name for p in errors.iterdir()) if errors.is_dir() else []


def _load_rows(run_id: str) -> list[dict]:
    from bench import workbench_contract as wc
    raw = _read(RUNS_ROOT / run_id / "results.jsonl")
    try:
        return [wc.FrameResult.from_dict(_loads(line, "results row")).to_dict() for line in raw.splitlines() if line.strip()]
    except ValueError as exc:
        raise ReplayError(f"{run_id}: results row violates the FrameResult contract ({exc})") from exc


def hash_receipts(receipts: Sequence[Mapping]) -> dict[str, str]:
    """Rehash the parent's observation receipts (small text/JSON files inside the workspace only)."""
    out: dict[str, str] = {}
    forbidden = [MASKS_ROOT.resolve(), (WORKSPACE / "models").resolve(), (DERIVED_ROOT / "dataset-reference").resolve()]
    for receipt in receipts:
        path = _safe_rel(receipt["path"], "observation receipt")
        resolved = path.resolve()
        if (WORKSPACE.resolve() not in resolved.parents or any(f == resolved or f in resolved.parents for f in forbidden)
                or path.suffix.lower() not in RECEIPT_SUFFIXES):
            raise ReplayError(f"observation receipt {receipt['path']} is outside the allowed receipt locations/types")
        if path.is_symlink() or not path.is_file():
            raise ReplayError(f"observation receipt {receipt['path']} is missing")
        digest = sha256_file(path)
        if digest != receipt["sha256"]:
            raise ReplayError(f"observation receipt {receipt['path']} differs from its attested SHA-256")
        out[receipt["path"]] = digest
    if out.get(_rel(TERMINAL_OBSERVATION[0])) != TERMINAL_OBSERVATION[1]:
        raise ReplayError("the pinned parent terminal observation receipt is not attested")
    return dict(sorted(out.items()))


def closure_snapshot(plan: Mapping, record: Mapping | None = None) -> dict:
    """Fresh file hashes of every evaluation input (excluding the sealed reference, hashed separately).
    GLB and mask PNG files are never opened; their closure is the parent attestation."""
    def build() -> dict:
        control = _safe_rel(plan["geometry"]["control_dir"], "control directory")
        snap = {
            "frozen_plan_sha256": sha256_file(FROZEN_PLAN),
            "accepted_evaluator_sha256": sha256_file(ACCEPTED_EVALUATOR),
            "original_run_files": _run_files(ORIGINAL_RUN), "new_run_files": _run_files(NEW_RUN),
            "new_run_error_receipts": _error_receipts(NEW_RUN),
            "original_mask_manifest_sha256": sha256_file(_safe_rel(plan["original_run"]["mask_manifest"], "manifest")),
            "new_mask_manifest_sha256": sha256_file(_safe_rel(plan["new_run"]["mask_manifest"], "manifest")),
            "configs": {name: sha256_file(path) for name, (path, _sha) in CONFIG_FILES.items()},
            "admission": {name: sha256_file(ADMISSION_DIR.joinpath(*name.split("/"))) for name in ADMISSION_FILE_PINS},
            "reviewed_map_sha256": sha256_file(REVIEWED_MAP),
            "clip_sha256": sha256_file(_safe_rel(plan["original_run"]["clip"]["path"], "clip")),
            "control_json": {name: sha256_file(control / name)
                             for name in ("metric-bundle.json", "execution-control-sidecar.json")},
            "rgb_files": {b["path"]: sha256_file(_safe_rel(b["path"], "RGB")) for b in plan["rgb"].values()},
            "terminal_observation_sha256": sha256_file(TERMINAL_OBSERVATION[0]),
            "sources": source_closure(), "runtime": runtime_identity(),
        }
        if record is not None:
            snap["attested_receipts"] = hash_receipts(record["attested"]["inference_execution"]["observation_receipts"])
        return _jsonable(snap)
    try:
        return _guard(build, "closure snapshot")
    except OSError as exc:
        raise ReplayError(f"closure input missing or unreadable: {exc}") from exc


def validate_expected_closure(plan: Mapping, snapshot: Mapping) -> None:
    """Every observed digest must equal the prospective plan AND the tool's fixed constants."""
    def check() -> list[str]:
        p: list[str] = []
        pins = plan["pins"]
        if not (snapshot["frozen_plan_sha256"] == FROZEN_PLAN_SHA256 == pins["frozen_plan_sha256"]):
            p.append("frozen plan")
        if not (snapshot["accepted_evaluator_sha256"] == ACCEPTED_EVALUATOR_SHA256 == pins["accepted_evaluator_sha256"]):
            p.append("accepted evaluator source")
        if snapshot["original_run_files"] != plan["original_run"]["files"]:
            p.append("original run inventory")
        if snapshot["new_run_error_receipts"] != []:
            p.append("new run has error receipts")
        if snapshot["original_mask_manifest_sha256"] != plan["original_run"]["mask_manifest_file_sha256"]:
            p.append("original mask manifest file")
        if not (snapshot["new_mask_manifest_sha256"] == NEW_MASK_MANIFEST_SHA256
                == plan["new_run"]["mask_manifest_file_sha256"] == pins["new_mask_manifest_sha256"]):
            p.append("new mask manifest file")
        fixed_configs = {k: sha for k, (_path, sha) in CONFIG_FILES.items()}
        if not (snapshot["configs"] == fixed_configs == pins["configs"]):
            p.append("config digests")
        if not (snapshot["admission"] == ADMISSION_FILE_PINS == plan["admission"]["file_pins"]):
            p.append("admission files")
        if snapshot["reviewed_map_sha256"] != plan["geometry"]["reviewed_map_sha256"]:
            p.append("reviewed control map")
        if snapshot["clip_sha256"] != plan["original_run"]["clip"]["sha256"]:
            p.append("clip")
        if snapshot["control_json"] != {"metric-bundle.json": plan["geometry"]["bundle_sha256"],
                                        "execution-control-sidecar.json": plan["original_run"]["sidecar_sha256"]}:
            p.append("metric bundle / control sidecar")
        if (set(plan["rgb"]) != set(SCHEDULE)
                or snapshot["rgb_files"] != {b["path"]: b["file_sha256"] for b in plan["rgb"].values()}):
            p.append("RGB file bindings")
        if not (snapshot["terminal_observation_sha256"] == TERMINAL_OBSERVATION[1] == plan["terminal_observation"]["sha256"]):
            p.append("parent terminal observation receipt")
        if snapshot["sources"] != plan["sources"] or snapshot["runtime"] != plan["runtime"]:
            p.append("evaluator sources/runtime")
        if "attested_receipts" in snapshot and snapshot["attested_receipts"].get(_rel(TERMINAL_OBSERVATION[0])) != TERMINAL_OBSERVATION[1]:
            p.append("attested observation receipt")
        return p
    problems = _guard(check, "expected closure")
    if problems:
        raise ReplayError("expected closure differs: " + "; ".join(problems))


def authenticate_static_inputs(plan: Mapping) -> dict:
    """Pinned small JSON only: metric bundle, reviewed map entry, admission contract/parity receipt."""
    from bench import workbench_contract as wc

    def check() -> dict:
        geometry = plan["geometry"]
        control = _safe_rel(geometry["control_dir"], "control directory")
        bundle = _read_pinned_json(control / "metric-bundle.json", geometry["bundle_sha256"], "metric bundle")
        if bundle.get("glb_sha256") != geometry["candidate_glb_sha256"] or bundle.get("principal_point_origin") != ORIGIN:
            raise ReplayError("metric bundle does not bind the pinned GLB/origin")
        reviewed = _read_pinned_json(REVIEWED_MAP, geometry["reviewed_map_sha256"], "reviewed control map")
        entry = reviewed["entries"][OBJECT_KEY]
        if (sha256_bytes(wc.canonical_json(entry)) != geometry["map_entry_sha256"]
                or entry.get("candidate_glb_sha256") != geometry["candidate_glb_sha256"]
                or entry.get("bundle_sha256") != geometry["bundle_sha256"] or entry.get("planned_out") != geometry["control_dir"]):
            raise ReplayError("reviewed map entry differs from the frozen original plan")
        contract = _read_pinned_json(ADMISSION_DIR / "contract.json", ADMISSION_FILE_PINS["contract.json"], "admission contract")
        parity = _read_pinned_json(ADMISSION_DIR / "parity/parity-receipt.json",
                                   ADMISSION_FILE_PINS["parity/parity-receipt.json"], "parity receipt")
        if contract.get("contract_sha256") != ADMISSION_CONTRACT_SHA256 or parity.get("receipt_sha256") != ADMISSION_PARITY_RECEIPT_SHA256:
            raise ReplayError("admission contract/parity semantic digests differ from the pins")
        return {"bundle_glb_sha256": bundle["glb_sha256"], "map_entry_sha256": geometry["map_entry_sha256"],
                "admission_contract_sha256": ADMISSION_CONTRACT_SHA256,
                "parity_receipt_sha256": ADMISSION_PARITY_RECEIPT_SHA256}
    return _guard(check, "static inputs")


def mask_inventory_digest(manifest: Mapping) -> dict:
    """Canonical digest of the manifest's complete physical inventory [frame, path, file sha, content sha]."""
    from bench import workbench_contract as wc
    masks = manifest.get("masks") if isinstance(manifest, Mapping) else None
    if not isinstance(masks, Mapping) or not masks:
        raise ReplayError("mask manifest has no mask inventory")
    rows = []
    for fid in sorted(masks):
        entry = masks[fid]
        if (not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not _hex(entry.get("sha256"))
                or not _hex(entry.get("content_sha256"))):
            raise ReplayError(f"mask manifest entry {fid} lacks path/sha256/content_sha256")
        if "/" in entry["path"] or "\\" in entry["path"] or entry["path"] in ("", ".", ".."):
            raise ReplayError(f"mask manifest entry {fid} has an unsafe path")
        rows.append([fid, entry["path"], entry["sha256"], entry["content_sha256"]])
    return {"frames": len(rows), "inventory_sha256": wc.canonical_sha256(rows)}


def validate_mask_consumption(rows_by_role: Mapping[str, Mapping[str, Mapping]], manifest: Mapping, stage: Any, *,
                              mask_set: str, clip_id: str = CLIP) -> dict:
    """The masks the rows ACTUALLY consumed: completed mask stage bound to this manifest's canonical
    identity, and every observed row's (mask_frame_id, mask_sha256) equal to (frame, manifest
    content_sha256). Non-observed rows keep their explicit state with no mask identity."""
    from bench import workbench_contract as wc

    def check() -> dict:
        if (manifest.get("schema") != MASK_SCHEMA or manifest.get("clip_id") != clip_id
                or manifest.get("mask_set") != mask_set or manifest.get("evaluator_inputs_read") is not False
                or not isinstance(manifest.get("masks"), Mapping)):
            raise ReplayError(f"{mask_set}: manifest schema/clip/mask set/evaluator flag differs")
        canonical = wc.canonical_sha256(manifest)
        identity = stage["identity"]
        if (stage["status"] != "completed" or identity["mask_set"] != mask_set or identity["manifest_sha256"] != canonical
                or identity.get("producer") != manifest.get("producer")):
            raise ReplayError(f"{mask_set}: run mask stage is not completed with this manifest's canonical identity")
        counts = {}
        for role in ROLES:
            states: dict[str, int] = {}
            for fid in SCHEDULE:
                row = rows_by_role[role][fid]
                state = row["mask_state"]
                if state == "observed":
                    entry = manifest["masks"].get(fid)
                    if (not isinstance(entry, Mapping) or not _hex(entry.get("content_sha256"))
                            or row["mask_frame_id"] != fid or row["mask_sha256"] != entry["content_sha256"]):
                        raise ReplayError(f"{mask_set} {role} {fid}: consumed mask identity differs from the manifest")
                elif state in ("unavailable", "rejected", "blocked"):
                    if row["mask_frame_id"] is not None or row["mask_sha256"] is not None:
                        raise ReplayError(f"{mask_set} {role} {fid}: non-observed mask carries a mask identity")
                else:
                    raise ReplayError(f"{mask_set} {role} {fid}: unknown mask state {state!r}")
                states[state] = states.get(state, 0) + 1
            counts[role] = dict(sorted(states.items()))
        return {"mask_set": mask_set, "manifest_canonical_sha256": canonical, "mask_state_counts": counts}
    return _guard(check, f"{mask_set} mask consumption")


# --------------------------------------------------------------------------
# run identity validation (no reference content)


def validate_run_documents(run_doc: Mapping, report: Mapping, run_id: str) -> dict:
    """Terminal-completed regression run: exact run/clip/mode/roles/schedule and 110 rows per role."""
    from bench import workbench_contract as wc

    def check() -> dict:
        run = wc.RunSpec.from_dict(run_doc["run"])
        if (run.run_id != run_id or run.clip_id != CLIP or run_doc.get("clip_id") != CLIP or run.mode != RUN_MODE
                or tuple(run.roles) != ROLES or list(run.scheduled_frame_ids) != list(SCHEDULE)):
            raise ReplayError(f"{run_id}: run identity/mode/roles/schedule differ")
        stages = report["stages"]
        if (report.get("status") != "completed" or report.get("run_id") != run_id or report.get("clip_id") != CLIP
                or report.get("mode") != RUN_MODE or report.get("scheduled_frames") != len(SCHEDULE)
                or set(report["roles"]) != set(ROLES)
                or any(report["roles"][r]["result_rows"] != len(SCHEDULE) or report["roles"][r]["missing_rows"] != []
                       for r in ROLES)
                or any(stages[r]["status"] != "completed" or stages[r]["frames"] != len(SCHEDULE) for r in ROLES)
                or stages["masks"]["status"] != "completed"):
            raise ReplayError(f"{run_id}: report is not terminal-completed with 110 rows for both roles")
        return dict(run_doc["run"]["config"])
    return _guard(check, f"{run_id} run documents")


def verify_rgb(plan: Mapping, decode: Callable[[bytes], np.ndarray] | None = None) -> dict[str, str]:
    """Rehash and decode every scheduled RGB (FrameReader convention); bound to the frozen plan."""
    rgb = plan["rgb"]
    if set(rgb) != set(SCHEDULE):
        raise ReplayError("frozen RGB bindings do not cover exactly the schedule")
    out = {}
    for fid in SCHEDULE:
        binding = rgb[fid]
        raw = _read(_safe_rel(binding["path"], "RGB"), 16 << 20)
        if sha256_bytes(raw) != binding["file_sha256"] or decode_pixel_sha256(raw, decode) != binding["pixel_sha256"]:
            raise ReplayError(f"RGB {fid} file/pixel identity differs from the frozen plan")
        out[fid] = binding["pixel_sha256"]
    return out


def validate_artwork(art: Mapping, plan: Mapping, what: str) -> None:
    from bench import workbench_contract as wc

    def check() -> None:
        expected = plan["artwork"]
        body = {k: v for k, v in art.items() if k != "record_sha256"}
        K_art = np.asarray(art["effective_K"], dtype=np.float64)
        if (art.get("schema") != ARTWORK_SCHEMA or art.get("units") != "metres"
                or art.get("glb_sha256") != plan["geometry"]["candidate_glb_sha256"]
                or art.get("principal_point_origin") != ORIGIN
                or art.get("record_sha256") != wc.canonical_sha256(body) or art["record_sha256"] != expected["record_sha256"]
                or int(art["marker"]["face_id"]) != expected["face_id"]
                or [float(b) for b in art["marker"]["barycentric"]] != expected["barycentric"]
                or int(art["mesh_faces"]) != expected["mesh_faces"] or K_art.size != 9):
            raise ReplayError(f"{what} neural-artwork.json schema/record/anchor/units/origin/GLB differ from the frozen plan")
        for fid in SCHEDULE:
            if not np.array_equal(K_art.reshape(3, 3), np.asarray(plan["K_by_frame"][fid], dtype=np.float64).reshape(3, 3)):
                raise ReplayError(f"{what} artwork effective K differs from the frozen K of {fid}")
    _guard(check, f"{what} artwork")


def validate_backend_identities(original: Mapping, new: Mapping, plan: Mapping) -> dict:
    """Complete per-role backend identity: constant across frames, all required fields present,
    pinned head/base/admission/geometry/origin/K, and EXACTLY equal between the two runs."""
    from bench import workbench_contract as wc

    def check() -> dict:
        out = {}
        for role in ROLES:
            idents = {}
            for name, table in (("original", original), ("new", new)):
                if len({wc.canonical_sha256(table[role][f]["backend_identity"]) for f in SCHEDULE}) != 1:
                    raise ReplayError(f"{name} {role}: backend identity varies across frames")
                ident = table[role][SCHEDULE[0]]["backend_identity"]
                required = IDENTITY_REQUIRED + (("export_sha256",) if role == "primary_neural" else ("head",))
                missing = [k for k in required if k not in ident]
                if missing:
                    raise ReplayError(f"{name} {role}: backend identity lacks {missing}")
                if (ident["base_checkpoint_sha256"] != BASE_SHA256 or ident["admission_contract_sha256"] != ADMISSION_CONTRACT_SHA256
                        or ident["principal_point_origin"] != ORIGIN or ident["metric_units"] != "metres"
                        or ident["glb_sha256"] != plan["geometry"]["candidate_glb_sha256"] or ident["quality_claim"] is not False
                        or not _hex(ident["installed_head_state_sha256"]) or not _hex(ident["install_digest"])
                        or not _text(ident["initialization"]) or not _text(ident["seed_frame_id"])
                        or (role == "primary_neural" and (ident["installed_head_sha256"] != HEAD_SHA256
                                                          or ident["export_sha256"] != HEAD_SHA256))
                        or (role == "frozen_baseline" and (ident["installed_head_sha256"] is not None
                                                           or ident["head"] != "frozen_base_pose_head"))):
                    raise ReplayError(f"{name} {role}: backend identity differs from the head/base/admission/geometry pins")
                K = np.asarray(ident["effective_K"], dtype=np.float64)
                for fid in SCHEDULE:
                    if K.size != 9 or not np.array_equal(K.reshape(3, 3), np.asarray(plan["K_by_frame"][fid], dtype=np.float64).reshape(3, 3)):
                        raise ReplayError(f"{name} {role}: effective K differs from the frozen K of {fid}")
                idents[name] = ident
            if idents["original"] != idents["new"]:
                diff = sorted(k for k in set(idents["original"]) | set(idents["new"])
                              if idents["original"].get(k) != idents["new"].get(k))
                raise ReplayError(f"{role}: backend identity differs between the original and new runs: {diff}")
            out[role] = {"identity_sha256": wc.canonical_sha256(idents["new"]), "install_digest": idents["new"]["install_digest"],
                         "installed_head_state_sha256": idents["new"]["installed_head_state_sha256"]}
        return out
    return _guard(check, "backend identity")


def verify_run_bindings(plan: Mapping, original_doc: Mapping, new_doc: Mapping, original: Mapping, new: Mapping,
                        original_art: Mapping, new_art: Mapping, pixel_sha: Mapping[str, str]) -> dict:
    """Config/prompt/mask/control/admission pins, per-row run/RGB identity, complete backend identity
    and exact artwork equality (old 2D references cannot be warped to a changed anchor)."""
    def check() -> dict:
        cfg_old, cfg_new = original_doc["run"]["config"], new_doc["run"]["config"]
        pins, geometry, sidecar = plan["pins"]["configs"], plan["geometry"], plan["original_run"]["sidecar_sha256"]
        if (cfg_old.get("mask_set") != ORIGINAL_MASK_SET or cfg_old.get("surface_control_map_sha256") != geometry["reviewed_map_sha256"]
                or cfg_old.get("surface_control_sidecar_sha256") != sidecar
                or cfg_old.get("execution_only_prepared_cad") is not True
                or cfg_old.get("independent_correspondence_verified") is not False):
            raise ReplayError("original run config differs from the frozen original plan")
        if (cfg_new.get("mask_set") != NEW_MASK_SET
                or cfg_new.get("segmentation_manifest_sha256") != NEW_MASK_MANIFEST_SHA256
                or cfg_new.get("segmentation_prompt_sha256") != pins["prompt"]
                or cfg_new.get("segmentation_registration_sha256") != pins["registration"]
                or cfg_new.get("segmentation_seed_rule_clarification_sha256") != pins["seed_rule_clarification"]
                or cfg_new.get("selection_receipt_sha256") != pins["selection_receipt"]
                or cfg_new.get("original_comparison_run_id") != ORIGINAL_RUN
                or cfg_new.get("surface_control_map_sha256") != geometry["reviewed_map_sha256"]
                or cfg_new.get("surface_control_sidecar_sha256") != sidecar
                or cfg_new.get("serving_admission_file_hashes") != plan["admission"]["file_pins"]
                or not _hex(cfg_new.get("surface_control_adapter_sha256"))
                or ("surface_control_adapter_sha256" in cfg_old
                    and cfg_new["surface_control_adapter_sha256"] != cfg_old["surface_control_adapter_sha256"])
                or cfg_new.get("execution_only_prepared_cad") is not True
                or cfg_new.get("independent_correspondence_verified") is not False
                or cfg_new.get("development_diagnostic_only") is not True):
            raise ReplayError("new run config does not bind the pinned prompt/mask/registration/control/admission identities")
        for name, run_id, table in (("original", ORIGINAL_RUN, original), ("new", NEW_RUN, new)):
            for role in ROLES:
                for fid in SCHEDULE:
                    row = table[role][fid]
                    if (row["run_id"] != run_id or row["clip_id"] != CLIP or row["backend_role"] != role
                            or row["frame_id"] != fid or row["source_frame_index"] != int(fid[1:])
                            or row["image_sha256"] != pixel_sha[fid]):
                        raise ReplayError(f"{name} {role} {fid}: run/clip/frame/decoded-RGB identity differs")
        identities = validate_backend_identities(original, new, plan)
        validate_artwork(original_art, plan, "original")
        validate_artwork(new_art, plan, "new")
        if new_art != original_art:
            raise ReplayError("new artwork differs from the original (old 2D references cannot be warped)")
        return {"config": "bound", "artwork_identical": True, "rgb_pixels_bound": len(pixel_sha),
                "backend_identities": identities,
                "surface_control_adapter_sha256": cfg_new["surface_control_adapter_sha256"],
                "surface_control_adapter_pinned_by_original_run": "surface_control_adapter_sha256" in cfg_old}
    return _guard(check, "run bindings")


def validate_inference_sources(report: Mapping) -> dict:
    """The run's own dependency receipt (module hashes and head identities recorded by the run)."""
    from bench import workbench_contract as wc

    def check() -> dict:
        deps = report["dependencies"]
        sources, heads = deps["sources"], deps["heads"]
        if not isinstance(sources, Mapping) or any(not _hex(sources.get(m)) for m in INFERENCE_REQUIRED_SOURCES):
            raise ReplayError("run dependency receipt lacks required inference source hashes")
        p, b = heads["primary_neural"], heads["frozen_baseline"]
        if (p.get("installed_head_sha256") != HEAD_SHA256 or p.get("export_sha256") != HEAD_SHA256
                or p.get("base_checkpoint_sha256") != BASE_SHA256 or b.get("base_checkpoint_sha256") != BASE_SHA256
                or b.get("installed_head_sha256") is not None):
            raise ReplayError("run dependency receipt head/base identities differ from the pins")
        return {"report_dependencies_sha256": wc.canonical_sha256(deps), "sources": dict(sorted(sources.items())),
                "versions": deps.get("versions")}
    return _guard(check, "inference sources")


# --------------------------------------------------------------------------
# parent terminal record and attestation

TERMINAL_KEYS = {"schema", "run_id", "execution_id", "command", "exit_code", "worker_process_exited", "observed_by",
                 "observed_at", "attested"}
ATTESTED_KEYS = {"geometry", "mask_inventory", "inference_execution"}
GEOMETRY_KEYS = {"glb_path", "glb_sha256", "physical_bytes_hashed_by_parent"}
MASK_INVENTORY_KEYS = {"manifest_path", "manifest_file_sha256", "manifest_canonical_sha256", "frames", "inventory_sha256",
                       "physical_files_hashed_by_parent"}
INFERENCE_KEYS = {"launch_path", "launch_sha256", "admission_dir", "admission_file_hashes", "report_dependencies_sha256",
                  "surface_control_adapter_sha256", "observation_receipts"}
RECEIPT_KEYS = {"path", "sha256"}


def validate_terminal_record(record: Any) -> dict:
    """Exact typed parent record of an observed GPU worker exit 0. A ``True`` field is not success by
    itself: seal/evaluate still validate the actual completed artifacts."""
    _exact_keys(record, TERMINAL_KEYS, "terminal record")
    if record["schema"] != SCHEMA_TERMINAL or record["run_id"] != NEW_RUN:
        raise ReplayError("terminal record schema/run differs")
    if type(record["exit_code"]) is not int or record["exit_code"] != 0:
        raise ReplayError("terminal record exit_code must be the integer 0")
    if record["worker_process_exited"] is not True:
        raise ReplayError("terminal record must observe the worker process exit")
    if not all(_text(record[k]) for k in ("execution_id", "command", "observed_by", "observed_at")):
        raise ReplayError("terminal record execution/command/observer/time must be non-empty strings")
    att = _exact_keys(record["attested"], ATTESTED_KEYS, "attested")
    g = _exact_keys(att["geometry"], GEOMETRY_KEYS, "attested.geometry")
    m = _exact_keys(att["mask_inventory"], MASK_INVENTORY_KEYS, "attested.mask_inventory")
    i = _exact_keys(att["inference_execution"], INFERENCE_KEYS, "attested.inference_execution")
    if not _text(g["glb_path"]) or not _hex(g["glb_sha256"]) or g["physical_bytes_hashed_by_parent"] is not True:
        raise ReplayError("attested.geometry is malformed")
    if (not _text(m["manifest_path"]) or not all(_hex(m[k]) for k in ("manifest_file_sha256", "manifest_canonical_sha256",
                                                                       "inventory_sha256"))
            or type(m["frames"]) is not int or m["frames"] <= 0 or m["physical_files_hashed_by_parent"] is not True):
        raise ReplayError("attested.mask_inventory is malformed")
    receipts = i["observation_receipts"]
    if (not _text(i["launch_path"]) or not _text(i["admission_dir"])
            or not all(_hex(i[k]) for k in ("launch_sha256", "report_dependencies_sha256", "surface_control_adapter_sha256"))
            or not isinstance(i["admission_file_hashes"], Mapping)
            or not all(isinstance(k, str) and _hex(v) for k, v in i["admission_file_hashes"].items())
            or not isinstance(receipts, list) or not receipts):
        raise ReplayError("attested.inference_execution is malformed")
    for r in receipts:
        _exact_keys(r, RECEIPT_KEYS, "observation receipt")
        if not _text(r["path"]) or not _hex(r["sha256"]):
            raise ReplayError("observation receipt is malformed")
    if len({r["path"] for r in receipts}) != len(receipts):
        raise ReplayError("duplicate observation receipt paths")
    return dict(record)


def expected_attestation(plan: Mapping, new_manifest: Mapping, new_report: Mapping, new_config: Mapping) -> dict:
    """What the parent attestation must say, from pins, the pinned manifest JSON and the run receipts."""
    from bench import workbench_contract as wc

    def build() -> dict:
        inventory = mask_inventory_digest(new_manifest)
        return {
            "geometry": {"glb_path": plan["geometry"]["glb_path"], "glb_sha256": plan["geometry"]["candidate_glb_sha256"]},
            "mask_inventory": {"manifest_path": plan["new_run"]["mask_manifest"],
                               "manifest_file_sha256": plan["new_run"]["mask_manifest_file_sha256"],
                               "manifest_canonical_sha256": wc.canonical_sha256(new_manifest),
                               "frames": inventory["frames"], "inventory_sha256": inventory["inventory_sha256"]},
            "inference_execution": {"launch_path": _rel(CONFIG_FILES["launch"][0]),
                                    "launch_sha256": plan["pins"]["configs"]["launch"],
                                    "admission_dir": plan["admission"]["dir"],
                                    "admission_file_hashes": plan["admission"]["file_pins"],
                                    "report_dependencies_sha256": wc.canonical_sha256(new_report["dependencies"]),
                                    "surface_control_adapter_sha256": new_config["surface_control_adapter_sha256"]},
        }
    return _guard(build, "expected attestation")


def validate_attestation(record: Mapping, expected: Mapping) -> None:
    att = record["attested"]
    for section, fields in expected.items():
        for key, value in fields.items():
            if att[section][key] != value:
                raise ReplayError(f"parent attestation {section}.{key} differs from the pinned/observed value")


def authenticate_runs(plan: Mapping, record: Mapping, decode: Callable[[bytes], np.ndarray] | None = None) -> dict:
    """Fresh reads of both runs (never cached from an earlier phase); no reference content."""
    static = authenticate_static_inputs(plan)
    loaded = {}
    for name, run_id in (("original", ORIGINAL_RUN), ("new", NEW_RUN)):
        run_dir = RUNS_ROOT / run_id
        run_doc = _loads(_read(run_dir / "run.json"), f"{name} run.json")
        report = _loads(_read(run_dir / "report.json"), f"{name} report.json")
        config = validate_run_documents(run_doc, report, run_id)
        rows = validate_schedule(_load_rows(run_id), SCHEDULE)
        art = _loads(_read(run_dir / "neural-artwork.json"), f"{name} neural-artwork.json")
        loaded[name] = {"doc": run_doc, "report": report, "config": config, "rows": rows, "art": art}
    o, n = loaded["original"], loaded["new"]
    pixel_sha = verify_rgb(plan, decode)
    bindings = verify_run_bindings(plan, o["doc"], n["doc"], o["rows"], n["rows"], o["art"], n["art"], pixel_sha)
    from bench import workbench_contract as wc
    original_manifest = _read_pinned_json(_safe_rel(plan["original_run"]["mask_manifest"], "manifest"),
                                          plan["original_run"]["mask_manifest_file_sha256"], "original mask manifest")
    if wc.canonical_sha256(original_manifest) != plan["original_run"]["mask_manifest_canonical_sha256"]:
        raise ReplayError("original mask manifest canonical identity differs from the frozen plan")
    new_manifest = _read_pinned_json(_safe_rel(plan["new_run"]["mask_manifest"], "manifest"),
                                     plan["new_run"]["mask_manifest_file_sha256"], "new mask manifest")
    masks = {"original_mask": validate_mask_consumption(o["rows"], original_manifest, _guard(lambda: o["report"]["stages"]["masks"], "stage"),
                                                        mask_set=ORIGINAL_MASK_SET),
             "new_qualified_mask": validate_mask_consumption(n["rows"], new_manifest, _guard(lambda: n["report"]["stages"]["masks"], "stage"),
                                                             mask_set=NEW_MASK_SET)}
    inference = validate_inference_sources(n["report"])
    validate_attestation(record, expected_attestation(plan, new_manifest, n["report"], n["config"]))
    original_sources = (o["report"].get("dependencies") or {}).get("sources")
    summary = {"static_inputs": static, "bindings": bindings, "mask_consumption": masks,
               "inference_sources": {"new_run": inference,
                                     "original_run_sources": original_sources if isinstance(original_sources, Mapping) else None,
                                     "module_hash_equal": ({m: original_sources.get(m) == inference["sources"].get(m)
                                                            for m in sorted(inference["sources"])}
                                                           if isinstance(original_sources, Mapping) else None),
                                     "label": FLAGS["inference_source_closure"]},
               "new_mask_inventory": mask_inventory_digest(new_manifest)}
    return {"original_rows": o["rows"], "new_rows": n["rows"], "summary": _jsonable(summary)}


# --------------------------------------------------------------------------
# outputs and budgets


def check_out(out: Path) -> Path:
    out = Path(os.path.abspath(out))
    if DERIVED_ROOT.resolve() not in out.parents:
        raise ReplayError("output must be a new directory inside .cache/video-workbench/derived")
    for part in [out, *out.parents]:
        if part.exists() and part.is_symlink():
            raise ReplayError("output path traverses a symlink")
    if os.path.lexists(out):
        raise ReplayError(f"output {out} already exists; no overwrite")
    return out


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) if Path(path).is_dir() else 0


def workspace_budget(extra_derived: int) -> None:
    """Complete workspace accounting (derived + scratch + runs + masks + suite, inputs, free reserve)."""
    from bench.workbench_cli import Workspace, WorkbenchCLIError
    try:
        Workspace(WORKSPACE).require_budget(extra_derived=int(extra_derived))
    except WorkbenchCLIError as exc:
        raise ReplayError(f"workspace budget: {exc}") from exc


def check_budget(planned: int, related: Sequence[Path] = ()) -> None:
    combined = planned + sum(_dir_bytes(p) for p in related)
    if combined > MAX_COMBINED_BYTES:
        raise ReplayError(f"combined plan/seal/evaluation bytes {combined} exceed 4 MiB")
    workspace_budget(planned)


def preflight_eval_budget(related: Sequence[Path]) -> int:
    """Reserve the remaining 4 MiB packet allowance before the sealed reference is read."""
    remaining = MAX_COMBINED_BYTES - sum(_dir_bytes(p) for p in related)
    if remaining < EVAL_MIN_RESERVE_BYTES:
        raise ReplayError(f"only {remaining} bytes of the 4 MiB packet allowance remain")
    workspace_budget(remaining)
    return remaining


def publish(out: Path, files: Mapping[str, bytes]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{secrets.token_hex(6)}"
    staging.mkdir()
    try:
        for name, data in files.items():
            with os.fdopen(os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                                   0o644), "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if sha256_file(staging / name) != sha256_bytes(data):
                raise ReplayError(f"staged {name} does not verify")
        if os.path.lexists(out):
            raise ReplayError("output appeared during publication")
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _external_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    try:
        return {name: sha256_file(path) for name, path in paths.items()}
    except OSError as exc:
        raise ReplayError(f"external closure file missing: {exc}") from exc


# --------------------------------------------------------------------------
# phase 1: prepare (no new-run outputs, no reference content, no metrics)


def build_plan() -> dict:
    frozen = _read_pinned_json(FROZEN_PLAN, FROZEN_PLAN_SHA256, "frozen original plan")
    if sha256_file(ACCEPTED_EVALUATOR) != ACCEPTED_EVALUATOR_SHA256 or frozen.get("tool_sha256") != ACCEPTED_EVALUATOR_SHA256:
        raise ReplayError("accepted evaluator source or the frozen plan's tool pin differs")
    for name, (path, sha) in CONFIG_FILES.items():
        if sha256_file(path) != sha:
            raise ReplayError(f"config {name} differs from its pin")
    for name, sha in ADMISSION_FILE_PINS.items():
        if sha256_file(ADMISSION_DIR.joinpath(*name.split("/"))) != sha:
            raise ReplayError(f"admission file {name} differs from its pin")

    def from_frozen() -> dict:
        constants, pins = frozen["constants"], frozen["pins"]
        obj = frozen["objects"][OBJECT_KEY]
        run, frames = obj["run"], obj["frames"]
        if (constants["roles"] != list(ROLES) or constants["principal_point_origin"] != ORIGIN
                or constants["frames_per_run"] != len(SCHEDULE) or constants["rotation_tolerance"] != ROTATION_TOL
                or list(constants["coverage_thresholds_px720"]) != list(COVERAGE_THRESHOLDS_PX720)
                or pins["head_sha256"] != HEAD_SHA256 or pins["base_sha256"] != BASE_SHA256
                or run["run_id"] != ORIGINAL_RUN or run["clip_id"] != CLIP or list(run["scheduled"]) != list(SCHEDULE)
                or [f["frame_id"] for f in frames] != list(SCHEDULE) or obj["image_size"] != [640, 480]
                or set(run["rgb"]) != set(SCHEDULE)
                or run["mask_set"]["manifest"] != _rel(MASKS_ROOT / CLIP / ORIGINAL_MASK_SET / "manifest.json")):
            raise ReplayError("frozen plan does not describe the original cracker-box run/schedule/mask set/pins")
        control_dir = run["control_dir"]
        return {
            "original_run": {"run_id": ORIGINAL_RUN, "files": run["files"], "mask_set": ORIGINAL_MASK_SET,
                             "mask_manifest": run["mask_set"]["manifest"],
                             "mask_manifest_file_sha256": run["mask_set"]["manifest_file_sha256"],
                             "mask_manifest_canonical_sha256": run["mask_set"]["manifest_canonical_sha256"],
                             "clip": {"path": run["clip"]["path"], "sha256": run["clip"]["sha256"]},
                             "sidecar_sha256": run["sidecar_sha256"]},
            "geometry": {"control_dir": control_dir, "glb_path": f"{control_dir}/model.glb",
                         "candidate_glb_sha256": obj["candidate_glb_sha256"], "bundle_sha256": obj["bundle_sha256"],
                         "map_entry_sha256": obj["map_entry_sha256"], "reviewed_map_sha256": pins["reviewed_map_sha256"]},
            "anchor_m": obj["anchor_m"], "image_size": obj["image_size"],
            "K_by_frame": {f["frame_id"]: f["cam_K"] for f in frames}, "rgb": run["rgb"], "artwork": run["artwork"],
        }
    frozen_part = _guard(from_frozen, "frozen original plan")
    plan = {
        "schema": SCHEMA_PLAN, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__)), "flags": FLAGS,
        "pins": {"accepted_evaluator_sha256": ACCEPTED_EVALUATOR_SHA256, "frozen_plan_sha256": FROZEN_PLAN_SHA256,
                 "sealed_result_sha256": SEALED_RESULT_SHA256, "head_sha256": HEAD_SHA256, "base_sha256": BASE_SHA256,
                 "admission_contract_sha256": ADMISSION_CONTRACT_SHA256, "new_mask_manifest_sha256": NEW_MASK_MANIFEST_SHA256,
                 "configs": {k: sha for k, (_p, sha) in CONFIG_FILES.items()}},
        **frozen_part,
        "new_run": {"run_id": NEW_RUN, "mask_set": NEW_MASK_SET, "mask_manifest": _rel(NEW_MASK_MANIFEST),
                    "mask_manifest_file_sha256": NEW_MASK_MANIFEST_SHA256,
                    "binding": "deferred to the parent terminal seal; no new-run output is read by prepare"},
        "admission": {"dir": _rel(ADMISSION_DIR), "file_pins": dict(ADMISSION_FILE_PINS),
                      "contract_sha256": ADMISSION_CONTRACT_SHA256, "parity_receipt_sha256": ADMISSION_PARITY_RECEIPT_SHA256},
        "terminal_observation": {"path": _rel(TERMINAL_OBSERVATION[0]), "sha256": TERMINAL_OBSERVATION[1]},
        "clip_id": CLIP, "mode": RUN_MODE, "schedule": list(SCHEDULE), "roles": list(ROLES),
        "principal_point_origin": ORIGIN,
        "metric_spec": {"projection": "legacy accepted-evaluator semantics, independently implemented",
                        "px720": "native * 720 / 480", "coverage_thresholds_px720": list(COVERAGE_THRESHOLDS_PX720),
                        "control_tolerance": {"rtol": 0.0, "atol_px": CONTROL_ATOL_PX},
                        "subsets": ["all_reference_eligible", "fixed_visible"],
                        "percentiles": "numpy linear", "empty": "null (never 0/NaN/Inf)"},
        "sources": source_closure(), "runtime": runtime_identity(),
    }
    authenticate_static_inputs(_jsonable(plan))
    return _jsonable(plan)


def validate_plan(plan: Any, expected: Mapping) -> dict:
    """Full semantic check: the plan must equal the prospective rebuild (schema, every pin, sources,
    runtime), so a rewritten plan with a matching outer hash cannot change any pin."""
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA_PLAN or plan != expected:
        raise ReplayError("plan content differs from the prospective rebuild (schema/pins/sources/runtime)")
    return plan


def prepare(out: Path) -> dict:
    out = check_out(out)
    blob = _dump(build_plan())
    check_budget(len(blob))
    publish(out, {"plan.json": blob})
    return {"directory": out.as_posix(), "plan_sha256": sha256_bytes(blob)}


def _load_plan(plan_path: Path, expected: str) -> dict:
    raw = _read(plan_path)
    if sha256_bytes(raw) != expected:
        raise ReplayError("plan.json differs from the externally provided SHA-256")
    return validate_plan(_loads(raw, "plan.json"), build_plan())


# --------------------------------------------------------------------------
# phase 2: parent-only terminal seal

SEAL_KEYS = {"schema", "tool_version", "plan_sha256", "tool_sha256", "terminal_record", "closure", "run_authentication",
             "sealed_result_sha256_expected", "labels", "statement"}


def validate_seal(seal: Any, plan: Mapping, plan_sha: str) -> dict:
    _exact_keys(seal, SEAL_KEYS, "seal")
    terminal = _exact_keys(seal["terminal_record"], {"path", "sha256"}, "seal.terminal_record")
    if (seal["schema"] != SCHEMA_SEAL or seal["tool_version"] != TOOL_VERSION or seal["plan_sha256"] != plan_sha
            or seal["tool_sha256"] != plan["tool_sha256"] or seal["sealed_result_sha256_expected"] != SEALED_RESULT_SHA256
            or not _hex(terminal["sha256"]) or not isinstance(seal["closure"], Mapping)
            or not isinstance(seal["run_authentication"], Mapping)):
        raise ReplayError("seal is stale or not bound to this plan/tool/reference pin")
    _safe_rel(terminal["path"], "terminal record")
    return dict(seal)


def build_seal(plan_path: Path, plan_sha: str, terminal_record: Path, terminal_record_sha: str, out: Path) -> dict:
    """PARENT-ONLY: bind the completed replay after the parent observed the GPU worker exit 0."""
    out = check_out(out)
    plan = _load_plan(plan_path, plan_sha)
    terminal_rel = _guard(lambda: _rel(terminal_record), "terminal record path")
    _safe_rel(terminal_rel, "terminal record")
    record = validate_terminal_record(_read_pinned_json(terminal_record, terminal_record_sha, "terminal record"))
    snapshot = closure_snapshot(plan, record)
    validate_expected_closure(plan, snapshot)
    auth = authenticate_runs(plan, record)
    after = closure_snapshot(plan, record)
    if after != snapshot or _external_hashes({"plan": Path(plan_path), "terminal": Path(terminal_record)}) != {
            "plan": plan_sha, "terminal": terminal_record_sha}:
        raise ReplayError("inputs changed while sealing")
    seal = {"schema": SCHEMA_SEAL, "tool_version": TOOL_VERSION, "plan_sha256": plan_sha, "tool_sha256": plan["tool_sha256"],
            "terminal_record": {"path": terminal_rel, "sha256": terminal_record_sha},
            "closure": snapshot, "run_authentication": auth["summary"], "sealed_result_sha256_expected": SEALED_RESULT_SHA256,
            "labels": {"geometry_and_mask_closure": FLAGS["geometry_and_mask_closure"],
                       "inference_source_closure": FLAGS["inference_source_closure"]},
            "statement": "parent seal after the parent-observed GPU worker exit 0; this tool validated the completed artifacts"}
    blob = _dump(seal)
    check_budget(len(blob), related=[Path(plan_path).parent])
    publish(out, {"seal.json": blob})
    return {"directory": out.as_posix(), "seal_sha256": sha256_bytes(blob)}


# --------------------------------------------------------------------------
# phase 3: evaluate


def evaluate(plan_path: Path, plan_sha: str, seal_path: Path, seal_sha: str, out: Path) -> dict:
    out = check_out(out)
    plan = _load_plan(plan_path, plan_sha)
    seal = validate_seal(_read_pinned_json(seal_path, seal_sha, "seal"), plan, plan_sha)
    terminal_path = _safe_rel(seal["terminal_record"]["path"], "terminal record")
    externals = {"plan": Path(plan_path), "seal": Path(seal_path), "terminal": terminal_path}
    expected_externals = {"plan": plan_sha, "seal": seal_sha, "terminal": seal["terminal_record"]["sha256"]}
    related = [Path(plan_path).parent, Path(seal_path).parent]
    stage = "closure"
    try:
        record = validate_terminal_record(_read_pinned_json(terminal_path, expected_externals["terminal"], "terminal record"))
        before = closure_snapshot(plan, record)
        validate_expected_closure(plan, before)
        if before != seal["closure"]:
            raise ReplayError("inputs, sources or runtime changed since the seal; refusing to parse the reference")
        stage = "run_identity"
        auth = authenticate_runs(plan, record)
        if auth["summary"] != seal["run_authentication"]:
            raise ReplayError("run authentication differs from the seal")
        stage = "budget_preflight"
        remaining = preflight_eval_budget(related)
        stage = "pre_reference_recheck"
        if _external_hashes(externals) != expected_externals or closure_snapshot(plan, record) != before:
            raise ReplayError("plan/seal/terminal record or closure changed before the reference parse")
        stage = "reference_parse"
        sealed = _read_pinned_json(SEALED_RESULT, SEALED_RESULT_SHA256, "sealed reference result")
        frozen = _read_pinned_json(FROZEN_PLAN, FROZEN_PLAN_SHA256, "frozen plan")
        reference = extract_reference(sealed, frozen)
        stage = "original_control"
        control = verify_original_control(auth["original_rows"], reference, sealed, anchor_m=plan["anchor_m"],
                                          K_by_frame=plan["K_by_frame"], image_size=plan["image_size"])
        if not control["passed"]:
            raise ReplayError(f"original control failed ({control['mismatch_count']} mismatches): {control['mismatches'][:5]}")
        stage = "new_metrics"
        new_scores = {role: score_role(auth["new_rows"][role], reference, plan["anchor_m"], plan["K_by_frame"],
                                       plan["image_size"]) for role in ROLES}
        cells = build_cells(control["scores"], new_scores, auth["original_rows"], auth["new_rows"])
        comparison = compare_replays(control["scores"], new_scores)
        stage = "post_closure"
        after = closure_snapshot(plan, record)
        validate_expected_closure(plan, after)
        if (after != before or sha256_file(SEALED_RESULT) != SEALED_RESULT_SHA256
                or _external_hashes(externals) != expected_externals
                or validate_terminal_record(_read_pinned_json(terminal_path, expected_externals["terminal"],
                                                              "terminal record")) != record):
            raise ReplayError("inputs changed during evaluation")
        document = {"schema": SCHEMA_RESULT, "flags": FLAGS, "plan_sha256": plan_sha, "seal_sha256": seal_sha,
                    "terminal_record_sha256": expected_externals["terminal"], "pins": plan["pins"],
                    "run_authentication": auth["summary"], "parent_attestation": record["attested"],
                    "original_control": {"passed": True, "rows_reproduced": 2 * len(SCHEDULE),
                                         "tolerance": control["tolerance"]},
                    "cells": cells, "original_to_new": comparison,
                    "rows": {"original": control["scores"], "new": new_scores},
                    "notes": ["new replay uses a different opening-box prompt and SAM seed; weights unchanged",
                              "legacy_pose_present is the old D tracking_pose (pose present), not validated tracking",
                              "geometry/mask closure is parent-attested; inference source closure is the run's own "
                              "dependency receipt attested by the parent",
                              "no pooled objects, significance, human, phone, orientation, footprint or isolated-head claim"]}
        blob, csv_blob = _dump(document), render_csv(cells)
        stage = "publish"
        if len(blob) + len(csv_blob) > remaining:
            raise ReplayError("evaluation outputs exceed the reserved packet allowance")
        check_budget(len(blob) + len(csv_blob), related=related)
        publish(out, {"replay-evaluation.json": blob, "replay-evaluation.csv": csv_blob})
        return {"directory": out.as_posix(), "json_sha256": sha256_bytes(blob), "csv_sha256": sha256_bytes(csv_blob)}
    except BaseException as exc:
        failure = {"schema": SCHEMA_FAILURE, "status": "failed_closed", "stage": stage, "plan_sha256": plan_sha,
                   "seal_sha256": seal_sha, "error": f"{type(exc).__name__}: {exc}"[:2000], "comparison_emitted": False}
        try:
            blob = _dump(failure)
            check_budget(len(blob), related=related)
            publish(out, {"failure.json": blob})
        except Exception:  # noqa: BLE001 - never hide the original failure
            pass
        raise


def build_cells(original_scores: Mapping, new_scores: Mapping, original_rows: Mapping, new_rows: Mapping) -> dict:
    """Four cells: {original, new} masks x {primary, baseline}; all 110 frames accounted."""
    cells = {}
    for mask, scores, rows in (("original_mask", original_scores, original_rows), ("new_qualified_mask", new_scores, new_rows)):
        for role in ROLES:
            scored = scores[role]
            eligible = [r["frame_id"] for r in scored if r["reference_status"] == "eligible"]
            visible = [r["frame_id"] for r in scored if r["reference_status"] == "eligible" and r["reference_visible"] == "visible"]
            cells[f"{mask}/{role}"] = {
                "frames": len(scored),
                "all_reference_eligible": summarize_anchor(scored, eligible),
                "fixed_visible": summarize_anchor(scored, visible),
                "prediction_status_counts": _counts(r["prediction_status"] for r in scored),
                "reference_status_counts": _counts(r["reference_status"] for r in scored),
                "availability": summarize_availability([rows[role][f] for f in SCHEDULE]),
            }
    return cells


CSV_FIELDS = ("cell", "subset", "denominator", "measured", "conditional_median_px720", "conditional_p95_px720",
              "coverage_le_5px720", "coverage_le_10px720", "frames", "validated_tracking_poses", "legacy_pose_present",
              "render_state_rendered", "rendered_pixels_positive", "rendered_pixels_median_when_positive")


def render_csv(cells: Mapping) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_FIELDS)

    def fmt(v: Any) -> str:
        return "" if v is None else repr(float(v)) if isinstance(v, float) else str(v)

    for cell in sorted(cells):
        c = cells[cell]
        for subset in ("all_reference_eligible", "fixed_visible"):
            s, a = c[subset], c["availability"]
            writer.writerow([cell, subset] + [fmt(s[k]) for k in CSV_FIELDS[2:8]]
                            + [fmt(a[k]) if subset == "all_reference_eligible" else "" for k in
                               ("frames", "validated_tracking_poses", "legacy_pose_present", "render_state_rendered",
                                "rendered_pixels_positive", "rendered_pixels_median_when_positive")])
    return buf.getvalue().encode("ascii")


# --------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Qualified-mask replay anchor evaluator (CPU; three phases).")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--out", required=True)
    s = sub.add_parser("seal", help="PARENT-ONLY after observing the GPU worker exit 0")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--terminal-record", required=True)
    s.add_argument("--expected-terminal-record-sha256", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("evaluate")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--seal", required=True)
    s.add_argument("--expected-seal-sha256", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(Path(args.out))
        elif args.command == "seal":
            result = build_seal(Path(args.plan), args.expected_plan_sha256, Path(args.terminal_record),
                                args.expected_terminal_record_sha256, Path(args.out))
        else:
            result = evaluate(Path(args.plan), args.expected_plan_sha256, Path(args.seal), args.expected_seal_sha256,
                              Path(args.out))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ReplayError, OSError, KeyError, ValueError, TypeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
