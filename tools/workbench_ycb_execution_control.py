"""Execution-only supplemental YCB CAD-control adapter (no GPU, no inference).

Turns one *reviewed* supplemental YCB candidate into a
``workbench_metric_bundle_v1`` for execution-only experiments, under the
separately accepted prospective contract:

* both fixed opening frames PASS the raw-PLY vs loaded-GLTF GPU reference
  (``tools/workbench_ycb_gpu_reference.py``). Its unchanged relative criteria
  include informative opposite-UV discrimination;
* the geometry, PNG, prepared-proof, runtime and tool provenance is exact;
* the parent has reviewed qualitative RGB/render placement on both opening
  frames.

This contract does **not** set the accepted proof's full-UV
``serving_render_pass`` or independent correspondence to true. It makes no
accuracy, mobile-readiness or automatic-preparation claim. The original
``blocked_unverified_uv`` candidates and every proof/GPU record stay
unchanged: they are only read.

Modes:

``inspect`` (CPU only; writes nothing; needs no approval)
    Re-validates the prepared proof, the runtime closure and the complete GPU
    evidence closure. It recomputes both frames' comparisons from the stored
    artifacts instead of trusting recorded pass booleans, then prints the
    immutable planned bundle and every hash the reviewed map must pin.
``emit``
    Requires an exact entry for the object in the fixed project-local reviewed
    map ``tools/workbench_ycb_execution_controls.reviewed.json``. Every pin is
    mandatory. The map also pins this adapter's SHA-256, so there is no
    self-hash cycle in the code. It re-runs ``inspect`` against the pinned
    paths, validates the parent's qualitative receipt and overlay hashes, then
    writes a fresh immutable directory under ``.cache/video-workbench/derived``
    containing:
    - ``model.glb``: the original candidate bytes, unchanged;
    - ``metric-bundle.json``;
    - ``execution-control-sidecar.json``;
    - ``emission-receipt.json``.

    The output directory must be the entry's ``planned_out``, and the
    independently re-derived bundle must equal the entry's ``bundle_sha256``.
    Both are printed by ``inspect --planned-out``.

    There is no ``--map`` and no ``--force``. A missing map or entry blocks
    emission.
``validate``
    The launch gate, run immediately before loading. It finds the entry whose
    ``planned_out`` is this directory, re-runs the full plan and qualitative
    review from that entry's closure, and requires all four files to equal the
    re-derived bytes. The bundle must equal the reviewed hash, including the
    exact seed pose.

Units: PLY millimetres become candidate GLB metres. The geometry is the
authenticated candidate, checked against ``float32(PLY mm x 0.001)``. The seed
is the final onboarding pose (logicalIndex 9) read only from the authenticated
inference manifest, with its translation in metres. ``evaluator.json``, scored
poses, masks and depth are never read. The principal-point origin
``integer_centers`` is an explicit *assumed* provenance. The 640x480 neural
serving admission remains a separate requirement; this adapter neither
satisfies nor alters it, and launches nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_ycb_gpu_reference as gpu  # noqa: E402  (read-only reuse; never modified)

proof = gpu.proof
yc = proof.yc

ADAPTER_VERSION = "1"
SCHEMA_MAP = "workbench_ycb_execution_controls_reviewed_v1"
SCHEMA_QUALITATIVE = "workbench_ycb_execution_qualitative_review_v1"
SCHEMA_PLAN = "workbench_ycb_execution_control_plan_v1"
SCHEMA_SIDECAR = "workbench_ycb_execution_control_sidecar_v1"
SCHEMA_RECEIPT = "workbench_ycb_execution_control_emission_v1"
MAP_PATH = ROOT / "tools/workbench_ycb_execution_controls.reviewed.json"
DERIVED_ROOT = ROOT / ".cache/video-workbench/derived"
DERIVED_BUDGET_BYTES = 1 << 30
FREE_RESERVE_BYTES = 1 << 30
NEW_INPUT_BUDGET_BYTES = 250_000_000
MAX_OUTPUT_BYTES = 32 << 20
MAX_EVIDENCE_FILE_BYTES = 64 << 20
MAX_JSON_BYTES = 4 << 20
ORIGIN = "integer_centers"
FINAL_ONBOARDING_LOGICAL = 9
RGB_TOLERANCE = 1.0 / 255.0 + 1e-9      # stored PNGs quantise the float render to 8 bits
DEPTH_TOLERANCE_MM = 1e-3               # stored depth is float32
EVIDENCE_RECORD = "gpu-reference.json"
OUTPUT_FILES = ("model.glb", "metric-bundle.json", "execution-control-sidecar.json", "emission-receipt.json")
SIDECAR_FLAGS = {"execution_only": True, "independent_correspondence_verified": False,
                 "automatic_rgb_preparation": False, "independent_tracking_accuracy": False,
                 "mobile_ready": False, "full_uv_diagnostic_pass": False}
RECORD_KEYS = {"schema", "binding", "thresholds", "frames", "relative_serving_uv_convention_pass",
               "onboarding_correspondence_pass", "final_status", "promotion", "claims", "source_reference", "texture",
               "opposite_hypothesis", "scratch", "proof_module", "proof_source_lines", "runtime_pins",
               "renderer_factory", "budgets", "tool_sha256", "limitations", "runtime", "files", "label"}
FRAME_KEYS = {"frame_id", "tag", "hypotheses", "reference_construction", "pre_draw_equality", "texture_sources",
              "gl_texture_state_equal", "comparison", "close_errors"}
HYPOTHESIS_KEYS = {"allocation_proof", "pre_draw_state", "post_draw_equal", "gl_texture_state", "draw_seconds_measured"}
FALSE_CLAIMS = ("promotion", "metric_bundle", "verified_override", "independent_tracking_accuracy", "mobile",
                "automatic_rgb_preparation", "accepted_proof_serving_render_pass_rewritten")
MAP_KEYS = {"schema", "adapter_sha256", "gpu_reference_tool_sha256", "proof_tool_sha256", "controls_tool_sha256",
            "contract", "entries"}
ENTRY_KEYS = {"object_id", "canonical_ycb_id", "planned_out", "bundle_sha256", "prepared_dir", "prepared_sha256",
              "evidence_dir",
              "evidence_record_sha256", "candidate_glb_sha256", "candidate_receipt_sha256", "png_sha256",
              "ply_sha256", "manifest_sha256", "runtime_pins_sha256", "seed_frame_id", "principal_point_origin",
              "qualitative_receipt", "qualitative_receipt_sha256", "overlays", "reviewed_by", "reviewed_at"}
QUALITATIVE_KEYS = {"schema", "object_id", "evidence_record_sha256", "prepared_sha256", "candidate_glb_sha256",
                    "reviewer", "reviewed_at", "decision", "frames", "claims"}
QUALITATIVE_FRAME_KEYS = {"frame_id", "rgb_sha256", "overlay_sha256", "placement", "notes"}
QUALITATIVE_DECISION = "placement_acceptable_for_execution_only"
QUALITATIVE_CLAIMS = {"accuracy": False, "independent_correspondence": False, "mobile": False}
SIDECAR_KEYS = {"schema", "object_id", "canonical_ycb_id", *SIDECAR_FLAGS, "serving_render_pass_unchanged",
                "principal_point_origin", "units", "seed", "bundle_sha256", "glb_sha256", "evidence_record_sha256",
                "prepared_sha256", "qualitative_receipt_sha256", "overlays", "map_entry_sha256",
                "adapter_sha256", "gpu_reference_tool_sha256", "proof_tool_sha256", "controls_tool_sha256",
                "contract", "admission", "claims"}
CONTRACT = ("execution-only supplemental YCB CAD control: both fixed opening frames pass the raw-PLY vs loaded-GLTF "
            "GPU reference with unchanged relative criteria and informative opposite discrimination, exact "
            "geometry/PNG/runtime provenance and parent qualitative RGB/render placement review; not independent "
            "correspondence, not full-UV serving_render_pass, no accuracy/mobile/automatic-preparation claim")
ADMISSION_NOTE = ("the 640x480 neural serving admission is a separate requirement; this control neither satisfies "
                  "nor alters it and launches no inference")


class AdapterError(Exception):
    """Inputs do not support an execution-only control; nothing is emitted."""


# --------------------------------------------------------------------------
# helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return proof.sha256_file(Path(path))


def canonical(value: Any) -> bytes:
    return proof._canonical(proof._jsonable(value))


def _same(a: Any, b: Any) -> bool:
    return proof._same(a, b)


def _hex(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AdapterError(f"{what} must be a lowercase SHA-256")
    return value


def _loads(raw: bytes, what: str) -> Any:
    try:
        return yc._loads_strict(raw, what)
    except yc.ControlError as exc:
        raise AdapterError(str(exc)) from exc


def _read_regular(path: Path, limit: int, what: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"{what} is missing or not a regular file: {path}")
    if path.stat().st_size > limit:
        raise AdapterError(f"{what} exceeds {limit} bytes")
    return path.read_bytes()


def repo_path(rel: Any, what: str, *, directory: bool) -> Path:
    """A project-relative POSIX path inside the repository without symlink components."""
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel or "\x00" in rel:
        raise AdapterError(f"{what} must be a project-relative POSIX path")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise AdapterError(f"{what} must stay inside the project")
    current = ROOT
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise AdapterError(f"{what} traverses a symlink")
    resolved = current.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise AdapterError(f"{what} escapes the project")
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise AdapterError(f"{what} does not exist as a {'directory' if directory else 'file'}: {rel}")
    return resolved


def tool_hashes() -> dict:
    return {"adapter_sha256": sha256_file(Path(__file__)),
            "gpu_reference_tool_sha256": sha256_file(Path(gpu.__file__)),
            "proof_tool_sha256": sha256_file(Path(proof.__file__)),
            "controls_tool_sha256": sha256_file(Path(yc.__file__))}


# --------------------------------------------------------------------------
# GPU evidence closure (never trusts the top-level pass boolean)


def evidence_names(object_id: int) -> set[str]:
    names = set()
    for frame_id in proof.ONBOARDING[object_id].values():
        tag = f"frame-{frame_id:06d}"
        names.add(f"{tag}-comparison.png")
        for hyp in gpu.HYPOTHESES:
            names.update({f"{tag}-{hyp}-rgb.png", f"{tag}-{hyp}-depth-mm.npy"})
    return names


def load_evidence(evidence_dir: Path, expected_record_sha256: str, object_id: int) -> tuple[dict, dict[str, bytes]]:
    d = Path(evidence_dir)
    if d.is_symlink() or not d.is_dir():
        raise AdapterError("GPU evidence directory is missing or a symlink")
    raw = _read_regular(d / EVIDENCE_RECORD, MAX_JSON_BYTES, EVIDENCE_RECORD)
    if sha256_bytes(raw) != _hex(expected_record_sha256, "expected evidence record SHA-256"):
        raise AdapterError("gpu-reference.json differs from the expected SHA-256")
    record = _loads(raw, EVIDENCE_RECORD)
    if not isinstance(record, dict) or set(record) != RECORD_KEYS or record.get("schema") != gpu.SCHEMA:
        raise AdapterError("gpu-reference.json is not an exact workbench_ycb_gpu_reference_v1 record")
    expected = evidence_names(object_id)
    files = record["files"]
    if not isinstance(files, dict) or set(files) != expected:
        raise AdapterError("GPU evidence file table differs from the exact two-frame inventory")
    if set(os.listdir(d)) != expected | {EVIDENCE_RECORD}:
        raise AdapterError("GPU evidence directory inventory differs (missing, extra, scratch or failure files)")
    artifacts = {}
    for name in sorted(expected):
        data = _read_regular(d / name, MAX_EVIDENCE_FILE_BYTES, name)
        if sha256_bytes(data) != _hex(files[name], f"files[{name}]"):
            raise AdapterError(f"GPU evidence file {name} changed")
        artifacts[name] = data
    return record, artifacts


def _render_from_artifacts(artifacts: Mapping[str, bytes], tag: str, hyp: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = proof.decode_png_rgb(artifacts[f"{tag}-{hyp}-rgb.png"]).astype(np.float64) / 255.0
    depth = np.load(io.BytesIO(artifacts[f"{tag}-{hyp}-depth-mm.npy"]), allow_pickle=False)
    W, H = proof.IMAGE_SIZE
    if rgb.shape != (H, W, 3) or depth.shape != (H, W) or depth.dtype != np.float32:
        raise AdapterError(f"{tag}/{hyp}: stored render has a wrong shape or dtype")
    return rgb, depth.astype(np.float64)


def _close(recorded: Any, recomputed: Any, tolerance: float, what: str) -> None:
    a = np.asarray(recorded, dtype=np.float64)
    b = np.asarray(recomputed, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or np.abs(a - b).max(initial=0.0) > tolerance:
        raise AdapterError(f"recorded {what} is inconsistent with the stored artifacts")


def _metrics_consistent(recorded: Mapping, recomputed: Mapping) -> None:
    for block in ("candidate_vs_reference", "opposite_vs_reference"):
        r, n = recorded.get(block), recomputed.get(block)
        if not isinstance(r, Mapping) or set(r) != set(n):
            raise AdapterError(f"recorded {block} metrics missing or malformed")
        for key in ("depth_median_mm", "depth_p99_mm"):
            _close(r[key], n[key], DEPTH_TOLERANCE_MM, f"{block}.{key}")
        for key in ("rgb_median_per_channel", "rgb_p99_per_channel", "rgb_median_pooled"):
            _close(r[key], n[key], RGB_TOLERANCE, f"{block}.{key}")
    r, n = recorded.get("opposite_vs_candidate_geometry"), recomputed.get("opposite_vs_candidate_geometry")
    if not isinstance(r, Mapping) or set(r) != set(n):
        raise AdapterError("recorded opposite_vs_candidate_geometry missing or malformed")
    for key in ("depth_median_mm", "depth_p99_mm"):
        _close(r[key], n[key], DEPTH_TOLERANCE_MM, f"opposite_vs_candidate_geometry.{key}")
    if r["full_image_mask_differences"] != n["full_image_mask_differences"]:
        raise AdapterError("recorded opposite/candidate mask differences are inconsistent with the artifacts")


def validate_evidence(record: Mapping, artifacts: Mapping[str, bytes], pre: Mapping) -> dict:
    """Complete closure against the freshly re-derived preflight; recompute both frames."""
    binding, valid = pre["binding"], pre["valid"]
    object_id = binding["object_id"]
    if record["tool_sha256"] != sha256_file(Path(gpu.__file__)):
        raise AdapterError("evidence was produced by a different GPU reference tool version")
    checks = {
        "proof_module": _same(record["proof_module"], pre["proof_module"]),
        "binding": _same(record["binding"], binding),
        "thresholds": _same(record["thresholds"], gpu.THRESHOLDS)
                      and all(gpu.THRESHOLDS[k] == proof.THRESHOLDS[k] for k in gpu.THRESHOLD_KEYS),
        "runtime_pins": _same(record["runtime_pins"], pre["runtime"]) and record["runtime_pins"].get("passed") is True,
        "proof_source_lines": _same(record["proof_source_lines"], valid["source"]["source_lines"])
                              and record["proof_source_lines"].get("passed") is True,
        "source_reference": _same(record["source_reference"], {k: v for k, v in pre["arrays"].items()
                                                               if k not in ("positions_m", "normals", "texcoord_0")}),
        "texture": _same(record["texture"], {"png_sha256": binding["png_sha256"], "rgba_sha256": pre["rgba_sha256"],
                                             "rgba_shape": list(pre["rgba"].shape)}),
        "opposite_hypothesis": _same(record["opposite_hypothesis"], pre["opposite_info"]),
        "scratch_removed": isinstance(record["scratch"], Mapping) and record["scratch"].get("removed") is True
                           and record["scratch"].get("candidate", {}).get("sha256") == binding["candidate_glb_sha256"]
                           and record["scratch"].get("opposite", {}).get("sha256") == pre["opposite_info"]["sha256"],
        "claims_false": record["promotion"] is False and isinstance(record["claims"], Mapping)
                        and set(record["claims"]) == set(FALSE_CLAIMS)
                        and all(record["claims"][k] is False for k in FALSE_CLAIMS),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise AdapterError(f"GPU evidence closure differs from the re-derived inputs: {failed}")
    frames_expected = list(proof.ONBOARDING[object_id].values())
    frames = record["frames"]
    if not isinstance(frames, list) or len(frames) != len(frames_expected):
        raise AdapterError("GPU evidence must hold exactly the two fixed opening frames")
    rows = []
    for k, (row, frame_id, ref) in enumerate(zip(frames, frames_expected, valid["references"])):
        tag = f"frame-{frame_id:06d}"
        if not isinstance(row, Mapping) or set(row) != FRAME_KEYS or row["frame_id"] != frame_id or row["tag"] != tag:
            raise AdapterError(f"frame row {k} is not the fixed opening frame {frame_id}")
        hyps = row["hypotheses"]
        if (not isinstance(hyps, Mapping) or set(hyps) != set(gpu.HYPOTHESES)
                or any(not isinstance(h, Mapping) or set(h) != HYPOTHESIS_KEYS or h["post_draw_equal"] is not True
                       for h in hyps.values())):
            raise AdapterError(f"{tag}: hypothesis rows missing or post-draw state changed")
        sources = row["texture_sources"]
        if (row["close_errors"] != [] or row["gl_texture_state_equal"] is not True
                or not isinstance(row["pre_draw_equality"], Mapping)
                or row["pre_draw_equality"].get("equal_excluding_content") is not True
                or not isinstance(sources, Mapping) or sources.get("expected_sha256") != pre["rgba_sha256"]
                or set(sources.get("hypotheses", {})) != set(gpu.HYPOTHESES)
                or any(v.get("equal_to_png_rgba") is not True for v in sources["hypotheses"].values())):
            raise AdapterError(f"{tag}: backend equality, texture source or renderer close evidence is not clean")
        count = ref["count"]
        if not gpu.THRESHOLDS["min_valid_pixels"] <= count <= gpu.THRESHOLDS["max_sample_pixels"]:
            raise AdapterError(f"{tag}: {count} samples outside 128..1024")
        recorded = row["comparison"]
        renders = {hyp: _render_from_artifacts(artifacts, tag, hyp) for hyp in gpu.HYPOTHESES}
        recomputed = gpu.compare_hypotheses(ref["samples"], renders)
        if (not isinstance(recorded, Mapping) or recorded.get("status") != "pass" or recorded.get("reason") is not None
                or recorded.get("samples") != count or recomputed["samples"] != count
                or not _same(recorded.get("coverage"), recomputed["coverage"])
                or any(not c["complete"] for c in recomputed["coverage"].values())):
            raise AdapterError(f"{tag}: recorded comparison is not a clean pass at the re-derived samples")
        if recomputed["status"] != "pass":
            raise AdapterError(f"{tag}: comparison recomputed from stored artifacts is {recomputed['status']} "
                               f"({recomputed['reason']}), not pass")
        _metrics_consistent(recorded, recomputed)
        rows.append({"frame_id": frame_id, "tag": tag, "samples": count, "status_recorded": "pass",
                     "status_recomputed_from_artifacts": "pass",
                     "candidate_rgb_median_pooled": recomputed["candidate_vs_reference"]["rgb_median_pooled"],
                     "opposite_rgb_median_pooled": recomputed["opposite_vs_reference"]["rgb_median_pooled"]})
    relative = gpu.relative_convention([r["comparison"]["status"] for r in frames], len(frames_expected))
    corr, final = gpu.final_status(relative, valid["correspondence"]["status"])
    if (relative is not True or record["relative_serving_uv_convention_pass"] is not True
            or not _same(record["onboarding_correspondence_pass"], corr) or record["final_status"] != final):
        raise AdapterError("recorded relative/correspondence/final status differs from the re-derived values")
    return {"frames": rows, "relative_serving_uv_convention_pass": True,
            "onboarding_correspondence_pass_recorded": corr, "final_status_recorded": final,
            "checks": sorted(checks), "artifact_files_verified": sorted(artifacts),
            "runtime_pins_sha256": sha256_bytes(canonical(record["runtime_pins"])),
            "renderer_factory": "recorded only (renderer not imported by this CPU adapter)"}


# --------------------------------------------------------------------------
# planning (CPU only; no output)


def plan(object_id: int, prepared_dir: Path, prepared_sha256: str, evidence_dir: Path, evidence_sha256: str,
         *, out: Path | None = None) -> dict:
    """Validated execution-only plan; never writes and never approves."""
    if object_id not in proof.ONBOARDING:
        raise AdapterError(f"object {object_id} is not one of {sorted(proof.ONBOARDING)}")
    before_torch = "torch" in sys.modules
    try:
        pre = gpu.preflight(Path(prepared_dir), _hex(prepared_sha256, "expected prepared SHA-256"))
    except (proof.ProofError, yc.ControlError) as exc:
        raise AdapterError(f"prepared proof/runtime preflight failed: {exc}") from exc
    binding, inputs = pre["binding"], pre["valid"]["inputs"]
    if binding["object_id"] != object_id:
        raise AdapterError("prepared proof belongs to another object")
    record, artifacts = load_evidence(Path(evidence_dir), evidence_sha256, object_id)
    closure = validate_evidence(record, artifacts, pre)

    canonical_id, ply_pin = yc.OBJECTS[object_id]
    glb = inputs["glb"]
    glb_pin, receipt_pin = proof.CANDIDATE_PINS[object_id]
    if sha256_bytes(glb) != glb_pin or binding["candidate_glb_sha256"] != glb_pin or inputs["ply_sha256"] != ply_pin:
        raise AdapterError("candidate GLB or source PLY differs from the original pins")
    parsed = proof.parse_glb(glb)
    mesh = inputs["mesh"]
    expected_m = (np.asarray(mesh["positions_mm"], dtype=np.float64) * yc.MM_TO_M).astype("<f4").astype(np.float64)
    if (parsed["positions_m"].shape != expected_m.shape or not np.array_equal(parsed["positions_m"], expected_m)
            or not np.array_equal(parsed["triangles"], np.asarray(mesh["triangles"], dtype=np.int64))):
        raise AdapterError("candidate GLB geometry is not float32(PLY mm x 0.001) metres in source face order")
    unit_error_m = float(np.abs(expected_m - np.asarray(mesh["positions_mm"], dtype=np.float64) / 1000.0).max())

    manifest, manifest_sha = yc.load_manifest(yc.DEFAULT_MANIFEST)
    if manifest_sha != inputs["manifest_sha256"]:
        raise AdapterError("inference manifest differs from the authenticated proof input")
    seed = yc.onboarding_seed(yc.select_object(manifest, object_id), object_id)
    final_frame = proof.ONBOARDING[object_id][FINAL_ONBOARDING_LOGICAL]
    final_image = inputs["images"][-1]
    if (seed["frame_id"] != final_frame or seed["logical_index"] != FINAL_ONBOARDING_LOGICAL
            or final_image["frame_id"] != final_frame
            or not np.array_equal(np.asarray(seed["pose_m"])[:3, :3], final_image["R"])
            or not np.array_equal(np.asarray(seed["translation_mm"]), final_image["t_mm"])):
        raise AdapterError("final onboarding seed differs from the authenticated final opening frame")
    seed_frame_id = f"s{final_frame:06d}"
    glb_path = "<OUT>/model.glb" if out is None else _bundle_path(Path(out) / "model.glb")
    bundle = {"schema": yc.BUNDLE_SCHEMA, "family": yc.FAMILY, "glb_path": glb_path, "glb_sha256": glb_pin,
              "obj_id": object_id, "principal_point_origin": ORIGIN, "seed_pose_m": seed["pose_m"],
              "seed_label": (f"{yc.SEED_PREFIX}{canonical_id}:inference.json final onboarding pose frameId "
                             f"{final_frame} (execution-only supplemental CAD control; {ORIGIN} origin assumed)"),
              "seed_frame_id": seed_frame_id, "metric_provenance": yc.METRIC_PROVENANCE}
    if "torch" in sys.modules and not before_torch:
        raise AdapterError("planning unexpectedly imported torch")
    return {
        "schema": SCHEMA_PLAN, "object_id": object_id, "canonical_ycb_id": canonical_id,
        "bundle": bundle, "bundle_sha256": sha256_bytes(canonical(bundle)) if out is not None else None,
        "glb": glb, "seed": {"frame_id": final_frame, "seed_frame_id": seed_frame_id,
                             "logical_index": FINAL_ONBOARDING_LOGICAL, "translation_mm": seed["translation_mm"],
                             "source": "authenticated inference.json final onboarding row (logicalIndex 9)"},
        "units": {"source": "PLY millimetres", "glb": "metres = float32(PLY mm x 0.001)",
                  "max_float32_rounding_m": unit_error_m, "seed_translation": "metres"},
        "closure": closure,
        "pins": {"prepared_sha256": binding["prepared_sha256"], "evidence_record_sha256": evidence_sha256,
                 "candidate_glb_sha256": glb_pin, "candidate_receipt_sha256": receipt_pin,
                 "png_sha256": binding["png_sha256"], "ply_sha256": ply_pin, "manifest_sha256": manifest_sha,
                 "runtime_pins_sha256": closure["runtime_pins_sha256"], "seed_frame_id": seed_frame_id,
                 "principal_point_origin": ORIGIN,
                 "planned_out": None if out is None else Path(os.path.abspath(out)).relative_to(ROOT.resolve()).as_posix(),
                 "bundle_sha256": sha256_bytes(canonical(bundle)) if out is not None else None},
        "opening_rgb": {str(img["frame_id"]): img["rgb_sha256"] for img in inputs["images"]},
        "tools": tool_hashes(),
        "not_read": ["evaluator.json", "scored frames/poses/labels", "masks", "depth (dataset)"],
    }


def _bundle_path(glb: Path) -> str:
    path = Path(os.path.abspath(glb))
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise AdapterError("bundle GLB path must be inside the project") from None


# --------------------------------------------------------------------------
# reviewed map and qualitative receipt (emit only)


def load_map() -> tuple[dict, str]:
    if not MAP_PATH.is_file() or MAP_PATH.is_symlink():
        raise AdapterError(f"reviewed map {MAP_PATH.relative_to(ROOT).as_posix()} is missing; emission blocked")
    raw = _read_regular(MAP_PATH, MAX_JSON_BYTES, "reviewed map")
    reviewed = _loads(raw, "reviewed map")
    if not isinstance(reviewed, dict) or set(reviewed) != MAP_KEYS or reviewed["schema"] != SCHEMA_MAP:
        raise AdapterError("reviewed map is not an exact workbench_ycb_execution_controls_reviewed_v1 document")
    current = tool_hashes()
    for key in ("adapter_sha256", "gpu_reference_tool_sha256", "proof_tool_sha256", "controls_tool_sha256"):
        if _hex(reviewed[key], key) != current[key]:
            raise AdapterError(f"reviewed map {key} differs from the current file")
    if reviewed["contract"] != CONTRACT or not isinstance(reviewed["entries"], dict):
        raise AdapterError("reviewed map contract text or entries differ")
    return reviewed, sha256_bytes(raw)


def map_entry(reviewed: Mapping, object_id: int) -> dict:
    entry = reviewed["entries"].get(str(object_id))
    if entry is None:
        raise AdapterError(f"no reviewed map entry for object {object_id}; emission blocked")
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        raise AdapterError("reviewed map entry does not have exactly the mandatory pins")
    if entry["object_id"] != object_id or entry["canonical_ycb_id"] != yc.OBJECTS[object_id][0]:
        raise AdapterError("reviewed map entry names another object")
    for key in ("prepared_sha256", "evidence_record_sha256", "candidate_glb_sha256", "candidate_receipt_sha256",
                "png_sha256", "ply_sha256", "manifest_sha256", "runtime_pins_sha256", "qualitative_receipt_sha256",
                "bundle_sha256"):
        _hex(entry[key], key)
    planned_out_path(entry["planned_out"])
    frames = [str(f) for f in proof.ONBOARDING[object_id].values()]
    overlays = entry["overlays"]
    if (not isinstance(overlays, dict) or sorted(overlays) != sorted(frames)
            or any(not isinstance(v, dict) or set(v) != {"path", "sha256"} for v in overlays.values())):
        raise AdapterError("reviewed map overlays must pin exactly the two opening frames (path, sha256)")
    for value in overlays.values():
        _hex(value["sha256"], "overlay sha256")
    if entry["principal_point_origin"] != ORIGIN:
        raise AdapterError("reviewed map must pin the assumed integer_centers origin")
    for key in ("reviewed_by", "reviewed_at"):
        if not isinstance(entry[key], str) or not entry[key].strip() or len(entry[key]) > 200:
            raise AdapterError(f"reviewed map {key} must be a non-empty string")
    return entry


def validate_qualitative(entry: Mapping, planned: Mapping) -> dict:
    path = repo_path(entry["qualitative_receipt"], "qualitative receipt", directory=False)
    raw = _read_regular(path, MAX_JSON_BYTES, "qualitative receipt")
    if sha256_bytes(raw) != entry["qualitative_receipt_sha256"]:
        raise AdapterError("qualitative receipt differs from its reviewed SHA-256")
    q = _loads(raw, "qualitative receipt")
    if not isinstance(q, dict) or set(q) != QUALITATIVE_KEYS or q["schema"] != SCHEMA_QUALITATIVE:
        raise AdapterError("qualitative receipt is not an exact workbench_ycb_execution_qualitative_review_v1")
    pins = planned["pins"]
    if (q["object_id"] != planned["object_id"] or q["evidence_record_sha256"] != pins["evidence_record_sha256"]
            or q["prepared_sha256"] != pins["prepared_sha256"]
            or q["candidate_glb_sha256"] != pins["candidate_glb_sha256"]
            or q["decision"] != QUALITATIVE_DECISION or q["claims"] != QUALITATIVE_CLAIMS
            or not isinstance(q["reviewer"], str) or not q["reviewer"].strip()
            or not isinstance(q["reviewed_at"], str) or not q["reviewed_at"].strip()):
        raise AdapterError("qualitative receipt does not bind this evidence/candidate with an acceptable decision")
    frames = [int(f) for f in proof.ONBOARDING[planned["object_id"]].values()]
    rows = q["frames"]
    if not isinstance(rows, list) or len(rows) != len(frames):
        raise AdapterError("qualitative receipt must review exactly the two opening frames")
    overlays = {}
    for row, frame_id in zip(rows, frames):
        pinned = entry["overlays"][str(frame_id)]
        if (not isinstance(row, dict) or set(row) != QUALITATIVE_FRAME_KEYS or row["frame_id"] != frame_id
                or row["rgb_sha256"] != planned["opening_rgb"][str(frame_id)]
                or row["overlay_sha256"] != pinned["sha256"] or row["placement"] != "acceptable"
                or not isinstance(row["notes"], str)):
            raise AdapterError(f"qualitative review of frame {frame_id} is missing, mismatched or not acceptable")
        overlay = _read_regular(repo_path(pinned["path"], f"overlay {frame_id}", directory=False), 32 << 20, "overlay")
        if sha256_bytes(overlay) != pinned["sha256"]:
            raise AdapterError(f"overlay for frame {frame_id} differs from its reviewed SHA-256")
        image = proof.decode_png_rgb(overlay)
        if image.ndim != 3 or image.size == 0:
            raise AdapterError(f"overlay for frame {frame_id} is not a decodable image")
        overlays[str(frame_id)] = {"sha256": pinned["sha256"], "shape": list(image.shape)}
    return {"sha256": entry["qualitative_receipt_sha256"], "reviewer": q["reviewer"], "reviewed_at": q["reviewed_at"],
            "decision": q["decision"], "overlays": overlays}


# --------------------------------------------------------------------------
# output


def _derived_usage() -> int:
    root = DERIVED_ROOT.resolve()
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()) if root.exists() else 0


def check_out(out: Path) -> Path:
    out = Path(os.path.abspath(out))
    root = DERIVED_ROOT.resolve()
    if root not in out.parents:
        raise AdapterError(f"output must be a new directory inside {root}")
    for part in [out.parent, *out.parent.parents]:
        if part.exists() and part.is_symlink():
            raise AdapterError("output path traverses a symlink")
    if os.path.lexists(out):
        raise AdapterError(f"output {out} already exists; outputs are immutable")
    return out


def planned_out_path(rel: Any) -> Path:
    """The reviewed planned output: a project-relative path strictly inside the derived root."""
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel or "\x00" in rel:
        raise AdapterError("planned_out must be a project-relative POSIX path")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise AdapterError("planned_out must stay inside the project")
    path = Path(os.path.abspath(ROOT.resolve().joinpath(*pure.parts)))
    if DERIVED_ROOT.resolve() not in path.parents:
        raise AdapterError("planned_out must be inside .cache/video-workbench/derived")
    for part in [path, *path.parents]:
        if part.exists() and part.is_symlink():
            raise AdapterError("planned_out traverses a symlink")
    return path


def build_outputs(planned: Mapping, entry: Mapping, qualitative: Mapping) -> dict[str, bytes]:
    """Deterministic bytes of every emitted file, re-derivable at validation time."""
    tools = planned["tools"]
    bundle_bytes = canonical(planned["bundle"])
    glb = planned["glb"]
    sidecar = {
        "schema": SCHEMA_SIDECAR, "object_id": planned["object_id"], "canonical_ycb_id": planned["canonical_ycb_id"],
        **SIDECAR_FLAGS, "serving_render_pass_unchanged": "pending_render_phase (accepted proof record not modified)",
        "principal_point_origin": {"value": ORIGIN, "provenance": "assumed"},
        "units": planned["units"], "seed": planned["seed"],
        "bundle_sha256": sha256_bytes(bundle_bytes), "glb_sha256": sha256_bytes(glb),
        "evidence_record_sha256": entry["evidence_record_sha256"], "prepared_sha256": entry["prepared_sha256"],
        "qualitative_receipt_sha256": qualitative["sha256"], "overlays": qualitative["overlays"],
        "map_entry_sha256": sha256_bytes(canonical(entry)), **tools,
        "contract": CONTRACT, "admission": ADMISSION_NOTE,
        "claims": {"accuracy": False, "mobile": False, "automatic_rgb_preparation": False,
                   "independent_correspondence": False, "promotion_of_original_candidate": False},
    }
    sidecar_bytes = canonical(sidecar)
    receipt = {
        "schema": SCHEMA_RECEIPT, "status": "emitted_execution_only_control", "object_id": planned["object_id"],
        "files": {"model.glb": sha256_bytes(glb), "metric-bundle.json": sha256_bytes(bundle_bytes),
                  "execution-control-sidecar.json": sha256_bytes(sidecar_bytes)},
        "evidence_closure": planned["closure"], "qualitative_review": qualitative, "pins": planned["pins"],
        "map_entry_sha256": sidecar["map_entry_sha256"], "tools": tools, "not_read": planned["not_read"],
        "budgets": {"derived_budget_bytes": DERIVED_BUDGET_BYTES, "free_reserve_bytes": FREE_RESERVE_BYTES,
                    "new_input_bytes_acquired": 0, "new_input_budget_bytes": NEW_INPUT_BUDGET_BYTES},
        "original_records_modified": False, "admission": ADMISSION_NOTE,
    }
    return {"model.glb": glb, "metric-bundle.json": bundle_bytes, "execution-control-sidecar.json": sidecar_bytes,
            "emission-receipt.json": canonical(receipt)}


def approved_closure(entry: Mapping, out: Path) -> tuple[dict, dict, dict[str, bytes]]:
    """Re-run the full plan and qualitative review for the map's exact closure and output path."""
    object_id = entry["object_id"]
    if planned_out_path(entry["planned_out"]) != Path(os.path.abspath(out)):
        raise AdapterError("output directory differs from the reviewed planned_out")
    prepared_dir = repo_path(entry["prepared_dir"], "prepared_dir", directory=True)
    evidence_dir = repo_path(entry["evidence_dir"], "evidence_dir", directory=True)
    planned = plan(object_id, prepared_dir, entry["prepared_sha256"], evidence_dir, entry["evidence_record_sha256"],
                   out=Path(os.path.abspath(out)))
    mismatched = sorted(k for k, v in planned["pins"].items() if entry.get(k) != v)
    if mismatched:
        raise AdapterError(f"reviewed map pins differ from the re-derived values: {mismatched}")
    qualitative = validate_qualitative(entry, planned)
    blobs = build_outputs(planned, entry, qualitative)
    if sha256_bytes(blobs["metric-bundle.json"]) != entry["bundle_sha256"]:
        raise AdapterError("re-derived bundle differs from the reviewed bundle_sha256")
    return planned, qualitative, blobs


def emit(object_id: int, out: Path) -> dict:
    out = check_out(out)
    reviewed, _map_sha = load_map()
    entry = map_entry(reviewed, object_id)
    _planned, _qualitative, blobs = approved_closure(entry, out)
    planned_bytes = sum(len(b) for b in blobs.values())
    if planned_bytes > MAX_OUTPUT_BYTES:
        raise AdapterError("emission exceeds its output cap")
    if _derived_usage() + planned_bytes > DERIVED_BUDGET_BYTES:
        raise AdapterError("1 GiB derived budget would be exceeded")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free - planned_bytes < FREE_RESERVE_BYTES:
        raise AdapterError("1 GiB free-disk reserve would be violated")
    _publish(out, blobs, object_id)
    return {"directory": out.as_posix(), "files": {k: sha256_bytes(v) for k, v in blobs.items()},
            "validation": validate_output(out)}


def _publish(out: Path, blobs: Mapping[str, bytes], object_id: int) -> None:
    staging = out.parent / f".{out.name}.staging-{secrets.token_hex(6)}"
    stage = "staging"
    try:
        staging.mkdir()
        for name, data in blobs.items():
            stage = f"write {name}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(staging / name, flags, 0o644), "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if sha256_file(staging / name) != sha256_bytes(data):
                raise AdapterError(f"staged {name} does not verify")
        stage = "publish"
        if os.path.lexists(out):
            raise AdapterError("output appeared during emission")
        os.rename(staging, out)
    except BaseException as exc:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            out.mkdir()
            failure = {"schema": SCHEMA_RECEIPT, "status": "failed", "object_id": object_id, "stage": stage,
                       "error": f"{type(exc).__name__}: {exc}"[:2000], "bundle_emitted": False,
                       "note": "no usable control; staging removed"}
            proof.write_new(out, "failure.json", canonical(failure))
        except OSError:
            pass
        raise


def validate_output(out: Path) -> dict:
    """Launch gate: run immediately before loading the bundle.

    The object is identified only by the reviewed entry whose ``planned_out`` is
    this directory. The full plan (prepared proof, runtime pins, GPU evidence
    recomputed from its artifacts, manifest seed) and the qualitative review are
    re-run from the map's exact closure. All four emitted files must equal the
    independently re-derived bytes, and the bundle must match the reviewed
    ``bundle_sha256``. Sidecar and receipt self-consistency is never trusted.
    """
    out = Path(os.path.abspath(out))
    if DERIVED_ROOT.resolve() not in out.parents or out.is_symlink() or not out.is_dir():
        raise AdapterError("control directory must be a real directory inside the derived root")
    if sorted(os.listdir(out)) != sorted(OUTPUT_FILES):
        raise AdapterError("control directory inventory differs (failed, partial or tampered emission)")
    reviewed, _map_sha = load_map()
    matches = []
    for object_id in sorted(proof.ONBOARDING):
        raw_entry = reviewed["entries"].get(str(object_id))
        if not isinstance(raw_entry, dict):
            continue
        try:
            same = planned_out_path(raw_entry.get("planned_out")) == out
        except AdapterError:
            continue
        if same:
            matches.append(map_entry(reviewed, object_id))
    if len(matches) != 1:
        raise AdapterError("no single reviewed map entry approves this output directory")
    entry = matches[0]
    planned, _qualitative, blobs = approved_closure(entry, out)
    for name in OUTPUT_FILES:
        actual = _read_regular(out / name, MAX_OUTPUT_BYTES, name)
        if actual != blobs[name]:
            raise AdapterError(f"{name} differs from the independently re-derived output")
    from bench import workbench_neural as wn
    bundle = wn.MetricBundle.from_file(out / "metric-bundle.json", root=ROOT)
    expected = planned["bundle"]
    if (Path(bundle.glb_path).resolve() != (out / "model.glb").resolve() or bundle.glb_sha256 != expected["glb_sha256"]
            or bundle.family != yc.FAMILY or bundle.principal_point_origin != ORIGIN
            or bundle.metric_provenance != yc.METRIC_PROVENANCE or bundle.obj_id != entry["object_id"]
            or bundle.seed_label != expected["seed_label"] or bundle.seed_frame_id != expected["seed_frame_id"]
            or bundle.seed_pose_m is None
            or not np.array_equal(np.asarray(bundle.seed_pose_m), np.asarray(expected["seed_pose_m"]))):
        raise AdapterError("loaded metric bundle differs from the authenticated plan")
    # Final rehash immediately before returning control to the launcher.
    for name in OUTPUT_FILES:
        if sha256_file(out / name) != sha256_bytes(blobs[name]):
            raise AdapterError(f"{name} changed during validation")
    return {"status": "valid_execution_only_control", "object_id": entry["object_id"],
            "bundle_sha256": entry["bundle_sha256"], "glb_sha256": expected["glb_sha256"],
            "seed_frame_id": expected["seed_frame_id"], "flags": dict(SIDECAR_FLAGS), "admission": ADMISSION_NOTE,
            "note": "validated against the re-derived closure; load the bundle immediately, without re-copying"}


# --------------------------------------------------------------------------
# CLI


def _map_status(object_id: int) -> dict:
    try:
        reviewed, map_sha = load_map()
    except AdapterError as exc:
        return {"status": "blocked", "reason": str(exc)}
    try:
        map_entry(reviewed, object_id)
    except AdapterError as exc:
        return {"status": "blocked", "map_sha256": map_sha, "reason": str(exc)}
    return {"status": "entry_present (emit re-validates every pin)", "map_sha256": map_sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execution-only supplemental YCB CAD-control adapter (no GPU).")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inspect", help="CPU-only plan and pins; writes nothing, approves nothing")
    s.add_argument("--object-id", type=int, required=True, choices=sorted(proof.ONBOARDING))
    s.add_argument("--prepared", required=True)
    s.add_argument("--expected-prepared-sha256", required=True)
    s.add_argument("--evidence", required=True, help="GPU reference output directory")
    s.add_argument("--expected-evidence-sha256", required=True, help="SHA-256 of its gpu-reference.json")
    s.add_argument("--planned-out", help="planned output directory (only to compute the bundle's glb_path/hash)")
    s = sub.add_parser("emit", help="emit from the fixed reviewed map only")
    s.add_argument("--object-id", type=int, required=True, choices=sorted(proof.ONBOARDING))
    s.add_argument("--out", required=True, help="new directory under .cache/video-workbench/derived")
    s = sub.add_parser("validate", help="CPU validation before a launcher consumes the bundle")
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            planned_out = check_out(Path(args.planned_out)) if args.planned_out else None
            result = plan(args.object_id, Path(args.prepared), args.expected_prepared_sha256, Path(args.evidence),
                          args.expected_evidence_sha256, out=planned_out)
            result = {k: v for k, v in result.items() if k != "glb"}
            result["map"] = _map_status(args.object_id)
            result["emitted"] = False
        elif args.command == "emit":
            result = emit(args.object_id, Path(args.out))
        else:
            result = validate_output(Path(args.out))
        print(json.dumps(proof._jsonable(result), indent=2, sort_keys=True))
        return 0
    except (AdapterError, proof.ProofError, yc.ControlError, OSError, KeyError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
