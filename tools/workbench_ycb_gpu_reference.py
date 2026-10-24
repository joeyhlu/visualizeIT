"""Raw-source GPU reference for the YCB UV convention (shared-backend relative evidence).

The accepted CPU proof (``tools/workbench_ycb_uv_proof.py``, frozen) cannot
decide the serving render, because the serving texture sampler is mipmapped and
repeat-wrapped while its CPU oracle is base-level clamp bilinear. This
experiment builds an independent reference **directly from the authenticated
original PLY** (positions in mm -> float32 metres, source face order, raw
``texture_u/texture_v``) and the official PNG. It does not use any GLB vertex
array, exporter/controls UV helper or glTF round trip. The reference is drawn
through the **same** pyrender sampler, shader, material settings and
zero-storage framebuffer as the GLB serving path. Rendering the reference, the
original candidate and an opposite-UV hypothesis at both onboarding poses
therefore tests the GLB loader/conversion boundary without modelling mipmaps.

What this is NOT:

* It is relative UV-convention evidence only. Reference and candidate share the
  pyrender rasterizer, texture upload, mipmap generation, sampler, shader and
  framebuffer, so common-mode backend faults remain invisible.
* It is not printed-landmark correspondence, which is still required before any
  promotion. It is not independent tracking accuracy, mobile readiness or
  automatic RGB preparation.
* It never promotes anything and writes no bundle or verified flag. It never
  rewrites the accepted proof's ``serving_render_pass``.

Phases:

``preflight`` (CPU only; no output, no GPU or renderer import):

* enforces the frozen proof module SHA and its reviewed runtime closure;
* calls the frozen ``proof.validate_prepared`` with the required externally
  expected prepared.json SHA;
* builds and validates the raw-source primitive arrays and texture;
* checks the runtime pins.

``run`` (explicit parent-scheduled GPU command): preflight, then a fresh
output directory, then one ``gpu_stage`` around the whole experiment. Per
onboarding frame it constructs the reference, candidate and opposite renderers
and asserts material/sampler/primitive-format/scene equality BEFORE drawing.
It then draws each with unchanged default serving samplers, records the actual
GL texture state and the pre/post-draw equality, and compares at the frozen CPU
interior sample pixels only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_ycb_uv_proof as proof  # noqa: E402  (frozen accepted proof; read-only use)

yc = proof.yc
ProofError = proof.ProofError

SCHEMA = "workbench_ycb_gpu_reference_v1"
PROOF_TOOL_SHA256 = "764f58f6ac93d4b85e89fdf41acb6dba944ddc4520a583439c3a3232c9676959"
PROOF_TOOL_PATH = ROOT / "tools/workbench_ycb_uv_proof.py"
HYPOTHESES = ("reference", "candidate", "opposite")
MAX_SCRATCH_GLB_BYTES = 32 << 20
MAX_OUTPUT_BYTES = 128 << 20          # this experiment's own cap (renders, depth, strips, receipts, scratch)
DERIVED_ROOT = ROOT / ".cache/video-workbench/derived"
DERIVED_BUDGET_BYTES = 1 << 30
# Fixed thresholds, reused unchanged from the frozen proof (declared before any run).
THRESHOLD_KEYS = ("min_valid_pixels", "max_sample_pixels", "depth_median_mm", "depth_p99_mm", "rgb_median",
                  "rgb_p99", "candidate_vs_opposite_factor", "opposite_min_median")
THRESHOLDS = {k: proof.THRESHOLDS[k] for k in THRESHOLD_KEYS}

# Backend files the raw-source reference construction relies on, read from the
# actual GPU env by the implementer (pending Sol review). Enforced exactly; any
# drift or absence refuses to run. They are shared by every hypothesis.
RUNTIME_PINS = {
    "pyrender/primitive.py": "c8037fb40905661423f365be594d1eef10f4c083ddd90e29f04c139c45d16e28",
    "pyrender/utils.py": "7c6b0d72ece9287f7b871befa94ce14772f73cc22bc81819be2dcaf536c8cabc",
    "pyrender/shader_program.py": "1f6969813351e1625d4110172b74abe26e17a147a8a72f0bccfbb7e4536673b4",
    "pyrender/shaders/flat.vert": "0aca30981e2752e840782430387ba48af53889a2008836332ff631efaa301ef8",
    "pyrender/scene.py": "310f8f92714bfa7277554f5498bf7528fe76bebe3459394dcb0af48980a87ac6",
    "pyrender/renderer.py": "364c16c98487e808b3124bdaa652ca8592bbf200cc63b90049057912c1c8f1b6",
    "pyrender/offscreen.py": "dab44974b08eeee46774653714524a500ff49bdc59cb0657f9720e00d21d3099",
    "bench/workbench_render.py": "da9d79157289d093399cc055ab98e8c8534f3aa2f0cbc52ac288c327d2fdf57a",
    "bench/workbench_telemetry.py": "4610b1f922d64b733b9f1d301712b386332d44af156635bf2ec1f76826663907",
    "bench/quality_render_stability.py": "66f907f0c6c3f9ed32c6edbedb996a9c45e5ab1aea1d1914f99026a7f725c6bd",
}
# Keys excluded from cross-hypothesis equality (array/pixel content and labels)
# and from pre/post-draw equality (GL residency).
CONTENT_KEYS = frozenset({"source_sha256", "positions_sha256", "normals_sha256", "texcoord_0_sha256", "asset_label"})
VOLATILE_KEYS = frozenset({"offscreen_present", "texture_id"})
LIMITATIONS = [
    "shared backend: reference, candidate and opposite use the same pyrender rasterizer, texture upload, mipmap "
    "generation, sampler, flat shader and zero-storage framebuffer; common-mode faults there are invisible",
    "relative evidence of the GLB loader/exporter conversion boundary only; not printed-landmark correspondence",
    "reference geometry is float32(float64(PLY mm)/1000) metres; candidate geometry is the exporter's GLB float32",
    "reference normals are independent face normals; the flat shader's colour does not use them",
    "the reference renderer's constructor necessarily loads the candidate GLB; that mesh node is removed before "
    "the first draw and is never used for reference arrays (material scalar settings only are copied)",
    "comparison only at frozen CPU interior sample pixels; no natural-RGB full-image metric",
]


# --------------------------------------------------------------------------
# small helpers

def _sha_array(a: np.ndarray, dtype: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    return proof._jsonable(value)


def verify_proof_module() -> dict:
    """The imported accepted proof must be the frozen file at its repository path."""
    actual_path = Path(proof.__file__).resolve()
    if actual_path != PROOF_TOOL_PATH.resolve():
        raise ProofError(f"imported proof module {actual_path} is not {PROOF_TOOL_PATH}")
    actual = proof.sha256_file(actual_path)
    if actual != PROOF_TOOL_SHA256:
        raise ProofError("accepted proof tool differs from its frozen SHA-256")
    return {"path": str(actual_path), "sha256": actual}


def runtime_pin_paths() -> dict[str, Path | None]:
    pyrender_dir = proof._runtime_package_dir("pyrender")
    out: dict[str, Path | None] = {}
    for key in RUNTIME_PINS:
        if key.startswith("pyrender/"):
            out[key] = None if pyrender_dir is None else pyrender_dir / key.split("/", 1)[1]
        else:
            out[key] = ROOT / key
    return out


def runtime_pin_checks() -> dict:
    rows, ok = {}, True
    for key, path in runtime_pin_paths().items():
        actual = proof.sha256_file(path) if path is not None and path.is_file() else None
        rows[key] = {"path": None if path is None else str(path), "sha256": actual, "pinned_sha256": RUNTIME_PINS[key],
                     "passed": actual == RUNTIME_PINS[key]}
        ok &= rows[key]["passed"]
    return {"files": rows, "passed": bool(ok), "status": "implementer-recorded pins (pending Sol review); enforced"}


# --------------------------------------------------------------------------
# pure: raw-source primitive and texture

def source_primitive_arrays(positions_mm: Any, triangles: Any, uv: Any) -> dict:
    """Unindexed per-corner arrays in the PLY face order, exactly like
    ``pyrender.Mesh.from_trimesh(smooth=False)`` lays out the serving mesh, but
    built only from the raw PLY: positions float32 metres, raw (u, v_ply)."""
    if uv is None:
        raise ProofError("source PLY has no texture_u/texture_v")
    P, T, UV = np.asarray(positions_mm), np.asarray(triangles), np.asarray(uv)
    if P.ndim != 2 or P.shape[1] != 3 or P.dtype.kind != "f" or len(P) == 0:
        raise ProofError("source positions must be a non-empty (n, 3) float array in mm")
    if T.ndim != 2 or T.shape[1] != 3 or T.dtype.kind not in "iu" or len(T) == 0:
        raise ProofError("source triangles must be a non-empty (m, 3) integer array")
    if UV.ndim != 2 or UV.shape != (len(P), 2) or UV.dtype.kind != "f":
        raise ProofError("source UV must be an (n, 2) float array aligned with the vertices")
    if not np.isfinite(P).all() or not np.isfinite(UV).all():
        raise ProofError("source positions/UV must be finite")
    T = T.astype(np.int64)
    if T.min() < 0 or T.max() >= len(P):
        raise ProofError("source triangle index out of range")
    vertices_m = (P.astype(np.float64) / 1000.0).astype(np.float32)
    if not np.isfinite(vertices_m).all():
        raise ProofError("metre conversion is not finite")
    extent = float(np.ptp(vertices_m.astype(np.float64), axis=0).max())
    if not 0 < extent <= 2:
        raise ProofError("expected a metric everyday object (0 < extent <= 2 m)")
    corners = np.ascontiguousarray(vertices_m[T].reshape(-1, 3))
    texcoord = np.ascontiguousarray(UV.astype(np.float64)[T].reshape(-1, 2).astype(np.float32))
    tri = corners.reshape(-1, 3, 3).astype(np.float64)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    normals = np.ascontiguousarray(np.repeat(n, 3, axis=0))
    return {
        "positions_m": corners, "normals": normals, "texcoord_0": texcoord,
        "vertex_count": int(len(P)), "face_count": int(len(T)), "corner_count": int(len(corners)),
        "extent_m": extent, "degenerate_faces": int((norm[:, 0] == 0).sum()),
        "hashes": {"positions_m_float32": _sha_array(corners, np.float32),
                   "texcoord_0_float32": _sha_array(texcoord, np.float32),
                   "normals_float64": _sha_array(normals, np.float64),
                   "triangles_int64": _sha_array(T, np.int64)},
        "units": "metres, float32(float64(PLY mm) / 1000)",
        "uv": "raw PLY (texture_u, texture_v), no v flip; no GLB, glTF or exporter/controls helper",
        "basis": "PLY model frame; identity scene node (the candidate's single identity node is verified by the proof)",
        "layout": "unindexed per-corner arrays in source face order (as Mesh.from_trimesh(smooth=False))",
    }


def texture_rgba(png_bytes: bytes, decoded_rgb: Any) -> np.ndarray:
    """Authenticated official PNG -> RGBA uint8 (alpha 255), as pyrender's
    format_texture_source pads an RGB source for a base-colour texture."""
    from PIL import Image
    with Image.open(io.BytesIO(png_bytes)) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ProofError("official texture must be an RGB PNG (no alpha/palette ambiguity)")
    rgb = np.asarray(decoded_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or rgb.size == 0:
        raise ProofError("decoded texture must be a non-empty (h, w, 3) uint8 array")
    rgba = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


# --------------------------------------------------------------------------
# pure: backend state capture and comparison (duck-typed; testable without GL)

def _floats(value: Any) -> list[float] | None:
    return None if value is None else [float(v) for v in np.asarray(value, dtype=np.float64).ravel()]


def sampler_state(sampler: Any) -> dict:
    return {k: (None if getattr(sampler, k, None) is None else int(getattr(sampler, k)))
            for k in ("magFilter", "minFilter", "wrapS", "wrapT")}


def material_state(material: Any) -> dict:
    tex = getattr(material, "baseColorTexture", None)
    texture = None
    if tex is not None:
        source = getattr(tex, "source", None)
        texture = {"type": type(tex).__name__, "source_channels": getattr(tex, "source_channels", None),
                   "tex_type": int(getattr(tex, "tex_type")), "data_format": int(getattr(tex, "data_format")),
                   "source_shape": None if source is None else list(np.asarray(source).shape),
                   "source_dtype": None if source is None else str(np.asarray(source).dtype),
                   "source_sha256": None if source is None else hashlib.sha256(
                       np.ascontiguousarray(source).tobytes()).hexdigest(),
                   "sampler": sampler_state(getattr(tex, "sampler", None)),
                   "is_transparent": bool(tex.is_transparent())}
    return {"type": type(material).__name__, "alphaMode": getattr(material, "alphaMode", None),
            "alphaCutoff": float(getattr(material, "alphaCutoff")), "doubleSided": bool(getattr(material, "doubleSided")),
            "smooth": bool(getattr(material, "smooth")), "wireframe": bool(getattr(material, "wireframe")),
            "baseColorFactor": _floats(getattr(material, "baseColorFactor", None)),
            "metallicFactor": float(getattr(material, "metallicFactor")),
            "roughnessFactor": float(getattr(material, "roughnessFactor")),
            "emissiveFactor": _floats(getattr(material, "emissiveFactor", None)),
            "other_textures_present": {k: getattr(material, k, None) is not None for k in (
                "normalTexture", "occlusionTexture", "emissiveTexture", "metallicRoughnessTexture")},
            "is_transparent": bool(getattr(material, "is_transparent")), "tex_flags": int(getattr(material, "tex_flags")),
            "base_color_texture": texture}


def primitive_state(prim: Any) -> dict:
    def shape(name: str) -> list[int] | None:
        value = getattr(prim, name, None)
        return None if value is None else list(np.asarray(value).shape)

    return {"mode": int(getattr(prim, "mode")), "buf_flags": int(getattr(prim, "buf_flags")),
            "positions_shape": shape("positions"), "normals_shape": shape("normals"),
            "texcoord_0_shape": shape("texcoord_0"), "texcoord_1_shape": shape("texcoord_1"),
            "tangents_shape": shape("tangents"), "color_0_shape": shape("color_0"), "indices_shape": shape("indices"),
            "poses_shape": shape("poses"), "is_transparent": bool(getattr(prim, "is_transparent")),
            "upload_dtype": "float32 (pyrender Primitive._add_to_context casts every attribute)",
            "positions_sha256": _sha_array(prim.positions, np.float32),
            "normals_sha256": None if prim.normals is None else _sha_array(prim.normals, np.float32),
            "texcoord_0_sha256": None if prim.texcoord_0 is None else _sha_array(prim.texcoord_0, np.float32),
            "material": material_state(prim.material)}


def scene_state(inner: Any) -> dict:
    scene = inner.scene
    nodes = list(scene.mesh_nodes)
    primitives, poses = [], []
    for node in nodes:
        poses.append(bool(np.array_equal(np.asarray(scene.get_pose(node), dtype=np.float64), np.eye(4))))
        mesh = node.mesh
        primitives.extend(primitive_state(p) for p in mesh.primitives)
    return {"mesh_nodes": len(nodes), "node_pose_identity": poses, "primitives": primitives,
            "mesh_visible": [bool(n.mesh.is_visible) for n in nodes],
            "bg_color": _floats(scene.bg_color), "ambient_light": _floats(scene.ambient_light),
            "unlit": bool(inner.unlit), "disable_multisampling": bool(inner.disable_multisampling),
            "coordinate_mode": inner.coordinate_mode, "render_policy": inner.render_policy,
            "offscreen_present": inner.offscreen is not None, "asset_label": str(inner.asset_sha256)}


def compare_states(a: Any, b: Any, ignore: Iterable[str] = (), _path: str = "") -> list[str]:
    """Dotted paths whose values differ (exact equality), skipping ``ignore`` keys."""
    ignore = frozenset(ignore)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            if key in ignore:
                continue
            if key not in a or key not in b:
                out.append(f"{_path}{key}")
            else:
                out.extend(compare_states(a[key], b[key], ignore, f"{_path}{key}."))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [_path.rstrip(".") or "<root>"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(compare_states(x, y, ignore, f"{_path}{i}."))
        return out
    same = json.dumps(_jsonable(a), sort_keys=True) == json.dumps(_jsonable(b), sort_keys=True)
    return [] if same else [_path.rstrip(".") or "<root>"]


def require_backend_equality(states: Mapping[str, Mapping]) -> dict:
    """Reference, candidate and opposite must share every non-content backend setting."""
    ignore = CONTENT_KEYS | VOLATILE_KEYS
    diffs = {f"reference_vs_{h}": compare_states(states["reference"], states[h], ignore) for h in ("candidate", "opposite")}
    if any(diffs.values()):
        raise ProofError(f"backend state differs between hypotheses (cannot adapt safely): {diffs}")
    for name, state in states.items():
        if state["mesh_nodes"] != 1 or len(state["primitives"]) != 1 or state["node_pose_identity"] != [True]:
            raise ProofError(f"{name}: expected exactly one identity mesh node with one primitive")
        mat = state["primitives"][0]["material"]
        if mat["base_color_texture"] is None or any(mat["other_textures_present"].values()):
            raise ProofError(f"{name}: expected exactly one base-colour texture and no other textures")
    return {"equal_excluding_content": True, "ignored_keys": sorted(ignore)}


def require_texture_sources(sources: Mapping[str, Any], expected_rgba: Any) -> dict:
    """Every hypothesis' base-colour Texture.source must equal the independently
    decoded authenticated official PNG RGBA exactly (shape, dtype, content)."""
    expected = np.asarray(expected_rgba)
    if expected.ndim != 3 or expected.shape[2] != 4 or expected.dtype != np.uint8:
        raise ProofError("expected texture must be an (h, w, 4) uint8 RGBA array")
    expected_sha = hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()
    if set(sources) != set(HYPOTHESES):
        raise ProofError(f"texture sources must be given for exactly {list(HYPOTHESES)}")
    evidence, bad = {}, []
    for hyp in HYPOTHESES:
        source = sources[hyp]
        row = {"type": type(source).__name__, "shape": None, "dtype": None, "sha256": None, "equal_to_png_rgba": False}
        if isinstance(source, np.ndarray):
            row.update(shape=list(source.shape), dtype=str(source.dtype),
                       sha256=hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest())
            row["equal_to_png_rgba"] = bool(source.shape == expected.shape and source.dtype == expected.dtype
                                            and np.array_equal(source, expected))
        evidence[hyp] = row
        if not row["equal_to_png_rgba"]:
            bad.append(hyp)
    if bad:
        raise ProofError(f"texture source differs from the authenticated PNG RGBA for {bad}: {evidence}")
    return {"expected_sha256": expected_sha, "expected_shape": list(expected.shape), "hypotheses": evidence}


# --------------------------------------------------------------------------
# pure: source bindings and comparison

def source_bindings(valid: Mapping) -> dict:
    """Bind everything this experiment uses to the validated, re-derived proof."""
    rec, inputs = valid["record"], valid["inputs"]
    object_id = rec.get("object_id")
    if object_id not in proof.ONBOARDING or inputs.get("object_id") != object_id:
        raise ProofError("validated object differs")
    pin_glb, pin_receipt = proof.CANDIDATE_PINS[object_id]
    if inputs["glb_sha256"] != pin_glb or proof.sha256_bytes(inputs["glb"]) != pin_glb:
        raise ProofError("candidate bytes differ from the original pin")
    if inputs["receipt_sha256"] != pin_receipt:
        raise ProofError("candidate receipt differs from the original pin")
    if proof.sha256_bytes(inputs["texture_bytes"]) != inputs["texture_info"]["sha256"]:
        raise ProofError("official PNG bytes differ from the admitted texture")
    if valid["source"].get("passed") is not True:
        raise ProofError("re-derived source_convention_pass is false; no GPU reference is started")
    frames = list(proof.ONBOARDING[object_id].values())
    images, refs = inputs["images"], valid["references"]
    if [i["frame_id"] for i in images] != frames or len(refs) != len(frames):
        raise ProofError("validated onboarding frames differ from the fixed rows 0/9")
    for img, ref in zip(images, refs):
        if ref["tag"] != f"frame-{img['frame_id']:06d}":
            raise ProofError("reference sample set is bound to another frame")
        proof.validate_samples(ref["samples"])
    return {"object_id": object_id, "prepared_sha256": valid["prepared_sha256"],
            "candidate_glb_path": valid["candidate_glb_path"], "candidate_glb_sha256": pin_glb,
            "candidate_receipt_sha256": pin_receipt, "manifest_sha256": inputs["manifest_sha256"],
            "ply_sha256": inputs["ply_sha256"], "png_sha256": inputs["texture_info"]["sha256"],
            "frames": [{"frame_id": i["frame_id"], "logical_index": i["logical_index"], "rgb_sha256": i["rgb_sha256"],
                        "K": i["K"], "R": i["R"], "t_mm": i["t_mm"], "samples": r["count"]} for i, r in zip(images, refs)],
            "not_read": inputs["not_read"]}


def _coverage(depth_s: np.ndarray, rgb_s: np.ndarray) -> dict:
    finite = np.isfinite(depth_s) & np.isfinite(rgb_s).all(axis=1)
    valid = finite & (depth_s > 0)
    return {"valid": int(valid.sum()), "missing": int((finite & ~(depth_s > 0)).sum()), "invalid": int((~finite).sum()),
            "complete": bool(valid.all())}


def _diff(a_rgb, a_depth, b_rgb, b_depth) -> dict:
    d = np.abs(a_depth - b_depth)
    c = np.abs(a_rgb - b_rgb)
    return {"depth_median_mm": float(np.median(d)), "depth_p99_mm": float(np.percentile(d, 99)),
            "rgb_median_per_channel": [float(v) for v in np.median(c, axis=0)],
            "rgb_p99_per_channel": [float(v) for v in np.percentile(c, 99, axis=0)],
            "rgb_median_pooled": float(np.median(c))}


def compare_hypotheses(samples: Mapping[str, np.ndarray], renders: Mapping[str, tuple], thresholds: Mapping = THRESHOLDS,
                       image_size: tuple[int, int] = proof.IMAGE_SIZE) -> dict:
    """Candidate and opposite against the GPU raw-source reference at ALL frozen
    sample pixels; nothing is dropped, registered, rescaled or shifted."""
    count = proof.validate_samples(samples)["count"]
    y, x = np.asarray(samples["y"]), np.asarray(samples["x"])
    W, H = image_size
    values, cover = {}, {}
    for hyp in HYPOTHESES:
        color, depth = renders[hyp]
        color, depth = np.asarray(color, dtype=np.float64), np.asarray(depth, dtype=np.float64)
        if color.ndim != 3 or color.shape[:2] != (H, W) or color.shape[2] < 3 or depth.shape != (H, W):
            raise ProofError(f"{hyp} render has a wrong shape")
        values[hyp] = (color[y, x, :3], depth[y, x], depth)
        cover[hyp] = _coverage(values[hyp][1], values[hyp][0])
    result: dict[str, Any] = {"samples": count, "coverage": cover}
    enough = thresholds["min_valid_pixels"] <= count <= thresholds["max_sample_pixels"]
    ref_rgb, ref_d, _ = values["reference"]
    cand_rgb, cand_d, cand_full = values["candidate"]
    opp_rgb, opp_d, opp_full = values["opposite"]
    complete = cover["reference"]["complete"] and cover["candidate"]["complete"] and cover["opposite"]["complete"]
    if complete:
        result["candidate_vs_reference"] = _diff(cand_rgb, cand_d, ref_rgb, ref_d)
        result["opposite_vs_reference"] = _diff(opp_rgb, opp_d, ref_rgb, ref_d)
        result["opposite_vs_candidate_geometry"] = {
            **{k: v for k, v in _diff(opp_rgb, opp_d, cand_rgb, cand_d).items() if k.startswith("depth")},
            "full_image_mask_differences": int(((opp_full > 0) != (cand_full > 0)).sum())}
    c = result.get("candidate_vs_reference")
    o = result.get("opposite_vs_reference")
    g = result.get("opposite_vs_candidate_geometry")

    def depth_ok(m):
        return m["depth_median_mm"] <= thresholds["depth_median_mm"] and m["depth_p99_mm"] <= thresholds["depth_p99_mm"]

    if not enough:
        status, reason = "inconclusive", "insufficient_samples"
    elif not cover["reference"]["complete"]:
        status, reason = "inconclusive", "reference_coverage_incomplete"
    elif not cover["candidate"]["complete"]:
        status, reason = "fail", "candidate_coverage_missing"
    elif not cover["opposite"]["complete"] or g["full_image_mask_differences"] != 0 or not depth_ok(g):
        status, reason = "inconclusive", "opposite_geometry_differs"
    elif not depth_ok(c):
        status, reason = "fail", "candidate_depth_mismatch"
    elif np.array_equal(cand_rgb, opp_rgb):
        status, reason = "inconclusive", "texture_ambiguous"
    elif not (o["rgb_median_pooled"] > thresholds["opposite_min_median"]
              and c["rgb_median_pooled"] * thresholds["candidate_vs_opposite_factor"] <= o["rgb_median_pooled"]):
        status, reason = "inconclusive", "not_discriminated"
    elif (max(c["rgb_median_per_channel"]) <= thresholds["rgb_median"]
          and max(c["rgb_p99_per_channel"]) <= thresholds["rgb_p99"]):
        status, reason = "pass", None
    else:
        status, reason = "fail", "candidate_rgb_mismatch"
    result.update(status=status, reason=reason)
    return result


def relative_convention(frame_statuses: Sequence[str], expected_frames: int = 2) -> Any:
    if any(s == "fail" for s in frame_statuses):
        return "fail"
    if len(frame_statuses) == expected_frames and all(s == "pass" for s in frame_statuses):
        return True
    return "inconclusive"


def final_status(relative: Any, correspondence_status: str) -> tuple[Any, str]:
    corr = True if correspondence_status == "pass" else correspondence_status
    if relative is True and corr is True:
        return corr, "eligible_for_review"
    if relative == "fail" or corr == "fail":
        return corr, "failed"
    return corr, "inconclusive"


# --------------------------------------------------------------------------
# preflight (CPU only; no output; no renderer/GL import)

def preflight(prepared_dir: Path, expected_prepared_sha256: str, *, texture_dir: Path = proof.DEFAULT_TEXTURE_DIR,
              manifest: Path = yc.DEFAULT_MANIFEST) -> dict:
    proof_module = verify_proof_module()
    valid = proof.validate_prepared(prepared_dir, expected_prepared_sha256, texture_dir=texture_dir, manifest=manifest)
    binding = source_bindings(valid)
    runtime = runtime_pin_checks()
    if not runtime["passed"]:
        raise ProofError("backend runtime files differ from their pins: "
                         + ", ".join(k for k, v in runtime["files"].items() if not v["passed"]))
    inputs = valid["inputs"]
    arrays = source_primitive_arrays(inputs["mesh"]["positions_mm"], inputs["mesh"]["triangles"], inputs["mesh"]["uv"])
    rgba = texture_rgba(inputs["texture_bytes"], inputs["texture"])
    opposite_bytes, opposite_info = proof.opposite_uv_glb(inputs["glb"])
    for name, data in (("candidate", inputs["glb"]), ("opposite", opposite_bytes)):
        if len(data) > MAX_SCRATCH_GLB_BYTES:
            raise ProofError(f"{name} GLB exceeds the 32 MiB scratch bound")
    parsed = proof.parse_glb(inputs["glb"])
    if len(parsed["triangles"]) != arrays["face_count"]:
        raise ProofError("candidate and source face counts differ")
    return {"valid": valid, "binding": binding, "runtime": runtime, "proof_module": proof_module, "arrays": arrays,
            "rgba": rgba, "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
            "opposite_bytes": opposite_bytes, "opposite_info": opposite_info}


def _check_budgets(out: Path) -> dict:
    """The output must resolve strictly inside the designated derived root, so
    the global 1 GiB derived budget always applies."""
    out = Path(out).resolve()
    root = DERIVED_ROOT.resolve()
    try:
        out.relative_to(root)
    except ValueError:
        raise ProofError(f"output {out} is outside the designated derived root {root}") from None
    if out == root:
        raise ProofError("output must be a new directory inside the derived root, not the root itself")
    usage = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    if usage + MAX_OUTPUT_BYTES > DERIVED_BUDGET_BYTES:
        raise ProofError("1 GiB derived budget would be exceeded")
    return {"derived_root": str(root), "derived_usage_bytes": usage, "output_cap_bytes": MAX_OUTPUT_BYTES,
            "derived_budget_bytes": DERIVED_BUDGET_BYTES}


def _write(out: Path, name: str, data: bytes) -> str:
    digest = proof.write_new(out, name, data)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if total > MAX_OUTPUT_BYTES:
        raise ProofError("experiment output exceeds its byte cap")
    return digest


def _png(image: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buf, format="PNG")
    return buf.getvalue()


def _to_u8(color: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(color, dtype=np.float64)[:, :, :3], 0, 1) * 255).round().astype(np.uint8)


# --------------------------------------------------------------------------
# GPU (explicit parent-scheduled run only)

def gl_texture_state(offscreen: Any, gl: Any, texture: Any) -> dict:
    """Actual GL state of an uploaded base-colour texture; bindings and the
    context are restored on every path."""
    from bench import workbench_render as wrender
    texid = getattr(texture, "_texid", None)
    platform_ = getattr(offscreen, "_platform", None)
    if texid is None or platform_ is None:
        raise ProofError("texture was not uploaded or the offscreen platform is unavailable")
    saved, state, primary, cleanup = {}, {"texture_id": int(texid)}, None, []
    try:
        platform_.make_current()
        saved["binding"] = wrender._gl_int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
        gl.glBindTexture(gl.GL_TEXTURE_2D, texid)
        for name in ("GL_TEXTURE_MIN_FILTER", "GL_TEXTURE_MAG_FILTER", "GL_TEXTURE_WRAP_S", "GL_TEXTURE_WRAP_T",
                     "GL_TEXTURE_BASE_LEVEL", "GL_TEXTURE_MAX_LEVEL"):
            state[name] = wrender._gl_int(gl.glGetTexParameteriv(gl.GL_TEXTURE_2D, getattr(gl, name)))
        for level in (0, 1):
            for name in ("GL_TEXTURE_WIDTH", "GL_TEXTURE_HEIGHT", "GL_TEXTURE_INTERNAL_FORMAT"):
                state[f"level{level}_{name}"] = wrender._gl_int(
                    gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, level, getattr(gl, name)))
    except BaseException as exc:  # noqa: BLE001
        primary = exc
    finally:
        if "binding" in saved:
            try:
                gl.glBindTexture(gl.GL_TEXTURE_2D, saved["binding"])
            except BaseException as exc:  # noqa: BLE001
                cleanup.append(exc)
        try:
            platform_.make_uncurrent()
        except BaseException as exc:  # noqa: BLE001
            cleanup.append(exc)
    if primary is not None:
        raise primary
    if cleanup:
        raise ProofError(f"GL texture binding/context restoration failed: {cleanup[0]}")
    return state


def _reference_mesh(pyrender: Any, arrays: Mapping, rgba: np.ndarray, constructor_material: Any) -> Any:
    """Fresh primitive/texture from raw arrays; only scalar material settings are
    copied from the constructor-loaded candidate material (for state equality)."""
    texture = pyrender.Texture(source=rgba, source_channels="RGBA", sampler=pyrender.Sampler())
    if not np.array_equal(texture.source, rgba):
        raise ProofError("reference Texture.source differs from the authenticated PNG RGBA")
    m = constructor_material
    material = pyrender.MetallicRoughnessMaterial(
        name="raw-ply-source-reference", normalTexture=None, occlusionTexture=None, emissiveTexture=None,
        emissiveFactor=np.array(m.emissiveFactor, copy=True), alphaMode=m.alphaMode, alphaCutoff=m.alphaCutoff,
        doubleSided=m.doubleSided, smooth=m.smooth, wireframe=m.wireframe,
        baseColorFactor=np.array(m.baseColorFactor, copy=True), baseColorTexture=texture,
        metallicFactor=m.metallicFactor, roughnessFactor=m.roughnessFactor, metallicRoughnessTexture=None)
    prim = pyrender.Primitive(positions=arrays["positions_m"], normals=arrays["normals"],
                              texcoord_0=arrays["texcoord_0"], material=material, mode=pyrender.GLTF.TRIANGLES)
    if (not np.array_equal(prim.positions, arrays["positions_m"]) or not np.array_equal(prim.texcoord_0, arrays["texcoord_0"])
            or prim.indices is not None or prim.material.baseColorTexture is not texture):
        raise ProofError("reference primitive does not hold the raw-source arrays")
    return pyrender.Mesh(primitives=[prim])


def _install_reference(inner: Any, pyrender: Any, arrays: Mapping, rgba: np.ndarray) -> dict:
    """Replace the constructor's GLB mesh node by the raw-PLY reference before the first draw."""
    if inner.offscreen is not None:
        raise ProofError("reference renderer was already drawn")
    nodes = list(inner.scene.mesh_nodes)
    if len(nodes) != 1 or len(nodes[0].mesh.primitives) != 1:
        raise ProofError("constructor scene must hold exactly one mesh with one primitive")
    constructor_material = nodes[0].mesh.primitives[0].material
    mesh = _reference_mesh(pyrender, arrays, rgba, constructor_material)
    inner.scene.remove_node(nodes[0])
    inner.scene.add(mesh)
    after = list(inner.scene.mesh_nodes)
    if len(after) != 1 or after[0].mesh is not mesh:
        raise ProofError("reference mesh was not installed as the only mesh")
    inner.asset_sha256 = "raw_ply_source_reference:" + arrays["hashes"]["positions_m_float32"]
    inner.vertices_m = arrays["positions_m"].astype(np.float64)
    return {"constructor_glb": "authenticated candidate scratch copy (constructor requirement only)",
            "constructor_mesh_node": "removed before first draw; never used for reference arrays",
            "copied_from_constructor_material": ["alphaMode", "alphaCutoff", "doubleSided", "smooth", "wireframe",
                                                 "baseColorFactor", "metallicFactor", "roughnessFactor", "emissiveFactor"],
            "independent": ["positions", "normals", "texcoord_0", "face order", "PNG texture source", "sampler object"],
            "asset_label": inner.asset_sha256}


def run(prepared_dir: Path, out: Path, *, expected_prepared_sha256: str, texture_dir: Path = proof.DEFAULT_TEXTURE_DIR,
        manifest: Path = yc.DEFAULT_MANIFEST) -> dict:
    budgets = _check_budgets(out)  # outside the derived root -> rejected before any work or output
    pre = preflight(prepared_dir, expected_prepared_sha256, texture_dir=texture_dir, manifest=manifest)
    valid, binding, arrays, rgba = pre["valid"], pre["binding"], pre["arrays"], pre["rgba"]
    inputs = valid["inputs"]
    budgets = _check_budgets(out)
    out = proof.new_output_dir(out)
    scratch = out / "scratch"
    stage, files, frames = "setup", {}, []
    scratch_receipt: dict = {}
    try:
        scratch.mkdir()
        glbs = {"candidate": scratch / "candidate-model.glb", "opposite": scratch / "opposite-uv.glb"}
        glbs["candidate"].write_bytes(inputs["glb"])
        glbs["opposite"].write_bytes(pre["opposite_bytes"])
        for name, expected in (("candidate", binding["candidate_glb_sha256"]), ("opposite", pre["opposite_info"]["sha256"])):
            if proof.sha256_file(glbs[name]) != expected:
                raise ProofError(f"scratch {name} GLB differs from its verified SHA-256")
            scratch_receipt[name] = {"sha256": expected, "bytes": glbs[name].stat().st_size}
        from bench import quality_gotrack as qg
        from bench import workbench_render as wrender
        from bench import workbench_telemetry as wt
        qg.upstream_path()
        from utils import structs  # type: ignore
        import pyrender
        import trimesh
        from OpenGL import GL as gl
        dirs = valid["source"]["source_lines"]["package_dirs"]
        if (str(Path(pyrender.__file__).resolve().parent) != dirs["pyrender"]
                or str(Path(trimesh.__file__).resolve().parent) != dirs["trimesh"]):
            raise ProofError("imported pyrender/trimesh are not the runtime packages whose sources were pinned")
        if not proof._same(proof.source_line_checks(), valid["source"]["source_lines"]):
            raise ProofError("reviewed proof source pins drifted after validation")
        if not proof._same(runtime_pin_checks(), pre["runtime"]):
            raise ProofError("backend runtime pins drifted after preflight")
        factory = wrender.factory_identity(require_pins=True)

        def camera_for(img: Mapping) -> Any:
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = img["R"], img["t_mm"]
            K = img["K"]
            return structs.PinholePlaneCameraModel(width=proof.IMAGE_SIZE[0], height=proof.IMAGE_SIZE[1],
                                                   f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]),
                                                   T_world_from_eye=np.linalg.inv(pose))

        def make_inner(path: Path) -> Any:
            return qg.TexturedRenderer(str(path), 1, unlit=True, disable_multisampling=True,
                                       coordinate_mode="integer_centers_v1", render_policy="legacy_v1")

        stage = "gpu_stage"
        with wt.gpu_stage("ycb_gpu_reference"):
            for img, ref in zip(inputs["images"], valid["references"]):
                tag = ref["tag"]
                stage = f"{tag}:construct"
                renderers: dict[str, Any] = {}
                row: dict[str, Any] = {"frame_id": img["frame_id"], "tag": tag, "hypotheses": {}}
                frames.append(row)
                try:
                    inner = make_inner(glbs["candidate"])
                    try:
                        row["reference_construction"] = _install_reference(inner, pyrender, arrays, rgba)
                        renderers["reference"] = wrender.WorkbenchZeroStorageRenderer(str(glbs["candidate"]), 1, inner=inner)
                    except BaseException:
                        inner.close()
                        raise
                    for hyp in ("candidate", "opposite"):
                        inner = make_inner(glbs[hyp])
                        try:
                            renderers[hyp] = wrender.WorkbenchZeroStorageRenderer(str(glbs[hyp]), 1, inner=inner)
                        except BaseException:
                            inner.close()
                            raise
                    pre_states = {h: scene_state(r._inner) for h, r in renderers.items()}
                    if any(s["offscreen_present"] for s in pre_states.values()):
                        raise ProofError("a renderer was drawn before the equality check")
                    row["pre_draw_equality"] = require_backend_equality(pre_states)
                    row["texture_sources"] = require_texture_sources(
                        {h: next(iter(r._inner.scene.meshes)).primitives[0].material.baseColorTexture.source
                         for h, r in renderers.items()}, rgba)
                    renders, gl_states = {}, {}
                    camera = camera_for(img)
                    for hyp in HYPOTHESES:
                        stage = f"{tag}:{hyp}:draw"
                        renderer = renderers[hyp]
                        t0 = time.perf_counter()
                        outmap = renderer.render_object_model(1, camera)
                        elapsed = time.perf_counter() - t0
                        allocation = renderer.metadata
                        post = scene_state(renderer._inner)
                        changed = compare_states(pre_states[hyp], post, VOLATILE_KEYS)
                        if changed:
                            raise ProofError(f"{hyp}: backend state changed during the draw: {changed}")
                        tex = next(iter(renderer._inner.scene.meshes)).primitives[0].material.baseColorTexture
                        gl_states[hyp] = gl_texture_state(renderer._inner.offscreen, gl, tex)
                        renderer.close()
                        color = depth = None
                        for key, value in outmap.items():
                            name = getattr(key, "name", str(key)).upper()
                            if name.endswith("COLOR"):
                                color = np.asarray(value, dtype=np.float64)
                            elif name.endswith("DEPTH"):
                                depth = np.asarray(value, dtype=np.float64)
                        if color is None or depth is None:
                            raise ProofError(f"{hyp}: renderer returned no colour/depth")
                        renders[hyp] = (color, depth)
                        files[f"{tag}-{hyp}-rgb.png"] = _write(out, f"{tag}-{hyp}-rgb.png", _png(_to_u8(color)))
                        buf = io.BytesIO()
                        np.save(buf, depth.astype(np.float32), allow_pickle=False)
                        files[f"{tag}-{hyp}-depth-mm.npy"] = _write(out, f"{tag}-{hyp}-depth-mm.npy", buf.getvalue())
                        row["hypotheses"][hyp] = {"allocation_proof": allocation, "pre_draw_state": pre_states[hyp],
                                                  "post_draw_equal": True, "gl_texture_state": gl_states[hyp],
                                                  "draw_seconds_measured": elapsed}
                    gl_diff = {f"reference_vs_{h}": compare_states(gl_states["reference"], gl_states[h], VOLATILE_KEYS)
                               for h in ("candidate", "opposite")}
                    if any(gl_diff.values()):
                        raise ProofError(f"actual GL texture/sampler state differs between hypotheses: {gl_diff}")
                    row["gl_texture_state_equal"] = True
                    stage = f"{tag}:compare"
                    row["comparison"] = compare_hypotheses(ref["samples"], renders)
                    strip = np.concatenate([img["rgb"]] + [_to_u8(renders[h][0]) for h in HYPOTHESES], axis=1)
                    files[f"{tag}-comparison.png"] = _write(out, f"{tag}-comparison.png", _png(strip))
                finally:
                    close_errors = []
                    for renderer in renderers.values():
                        try:
                            renderer.close()
                        except BaseException as exc:  # noqa: BLE001
                            close_errors.append(f"{type(exc).__name__}: {exc}")
                    row["close_errors"] = close_errors
                if close_errors:
                    raise ProofError(f"{tag}: renderer close failed: {close_errors}")
        stage = "cleanup"
        shutil.rmtree(scratch)
    except BaseException as exc:
        cleanup_error = None
        if scratch.exists() and scratch.parent == out:
            try:
                shutil.rmtree(scratch)
            except OSError as cexc:
                cleanup_error = f"{type(cexc).__name__}: {cexc}"
        failure = {"schema": SCHEMA, "status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}"[:4000],
                   "cleanup_error": cleanup_error, "binding": binding, "scratch": scratch_receipt,
                   "partial_frames": frames, "files_written_before_failure": files,
                   "note": "incomplete experiment; no pass of any kind"}
        try:
            proof.write_new(out, "failure.json", proof._canonical(_jsonable(failure)))
        except Exception:  # noqa: BLE001 - never hide the original failure
            pass
        raise
    statuses = [f["comparison"]["status"] for f in frames]
    relative = relative_convention(statuses, len(proof.ONBOARDING[binding["object_id"]]))
    corr_pass, status = final_status(relative, valid["correspondence"]["status"])
    record = {
        "schema": SCHEMA, "binding": binding, "thresholds": THRESHOLDS, "frames": frames,
        "relative_serving_uv_convention_pass": relative, "onboarding_correspondence_pass": corr_pass,
        "final_status": status, "promotion": False,
        "claims": {"promotion": False, "metric_bundle": False, "verified_override": False,
                   "independent_tracking_accuracy": False, "mobile": False, "automatic_rgb_preparation": False,
                   "accepted_proof_serving_render_pass_rewritten": False},
        "source_reference": {k: v for k, v in arrays.items() if k not in ("positions_m", "normals", "texcoord_0")},
        "texture": {"png_sha256": binding["png_sha256"], "rgba_sha256": pre["rgba_sha256"],
                    "rgba_shape": list(rgba.shape)},
        "opposite_hypothesis": pre["opposite_info"], "scratch": {**scratch_receipt, "removed": True},
        "proof_module": pre["proof_module"], "proof_source_lines": valid["source"]["source_lines"],
        "runtime_pins": pre["runtime"], "renderer_factory": factory, "budgets": budgets,
        "tool_sha256": proof.sha256_file(Path(__file__)), "limitations": LIMITATIONS,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "files": files,
        "label": "shared-backend raw-source relative UV-convention evidence; not landmark correspondence",
    }
    files["gpu-reference.json"] = _write(out, "gpu-reference.json", proof._canonical(_jsonable(record)))
    return {"directory": str(out), "record": record}


# --------------------------------------------------------------------------
# CLI

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Raw-source GPU UV reference (never promotes candidates).")
    sub = p.add_subparsers(dest="command", required=True)
    for name, text in (("preflight", "CPU-only validation; no output, no GPU"),
                       ("run", "explicit parent-scheduled GPU experiment")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--prepared", required=True)
        s.add_argument("--expected-prepared-sha256", required=True)
        s.add_argument("--texture-dir", default=str(proof.DEFAULT_TEXTURE_DIR))
        s.add_argument("--manifest", default=str(yc.DEFAULT_MANIFEST))
        if name == "run":
            s.add_argument("--out", required=True, help="new immutable output directory")
    args = p.parse_args(argv)
    try:
        if args.command == "preflight":
            pre = preflight(Path(args.prepared), args.expected_prepared_sha256, texture_dir=Path(args.texture_dir),
                            manifest=Path(args.manifest))
            a = pre["arrays"]
            print(json.dumps(_jsonable({"object_id": pre["binding"]["object_id"], "prepared_sha256": pre["binding"]["prepared_sha256"],
                                        "faces": a["face_count"], "vertices": a["vertex_count"], "extent_m": a["extent_m"],
                                        "degenerate_faces": a["degenerate_faces"], "source_hashes": a["hashes"],
                                        "rgba_sha256": pre["rgba_sha256"], "samples": [f["samples"] for f in pre["binding"]["frames"]],
                                        "runtime_pins_passed": pre["runtime"]["passed"],
                                        "correspondence": pre["valid"]["correspondence"]["status"]}), indent=2))
        else:
            result = run(Path(args.prepared), Path(args.out), expected_prepared_sha256=args.expected_prepared_sha256,
                         texture_dir=Path(args.texture_dir), manifest=Path(args.manifest))
            r = result["record"]
            print(json.dumps(_jsonable({"directory": result["directory"],
                                        "frames": [{"frame_id": f["frame_id"], "status": f["comparison"]["status"],
                                                    "reason": f["comparison"]["reason"]} for f in r["frames"]],
                                        "relative_serving_uv_convention_pass": r["relative_serving_uv_convention_pass"],
                                        "onboarding_correspondence_pass": r["onboarding_correspondence_pass"],
                                        "final_status": r["final_status"]}), indent=2))
        return 0
    except (ProofError, yc.ControlError, OSError, KeyError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
