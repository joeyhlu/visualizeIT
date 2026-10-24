"""Staged UV-convention proof for the three blocked official-YCB candidate GLBs.

Status: diagnostic tool. It never promotes a candidate, never writes a metric
bundle and has no override. Candidates stay ``blocked_unverified_uv`` until a
separate reviewed decision pins them.

Scope (known-onboarding metric-CAD diagnostic, NOT automatic RGB preparation
and NOT held-out accuracy): only onboarding rows with logicalIndex 0 and 9 of
the immutable inference manifest are used (object 2: frames 361/370, object 4:
177/186, object 18: 175/184): their RGB, ``cam_K`` and known onboarding pose.
Scored frames, ``evaluator.json``, masks and depth are never read.

Phases
------
``prepare`` (pure CPU; numpy + PIL only; no torch/GL/matplotlib):
  * authenticates the manifest, source PLY, official PNG and the candidate
    receipt/GLB (via the existing ``tools/workbench_ycb_controls`` pins);
  * ``source_convention_pass``: triangle indices/order exact; GLB positions =
    PLY_mm x 0.001 within 0.001 mm; single identity node; projection parity
    K(R*PLY_mm + t_mm) vs K(R*GLB_m + t_m) <= 0.001 px with positive depth,
    while geometry-only and translation-only x1000 mutations must fail; the
    candidate TEXCOORD_0 (read by this tool's own GLB parser) equals
    (u, 1 - v_ply); the referenced BOP / trimesh / pyrender source lines are
    hashed and their expected statements verified. If trimesh is importable
    the actual loader's UVs are compared to the raw PLY UVs; otherwise that
    check is recorded unavailable (it is not faked);
  * builds an independent BOP-convention source reference per onboarding
    image with the existing fixed-point (1/256 subpixel) rasterizer: projected
    vertices are quantized, then depth and raw PLY UV are interpolated
    perspective-correctly (NOT exact ray/triangle intersection; a limitation
    especially for high-frequency texture), sampled with OpenGL normalized clamp bilinear
    texel centres x = u*W - 0.5, y = (1 - v)*H - 0.5 on the original top-down
    PNG; and freezes deterministic interior sample pixels (away from triangle
    edges, depth discontinuities and UV seams). Fewer than 128 -> inconclusive;
  * ``onboarding_correspondence_pass`` from an optional frozen landmark file
    only (>= 3 visible landmarks per image, median <= 5 px, max <= 10 px);
    without it the result is ``inconclusive``. No labels are fabricated and no
    pose/correspondence is refined.

``render`` (explicit parent-only command; GPU env; requires the externally
expected prepared.json SHA-256): before creating output or importing any
renderer it strictly re-validates the prepared directory (exact inventory,
strict JSON, tool/threshold/source pins), re-authenticates every input and
re-derives all CPU checks, landmarks and reference samples. It then renders
each onboarding image with the serving ``WorkbenchZeroStorageRenderer`` for the
candidate and an opposite-UV hypothesis (only TEXCOORD_0 v flipped; everything
else byte identical) under the repository GPU stage. The serving texture
sampler (pyrender default: mipmapped minification, repeat wrap) differs from
the base-level clamp bilinear reference oracle, so the serving comparison is
recorded as ``inconclusive: reference_sampler_mismatch`` and can never set
``serving_render_pass`` to true. A separately labelled
``base_level_sampler_control`` re-renders the final onboarding image with a
diagnostic renderer INSTANCE whose textures use GL_LINEAR min/mag and
CLAMP_TO_EDGE; it can corroborate orientation but can never set
``serving_render_pass`` either.

Fixed acceptance (declared before any run): depth median <= 0.1 mm and P99 <=
0.5 mm; RGB per-channel median <= 2/255 and P99 <= 8/255; candidate median RGB
error at least 4x lower than the opposite hypothesis and opposite median >
8/255, else inconclusive. Final status is ``inconclusive`` unless all three
passes hold, and even then only ``eligible_for_review`` (never promoted).

    python tools/workbench_ycb_uv_proof.py prepare --object-id 2 \\
        --candidate-dir artifacts/video-workbench/ycb-controls/validation-authenticated-v1/object-000002 \\
        --out artifacts/video-workbench/ycb-uv-proof/prepare-object-000002-v1
    .cache/quality-windows/Scripts/python.exe tools/workbench_ycb_uv_proof.py render \\
        --prepared artifacts/video-workbench/ycb-uv-proof/prepare-object-000002-v1 \\
        --out artifacts/video-workbench/ycb-uv-proof/render-object-000002-v1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_ycb_controls as yc  # noqa: E402  (authentication pins; read-only use)

SCHEMA_PREPARE = "workbench_ycb_uv_proof_prepare_v1"
SCHEMA_RENDER = "workbench_ycb_uv_proof_render_v1"
LANDMARK_SCHEMA = "workbench_uv_landmarks_v1"
DATASET_ROOT = ROOT / ".cache/prototype-sprint/dataset-pilot-v1"
DEFAULT_TEXTURE_DIR = ROOT / ".cache/video-workbench/imports/ycb-textures-v1"
ONBOARDING = {2: {0: 361, 9: 370}, 4: {0: 177, 9: 186}, 18: {0: 175, 9: 184}}
IMAGE_SIZE = (640, 480)

THRESHOLDS = {  # fixed before any actual run; never adjusted after results
    "position_max_error_mm": 1e-3, "projection_max_error_px": 1e-3, "mutation_min_error_px": 1.0,
    "min_valid_pixels": 128, "max_sample_pixels": 1024, "interior_min_barycentric": 0.02,
    "depth_discontinuity_mm": 2.0, "uv_seam_max_delta": 0.01,
    "depth_median_mm": 0.1, "depth_p99_mm": 0.5, "rgb_median": 2 / 255, "rgb_p99": 8 / 255,
    "candidate_vs_opposite_factor": 4.0, "opposite_min_median": 8 / 255,
    "landmarks_min_visible": 3, "landmark_median_px": 5.0, "landmark_max_px": 10.0,
    "landmark_texture_binding_texels": 4.0,
    # Distinct-landmark separations (fixed): one physical point / RGB location
    # can never count as several landmarks, whatever IDs/triangles it is given.
    "landmark_min_separation_mm": 1.0, "landmark_min_separation_rgb_px": 0.05,
    "landmark_min_separation_texels": 0.25,
}
MAX_OUTPUT_BYTES = 256 << 20
MAX_PREPARED_FILE_BYTES = 64 << 20
REFERENCE_METHOD = ("bench.workbench_geometry.rasterize: fixed-point 1/256-subpixel quantized projected vertices, "
                    "perspective-correct barycentric interpolation of depth and raw PLY UV; not exact ray/triangle "
                    "intersection (limitation for high-frequency texture)")

# Exact reviewed source pins (actual files of the importing runtime must match).
REVIEWED_SOURCE_PINS = {
    "bench_workbench_geometry": "80a6369d01208f07b2dd8a78b437fa815fd2a51832fef89d3564926e5adca7ed",
    "tools_workbench_ycb_controls": "5e2b3700d311020848111b60edf3964655bf8ed1db248f2fc6d80e51c0be97a9",
    "bop_inout": "f43733ac5fb2c4b7947662fde26e1a1accfbf822dee9d3ca1cfc474d57ad972c",
    "bop_renderer_py": "f7c0396729c0d42362893c93ef376bb4893b695d6c1644b2e6768401f308bd0d",
    "trimesh_gltf": "f1b2502c687da1c4e2dcc5240555efc0ddb2c0b72c1a908aff99be9e649ccaca",
    "pyrender_mesh": "98f7db85e6c4ab3471ee630bf653c410049ec1ee4bcb23b40dff8e4971cdcb4f",
    "pyrender_texture": "228f397e9d55aa0fa1e043ff0d8a5727d908a47745145287a03e73f53ec3cb94",
    "pyrender_flat_frag": "b55c740cf48f33b246e03988858b7ad34bf8fd8f3b8a1f830d98c0a7c2661eaf",
    "quality_gotrack": "0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29",
    # Sampler/material sources of the actual GPU env, reviewed by Sol; enforced
    # like every other pin (drift or absence fails the source convention).
    "pyrender_sampler": "de90c8f39e1bb7badad007758ac1bd3f6bf4b1b21ac89b921d32594e75af1f94",
    "pyrender_material": "39f8ab13ca148dc2e3479d4849ac022f2bd86749df856645bed8db1fe143cd0a",
    "pyrender_constants": "3bbca717f32cc5e5898368602b8f7df3f1d331e394b7fe61a268b201d1b89bf5",
    "trimesh_visual_material": "387e9001787b3176717322ac201b9145036e6e2b44053f355eb4770751b99246",
    "trimesh_visual_texture": "4a236d245a3f9d6dd8a211700ea4e8d78b644fabf1fc12b9bb96721770709c55",
}
# Original candidate GLB / receipt pins.
CANDIDATE_PINS = {
    2: ("a5d6e179b5020406a5f3f0143d3958cb4a3fda749863a89e2133d866356b65de",
        "26f70830c28614333c2098899a34f12dee5e00158e902ea2c646ef12d96989f7"),
    4: ("2bc906380ceccda6f4524db667b4f90f13a43cf078f393ca6edf6751e9496e80",
        "d7fa080363df14b6acd9c99623372313f4c6dfb222ab364f497a57e5d6aefaec"),
    18: ("980c1615a22799b4c14b430bf4c239ca8460a492efb286c8774e34ab8bc216a8",
         "08b11f7e8c8d0980985c477aa7425467092c53defd772814ecb220dcb22a20cd"),
}
GL_LINEAR, GL_CLAMP_TO_EDGE = 9729, 33071
SAMPLE_KEYS = {"y": ("int", 1), "x": ("int", 1), "face_id": ("int", 1), "barycentric": ("float", 2),
               "depth_mm": ("float", 1), "uv": ("float", 2), "rgb": ("float", 2)}

# Referenced source statements (path relative to its root, 1-based line, required substring).
SOURCE_LINES = {
    "bop_inout": ("bop_toolkit_lib/inout.py", [(701, 'model["texture_uv"][pt_id, 0] = float(prop_vals["texture_u"])'),
                                               (702, 'model["texture_uv"][pt_id, 1] = float(prop_vals["texture_v"])')]),
    "bop_renderer_py": ("bop_toolkit_lib/renderer_py.py", [(345, "model_texture = np.flipud(model_texture)"),
                                                           (349, 'texture_uv = model["texture_uv"]')]),
    "trimesh_gltf": ("trimesh/exchange/gltf.py", [(1554, 'uv = access[attr["TEXCOORD_0"]].copy()'),
                                                  (1555, "uv[:, 1] = 1.0 - uv[:, 1]")]),
    "pyrender_mesh": ("pyrender/mesh.py", [(287, "uv = mesh.visual.uv.copy()")]),
    "pyrender_texture": ("pyrender/texture.py", [(192, "np.flip(self.source, axis=0)")]),
    "pyrender_flat_frag": ("pyrender/shaders/flat.frag", [(118, "texture(material.base_color_texture, uv_0)")]),
    "quality_gotrack": ("bench/quality_gotrack.py", [(362, "trimesh.load(path, force='scene', process=False)")]),
}


class ProofError(RuntimeError):
    """Inputs or invariants do not allow a proof; nothing is promoted."""


# --------------------------------------------------------------------------
# helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


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


def new_output_dir(out: Path) -> Path:
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise ProofError(f"output directory {out} already exists; outputs are immutable")
    for part in [out.parent, *out.parent.parents]:
        if part.exists() and part.is_symlink():
            raise ProofError("output path traverses a symlink")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < yc.FREE_RESERVE_BYTES + MAX_OUTPUT_BYTES:
        raise ProofError("1 GiB free-disk reserve (plus output allowance) would be violated")
    out.mkdir(exist_ok=False)
    return out


def write_new(directory: Path, name: str, data: bytes) -> str:
    tmp = directory / f".{name}.tmp"
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, directory / name)
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    if total > MAX_OUTPUT_BYTES:
        raise ProofError("proof output exceeds its byte allowance")
    return sha256_bytes(data)


def save_png(directory: Path, name: str, image: np.ndarray) -> str:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buf, format="PNG")
    return write_new(directory, name, buf.getvalue())


def decode_png_rgb(data: bytes) -> np.ndarray:
    from io import BytesIO
    from PIL import Image
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG":
            raise ProofError("expected a PNG")
        return np.asarray(image.convert("RGB"))


# --------------------------------------------------------------------------
# GLB (this tool's own parser; independent of the exporter's read_glb)

def parse_glb(glb: bytes) -> dict:
    if len(glb) < 20:
        raise ProofError("GLB too short")
    magic, version, length = struct.unpack_from("<4sII", glb, 0)
    if magic != b"glTF" or version != 2 or length != len(glb):
        raise ProofError("GLB header invalid")
    offset, doc, binary, bin_offset = 12, None, b"", None
    while offset < len(glb):
        size, kind = struct.unpack_from("<I4s", glb, offset)
        if kind == b"JSON":
            doc = json.loads(glb[offset + 8: offset + 8 + size].decode("utf-8"))
        elif kind == b"BIN\x00":
            binary, bin_offset = glb[offset + 8: offset + 8 + size], offset + 8
        offset += 8 + size
    if not isinstance(doc, dict) or bin_offset is None:
        raise ProofError("GLB lacks JSON or BIN chunk")
    for kind in ("buffers", "images"):
        for item in doc.get(kind, []) or []:
            if "uri" in item:
                raise ProofError(f"GLB {kind} must be embedded")
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    dtypes = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}

    def accessor(index: int) -> tuple[np.ndarray, int, int]:
        acc = doc["accessors"][index]
        view = doc["bufferViews"][acc["bufferView"]]
        if "byteStride" in view or "sparse" in acc:
            raise ProofError("strided/sparse accessors are not supported by this proof")
        start = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        dtype = np.dtype(dtypes[acc["componentType"]])
        count, width = int(acc["count"]), widths[acc["type"]]
        nbytes = count * width * dtype.itemsize
        arr = np.frombuffer(binary[start:start + nbytes], dtype=dtype).reshape(count, width)
        return arr, bin_offset + start, nbytes

    meshes = doc.get("meshes", [])
    if len(meshes) != 1 or len(meshes[0].get("primitives", [])) != 1:
        raise ProofError("expected exactly one mesh with one primitive")
    prim = meshes[0]["primitives"][0]
    if prim.get("mode", 4) != 4:
        raise ProofError("expected triangles")
    pos, _, _ = accessor(prim["attributes"]["POSITION"])
    uv, uv_offset, uv_bytes = accessor(prim["attributes"]["TEXCOORD_0"])
    idx, _, _ = accessor(prim["indices"])
    image = None
    if doc.get("images"):
        view = doc["bufferViews"][doc["images"][0]["bufferView"]]
        image = binary[int(view.get("byteOffset", 0)): int(view.get("byteOffset", 0)) + int(view["byteLength"])]
    return {"doc": doc, "positions_m": pos.astype(np.float64), "uv": uv.astype(np.float64),
            "uv_byte_range": (uv_offset, uv_bytes), "triangles": idx.reshape(-1, 3).astype(np.int64),
            "image": image, "nodes": doc.get("nodes", []), "scene_nodes": doc.get("scenes", [{}])[doc.get("scene", 0)].get("nodes")}


def opposite_uv_glb(glb: bytes) -> tuple[bytes, dict]:
    """Same bytes except the TEXCOORD_0 data region, where v -> 1 - v."""
    parsed = parse_glb(glb)
    start, nbytes = parsed["uv_byte_range"]
    uv = np.frombuffer(glb[start:start + nbytes], dtype="<f4").reshape(-1, 2).copy()
    uv[:, 1] = (1.0 - uv[:, 1].astype(np.float64)).astype("<f4")
    out = bytearray(glb)
    out[start:start + nbytes] = uv.tobytes()
    out = bytes(out)
    differing = [i for i in (0, start - 1, start + nbytes, len(glb) - 1)
                 if 0 <= i < len(glb) and not (start <= i < start + nbytes) and out[i] != glb[i]]
    if out[:start] != glb[:start] or out[start + nbytes:] != glb[start + nbytes:] or differing:
        raise ProofError("opposite-UV GLB changed bytes outside TEXCOORD_0")
    return out, {"changed_byte_range": [start, start + nbytes], "sha256": sha256_bytes(out),
                 "only_texcoord_v_flipped": True}


# --------------------------------------------------------------------------
# authenticated inputs

def authenticate(object_id: int, candidate_dir: Path, manifest_path: Path, texture_dir: Path) -> dict:
    if object_id not in ONBOARDING:
        raise ProofError(f"object {object_id} is not one of {sorted(ONBOARDING)}")
    manifest, manifest_sha = yc.load_manifest(manifest_path)
    obj = yc.select_object(manifest, object_id)
    ply_path = yc._safe_relative(DATASET_ROOT, obj["model"]["path"], "model path")
    ply_raw = yc._read_regular(ply_path, yc.MAX_PLY_BYTES, "source PLY")
    if sha256_bytes(ply_raw) != yc.OBJECTS[object_id][1]:
        raise ProofError("source PLY SHA-256 differs from the frozen pin")
    mesh = yc.parse_ascii_ply(ply_raw)
    texture_name = Path(yc.TRUSTED_TEXTURES[object_id]["member"]).name
    texture_bytes, texture_info = yc.admit_texture(object_id, mesh, manifest, Path(texture_dir) / texture_name)
    receipt_raw = yc._read_regular(Path(candidate_dir) / "provenance-receipt.json", 1 << 20, "candidate receipt")
    receipt = yc._loads_strict(receipt_raw, "candidate receipt")
    glb_raw = yc._read_regular(Path(candidate_dir) / "candidate-model.glb", yc.MAX_GLB_BYTES, "candidate GLB")
    pin_glb, pin_receipt = CANDIDATE_PINS[object_id]
    if sha256_bytes(glb_raw) != pin_glb or sha256_bytes(receipt_raw) != pin_receipt:
        raise ProofError("candidate GLB/receipt differ from the original pinned candidate")
    if receipt.get("status") != "blocked_unverified_uv" or receipt.get("metric_bundle") is not None:
        raise ProofError("candidate receipt must be blocked_unverified_uv without a metric bundle")
    if (receipt["object"]["obj_id"] != object_id or receipt["glb"]["sha256"] != sha256_bytes(glb_raw)
            or receipt["source"]["model_sha256"] != yc.OBJECTS[object_id][1]
            or receipt["source"]["manifest_sha256"] != manifest_sha
            or receipt["appearance"]["texture"]["supplied"]["sha256"] != texture_info["sha256"]):
        raise ProofError("candidate receipt does not bind this object, GLB, PLY, manifest and texture")
    # Only onboarding rows logicalIndex 0 and 9 (RGB, K, known pose). No depth, masks or scored data.
    rows = {r.get("logicalIndex"): r for r in obj["onboarding"] if isinstance(r, dict)}
    images = []
    for logical, frame_id in ONBOARDING[object_id].items():
        row = rows.get(logical)
        if row is None or row.get("frameId") != frame_id:
            raise ProofError(f"onboarding logicalIndex {logical} is not frame {frame_id}")
        rgb = row.get("rgb") or {}
        if rgb.get("available") is not True:
            raise ProofError(f"onboarding RGB {frame_id} unavailable")
        rgb_path = yc._safe_relative(DATASET_ROOT, rgb.get("path"), "onboarding RGB")
        rgb_raw = yc._read_regular(rgb_path, 16 << 20, "onboarding RGB")
        if sha256_bytes(rgb_raw) != rgb.get("sha256"):
            raise ProofError(f"onboarding RGB {frame_id} hash differs from the manifest")
        pose = row.get("evaluationPose") or {}
        if pose.get("obj_id") != object_id:
            raise ProofError("onboarding pose names another object")
        R = np.array(yc._finite_list(pose.get("cam_R_m2c"), 9, "cam_R_m2c")).reshape(3, 3)
        t_mm = np.array(yc._finite_list(pose.get("cam_t_m2c"), 3, "cam_t_m2c"))
        if abs(np.linalg.det(R) - 1) > 1e-4 or np.abs(R.T @ R - np.eye(3)).max() > 1e-4:
            raise ProofError("onboarding rotation is not proper")
        K = np.array(yc._finite_list(row["calibration"].get("cam_K"), 9, "cam_K")).reshape(3, 3)
        image = decode_png_rgb(rgb_raw)
        if image.shape[:2] != IMAGE_SIZE[::-1]:
            raise ProofError(f"onboarding RGB {frame_id} is not 640x480")
        images.append({"logical_index": logical, "frame_id": frame_id, "rgb_path": rgb_path, "rgb_sha256": rgb["sha256"],
                       "rgb": image, "K": K, "R": R, "t_mm": t_mm})
    return {"object_id": object_id, "manifest_sha256": manifest_sha, "mesh": mesh, "ply_sha256": yc.OBJECTS[object_id][1],
            "ply_path": ply_path, "texture_bytes": texture_bytes, "texture_info": texture_info,
            "texture": decode_png_rgb(texture_bytes), "glb": glb_raw, "glb_sha256": sha256_bytes(glb_raw),
            "receipt_sha256": sha256_bytes(receipt_raw), "candidate_dir": Path(candidate_dir), "images": images,
            "not_read": ["evaluator.json", "scored frames/poses/labels", "masks", "depth"]}


# --------------------------------------------------------------------------
# source convention

def _runtime_package_dir(name: str) -> Path | None:
    """Directory of the package the CURRENT interpreter would import (no execution)."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def source_paths() -> dict[str, Path | None]:
    trimesh_dir, pyrender_dir = _runtime_package_dir("trimesh"), _runtime_package_dir("pyrender")
    bop = ROOT / ".cache/model-quality/sources/gotrack/external/bop_toolkit"
    return {
        "bench_workbench_geometry": ROOT / "bench/workbench_geometry.py",
        "tools_workbench_ycb_controls": ROOT / "tools/workbench_ycb_controls.py",
        "bop_inout": bop / "bop_toolkit_lib/inout.py",
        "bop_renderer_py": bop / "bop_toolkit_lib/renderer_py.py",
        "trimesh_gltf": None if trimesh_dir is None else trimesh_dir / "exchange/gltf.py",
        "pyrender_mesh": None if pyrender_dir is None else pyrender_dir / "mesh.py",
        "pyrender_texture": None if pyrender_dir is None else pyrender_dir / "texture.py",
        "pyrender_flat_frag": None if pyrender_dir is None else pyrender_dir / "shaders/flat.frag",
        "quality_gotrack": ROOT / "bench/quality_gotrack.py",
        "pyrender_sampler": None if pyrender_dir is None else pyrender_dir / "sampler.py",
        "pyrender_material": None if pyrender_dir is None else pyrender_dir / "material.py",
        "pyrender_constants": None if pyrender_dir is None else pyrender_dir / "constants.py",
        "trimesh_visual_material": None if trimesh_dir is None else trimesh_dir / "visual/material.py",
        "trimesh_visual_texture": None if trimesh_dir is None else trimesh_dir / "visual/texture.py",
    }


def source_line_checks() -> dict:
    """Exact reviewed hashes of the files the current runtime would actually use;
    the quoted statements are informative only (the hash is the binding)."""
    paths = source_paths()
    out, ok = {}, True
    for key, path in paths.items():
        row = {"path": None if path is None else str(path), "sha256": None, "pinned_sha256": REVIEWED_SOURCE_PINS[key],
               "lines": {}, "passed": False}
        if path is not None and path.is_file():
            row["sha256"] = sha256_file(path)
            if key in SOURCE_LINES:
                text = path.read_text(encoding="utf-8", errors="replace").splitlines()
                row["lines"] = {str(n): (text[n - 1].strip() if n <= len(text) else None) for n, _ in SOURCE_LINES[key][1]}
                row["statements_present"] = all(n <= len(text) and needle in text[n - 1] for n, needle in SOURCE_LINES[key][1])
            row["passed"] = row["sha256"] == REVIEWED_SOURCE_PINS[key] and row.get("statements_present", True)
        else:
            row["reason"] = "file unavailable in the current runtime"
        ok &= row["passed"]
        out[key] = row
    if set(out) != set(REVIEWED_SOURCE_PINS):
        ok = False
    return {"files": out, "passed": bool(ok),
            "package_dirs": {name: (None if (p := _runtime_package_dir(name)) is None else str(p))
                             for name in ("trimesh", "pyrender")},
            "algebra": "exported TEXCOORD_0 = (u, 1 - v_ply); trimesh glTF loader flips v -> loaded uv = (u, v_ply); "
                       "pyrender keeps uv and flips PNG rows; flat.frag samples texture(uv_0). BOP keeps raw (u, v), "
                       "flips the PNG vertically and samples texture2D(raw uv). Both reduce to sampling the top-down "
                       "PNG at row (1 - v)*H - 0.5. Source algebra only; not an actual full-renderer proof."}


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = X @ R.T + t
    uvw = cam @ K.T
    return uvw[:, :2] / uvw[:, 2:3], cam[:, 2]


def source_convention(inputs: dict) -> dict:
    mesh, glb = inputs["mesh"], parse_glb(inputs["glb"])
    checks: dict[str, Any] = {}
    checks["triangles_exact"] = bool(np.array_equal(glb["triangles"], mesh["triangles"].astype(np.int64)))
    pos_err_mm = float(np.abs(glb["positions_m"] * 1000.0 - mesh["positions_mm"]).max())
    checks["position_max_error_mm"] = pos_err_mm
    checks["positions_ok"] = pos_err_mm <= THRESHOLDS["position_max_error_mm"]
    nodes = glb["nodes"]
    node_ok = (len(nodes) == 1 and glb["scene_nodes"] == [0]
               and not any(k in nodes[0] for k in ("matrix", "translation", "rotation", "scale", "children")))
    checks["single_identity_node"] = bool(node_ok)
    expected_uv = np.stack([mesh["uv"][:, 0], 1.0 - mesh["uv"][:, 1]], axis=1).astype("<f4").astype(np.float64)
    checks["candidate_texcoord_is_u_one_minus_v"] = bool(np.array_equal(glb["uv"], expected_uv))
    checks["embedded_texture_is_official_png"] = glb["image"] is not None and sha256_bytes(glb["image"]) == inputs["texture_info"]["sha256"]
    projection = []
    for img in inputs["images"]:
        K, R, t_mm = img["K"], img["R"], img["t_mm"]
        ref, z_ref = project(K, R, t_mm, mesh["positions_mm"])
        got, z_got = project(K, R, t_mm / 1000.0, glb["positions_m"])
        geo, _ = project(K, R, t_mm / 1000.0, glb["positions_m"] * 1000.0)   # geometry-only x1000
        trans, _ = project(K, R, t_mm, glb["positions_m"])                    # translation-only x1000
        row = {"frame_id": img["frame_id"], "max_error_px": float(np.abs(got - ref).max()),
               "min_depth_mm": float(z_ref.min()), "min_depth_m": float(z_got.min()),
               "geometry_only_x1000_error_px": float(np.nanmax(np.abs(geo - ref))),
               "translation_only_x1000_error_px": float(np.nanmax(np.abs(trans - ref)))}
        row["passed"] = (row["max_error_px"] <= THRESHOLDS["projection_max_error_px"] and z_ref.min() > 0 and z_got.min() > 0
                         and row["geometry_only_x1000_error_px"] > THRESHOLDS["mutation_min_error_px"]
                         and row["translation_only_x1000_error_px"] > THRESHOLDS["mutation_min_error_px"])
        projection.append(row)
    checks["projection"] = projection
    checks["projection_note"] = "scaling geometry and translation together is projection-invariant; it does not prove units"
    loader = {"status": "unavailable", "reason": "trimesh not importable in this interpreter (not faked)"}
    if importlib.util.find_spec("trimesh") is not None:
        try:
            import trimesh
            from io import BytesIO
            scene = trimesh.load(BytesIO(inputs["glb"]), file_type="glb", force="scene", process=False)
            geoms = list(scene.geometry.values())
            if len(geoms) != 1:
                raise ProofError("loader produced more than one geometry")
            luv = np.asarray(geoms[0].visual.uv, dtype=np.float64)
            same_order = np.array_equal(np.asarray(geoms[0].faces), mesh["triangles"].astype(np.int64))
            err = float(np.abs(luv - mesh["uv"]).max()) if luv.shape == mesh["uv"].shape else None
            loader = {"status": "measured", "trimesh_version": trimesh.__version__, "faces_identical": bool(same_order),
                      "max_abs_loaded_uv_minus_raw_ply_uv": err,
                      "passed": bool(same_order and err is not None and err <= 1e-6)}
        except Exception as exc:  # noqa: BLE001
            loader = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500], "passed": False}
    checks["actual_loader_uv"] = loader
    checks["source_lines"] = source_line_checks()
    passed = (checks["triangles_exact"] and checks["positions_ok"] and checks["single_identity_node"]
              and checks["candidate_texcoord_is_u_one_minus_v"] and checks["embedded_texture_is_official_png"]
              and all(r["passed"] for r in projection) and checks["source_lines"]["passed"]
              and loader.get("passed") is True)
    checks["passed"] = bool(passed)
    if loader["status"] != "measured":
        checks["incomplete"] = "actual loader UV comparison unavailable; source_convention_pass stays false"
    return checks


# --------------------------------------------------------------------------
# BOP-convention reference

def sample_bilinear_bop(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """OpenGL normalized clamp bilinear at texel centres on the top-down PNG:
    x = u*W - 0.5, y = (1 - v)*H - 0.5 (BOP flips the PNG then samples raw uv)."""
    H, W = texture.shape[:2]
    x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1)
    y = np.clip((1.0 - uv[:, 1]) * H - 0.5, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    t = texture.astype(np.float64) / 255.0
    return ((1 - fx) * (1 - fy) * t[y0, x0] + fx * (1 - fy) * t[y0, x1]
            + (1 - fx) * fy * t[y1, x0] + fx * fy * t[y1, x1])


def reference_image(inputs: dict, img: dict) -> dict:
    from bench import workbench_geometry as wg
    mesh = inputs["mesh"]
    W, H = IMAGE_SIZE
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = img["R"], img["t_mm"]
    uvp, z = project(img["K"], img["R"], img["t_mm"], mesh["positions_mm"])
    roi = (int(np.floor(uvp[:, 0].min())) - 1, int(np.floor(uvp[:, 1].min())) - 1,
           int(np.ceil(uvp[:, 0].max())) + 1, int(np.ceil(uvp[:, 1].max())) + 1) if (z > 0).all() else None
    ras = wg.rasterize(mesh["positions_mm"].astype(np.float32), mesh["triangles"].astype(np.int64), img["K"], pose,
                       W, H, roi=roi)
    cov = ras.face_id >= 0
    tri = mesh["triangles"].astype(np.int64)
    uv_img = np.full((H, W, 2), np.nan)
    ys, xs = np.nonzero(cov)
    bary = ras.barycentric[ys, xs]
    uv_img[ys, xs] = (bary[:, :, None] * mesh["uv"][tri[ras.face_id[ys, xs]]]).sum(axis=1)
    rgb_ref = np.zeros((H, W, 3))
    rgb_ref[ys, xs] = sample_bilinear_bop(inputs["texture"], uv_img[ys, xs])
    depth = np.where(cov, ras.depth, np.nan)
    # deterministic interior selection: away from triangle edges, depth discontinuities, UV seams, borders
    valid = cov.copy()
    valid[:1, :] = valid[-1:, :] = False
    valid[:, :1] = valid[:, -1:] = False
    minbary = np.where(cov, ras.barycentric.min(axis=2), -1)
    valid &= minbary >= THRESHOLDS["interior_min_barycentric"]
    pad_d = np.pad(depth, 1, constant_values=np.nan)
    pad_uv = np.pad(uv_img, ((1, 1), (1, 1), (0, 0)), constant_values=np.nan)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nd = pad_d[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            nuv = pad_uv[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            with np.errstate(invalid="ignore"):
                valid &= np.isfinite(nd) & (np.abs(nd - depth) <= THRESHOLDS["depth_discontinuity_mm"])
                valid &= (np.abs(nuv - uv_img) <= THRESHOLDS["uv_seam_max_delta"]).all(axis=2)
    vy, vx = np.nonzero(valid)
    n_valid = int(len(vy))
    take = np.linspace(0, n_valid - 1, min(n_valid, THRESHOLDS["max_sample_pixels"])).astype(int) if n_valid else np.zeros(0, int)
    sy, sx = vy[take], vx[take]
    return {"coverage": cov, "depth_mm": depth, "uv": uv_img, "rgb": rgb_ref, "valid": valid,
            "samples": {"y": sy, "x": sx, "face_id": ras.face_id[sy, sx], "barycentric": ras.barycentric[sy, sx],
                        "depth_mm": depth[sy, sx], "uv": uv_img[sy, sx], "rgb": rgb_ref[sy, sx]},
            "counts": {"covered": int(cov.sum()), "valid_interior": n_valid, "sampled": int(len(sy)),
                       "excluded": int(cov.sum()) - n_valid},
            "raster_stats": ras.stats}


# --------------------------------------------------------------------------
# landmarks (optional frozen evidence)

def _strict_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProofError(f"{what} must be a finite number")
    return float(value)


def _strict_vector(value: Any, n: int, what: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != n:
        raise ProofError(f"{what} must be a list of {n} finite numbers")
    return np.array([_strict_number(v, what) for v in value], dtype=np.float64)


def _strict_matrix(value: Any, rows: int, cols: int, what: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise ProofError(f"{what} must be a {rows}x{cols} list of finite numbers")
    return np.stack([_strict_vector(row, cols, what) for row in value])


def evaluate_landmarks(path: Path | None, inputs: dict) -> dict:
    """Strictly validated frozen landmarks; anything ambiguous is rejected or inconclusive."""
    if path is None:
        return {"status": "inconclusive", "passed": False, "reason": "no frozen landmark file supplied; no labels fabricated"}
    raw = Path(path).read_bytes()
    if len(raw) > (1 << 20):
        raise ProofError("landmark file too large")
    doc = yc._loads_strict(raw, "landmark file")  # rejects duplicate keys and non-finite constants
    if not isinstance(doc, dict) or doc.get("schema") != LANDMARK_SCHEMA or doc.get("object_id") != inputs["object_id"]:
        raise ProofError("landmark file schema/object differs")
    if doc.get("texture_sha256") != inputs["texture_info"]["sha256"]:
        raise ProofError("landmark file is bound to another texture")
    landmarks = doc.get("landmarks")
    if not isinstance(landmarks, list):
        raise ProofError("landmarks must be a list")
    mesh, tex = inputs["mesh"], inputs["texture"]
    H, W = tex.shape[:2]
    n_tri = len(mesh["triangles"])
    per_image = {img["frame_id"]: [] for img in inputs["images"]}
    rgb_sha = {img["frame_id"]: img["rgb_sha256"] for img in inputs["images"]}
    ids, points_mm, texture_points, binding_errors = set(), [], [], []
    frame_rgb: dict[int, list[np.ndarray]] = {fid: [] for fid in per_image}
    visible_rgb: dict[str, dict[int, np.ndarray]] = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            raise ProofError("each landmark must be an object")
        lid = lm.get("landmark_id")
        if not isinstance(lid, str) or not lid.strip():
            raise ProofError("landmark_id must be a nonempty string")
        if lid in ids:
            raise ProofError(f"duplicate landmark_id {lid}")
        ids.add(lid)
        tri = lm.get("triangle")
        if type(tri) is not int or not 0 <= tri < n_tri:
            raise ProofError(f"landmark {lid} triangle must be an integer index into the mesh")
        b = _strict_vector(lm.get("barycentric"), 3, f"landmark {lid} barycentric")
        if (b < 0).any() or abs(b.sum() - 1) > 1e-9:
            raise ProofError(f"landmark {lid} barycentric must be nonnegative and sum to 1")
        txy = _strict_vector(lm.get("texture_xy"), 2, f"landmark {lid} texture_xy")
        if not (-0.5 <= txy[0] <= W - 0.5 and -0.5 <= txy[1] <= H - 0.5):
            raise ProofError(f"landmark {lid} texture_xy outside the texture")
        verts = mesh["triangles"][tri].astype(np.int64)
        point_mm = (b[:, None] * mesh["positions_mm"][verts]).sum(axis=0)
        if not np.isfinite(point_mm).all():
            raise ProofError(f"landmark {lid} physical point is not finite")
        if any(np.linalg.norm(point_mm - p) < THRESHOLDS["landmark_min_separation_mm"] for p in points_mm):
            raise ProofError(f"landmark {lid} repeats another landmark's physical point")
        if any(np.linalg.norm(txy - q) < THRESHOLDS["landmark_min_separation_texels"] for q in texture_points):
            raise ProofError(f"landmark {lid} repeats another landmark's texture location")
        points_mm.append(point_mm)
        texture_points.append(txy)
        uv = (b[:, None] * mesh["uv"][verts]).sum(axis=0)
        texel = np.array([uv[0] * W - 0.5, (1 - uv[1]) * H - 0.5])
        bind = float(np.linalg.norm(texel - txy))
        binding_errors.append(bind)
        observations = lm.get("images")
        if not isinstance(observations, list):
            raise ProofError(f"landmark {lid} images must be a list")
        seen_frames = set()
        for obs in observations:
            if not isinstance(obs, dict):
                raise ProofError("each observation must be an object")
            fid = obs.get("frame_id")
            if type(fid) is not int or fid not in per_image:
                raise ProofError("landmark references a non-onboarding image")
            if fid in seen_frames:
                raise ProofError(f"landmark {lid} has more than one observation for frame {fid}")
            seen_frames.add(fid)
            if obs.get("rgb_sha256") != rgb_sha[fid]:
                raise ProofError("landmark observation is bound to another RGB")
            visible = obs.get("visible")
            if type(visible) is not bool:
                raise ProofError("visibility must be a boolean")
            if not visible:
                continue  # occlusion retained as declared, never guessed; any rgb_xy is ignored
            rxy = _strict_vector(obs.get("rgb_xy"), 2, f"landmark {lid} rgb_xy")
            if not (-0.5 <= rxy[0] <= IMAGE_SIZE[0] - 0.5 and -0.5 <= rxy[1] <= IMAGE_SIZE[1] - 0.5):
                raise ProofError(f"landmark {lid} rgb_xy outside the image")
            repeated_rgb = any(np.linalg.norm(rxy - q) < THRESHOLDS["landmark_min_separation_rgb_px"]
                               for q in frame_rgb[fid])
            frame_rgb[fid].append(rxy)
            visible_rgb.setdefault(lid, {})[fid] = rxy
            img = next(i for i in inputs["images"] if i["frame_id"] == fid)
            pix, z = project(img["K"], img["R"], img["t_mm"], point_mm[None])
            err = float(np.linalg.norm(pix[0] - rxy))
            if not math.isfinite(err) or not z[0] > 0:
                raise ProofError(f"landmark {lid} projection is not finite/in front of the camera")
            per_image[fid].append({"landmark_id": lid, "error_px": err, "depth_mm": float(z[0]),
                                   "texture_binding_texels": bind, "repeated_rgb_location": bool(repeated_rgb)})
    # Two IDs whose RGB observations coincide in every co-visible frame are one
    # labelled image point: reject. Coincidence in only some frames is handled
    # per frame below (it never counts as another distinct landmark there).
    labelled = sorted(visible_rgb)
    for i, a in enumerate(labelled):
        for b_id in labelled[i + 1:]:
            common = set(visible_rgb[a]) & set(visible_rgb[b_id])
            if common and all(np.linalg.norm(visible_rgb[a][f] - visible_rgb[b_id][f])
                              < THRESHOLDS["landmark_min_separation_rgb_px"] for f in common):
                raise ProofError(f"landmarks {a} and {b_id} share one RGB location in every co-visible frame")
    images = {}
    for fid, rows in per_image.items():
        errs = [r["error_px"] for r in rows]
        # A repeated RGB location in this frame never counts as another distinct landmark.
        distinct = len({r["landmark_id"] for r in rows if not r["repeated_rgb_location"]})
        ok_bind = all(r["texture_binding_texels"] <= THRESHOLDS["landmark_texture_binding_texels"] for r in rows)
        enough = distinct >= THRESHOLDS["landmarks_min_visible"]
        images[str(fid)] = {"visible_distinct": distinct, "median_px": float(np.median(errs)) if errs else None,
                            "max_px": float(np.max(errs)) if errs else None, "rows": rows,
                            "status": ("inconclusive" if not enough else
                                       "pass" if ok_bind and np.median(errs) <= THRESHOLDS["landmark_median_px"]
                                       and np.max(errs) <= THRESHOLDS["landmark_max_px"] else "fail")}
    statuses = {v["status"] for v in images.values()}
    status = "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass"
    return {"status": status, "passed": status == "pass", "file_sha256": sha256_bytes(raw),
            "distinct_landmarks": len(ids), "selected_by": doc.get("selected_by"), "images": images,
            "texture_binding_texels_max": max(binding_errors) if binding_errors else None,
            "note": "declared asymmetric printed-texture landmarks only; nothing refined or optimized"}


# --------------------------------------------------------------------------
# prepare

def prepare(object_id: int, candidate_dir: Path, out: Path, *, manifest: Path = yc.DEFAULT_MANIFEST,
            texture_dir: Path = DEFAULT_TEXTURE_DIR, landmarks: Path | None = None) -> dict:
    before_torch = "torch" in sys.modules
    inputs = authenticate(object_id, candidate_dir, manifest, texture_dir)
    source = source_convention(inputs)
    correspondence = evaluate_landmarks(landmarks, inputs)
    landmark_bytes = Path(landmarks).read_bytes() if landmarks is not None else None
    out = new_output_dir(out)
    files: dict[str, str] = {}
    if landmark_bytes is not None:
        files["landmarks.json"] = write_new(out, "landmarks.json", landmark_bytes)  # bound copy, re-validated by render
    images = []
    for img in inputs["images"]:
        ref = reference_image(inputs, img)
        tag = f"frame-{img['frame_id']:06d}"
        s = ref["samples"]
        from io import BytesIO
        buf = BytesIO()
        np.savez(buf, y=s["y"], x=s["x"], face_id=s["face_id"], barycentric=s["barycentric"],
                 depth_mm=s["depth_mm"], uv=s["uv"], rgb=s["rgb"])
        files[f"{tag}-samples.npz"] = write_new(out, f"{tag}-samples.npz", buf.getvalue())
        buf = BytesIO()
        np.save(buf, ref["depth_mm"].astype(np.float64), allow_pickle=False)
        files[f"{tag}-reference-depth-mm.npy"] = write_new(out, f"{tag}-reference-depth-mm.npy", buf.getvalue())
        ref_rgb8 = (np.clip(ref["rgb"], 0, 1) * 255).round().astype(np.uint8)
        files[f"{tag}-reference-rgb.png"] = save_png(out, f"{tag}-reference-rgb.png", ref_rgb8)
        overlay = img["rgb"].copy()
        overlay[ref["coverage"]] = (0.5 * overlay[ref["coverage"]] + 0.5 * ref_rgb8[ref["coverage"]]).astype(np.uint8)
        sel = overlay.copy()
        sel[s["y"], s["x"]] = (255, 0, 255)
        files[f"{tag}-side-by-side.png"] = save_png(out, f"{tag}-side-by-side.png",
                                                    np.concatenate([img["rgb"], ref_rgb8, overlay, sel], axis=1))
        enough = ref["counts"]["sampled"] >= THRESHOLDS["min_valid_pixels"]
        images.append({"frame_id": img["frame_id"], "logical_index": img["logical_index"], "rgb_sha256": img["rgb_sha256"],
                       "K": img["K"], "R": img["R"], "t_mm": img["t_mm"], "counts": ref["counts"],
                       "enough_valid_pixels": enough, "raster_stats": ref["raster_stats"]})
    if "torch" in sys.modules and not before_torch:
        raise ProofError("prepare unexpectedly imported torch")
    record = {
        "schema": SCHEMA_PREPARE, "object_id": object_id, "thresholds": THRESHOLDS,
        "inputs": {"manifest_sha256": inputs["manifest_sha256"], "ply_sha256": inputs["ply_sha256"],
                   "texture": {k: v for k, v in inputs["texture_info"].items() if k != "trusted_binding"},
                   "candidate_glb_sha256": inputs["glb_sha256"], "candidate_receipt_sha256": inputs["receipt_sha256"],
                   "candidate_glb_path": str(Path(candidate_dir) / "candidate-model.glb"),
                   "texture_dir": str(texture_dir), "not_read": inputs["not_read"]},
        "images": images, "source_convention": source, "onboarding_correspondence": correspondence,
        "source_convention_pass": bool(source["passed"]),
        "onboarding_correspondence_pass": correspondence["status"] if correspondence["status"] != "pass" else True,
        "serving_render_pass": "pending_render_phase",
        "final_status": "inconclusive",
        "reference_ready": all(i["enough_valid_pixels"] for i in images),
        "files": files,
        "tool_sha256": sha256_file(Path(__file__)),
        "reference_method": REFERENCE_METHOD,
        "reviewed_source_pins": REVIEWED_SOURCE_PINS, "candidate_pins": {str(k): list(v) for k, v in CANDIDATE_PINS.items()},
        "image_inventory": [{"frame_id": img["frame_id"], "logical_index": img["logical_index"],
                             "rgb_sha256": img["rgb_sha256"]} for img in inputs["images"]],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "claims": {"promotion": False, "metric_bundle": False, "automatic_rgb_preparation": False,
                   "independent_tracking_accuracy": False, "mobile": False},
        "label": "known-onboarding metric-CAD UV diagnostic; not automatic RGB preparation or held-out accuracy",
    }
    files["prepared.json"] = write_new(out, "prepared.json", _canonical(_jsonable(record)))
    return {"directory": str(out), "record": record, "prepared_sha256": files["prepared.json"]}


# --------------------------------------------------------------------------
# strict prepared-directory validation (before any output or renderer import)

PER_IMAGE_FILES = ("-samples.npz", "-reference-depth-mm.npy", "-reference-rgb.png", "-side-by-side.png")
IMAGE_ROW_KEYS = {"frame_id", "logical_index", "rgb_sha256", "K", "R", "t_mm", "counts", "enough_valid_pixels",
                  "raster_stats"}


def validate_samples(arrays: Mapping[str, np.ndarray]) -> dict:
    """Exact keys/shapes/dtypes/finite/in-bounds/unique sample arrays.
    Fewer than 128 legitimate samples is inconclusive, never padded."""
    if set(arrays) != set(SAMPLE_KEYS):
        raise ProofError(f"sample arrays must have exactly {sorted(SAMPLE_KEYS)}")
    n = int(arrays["y"].shape[0]) if arrays["y"].ndim == 1 else -1
    for key, (kind, ndim) in SAMPLE_KEYS.items():
        a = arrays[key]
        if a.ndim != ndim or a.shape[0] != n:
            raise ProofError(f"sample array {key} has a wrong shape")
        if ndim == 2 and a.shape[1] != (2 if key == "uv" else 3):
            raise ProofError(f"sample array {key} has a wrong width")
        if kind == "int" and a.dtype.kind not in "iu":
            raise ProofError(f"sample array {key} must be integer")
        if kind == "float" and a.dtype != np.float64:
            raise ProofError(f"sample array {key} must be float64")
        if kind == "float" and not np.isfinite(a).all():
            raise ProofError(f"sample array {key} contains non-finite values")
    if not 0 <= n <= THRESHOLDS["max_sample_pixels"]:
        raise ProofError("sample count outside 0..1024")
    W, H = IMAGE_SIZE
    if n and ((arrays["x"] < 0).any() or (arrays["x"] >= W).any() or (arrays["y"] < 0).any() or (arrays["y"] >= H).any()):
        raise ProofError("sample pixels outside the image")
    if len({(int(y), int(x)) for y, x in zip(arrays["y"], arrays["x"])}) != n:
        raise ProofError("sample pixels are not distinct")
    b = arrays["barycentric"]
    if n and ((b < 0).any() or np.abs(b.sum(axis=1) - 1).max() > 1e-9 or (arrays["depth_mm"] <= 0).any()
              or (arrays["rgb"] < 0).any() or (arrays["rgb"] > 1).any() or (arrays["face_id"] < 0).any()):
        raise ProofError("sample barycentric/depth/RGB/face values invalid")
    return {"count": n, "enough": n >= THRESHOLDS["min_valid_pixels"]}


def _read_bounded_regular(path: Path, limit: int, what: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ProofError(f"{what} is not a regular file")
    if path.stat().st_size > limit:
        raise ProofError(f"{what} exceeds its size bound")
    return path.read_bytes()


def _same(a: Any, b: Any) -> bool:
    return json.dumps(_jsonable(a), sort_keys=True) == json.dumps(_jsonable(b), sort_keys=True)


def validate_prepared(prepared_dir: Path, expected_sha256: str, *, texture_dir: Path, manifest: Path) -> dict:
    """Re-validate everything prepare claimed, from authenticated inputs. The
    expected SHA identifies the record; it is not itself proof."""
    d = Path(prepared_dir)
    if d.is_symlink() or not d.is_dir():
        raise ProofError("prepared directory missing or a symlink")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ProofError("an exact lowercase expected prepared.json SHA-256 is required")
    raw = _read_bounded_regular(d / "prepared.json", 4 << 20, "prepared.json")
    if sha256_bytes(raw) != expected_sha256:
        raise ProofError("prepared.json differs from the externally expected SHA-256")
    rec = yc._loads_strict(raw, "prepared.json")
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA_PREPARE:
        raise ProofError("not a prepare record")
    if rec.get("tool_sha256") != sha256_file(Path(__file__)):
        raise ProofError("prepared with a different tool version")
    if (not _same(rec.get("thresholds"), THRESHOLDS) or rec.get("reference_method") != REFERENCE_METHOD
            or rec.get("reviewed_source_pins") != REVIEWED_SOURCE_PINS
            or rec.get("candidate_pins") != {str(k): list(v) for k, v in CANDIDATE_PINS.items()}):
        raise ProofError("prepared thresholds/reference method/source or candidate pins differ")
    object_id = rec.get("object_id")
    if type(object_id) is not int or object_id not in ONBOARDING:
        raise ProofError("prepared object id invalid")
    frames = list(ONBOARDING[object_id].values())
    tags = [f"frame-{fid:06d}" for fid in frames]
    expected = {"prepared.json"} | {t + suffix for t in tags for suffix in PER_IMAGE_FILES}
    files = rec.get("files")
    if not isinstance(files, dict):
        raise ProofError("prepared file table missing")
    if "landmarks.json" in files:
        expected.add("landmarks.json")
    if set(os.listdir(d)) != expected or set(files) != expected - {"prepared.json"}:
        raise ProofError("prepared directory inventory differs from the exact generated inventory")
    for name in expected - {"prepared.json"}:
        data = _read_bounded_regular(d / name, MAX_PREPARED_FILE_BYTES, name)
        if sha256_bytes(data) != files[name]:
            raise ProofError(f"prepared file {name} changed")
    candidate_glb_raw = rec.get("inputs", {}).get("candidate_glb_path") if isinstance(rec.get("inputs"), dict) else None
    if (not isinstance(candidate_glb_raw, str) or Path(candidate_glb_raw).name != "candidate-model.glb"
            or str(Path(candidate_glb_raw).parent / "candidate-model.glb") != candidate_glb_raw):
        raise ProofError("recorded candidate path must be the canonical authenticated candidate-model.glb")
    candidate_dir = Path(candidate_glb_raw).parent
    inputs = authenticate(object_id, candidate_dir, manifest, texture_dir)
    recorded_inputs = rec["inputs"]
    if (recorded_inputs.get("candidate_glb_sha256") != inputs["glb_sha256"]
            or recorded_inputs.get("candidate_receipt_sha256") != inputs["receipt_sha256"]
            or recorded_inputs.get("manifest_sha256") != inputs["manifest_sha256"]
            or recorded_inputs.get("ply_sha256") != inputs["ply_sha256"]
            or recorded_inputs.get("texture", {}).get("sha256") != inputs["texture_info"]["sha256"]):
        raise ProofError("re-authenticated inputs differ from the prepare record")
    logicals = list(ONBOARDING[object_id].keys())
    inventory = [{"frame_id": i["frame_id"], "logical_index": i["logical_index"], "rgb_sha256": i["rgb_sha256"]}
                 for i in inputs["images"]]
    if (len(inputs["images"]) != len(frames) or [i["frame_id"] for i in inputs["images"]] != frames
            or [i["logical_index"] for i in inputs["images"]] != logicals or rec.get("image_inventory") != inventory):
        raise ProofError("onboarding image inventory/order differs")
    rec_images = rec.get("images")
    if not isinstance(rec_images, list) or len(rec_images) != len(frames):
        raise ProofError(f"prepare record must have exactly {len(frames)} onboarding image rows")
    for k, img in enumerate(inputs["images"]):
        rec_img = rec_images[k]
        if not isinstance(rec_img, dict) or set(rec_img) != IMAGE_ROW_KEYS:
            raise ProofError("prepare image row has unexpected keys")
        if (type(rec_img["frame_id"]) is not int or rec_img["frame_id"] != frames[k]
                or type(rec_img["logical_index"]) is not int or rec_img["logical_index"] != logicals[k]
                or rec_img["rgb_sha256"] != img["rgb_sha256"]
                or not np.array_equal(_strict_matrix(rec_img["K"], 3, 3, "K"), img["K"])
                or not np.array_equal(_strict_matrix(rec_img["R"], 3, 3, "R"), img["R"])
                or not np.array_equal(_strict_vector(rec_img["t_mm"], 3, "t_mm"), img["t_mm"])):
            raise ProofError(f"onboarding row {k} (frame/logical index/RGB/K/pose) differs from the authenticated input")
    # Re-derive every CPU claim instead of trusting recorded booleans.
    source = source_convention(inputs)
    if not _same(source, rec.get("source_convention")):
        raise ProofError("re-derived source-convention checks differ from the prepare record")
    corr = evaluate_landmarks(d / "landmarks.json" if "landmarks.json" in files else None, inputs)
    if not _same(corr, rec.get("onboarding_correspondence")):
        raise ProofError("re-derived landmark evaluation differs from the prepare record")
    expected_corr = True if corr["status"] == "pass" else corr["status"]
    if (rec.get("source_convention_pass") is not bool(source["passed"])
            or rec.get("onboarding_correspondence_pass") != expected_corr
            or rec.get("serving_render_pass") != "pending_render_phase" or rec.get("final_status") != "inconclusive"):
        raise ProofError("recorded pass flags differ from the re-derived results")
    references = []
    for k, (tag, img) in enumerate(zip(tags, inputs["images"])):
        rec_img = rec_images[k]
        ref = reference_image(inputs, img)
        with np.load(d / f"{tag}-samples.npz", allow_pickle=False) as data:
            stored = {key: data[key] for key in data.files}
        check = validate_samples(stored)
        recomputed = ref["samples"]
        if set(stored) != set(recomputed) or any(not np.array_equal(stored[key], recomputed[key]) for key in stored):
            raise ProofError(f"{tag}: stored samples differ from the CPU-recomputed reference")
        if (not _same(ref["counts"], rec_img.get("counts")) or rec_img.get("enough_valid_pixels") is not check["enough"]
                or not _same(ref["raster_stats"], rec_img.get("raster_stats"))):
            raise ProofError(f"{tag}: declared counts differ from the actual arrays")
        references.append({"tag": tag, "samples": recomputed, "enough": check["enough"], "count": check["count"]})
    if len(references) != len(frames) or rec.get("reference_ready") is not all(r["enough"] for r in references):
        raise ProofError("recorded reference readiness differs from the re-derived samples")
    return {"record": rec, "inputs": inputs, "source": source, "correspondence": corr, "references": references,
            "prepared_sha256": expected_sha256, "candidate_glb_path": str(candidate_dir / "candidate-model.glb")}


# --------------------------------------------------------------------------
# render (explicit parent-only phase)

def _sampler_state(renderer) -> list[dict]:
    """Actual texture sampler settings of an adapter's inner pyrender scene."""
    states = []
    scene = getattr(renderer._inner, "scene", None)
    for mesh in getattr(scene, "meshes", []) or []:
        for prim in getattr(mesh, "primitives", []) or []:
            tex = getattr(getattr(prim, "material", None), "baseColorTexture", None)
            sampler = getattr(tex, "sampler", None)
            if sampler is None:
                continue
            states.append({"magFilter": sampler.magFilter, "minFilter": sampler.minFilter,
                           "wrapS": sampler.wrapS, "wrapT": sampler.wrapT,
                           "effective_min": sampler.minFilter if sampler.minFilter is not None else 9987,
                           "effective_mag": sampler.magFilter if sampler.magFilter is not None else GL_LINEAR})
    return states


def _base_level(states: list[dict]) -> bool:
    return bool(states) and all(s["effective_min"] == GL_LINEAR and s["effective_mag"] == GL_LINEAR
                                and s["wrapS"] == GL_CLAMP_TO_EDGE and s["wrapT"] == GL_CLAMP_TO_EDGE for s in states)


def _compare(color: np.ndarray, depth: np.ndarray, s: Mapping[str, np.ndarray]) -> dict:
    d = np.abs(depth[s["y"], s["x"]] - s["depth_mm"])
    c = np.abs(color[s["y"], s["x"], :3] - s["rgb"])
    return {"depth_median_mm": float(np.median(d)), "depth_p99_mm": float(np.percentile(d, 99)),
            "rgb_median": float(np.median(c)), "rgb_p99": float(np.percentile(c, 99)), "samples": int(len(d))}


def _judge(cand: dict, opp: dict, enough: bool) -> str:
    matches = (cand["depth_median_mm"] <= THRESHOLDS["depth_median_mm"] and cand["depth_p99_mm"] <= THRESHOLDS["depth_p99_mm"]
               and cand["rgb_median"] <= THRESHOLDS["rgb_median"] and cand["rgb_p99"] <= THRESHOLDS["rgb_p99"])
    discriminated = (opp["rgb_median"] > THRESHOLDS["opposite_min_median"]
                     and cand["rgb_median"] * THRESHOLDS["candidate_vs_opposite_factor"] <= opp["rgb_median"])
    return "inconclusive" if not enough or not discriminated else "pass" if matches else "fail"


def render(prepared_dir: Path, out: Path, *, expected_prepared_sha256: str, texture_dir: Path = DEFAULT_TEXTURE_DIR,
           manifest: Path = yc.DEFAULT_MANIFEST) -> dict:
    # 1) Reject an invalid proof before creating any output or importing a renderer.
    valid = validate_prepared(prepared_dir, expected_prepared_sha256, texture_dir=texture_dir, manifest=manifest)
    rec, inputs = valid["record"], valid["inputs"]
    if valid["source"]["passed"] is not True:
        raise ProofError("re-derived source_convention_pass is false in this runtime; no GPU render is started")
    opposite_bytes, opposite_info = opposite_uv_glb(inputs["glb"])
    out = new_output_dir(out)
    scratch = out / "scratch"
    stage = "setup"
    files: dict[str, str] = {}
    try:
        scratch.mkdir()
        # Render only the authenticated bytes (never reopen the recorded literal path).
        glbs = {"candidate": scratch / "candidate-model.glb", "opposite": scratch / "opposite-uv.glb"}
        glbs["candidate"].write_bytes(inputs["glb"])
        if sha256_file(glbs["candidate"]) != CANDIDATE_PINS[rec["object_id"]][0]:
            raise ProofError("scratch candidate copy differs from the authenticated candidate pin")
        glbs["opposite"].write_bytes(opposite_bytes)
        from bench import quality_gotrack as qg
        from bench import workbench_render as wrender
        from bench import workbench_telemetry as wt
        qg.upstream_path()
        from utils import structs  # type: ignore
        import pyrender
        import trimesh
        # The modules actually imported must be the pinned files validated above.
        dirs = valid["source"]["source_lines"]["package_dirs"]
        if (str(Path(pyrender.__file__).resolve().parent) != dirs["pyrender"]
                or str(Path(trimesh.__file__).resolve().parent) != dirs["trimesh"]):
            raise ProofError("imported pyrender/trimesh are not the runtime packages whose sources were pinned")
        if not _same(source_line_checks(), valid["source"]["source_lines"]):
            raise ProofError("reviewed source pins drifted after validation")

        def draw(hyp: str, img: dict, *, base_level: bool) -> tuple[np.ndarray, np.ndarray, dict, list]:
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = img["R"], img["t_mm"]
            K = img["K"]
            camera = structs.PinholePlaneCameraModel(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], f=(K[0, 0], K[1, 1]),
                                                     c=(K[0, 2], K[1, 2]), T_world_from_eye=np.linalg.inv(pose))
            renderer = wrender.WorkbenchZeroStorageRenderer(str(glbs[hyp]), 1)
            try:
                if base_level:  # diagnostic instance only; production renderer/shaders untouched
                    for mesh in renderer._inner.scene.meshes:
                        for prim in mesh.primitives:
                            tex = getattr(prim.material, "baseColorTexture", None)
                            if tex is not None:
                                tex.sampler = pyrender.Sampler(magFilter=GL_LINEAR, minFilter=GL_LINEAR,
                                                               wrapS=GL_CLAMP_TO_EDGE, wrapT=GL_CLAMP_TO_EDGE)
                states = _sampler_state(renderer)
                outmap = renderer.render_object_model(1, camera)
                proof = renderer.metadata
            finally:
                renderer.close()
            color = depth = None
            for key, value in outmap.items():
                name = getattr(key, "name", str(key)).upper()
                if name.endswith("COLOR"):
                    color = np.asarray(value, dtype=np.float64)
                elif name.endswith("DEPTH"):
                    depth = np.asarray(value, dtype=np.float64)
            if color is None or depth is None:
                raise ProofError("renderer returned no colour/depth")
            return color, depth, proof, states

        stage = "render"
        serving, control = [], None
        with wt.gpu_stage("ycb_uv_proof_render"):
            for img, ref in zip(inputs["images"], valid["references"]):
                row = {"frame_id": img["frame_id"], "hypotheses": {}}
                colors = {}
                for hyp in ("candidate", "opposite"):
                    color, depth, proof, states = draw(hyp, img, base_level=False)
                    colors[hyp] = color
                    row["hypotheses"][hyp] = {**_compare(color, depth, ref["samples"]), "allocation_proof": proof,
                                              "sampler_state": states}
                    files[f"{ref['tag']}-{hyp}-serving-render.png"] = save_png(
                        out, f"{ref['tag']}-{hyp}-serving-render.png",
                        (np.clip(color[:, :, :3], 0, 1) * 255).round().astype(np.uint8))
                states = row["hypotheses"]["candidate"]["sampler_state"]
                row["numeric_status"] = _judge(row["hypotheses"]["candidate"], row["hypotheses"]["opposite"], ref["enough"])
                row["status"] = (row["numeric_status"] if _base_level(states)
                                 else "inconclusive: reference_sampler_mismatch")
                row["reference_sampler"] = "base-level clamp bilinear (CPU oracle)"
                ref_png = decode_png_rgb((Path(prepared_dir) / f"{ref['tag']}-reference-rgb.png").read_bytes())
                files[f"{ref['tag']}-serving-comparison.png"] = save_png(out, f"{ref['tag']}-serving-comparison.png",
                    np.concatenate([img["rgb"], ref_png] + [(np.clip(colors[h][:, :, :3], 0, 1) * 255).round()
                                                             .astype(np.uint8) for h in ("candidate", "opposite")], axis=1))
                serving.append(row)
            # Base-level sampler control: final onboarding image only, 2 renders.
            stage = "base_level_sampler_control"
            img, ref = inputs["images"][-1], valid["references"][-1]
            control = {"frame_id": img["frame_id"], "hypotheses": {},
                       "label": "diagnostic renderer instance with GL_LINEAR min/mag + CLAMP_TO_EDGE textures; "
                                "cannot set serving_render_pass"}
            colors = {}
            for hyp in ("candidate", "opposite"):
                color, depth, proof, states = draw(hyp, img, base_level=True)
                colors[hyp] = color
                control["hypotheses"][hyp] = {**_compare(color, depth, ref["samples"]), "allocation_proof": proof,
                                              "sampler_state": states}
            control["sampler_verified_base_level"] = _base_level(control["hypotheses"]["candidate"]["sampler_state"])
            control["status"] = (_judge(control["hypotheses"]["candidate"], control["hypotheses"]["opposite"], ref["enough"])
                                 if control["sampler_verified_base_level"] else "inconclusive: sampler_not_base_level")
            files[f"{ref['tag']}-base-level-control.png"] = save_png(out, f"{ref['tag']}-base-level-control.png",
                np.concatenate([(np.clip(colors[h][:, :, :3], 0, 1) * 255).round().astype(np.uint8)
                                for h in ("candidate", "opposite")], axis=1))
        stage = "cleanup"
        shutil.rmtree(scratch)  # only our own scratch; the opposite GLB hash is recorded
    except BaseException as exc:
        cleanup_error = None
        if scratch.exists() and scratch.parent == out:
            try:
                shutil.rmtree(scratch)
            except OSError as cexc:
                cleanup_error = f"{type(cexc).__name__}: {cexc}"
        failure = {"schema": SCHEMA_RENDER, "status": "failed", "stage": stage,
                   "error": f"{type(exc).__name__}: {exc}"[:2000], "cleanup_error": cleanup_error,
                   "prepared_sha256": valid["prepared_sha256"], "files_written_before_failure": files,
                   "note": "incomplete render attempt; no pass of any kind"}
        try:
            write_new(out, "failure.json", _canonical(_jsonable(failure)))
        except Exception:  # noqa: BLE001 - never hide the original failure
            pass
        raise
    statuses = [r["status"] for r in serving]
    serving_pass: Any
    if any(s == "fail" for s in statuses):
        serving_pass = "fail"
    elif len(statuses) == 2 and all(s == "pass" for s in statuses):
        serving_pass = True  # only reachable when the actual serving sampler is the base-level oracle
    else:
        serving_pass = "inconclusive"
    corr_pass = True if valid["correspondence"]["status"] == "pass" else valid["correspondence"]["status"]
    all_pass = valid["source"]["passed"] is True and corr_pass is True and serving_pass is True
    record = {"schema": SCHEMA_RENDER, "object_id": rec["object_id"], "prepared_sha256": valid["prepared_sha256"],
              "thresholds": THRESHOLDS, "reference_method": REFERENCE_METHOD, "opposite_hypothesis": opposite_info,
              "authenticated_candidate": {"path": valid["candidate_glb_path"], "sha256": inputs["glb_sha256"],
                                          "rendered_from": "scratch copy of the authenticated bytes (removed)"},
              "serving_images": serving, "base_level_sampler_control": control,
              "renderer_factory": wrender.factory_identity(),
              "source_convention_pass": bool(valid["source"]["passed"]),
              "onboarding_correspondence_pass": corr_pass, "serving_render_pass": serving_pass,
              "final_status": "eligible_for_review" if all_pass else "failed" if serving_pass == "fail" else "inconclusive",
              "promotion": False, "files": files, "tool_sha256": sha256_file(Path(__file__)),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    files["render.json"] = write_new(out, "render.json", _canonical(_jsonable(record)))
    return {"directory": str(out), "record": record}


# --------------------------------------------------------------------------
# CLI

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Staged YCB UV-convention proof (never promotes candidates).")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="pure CPU authentication, source convention and reference")
    s.add_argument("--object-id", type=int, required=True, choices=sorted(ONBOARDING))
    s.add_argument("--candidate-dir", required=True)
    s.add_argument("--manifest", default=str(yc.DEFAULT_MANIFEST))
    s.add_argument("--texture-dir", default=str(DEFAULT_TEXTURE_DIR))
    s.add_argument("--landmarks", help="frozen workbench_uv_landmarks_v1 file (optional)")
    s.add_argument("--out", required=True, help="new immutable output directory")
    s = sub.add_parser("render", help="explicit parent-only GPU render comparison")
    s.add_argument("--prepared", required=True)
    s.add_argument("--expected-prepared-sha256", required=True, help="exact SHA-256 of the prepare output prepared.json")
    s.add_argument("--texture-dir", default=str(DEFAULT_TEXTURE_DIR))
    s.add_argument("--manifest", default=str(yc.DEFAULT_MANIFEST))
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.object_id, Path(args.candidate_dir), Path(args.out), manifest=Path(args.manifest),
                             texture_dir=Path(args.texture_dir),
                             landmarks=Path(args.landmarks) if args.landmarks else None)
            r = result["record"]
            print(json.dumps({"directory": result["directory"], "prepared_sha256": result["prepared_sha256"],
                              "source_convention_pass": r["source_convention_pass"],
                              "onboarding_correspondence_pass": r["onboarding_correspondence_pass"],
                              "serving_render_pass": r["serving_render_pass"], "final_status": r["final_status"],
                              "images": [{"frame_id": i["frame_id"], **i["counts"]} for i in r["images"]]}, indent=2))
        else:
            result = render(Path(args.prepared), Path(args.out), expected_prepared_sha256=args.expected_prepared_sha256,
                            texture_dir=Path(args.texture_dir), manifest=Path(args.manifest))
            r = result["record"]
            print(json.dumps({"directory": result["directory"], "serving_render_pass": r["serving_render_pass"],
                              "base_level_sampler_control": (r["base_level_sampler_control"] or {}).get("status"),
                              "final_status": r["final_status"]}, indent=2))
        return 0
    except (ProofError, yc.ControlError, OSError, KeyError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
