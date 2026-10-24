"""Export prepared-metric CAD diagnostic controls for supplemental YCB-Video clips.

This is a *controlled diagnostic* exporter, not automatic RGB preparation. For
one of three pinned BOP YCB-Video objects it:

1. authenticates the pinned inference manifest and the frozen source PLY;
2. converts the PLY to an embedded-only glTF 2.0 binary (GLB) in metres,
   keeping the original right-handed BOP model basis, triangle order and
   source UVs, with a flat unlit material made only from real appearance data:
   - no appearance data: geometry-only GLB, ``blocked_unsupported_appearance``;
   - PLY vertex colours: ``prepared_control`` with a metric bundle;
   - the PLY's ``TextureFile`` PNG: admitted only if it matches the immutable
     ``TRUSTED_TEXTURES`` pin (object, dataset revision, archive member,
     SHA-256, size) and fully decodes to the pinned dimensions. Even then the
     UV orientation is an unverified assumption, so the run is
     ``blocked_unverified_uv``: a candidate GLB and receipt are kept, no
     metric bundle is written, and the candidate must not be consumed by the
     neural pipeline until a separate orientation proof has been reviewed.
     There is no flag that bypasses this;
3. optionally (``--allow-dataset-onboarding-control``) seeds the bundle with
   the *final onboarding* pose of the inference manifest, labelled
   ``controlled_init:dataset_onboarding:``. Scored frames, scored labels and
   ``evaluator.json`` are never read;
4. writes ``metric-bundle.json`` (``workbench_metric_bundle_v1``, loadable
   unchanged by ``bench.workbench_neural.MetricBundle``), the GLB and a
   separate provenance receipt into a brand-new directory, atomically.

Only the standard library and NumPy are used: no torch, model, renderer or
mesh-library imports, and no network access.

Unit conversion happens exactly once: BOP millimetres x 0.001 -> metres for
vertex positions and for the seed translation. glTF UVs use a top-left
origin; the exporter *assumes* (unverified) that the PLY ``texture_v`` uses a
bottom-left origin and writes ``v_gltf = 1 - v_ply``. The receipt records
``uv_orientation_verified: false``. The recorded signed volume is an aggregate
over the mesh; a positive value is not a per-face orientation proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import struct
import sys
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = "visualizeit.workbench_ycb_controls"
CONVERTER_VERSION = "1"
RECEIPT_SCHEMA = "workbench_ycb_control_receipt_v1"
BUNDLE_SCHEMA = "workbench_metric_bundle_v1"
FAMILY = "ycb_640x480"
METRIC_PROVENANCE = "dataset_metric_cad"
SEED_PREFIX = "controlled_init:dataset_onboarding:"

DEFAULT_MANIFEST = ROOT / ".cache/prototype-sprint/dataset-pilot-v1/inference.json"
MANIFEST_SHA256 = "9b936de31c5511e3644e065538dedf47d0cdcd22f2ba17cee083e75a25403945"
OBJECTS = {
    2: ("003_cracker_box", "121df764d2214a16d59fd77a31e168ca9e79ba0f30a5e529de86c8cd19c1686f"),
    4: ("005_tomato_soup_can", "fea7ae381b760b4852463fce4effa53fe9535dc5c8c1ee89de2be85a30b32211"),
    18: ("040_large_marker", "aa6f0f674e68e4affd95f7d21ae052539fe817590492d485f12e34dd80489a2a"),
}

DATASET_REVISION = "5c2c4aa229800355648cd268040aa814f8dc94f0"
TEXTURE_ARCHIVE = (f"https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{DATASET_REVISION}/"
                   "ycbv_models.zip")


def _trusted(member: str, sha256: str, size: int) -> MappingProxyType:
    return MappingProxyType({"revision": DATASET_REVISION, "archive": TEXTURE_ARCHIVE, "member": member,
                             "sha256": sha256, "bytes": size, "dimensions": (4096, 4096)})


# Official BOP YCB-Video textures; the binding is by object, revision, member and content hash.
TRUSTED_TEXTURES = MappingProxyType({
    2: _trusted("models/obj_000002.png", "9a04bc0e7a10766e44aee4bdfc719462b0b6cc53d186bc6d4e8b890691acdf59", 10406584),
    4: _trusted("models/obj_000004.png", "33183a894eb75ae04ae5188a3ae689fe8d06ec6b9f9624da1bf853c2a25e70d2", 9350703),
    18: _trusted("models/obj_000018.png", "0351acbfd666baabc23261c3f7d5d7645c5b55bd47cef0dbc101572552c888fb", 4403500),
})
UV_ASSUMPTION = "TEXCOORD_0 = (u, 1 - v), assuming a bottom-left PLY texture_v origin (UNVERIFIED)"

MM_TO_M = 0.001
MAX_MANIFEST_BYTES = 16 << 20
MAX_PLY_BYTES = 64 << 20
MAX_TEXTURE_BYTES = 16 << 20
MAX_GLB_BYTES = 20_000_000
MAX_DERIVED_BYTES = 1 << 30
FREE_RESERVE_BYTES = 1 << 30
MAX_ELEMENTS = 1_000_000

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_BLOCKED = 3

_PLY_SCALARS = {"float": np.float32, "float32": np.float32, "double": np.float64, "float64": np.float64,
                "uchar": np.uint8, "uint8": np.uint8}
_PLY_NAMES = {"x", "y", "z", "nx", "ny", "nz", "texture_u", "texture_v", "red", "green", "blue", "alpha"}
_FLOAT, _UINT, _UBYTE = 5126, 5125, 5121
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_DTYPES = {_FLOAT: np.dtype("<f4"), _UINT: np.dtype("<u4"), _UBYTE: np.dtype("u1")}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


class ControlError(Exception):
    """The control cannot be produced safely; nothing is published."""


# --------------------------------------------------------------------------
# small helpers


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _loads_strict(raw: bytes, what: str) -> Any:
    def pairs(items):
        out = {}
        for key, val in items:
            if key in out:
                raise ControlError(f"{what}: duplicate JSON key {key!r}")
            out[key] = val
        return out

    def constant(token):
        raise ControlError(f"{what}: non-finite JSON value {token}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"{what}: invalid JSON ({exc})") from exc


def _read_regular(path: Path, limit: int, what: str) -> bytes:
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ControlError(f"{what}: symlink in path {part}")
    if not path.is_file():
        raise ControlError(f"{what}: not a regular file: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ControlError(f"{what}: larger than {limit} bytes")
    return data


def _safe_relative(base: Path, rel: Any, what: str) -> Path:
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        raise ControlError(f"{what}: unsafe relative path {rel!r}")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ControlError(f"{what}: unsafe relative path {rel!r}")
    return base.joinpath(*pure.parts)


def _finite_list(value: Any, length: int, what: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ControlError(f"{what}: expected {length} numbers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ControlError(f"{what}: non-finite or non-numeric entry")
        out.append(float(item))
    return out


# --------------------------------------------------------------------------
# manifest


def load_manifest(path: Path) -> tuple[dict, str]:
    """Authenticate the pinned inference manifest (SHA checked before parsing)."""
    raw = _read_regular(path, MAX_MANIFEST_BYTES, "inference manifest")
    digest = _sha256(raw)
    if digest != MANIFEST_SHA256:
        raise ControlError(f"inference manifest SHA-256 {digest} != pinned {MANIFEST_SHA256}")
    manifest = _loads_strict(raw, "inference manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("objects"), list):
        raise ControlError("inference manifest has no objects list")
    return manifest, digest


def select_object(manifest: dict, object_id: int) -> dict:
    if object_id not in OBJECTS:
        raise ControlError(f"object {object_id} is not one of the allowed controls {sorted(OBJECTS)}")
    matches = [o for o in manifest["objects"] if isinstance(o, dict) and o.get("objectId") == object_id]
    if len(matches) != 1:
        raise ControlError(f"manifest must contain exactly one object {object_id}")
    obj = matches[0]
    canonical, model_sha = OBJECTS[object_id]
    if obj.get("canonicalYcbId") != canonical:
        raise ControlError(f"object {object_id} is {obj.get('canonicalYcbId')!r}, expected {canonical!r}")
    model = obj.get("model")
    if not isinstance(model, dict) or model.get("available") is not True or model.get("sha256") != model_sha:
        raise ControlError(f"object {object_id} model entry does not match the frozen PLY pin")
    return obj


def onboarding_seed(obj: dict, object_id: int) -> dict:
    """Final onboarding pose only (by logicalIndex). Scored frames contribute frame ids only."""
    rows = obj.get("onboarding")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ControlError("expected exactly 10 onboarding rows")
    indices = [row.get("logicalIndex") if isinstance(row, dict) else None for row in rows]
    if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
        raise ControlError("onboarding logicalIndex values must be unique integers")
    final = rows[indices.index(max(indices))]
    frame_id = final.get("frameId")
    if type(frame_id) is not int or frame_id < 0:
        raise ControlError("final onboarding frameId must be a non-negative integer")
    scored_ids = {f.get("frameId") for f in obj.get("frames", []) if isinstance(f, dict)}
    if frame_id in scored_ids:
        raise ControlError(f"final onboarding frame {frame_id} is also a scored frame")
    pose = final.get("evaluationPose")
    if not isinstance(pose, dict) or pose.get("obj_id") != object_id:
        raise ControlError("final onboarding pose is missing or names another object")
    rotation = np.array(_finite_list(pose.get("cam_R_m2c"), 9, "cam_R_m2c"), dtype=np.float64).reshape(3, 3)
    translation_mm = np.array(_finite_list(pose.get("cam_t_m2c"), 3, "cam_t_m2c"), dtype=np.float64)
    orthogonality = float(np.abs(rotation.T @ rotation - np.eye(3)).max())
    determinant = float(np.linalg.det(rotation))
    if orthogonality > 1e-4 or abs(determinant - 1.0) > 1e-4:
        raise ControlError("final onboarding rotation is not a proper rotation")
    pose_m = np.eye(4)
    pose_m[:3, :3] = rotation
    pose_m[:3, 3] = translation_mm * MM_TO_M
    calibration = final.get("calibration") if isinstance(final.get("calibration"), dict) else {}
    return {
        "pose_m": [[float(v) for v in row] for row in pose_m],
        "frame_id": frame_id,
        "logical_index": max(indices),
        "translation_mm": translation_mm.tolist(),
        "rotation_orthogonality_max_abs": orthogonality,
        "rotation_determinant": determinant,
        "cam_K": calibration.get("cam_K"),
        "rgb": final.get("rgb"),
    }


# --------------------------------------------------------------------------
# PLY


def parse_ascii_ply(raw: bytes) -> dict:
    """Strict ASCII PLY reader for the official BOP YCB-Video models."""
    marker = b"end_header"
    end = raw.find(marker)
    if end < 0:
        raise ControlError("PLY has no end_header")
    newline = raw.find(b"\n", end)
    if newline < 0:
        raise ControlError("PLY header is not terminated")
    try:
        header = raw[:newline].decode("ascii").splitlines()
        body = raw[newline + 1:].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ControlError("PLY is not ASCII") from exc
    if [line.strip() for line in header[:2]] != ["ply", "format ascii 1.0"]:
        raise ControlError("expected the ASCII BOP PLY variant")
    comments, elements, current = [], [], None
    for line in header[2:]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "comment":
            comments.append(line.strip()[len("comment"):].strip())
        elif tokens[0] == "element" and len(tokens) == 3:
            current = {"name": tokens[1], "count": int(tokens[2]), "props": []}
            elements.append(current)
        elif tokens[0] == "property" and current is not None:
            current["props"].append(tokens[1:])
        elif tokens[0] == "end_header":
            break
        else:
            raise ControlError(f"unsupported PLY header line {line!r}")
    if [e["name"] for e in elements] != ["vertex", "face"]:
        raise ControlError("PLY must contain exactly vertex then face elements")
    vertex, face = elements
    if not (0 < vertex["count"] <= MAX_ELEMENTS and 0 < face["count"] <= MAX_ELEMENTS):
        raise ControlError("PLY element counts out of bounds")
    names = []
    for prop in vertex["props"]:
        if len(prop) != 2 or prop[0] not in _PLY_SCALARS or prop[1] not in _PLY_NAMES or prop[1] in names:
            raise ControlError(f"unsupported vertex property {prop}")
        names.append(prop[1])
    if face["props"] not in ([["list", "uchar", "int", "vertex_indices"]], [["list", "uchar", "uint", "vertex_indices"]]):
        raise ControlError(f"unsupported face property {face['props']}")
    for required in ("x", "y", "z"):
        if required not in names:
            raise ControlError(f"PLY lacks vertex property {required}")

    tokens = body.split()
    n_vertex_tokens = vertex["count"] * len(names)
    n_face_tokens = face["count"] * 4
    if len(tokens) != n_vertex_tokens + n_face_tokens:
        raise ControlError("PLY body length does not match the header (non-triangle faces or trailing data)")
    try:
        data = np.asarray(tokens[:n_vertex_tokens], dtype=np.float64).reshape(vertex["count"], len(names))
        faces = np.asarray(tokens[n_vertex_tokens:], dtype=np.int64).reshape(face["count"], 4)
    except ValueError as exc:
        raise ControlError(f"PLY body is not numeric: {exc}") from exc
    if not np.isfinite(data).all():
        raise ControlError("PLY contains non-finite vertex data")
    if (faces[:, 0] != 3).any():
        raise ControlError("only triangle faces are supported")
    triangles = faces[:, 1:]
    if triangles.min() < 0 or triangles.max() >= vertex["count"]:
        raise ControlError("PLY face index out of range")
    column = {name: data[:, i] for i, name in enumerate(names)}
    texture_files = [c.split(None, 1)[1] for c in comments if c.startswith("TextureFile ") and len(c.split()) == 2]
    colors = None
    if {"red", "green", "blue"} <= set(names):
        channels = [column[c] for c in ("red", "green", "blue")] + ([column["alpha"]] if "alpha" in names else [])
        colors = np.stack(channels, axis=1)
        if (colors < 0).any() or (colors > 255).any() or (colors != np.round(colors)).any():
            raise ControlError("PLY vertex colours must be integers in [0, 255]")
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.full((len(colors), 1), 255.0)], axis=1)
        colors = colors.astype(np.uint8)
    return {
        "positions_mm": np.stack([column["x"], column["y"], column["z"]], axis=1),
        "normals": np.stack([column[c] for c in ("nx", "ny", "nz")], axis=1) if {"nx", "ny", "nz"} <= set(names) else None,
        "uv": np.stack([column["texture_u"], column["texture_v"]], axis=1) if {"texture_u", "texture_v"} <= set(names) else None,
        "colors": colors,
        "triangles": triangles.astype(np.uint32),
        "comments": comments,
        "texture_files": texture_files,
        "vertex_properties": names,
        "vertex_count": vertex["count"],
        "face_count": face["count"],
    }


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ControlError("texture is not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    if not (0 < width <= 16384 and 0 < height <= 16384):
        raise ControlError("texture dimensions out of bounds")
    return width, height


def admit_texture(object_id: int, mesh: dict, manifest: dict, texture_path: Path) -> tuple[bytes, dict]:
    """Authenticate and fully decode the official texture before any GLB is built."""
    trusted = TRUSTED_TEXTURES.get(object_id)
    if trusted is None:
        raise ControlError(f"no trusted texture pin for object {object_id}")
    if manifest.get("datasetRevision") != trusted["revision"]:
        raise ControlError("manifest dataset revision differs from the trusted texture revision")
    expected_name = PurePosixPath(trusted["member"]).name
    if mesh["uv"] is None or mesh["texture_files"] != [expected_name]:
        raise ControlError(f"PLY TextureFile {mesh['texture_files']} is not the trusted member {expected_name}")
    if texture_path.name != expected_name:
        raise ControlError(f"texture file name {texture_path.name} is not {expected_name}")
    data = _read_regular(texture_path, MAX_TEXTURE_BYTES, "texture PNG")
    digest = _sha256(data)
    if len(data) != trusted["bytes"] or digest != trusted["sha256"]:
        raise ControlError(f"texture {texture_path.name} SHA-256 {digest} ({len(data)} B) does not match the "
                           f"trusted pin {trusted['sha256']} ({trusted['bytes']} B)")
    header_size = _png_dimensions(data)
    try:
        from PIL import Image  # noqa: PLC0415 - existing dependency, imported only for decoding

        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                raise ControlError(f"texture decodes as {image.format}, not PNG")
            image.load()
            mode, size = image.mode, image.size
            pixels = np.asarray(image)
    except ControlError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decoder failure rejects the texture
        raise ControlError(f"texture PNG failed full decoding: {type(exc).__name__}: {exc}") from exc
    if size != tuple(trusted["dimensions"]) or header_size != size or pixels.shape[:2] != size[::-1]:
        raise ControlError(f"texture decoded to {size} / {pixels.shape}, expected {trusted['dimensions']}")
    return data, {
        "path": texture_path.as_posix(), "sha256": digest, "bytes": len(data),
        "width": size[0], "height": size[1], "mode": mode, "decoded_shape": list(pixels.shape),
        "trusted_binding": {k: (list(v) if isinstance(v, tuple) else v) for k, v in trusted.items()},
        "authentication": "object id + dataset revision + archive member + size + SHA-256 matched before GLB "
                          "construction; full PIL verify() and load() decode",
    }


# --------------------------------------------------------------------------
# GLB


def build_glb(mesh: dict, *, name: str, texture: bytes | None) -> tuple[bytes, dict]:
    """Return (GLB bytes, summary). Positions are metres in the BOP model basis."""
    positions = (mesh["positions_mm"] * MM_TO_M).astype("<f4")
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add_view(data: bytes, target: int | None) -> int:
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        blob.extend(data)
        blob.extend(b"\x00" * (-len(blob) % 4))
        views.append(view)
        return len(views) - 1

    def add_accessor(array: np.ndarray, component: int, kind: str, *, target: int,
                     normalized: bool = False, bounds: bool = False) -> int:
        array = np.ascontiguousarray(array.astype(_DTYPES[component]))
        accessor = {"bufferView": add_view(array.tobytes(), target), "componentType": component,
                    "count": int(array.shape[0]), "type": kind}
        if normalized:
            accessor["normalized"] = True
        if bounds:
            accessor["min"] = [float(v) for v in array.reshape(len(array), -1).min(axis=0)]
            accessor["max"] = [float(v) for v in array.reshape(len(array), -1).max(axis=0)]
        accessors.append(accessor)
        return len(accessors) - 1

    attributes = {"POSITION": add_accessor(positions, _FLOAT, "VEC3", target=_ARRAY_BUFFER, bounds=True)}
    normals_note = "absent in source"
    if mesh["normals"] is not None:
        lengths = np.linalg.norm(mesh["normals"], axis=1)
        if (lengths > 1e-12).all():
            attributes["NORMAL"] = add_accessor(mesh["normals"] / lengths[:, None], _FLOAT, "VEC3",
                                                target=_ARRAY_BUFFER)
            normals_note = f"source normals renormalised (max |len-1| before = {float(np.abs(lengths - 1).max()):.3g})"
        else:
            normals_note = "omitted: source contains zero-length normals"
    if mesh["uv"] is not None:
        uv = np.stack([mesh["uv"][:, 0], 1.0 - mesh["uv"][:, 1]], axis=1)
        attributes["TEXCOORD_0"] = add_accessor(uv, _FLOAT, "VEC2", target=_ARRAY_BUFFER)
    if mesh["colors"] is not None:
        attributes["COLOR_0"] = add_accessor(mesh["colors"], _UBYTE, "VEC4", target=_ARRAY_BUFFER, normalized=True)
    indices = add_accessor(mesh["triangles"].reshape(-1), _UINT, "SCALAR", target=_ELEMENT_ARRAY_BUFFER)

    primitive: dict[str, Any] = {"attributes": attributes, "indices": indices, "mode": 4}
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": f"{CONVERTER}/{CONVERTER_VERSION}",
                  "extras": {"units": "metres", "basis": "original right-handed BOP model frame"}},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [primitive]}],
        "accessors": accessors,
        "bufferViews": views,
    }
    appearance = "unsupported"
    if texture is not None or mesh["colors"] is not None:
        material: dict[str, Any] = {
            "name": f"{name}-unlit",
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                     "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_unlit": {}},
        }
        if texture is not None:
            gltf["images"] = [{"bufferView": add_view(texture, None), "mimeType": "image/png"}]
            gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}]
            gltf["textures"] = [{"source": 0, "sampler": 0}]
            material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0, "texCoord": 0}
            appearance = "source_texture_png"
        else:
            appearance = "source_vertex_colors"
        gltf["materials"] = [material]
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
        primitive["material"] = 0
    gltf["buffers"] = [{"byteLength": len(blob)}]

    json_chunk = _canonical(gltf)
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = bytes(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
           + struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk)
    return glb, {"appearance": appearance, "normals": normals_note}


def read_glb(glb: bytes) -> tuple[dict, dict[str, np.ndarray], bytes | None]:
    """Independent GLB reader for the round-trip check (embedded resources only)."""
    magic, version, length = struct.unpack_from("<4sII", glb, 0)
    if magic != b"glTF" or version != 2 or length != len(glb):
        raise ControlError("GLB header invalid")
    offset, gltf, binary = 12, None, b""
    while offset < len(glb):
        size, kind = struct.unpack_from("<I4s", glb, offset)
        chunk = glb[offset + 8: offset + 8 + size]
        if kind == b"JSON":
            gltf = _loads_strict(chunk, "GLB JSON")
        elif kind == b"BIN\x00":
            binary = chunk
        offset += 8 + size
    if not isinstance(gltf, dict):
        raise ControlError("GLB has no JSON chunk")
    for kind in ("buffers", "images"):
        for item in gltf.get(kind, []):
            if "uri" in item:
                raise ControlError(f"GLB {kind} must be embedded")
    views = gltf["bufferViews"]

    def view_bytes(index: int) -> bytes:
        view = views[index]
        return binary[view["byteOffset"]: view["byteOffset"] + view["byteLength"]]

    def accessor(index: int) -> np.ndarray:
        acc = gltf["accessors"][index]
        array = np.frombuffer(view_bytes(acc["bufferView"]), dtype=_DTYPES[acc["componentType"]])
        width = _WIDTH[acc["type"]]
        if array.size != acc["count"] * width:
            raise ControlError("GLB accessor length mismatch")
        return array.reshape(acc["count"], width) if width > 1 else array

    primitive = gltf["meshes"][0]["primitives"][0]
    arrays = {name: accessor(index) for name, index in primitive["attributes"].items()}
    arrays["indices"] = accessor(primitive["indices"])
    image = view_bytes(gltf["images"][0]["bufferView"]) if gltf.get("images") else None
    return gltf, arrays, image


def verify_roundtrip(glb: bytes, mesh: dict, texture: bytes | None) -> dict:
    gltf, arrays, image = read_glb(glb)
    expected = (mesh["positions_mm"] * MM_TO_M).astype("<f4")
    if not np.array_equal(arrays["POSITION"], expected):
        raise ControlError("round trip: positions differ")
    error_mm = float(np.abs(arrays["POSITION"].astype(np.float64) / MM_TO_M - mesh["positions_mm"]).max())
    if error_mm > 1e-3:
        raise ControlError(f"round trip: float32 metre positions deviate {error_mm} mm from source")
    if not np.array_equal(arrays["indices"].reshape(-1, 3), mesh["triangles"]):
        raise ControlError("round trip: triangle indices/winding differ")
    accessor = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    if accessor["min"] != [float(v) for v in expected.min(axis=0)] or accessor["max"] != [float(v) for v in expected.max(axis=0)]:
        raise ControlError("round trip: POSITION bounds differ")
    if mesh["uv"] is not None:
        uv = np.stack([mesh["uv"][:, 0], 1.0 - mesh["uv"][:, 1]], axis=1).astype("<f4")
        if not np.array_equal(arrays["TEXCOORD_0"], uv):
            raise ControlError("round trip: UVs differ from the assumed conversion")
    if mesh["colors"] is not None and not np.array_equal(arrays["COLOR_0"], mesh["colors"]):
        raise ControlError("round trip: vertex colours differ")
    if (texture is None) != (image is None) or (texture is not None and image != texture):
        raise ControlError("round trip: embedded texture differs")
    p = mesh["positions_mm"] * MM_TO_M
    tri = p[mesh["triangles"].astype(np.int64)]
    signed_volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    return {
        "positions_bitwise_equal_float32": True,
        "max_position_error_mm_vs_source": error_mm,
        "indices_and_vertex_order_equal": True,
        "uv_matches_assumed_conversion": mesh["uv"] is not None,
        "uv_check_note": "re-applies the assumed v-flip; serialisation check only, not orientation evidence",
        "vertex_colors_equal": mesh["colors"] is not None,
        "texture_bytes_equal": texture is not None,
        "signed_volume_m3": signed_volume,
        "aggregate_signed_volume_positive": signed_volume > 0,
        "signed_volume_note": "aggregate over all faces; not a per-face orientation proof",
    }


def neural_inventory(path: Path) -> dict | None:
    """Consumer-side GLB check via ``bench.workbench_neural.glb_inventory`` (numpy-only import)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    before = "torch" in sys.modules
    from bench.workbench_neural import glb_inventory  # noqa: PLC0415 - deliberate lazy import

    if not before and "torch" in sys.modules:
        raise ControlError("importing glb_inventory unexpectedly loaded torch")
    return glb_inventory(path)


# --------------------------------------------------------------------------
# publication


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish(out: Path, files: dict[str, bytes], verify) -> None:
    planned = sum(len(data) for data in files.values())
    if planned >= MAX_DERIVED_BYTES:
        raise ControlError(f"derived output {planned} B exceeds the 1 GiB bound")
    free = shutil.disk_usage(out.parent).free
    if free - planned < FREE_RESERVE_BYTES:
        raise ControlError(f"writing {planned} B would leave {free - planned} B free (< 1 GiB reserve)")
    staging = out.parent / f".{out.name}.staging-{secrets.token_hex(6)}"
    os.mkdir(staging)
    try:
        for name, data in files.items():
            _write_new(staging / name, data)
        verify(staging)
        if os.path.lexists(out):
            raise ControlError(f"output appeared during export: {out}")
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _bundle_glb_path(glb: Path) -> str:
    try:
        return glb.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return glb.resolve().as_posix()


def _versions() -> dict:
    return {"converter": CONVERTER, "converter_version": CONVERTER_VERSION,
            "python": platform.python_version(), "numpy": np.__version__}


def export_control(manifest_path: Path, object_id: int, out: Path, principal_origin: str,
                   allow_onboarding: bool, texture_path: Path | None) -> dict:
    out = Path(os.path.abspath(out))
    if os.path.lexists(out):
        raise ControlError(f"output directory already exists (no overwrites): {out}")
    if not out.parent.is_dir():
        raise ControlError(f"output parent does not exist: {out.parent}")
    if principal_origin not in ("integer_centers", "corner"):
        raise ControlError("--principal-origin must be integer_centers or corner")

    manifest_path = Path(os.path.abspath(manifest_path))
    manifest, manifest_sha = load_manifest(manifest_path)
    obj = select_object(manifest, object_id)
    canonical, pinned_ply = OBJECTS[object_id]
    model_path = _safe_relative(manifest_path.parent, obj["model"]["path"], "model path")
    ply = _read_regular(model_path, MAX_PLY_BYTES, "source PLY")
    if _sha256(ply) != pinned_ply:
        raise ControlError(f"source PLY hash differs from pin for object {object_id}")
    mesh = parse_ascii_ply(ply)

    issues: list[str] = []
    texture = None
    texture_info: dict[str, Any] = {"referenced": mesh["texture_files"], "supplied": None}
    if texture_path is not None:
        texture, texture_info["supplied"] = admit_texture(
            object_id, mesh, manifest, Path(os.path.abspath(texture_path)))
    if mesh["uv"] is not None:
        uv_min, uv_max = mesh["uv"].min(axis=0), mesh["uv"].max(axis=0)
        texture_info["uv_range"] = {"min": uv_min.tolist(), "max": uv_max.tolist()}
        if (uv_min < 0).any() or (uv_max > 1).any():
            issues.append("source UVs leave [0,1]; sampler uses clamp-to-edge")

    glb, glb_summary = build_glb(mesh, name=f"ycbv_obj_{object_id:06d}_{canonical}", texture=texture)
    if len(glb) > MAX_GLB_BYTES:
        raise ControlError(f"GLB is {len(glb)} B, above the 20 MB geometry bound")
    roundtrip = verify_roundtrip(glb, mesh, texture)
    if not roundtrip["aggregate_signed_volume_positive"]:
        issues.append("aggregate signed volume is non-positive; source face order preserved unchanged")
    appearance = glb_summary["appearance"]
    if appearance == "source_vertex_colors":
        status = "prepared_control"
    elif appearance == "source_texture_png":
        status = "blocked_unverified_uv"
        issues.append("UV orientation unverified: candidate GLB only, no metric bundle; not consumable by the "
                      "neural pipeline until a separate exact-source orientation proof is reviewed")
    else:
        status = "blocked_unsupported_appearance"
        issues.append(
            "unsupported appearance: the PLY has no vertex colours and its TextureFile "
            f"{mesh['texture_files'] or '(none)'} was not supplied; geometry-only GLB, no metric bundle emitted")
    supported = status == "prepared_control"

    seed = onboarding_seed(obj, object_id) if allow_onboarding else None
    glb_name = "candidate-model.glb" if status == "blocked_unverified_uv" else "model.glb"
    bundle_name, receipt_name = "metric-bundle.json", "provenance-receipt.json"
    glb_sha = _sha256(glb)
    positions_m = (mesh["positions_mm"] * MM_TO_M).astype("<f4").astype(np.float64)
    bounds_min, bounds_max = positions_m.min(axis=0), positions_m.max(axis=0)

    files: dict[str, bytes] = {glb_name: glb}
    bundle = None
    if supported:
        bundle = {
            "schema": BUNDLE_SCHEMA,
            "family": FAMILY,
            "glb_path": _bundle_glb_path(out / glb_name),
            "glb_sha256": glb_sha,
            "obj_id": object_id,
            "principal_point_origin": principal_origin,
            "seed_pose_m": seed["pose_m"] if seed else None,
            "seed_label": (f"{SEED_PREFIX}{canonical}:inference.json final onboarding pose frameId "
                           f"{seed['frame_id']} (known-pose controlled diagnostic)") if seed else None,
            "seed_frame_id": f"s{seed['frame_id']:06d}" if seed else None,
            "metric_provenance": METRIC_PROVENANCE,
        }
        files[bundle_name] = _pretty(bundle)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "label": "known-pose controlled diagnostic (dataset metric CAD); not automatic RGB preparation",
        "claims": {"rgb_automatic_preparation": False, "independent_accuracy": False,
                   "tracking_quality": False, "mobile_readiness": False},
        "object": {"obj_id": object_id, "canonical_ycb_id": canonical, "name": obj.get("name"),
                   "scene_id": obj.get("sceneId")},
        "source": {
            "manifest_path": manifest_path.as_posix(), "manifest_sha256": manifest_sha,
            "dataset": manifest.get("dataset"), "dataset_revision": manifest.get("datasetRevision"),
            "model_path": model_path.as_posix(), "model_sha256": pinned_ply, "model_bytes": len(ply),
            "ply_vertex_properties": mesh["vertex_properties"], "ply_comments": mesh["comments"],
            "not_read": ["evaluator.json", "scored labels", "scored-frame poses", "RGB/depth pixels"],
        },
        "versions": _versions(),
        "conversion": {
            "units": "BOP millimetres x 0.001 -> metres, applied once to positions and seed translation",
            "basis": "original right-handed BOP model frame; no axis swap, recentering or rescaling",
            "winding": "source face vertex order preserved (aggregate signed volume recorded, not per-face proof)",
            "uv": UV_ASSUMPTION if mesh["uv"] is not None else "no source UVs",
            "normals": glb_summary["normals"],
            "material": ("none" if appearance == "unsupported"
                         else "flat KHR_materials_unlit from authenticated source appearance only"),
        },
        "appearance": {"status": appearance, "texture": texture_info,
                       "vertex_colors": mesh["colors"] is not None},
        "uv_orientation": {
            "uv_orientation_verified": False,
            "assumption": UV_ASSUMPTION if mesh["uv"] is not None else None,
            "required_before_use": "separate exact-source orientation proof after onboarding RGB verification",
            "neural_consumption_permitted": supported,
        },
        "glb": {
            "path": (out / glb_name).as_posix(), "sha256": glb_sha, "bytes": len(glb),
            "vertex_count": mesh["vertex_count"], "triangle_count": mesh["face_count"],
            "bounds_min_m": bounds_min.tolist(), "bounds_max_m": bounds_max.tolist(),
            "extent_m": (bounds_max - bounds_min).tolist(), "roundtrip": roundtrip,
        },
        "principal_point_origin": principal_origin,
        "seed": None if seed is None else {
            "permitted_by": "--allow-dataset-onboarding-control",
            "source": "inference.json objects[].onboarding final row (max logicalIndex) evaluationPose",
            "frame_id": seed["frame_id"], "seed_frame_id": f"s{seed['frame_id']:06d}",
            "logical_index": seed["logical_index"], "translation_mm": seed["translation_mm"],
            "rotation_orthogonality_max_abs": seed["rotation_orthogonality_max_abs"],
            "rotation_determinant": seed["rotation_determinant"], "cam_K": seed["cam_K"],
            "rgb": seed["rgb"], "pose_convention": "4x4 object-to-camera, R = cam_R_m2c row-major, t in metres",
        },
        "metric_bundle": None if bundle is None else {"path": (out / bundle_name).as_posix(),
                                                       "sha256": _sha256(files[bundle_name])},
        "issues": issues,
    }
    files[receipt_name] = _pretty(receipt)
    inventory: dict = {}

    def verify(staging: Path) -> None:
        inventory.update(neural_inventory(staging / glb_name))
        if inventory["sha256"] != glb_sha:
            raise ControlError("glb_inventory hash differs from the staged GLB")

    _publish(out, files, verify)
    return {"out": out.as_posix(), "status": receipt["status"], "glb_sha256": glb_sha, "glb_bytes": len(glb),
            "extent_m": receipt["glb"]["extent_m"], "appearance": appearance,
            "bundle_sha256": receipt["metric_bundle"] and receipt["metric_bundle"]["sha256"],
            "receipt_sha256": _sha256(files[receipt_name]), "seed_frame_id": bundle and bundle["seed_frame_id"],
            "inventory": {k: inventory[k] for k in ("mesh_count", "texture_count", "image_count", "node_count")},
            "issues": issues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--object-id", type=int, required=True, choices=sorted(OBJECTS))
    parser.add_argument("--out", type=Path, required=True, help="new directory (must not exist)")
    parser.add_argument("--principal-origin", required=True, choices=("integer_centers", "corner"))
    parser.add_argument("--allow-dataset-onboarding-control", action="store_true",
                        help="seed with the final onboarding pose (known-pose controlled diagnostic)")
    parser.add_argument("--texture", type=Path, default=None,
                        help="the PLY's official TextureFile PNG; must match the trusted pin "
                             "(output stays blocked_unverified_uv)")
    args = parser.parse_args(argv)
    try:
        result = export_control(args.manifest, args.object_id, args.out, args.principal_origin,
                                args.allow_dataset_onboarding_control, args.texture)
    except (ControlError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(result, indent=2))
    return EXIT_OK if result["status"] == "prepared_control" else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
