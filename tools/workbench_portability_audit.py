"""Offline portability AUDIT of prepared metric CAD controls and SurfacePackages.

Accounting only. This tool authenticates the explicit runtime payload files
of each requested row and measures them against the workbench targets: 20,000,000
bytes, 20,000 triangles and a 1024 px texture edge. It reads ONLY explicit
inputs:

* the prepared-control inventory (provenance; not runtime payload);
* each row's metric bundle, the GLB it names, and the explicit
  ``neural-artwork.json`` record supplied per case;
* each explicit SurfacePackage directory (``surface.json`` plus its FileRefs).

It never discovers runs and never reads results, reports, evaluator data,
reference poses, masks, seed images or checkpoints. It never constructs
``NeuralArtwork``, never renders, never runs a model, the GPU, segmentation or
COLMAP, and never dereferences an external URI. Head/base pins are recorded as
``not_read`` and are excluded from the surface bytes.

It claims nothing about mobile readiness, the fit of the full runtime,
automatic geometry, or compression that preserves attachment.
``mobile_verified`` is always ``false``.

    --prepared-inventory PATH --artwork CASE=PATH [...] [--surface-package DIR ...] --out NEW.json
    --surface-package DIR [...] --out NEW.json          (surface-only is allowed)

Exit 0 only when every requested row is VERIFIED and all three targets PASS.
Exit 1 when corruption or a target failure is confirmed. Exit 3 when the result
is incomplete, unsupported or unavailable without a confirmed failure. Exit 2
for invalid CLI input or an output error.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import contextmanager
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import struct
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "workbench_portability_audit_v1"
INVENTORY_SCHEMA = "workbench_prepared_cad_target_inventory_v1"
BUNDLE_SCHEMA = "workbench_metric_bundle_v1"
ARTWORK_SCHEMA = "workbench_neural_artwork_v1"
WORKSPACE = ".cache/video-workbench"
LIMIT_BYTES = 20_000_000
LIMIT_TRIANGLES = 20_000
LIMIT_TEXTURE_EDGE = 1024
PRODUCER_MAX_FACES = 200_000            # NeuralArtwork(max_faces=200_000) -> glb_mesh
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
MAX_JSON_BYTES = 1 << 20
MAX_SURFACE_JSON_BYTES = 1_000_000
MAX_GLB_BYTES = 256 << 20
MAX_PACKAGE_FILE_BYTES = 64 << 20
MAX_OUTPUT_BYTES = 2 << 20
MAX_NPY_ROWS = 10_000_000
MAX_INSTANCES = 100_000
BUILTIN_GENERATION = "fixed_builtin_v1"
BUILTIN_KINDS = ("checkerboard", "directional")
BUNDLE_KEYS = ("schema", "family", "glb_path", "glb_sha256", "obj_id", "principal_point_origin",
               "seed_pose_m", "seed_label", "seed_frame_id", "metric_provenance")
SOURCE_MODULES = ("workbench_contract", "workbench_geometry", "workbench_neural", "workbench_reconstruction",
                  "workbench_image_io", "workbench_telemetry")
FORBIDDEN_IMPORTS = ("torch", "torchvision", "sam2", "pyrender", "OpenGL", "pycolmap",
                     "bench.quality_gotrack", "bench.quality_gotrack_inference", "bench.workbench_render")
TARGETS = ("bytes_target", "triangles_target", "artwork_target")
CASE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
CLAIMS = ("Portability accounting only: no mobile readiness, full-runtime fit, automatic geometry or "
          "attachment-preserving compression claim. Model checkpoints are pinned but not read and are excluded "
          "from the surface bytes; the full mobile runtime is unverified.")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class AuditFail(Exception):
    """Confirmed invalid, missing, tampered or inconsistent input (closure FAIL)."""


class AuditUnsupported(Exception):
    """Input outside the supported subset; nothing is concluded (closure UNVERIFIED)."""


class AuditViolation(BaseException):
    """A guarded route (model, GPU, render, segmentation) was attempted."""


class UsageError(Exception):
    """Invalid CLI request (exit 2)."""


# --------------------------------------------------------------------------
# small helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def strict_json(raw: bytes, what: str) -> Any:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise AuditFail(f"{what}: duplicate JSON key {k!r}")
            out[k] = v
        return out

    def constant(name):
        raise AuditFail(f"{what}: non-finite JSON value {name}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AuditFail(f"{what}: invalid JSON ({exc})") from None


def _is_sha(v: Any) -> bool:
    return isinstance(v, str) and bool(_SHA_RE.match(v))


def _int(v: Any, what: str, minimum: int = 0) -> int:
    if type(v) is not int or v < minimum:
        raise AuditFail(f"{what} must be an integer >= {minimum}")
    return v


def _is_link(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _rel(root: Path, path: Path) -> str:
    return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root))).as_posix()


def resolve_input(root: Path, value: str | Path, *, kind: str = "file") -> Path:
    """Contained regular file/dir under ``root``. Strings are root-relative (``\\`` is
    normalised to ``/``); absolute Paths must lie inside ``root``. Any symlink or
    junction on the path, traversal, drive/stream syntax or escape is rejected."""
    root_abs = Path(os.path.abspath(root))
    if isinstance(value, Path) and value.is_absolute():
        cand = Path(os.path.abspath(value))
    else:
        text = str(value).replace("\\", "/")
        if not text or "\x00" in text or ":" in text or text.startswith("/"):
            raise AuditFail(f"path {str(value)[:200]!r} is not a safe root-relative path")
        parts = text.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise AuditFail(f"path {str(value)[:200]!r} traverses or is not normalised")
        cand = root_abs.joinpath(*parts)
    try:
        rel = cand.relative_to(root_abs)
    except ValueError:
        raise AuditFail(f"path {str(value)[:200]!r} escapes the audit root") from None
    if not rel.parts:
        raise AuditFail("path names the audit root itself")
    cur = root_abs
    for part in rel.parts:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            raise AuditFail(f"{rel.as_posix()} does not exist") from None
        if _is_link(st):
            raise AuditFail(f"{rel.as_posix()} traverses a symlink/junction")
    st = os.lstat(cand)
    if kind == "file" and not stat.S_ISREG(st.st_mode):
        raise AuditFail(f"{rel.as_posix()} is not a regular file")
    if kind == "dir" and not stat.S_ISDIR(st.st_mode):
        raise AuditFail(f"{rel.as_posix()} is not a directory")
    real = os.path.normcase(os.path.realpath(cand))
    if real != os.path.normcase(os.path.join(os.path.realpath(root_abs), *rel.parts)):
        raise AuditFail(f"{rel.as_posix()} does not resolve to itself")
    return cand


class _Reads:
    """Every input byte the audit reads, with identity; re-hashed for the after-closure."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.items: list[dict] = []

    def read(self, path: Path, role: str, cap: int) -> bytes:
        size = os.lstat(path).st_size
        if size > cap:
            raise AuditUnsupported(f"{role} is {size} bytes, above the {cap} byte audit cap; not authenticated")
        with open(path, "rb") as stream:
            data = stream.read(cap + 1)
        if len(data) != size:
            raise AuditFail(f"{role} changed size while being read")
        rel = _rel(self.root, path)
        for item in self.items:
            if item["path"] == rel and item["sha256"] != sha256_bytes(data):
                raise AuditFail(f"{rel} changed between reads")
        self.items.append({"role": role, "path": rel, "sha256": sha256_bytes(data), "bytes": len(data)})
        return data


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return None if not st.st_ino else (int(st.st_dev), int(st.st_ino))


# --------------------------------------------------------------------------
# pure accounting

def account(references: Sequence[Mapping], generated_artwork_raw_bytes: int | None) -> dict:
    """``references``: [{role, path, bytes, file_id}] with duplicates retained.
    unique_file_object_bytes is diagnostic only (null if any identity is unknown);
    the portable budget never deducts hard links or equal content."""
    reference_bytes = sum(int(r["bytes"]) for r in references)
    by_path: dict[str, int] = {}
    for r in references:
        by_path.setdefault(os.path.normcase(str(r["path"])), int(r["bytes"]))
    ids = [r.get("file_id") for r in references]
    if references and all(i is not None for i in ids):
        by_obj: dict[tuple, int] = {}
        for r in references:
            by_obj.setdefault(tuple(r["file_id"]), int(r["bytes"]))
        unique_obj = sum(by_obj.values())
    else:
        unique_obj = None
    unique_path = sum(by_path.values())
    portable = None if generated_artwork_raw_bytes is None else unique_path + int(generated_artwork_raw_bytes)
    return {"references": [{"role": r["role"], "path": r["path"], "bytes": int(r["bytes"])} for r in references],
            "reference_bytes": reference_bytes, "unique_path_bytes": unique_path,
            "unique_file_object_bytes": unique_obj, "generated_artwork_raw_bytes": generated_artwork_raw_bytes,
            "portable_budget_bytes": portable, "portable_lower_bound_bytes": unique_path + int(generated_artwork_raw_bytes or 0),
            "allocated_disk_bytes": None, "limit_bytes": LIMIT_BYTES,
            "note": "encoded physical bytes from stat/read sizes; stat size is not disk allocation"}


def _ref(role: str, root: Path, path: Path, size: int) -> dict:
    return {"role": role, "path": _rel(root, path), "bytes": size, "file_id": _file_identity(path)}


# --------------------------------------------------------------------------
# strict pure GLB parser (dense embedded triangle subset)

_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_INDEX_TYPES = (5121, 5123, 5125)
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def image_dims(data: bytes) -> tuple[int, int] | None:
    """Width/height from a PNG IHDR or JPEG SOF header without decoding; None if unknown."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 4 <= len(data):
            if data[i] != 0xFF:
                return None
            marker = data[i + 1]
            if marker == 0xFF:
                i += 1
                continue
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                i += 2
                continue
            seg = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in _SOF and i + 9 <= len(data):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return int(w), int(h)
            if seg < 2:
                return None
            i += 2 + seg
    return None


def _node_matrix(node: Mapping, where: str) -> np.ndarray:
    def floats(key, n):
        v = node[key]
        if not isinstance(v, list) or len(v) != n or not all(type(x) in (int, float) for x in v):
            raise AuditFail(f"{where}.{key} must be {n} numbers")
        a = np.asarray(v, dtype=np.float64)
        if not np.isfinite(a).all():
            raise AuditFail(f"{where}.{key} is non-finite")
        return a
    if "matrix" in node:
        if any(k in node for k in ("rotation", "scale", "translation")):
            raise AuditFail(f"{where} has both matrix and TRS")
        return floats("matrix", 16).reshape(4, 4).T
    M = np.eye(4)
    if "rotation" in node:
        x, y, z, w = floats("rotation", 4)
        M[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                              [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                              [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    if "scale" in node:
        M[:3, :3] = M[:3, :3] @ np.diag(floats("scale", 3))
    if "translation" in node:
        M[:3, 3] = floats("translation", 3)
    return M


def parse_glb(raw: bytes) -> dict:
    """Authenticate the container and count triangles. Raises AuditFail for invalid
    content, AuditUnsupported when geometry uses an unsupported feature. Non-geometry
    unsupported items (e.g. external image URIs, never opened) are listed in
    ``unsupported`` and keep the closure UNVERIFIED."""
    out: dict[str, Any] = {"bytes": len(raw), "unsupported": []}
    if len(raw) < 20:
        raise AuditFail("GLB shorter than its header")
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise AuditFail("not a glTF 2.0 binary container of the stated length")
    chunks, offset = [], 12
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise AuditFail("truncated GLB chunk header")
        clen, ctype = struct.unpack_from("<I4s", raw, offset)
        if clen % 4 or offset + 8 + clen > len(raw):
            raise AuditFail("GLB chunk length misaligned or out of bounds")
        chunks.append((ctype, offset + 8, clen))
        offset += 8 + clen
    if not chunks or chunks[0][0] != b"JSON":
        raise AuditFail("first GLB chunk must be JSON")
    if sum(1 for c in chunks if c[0] == b"JSON") != 1 or sum(1 for c in chunks if c[0] == b"BIN\x00") > 1:
        raise AuditFail("GLB must hold one JSON chunk and at most one BIN chunk")
    if any(c[0] == b"BIN\x00" for c in chunks[2:]):
        raise AuditFail("GLB BIN chunk must directly follow JSON")
    out["chunks"] = [{"type": c[0].decode("latin-1"), "bytes": c[2]} for c in chunks]
    gltf = strict_json(raw[chunks[0][1]:chunks[0][1] + chunks[0][2]], "GLB JSON")
    if not isinstance(gltf, dict):
        raise AuditFail("GLB JSON is not an object")
    binary = next((raw[s:s + n] for t, s, n in chunks if t == b"BIN\x00"), None)
    if not isinstance((gltf.get("asset") or {}).get("version"), str) or not gltf["asset"]["version"].startswith("2."):
        raise AuditFail("glTF asset.version must be 2.x")
    out["extensions_used"] = list(gltf.get("extensionsUsed") or [])
    required = list(gltf.get("extensionsRequired") or [])
    out["extensions_required"] = required
    if required:
        raise AuditUnsupported(f"required glTF extensions are outside the supported subset: {required[:8]}")

    def lst(key):
        v = gltf.get(key, [])
        if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
            raise AuditFail(f"glTF {key} must be a list of objects")
        return v
    buffers, views, accessors = lst("buffers"), lst("bufferViews"), lst("accessors")
    images, meshes, nodes, scenes = lst("images"), lst("meshes"), lst("nodes"), lst("scenes")
    # Semantics outside the static dense subset: counted where possible, but never verified
    # and never handed to the producer decoder.
    geo_unsupported: list[str] = []

    def ext(obj: Mapping, where: str) -> None:
        e = obj.get("extensions")
        if e is None:
            return
        if not isinstance(e, dict):
            raise AuditFail(f"{where}.extensions must be an object")
        if e:
            geo_unsupported.append(f"{where} extensions {sorted(e)[:6]}")
    ext(gltf, "root")
    for key in ("skins", "animations"):
        if gltf.get(key):
            geo_unsupported.append(f"glTF {key} present")
    for n, node in enumerate(nodes):
        ext(node, f"nodes[{n}]")
        for key in ("skin", "weights"):
            if key in node:
                geo_unsupported.append(f"nodes[{n}].{key} present")
    for m, mesh in enumerate(meshes):
        ext(mesh, f"meshes[{m}]")
        if "weights" in mesh:
            geo_unsupported.append(f"meshes[{m}].weights present")
        for p, prim in enumerate(mesh.get("primitives") or []):
            if isinstance(prim, dict):
                ext(prim, f"meshes[{m}].primitives[{p}]")
                if "targets" in prim:
                    geo_unsupported.append(f"meshes[{m}].primitives[{p}] morph targets present")
    # buffers: only the GLB-stored buffer 0 is supported as payload data
    buffer_data: dict[int, bytes | None] = {}
    for i, b in enumerate(buffers):
        blen = _int(b.get("byteLength"), f"buffers[{i}].byteLength", 1)
        if "uri" in b:
            uri = b["uri"]
            kind = "data-URI" if isinstance(uri, str) and uri.startswith("data:") else "external URI (not opened)"
            out["unsupported"].append(f"buffers[{i}] uses a {kind}")
            buffer_data[i] = None
        elif i == 0:
            if binary is None or blen > len(binary) or len(binary) - blen > 3:
                raise AuditFail("buffer 0 byteLength does not match the BIN chunk")
            buffer_data[i] = binary[:blen]
        else:
            raise AuditFail(f"buffers[{i}] has no uri but only buffer 0 may use the BIN chunk")
    for i, v in enumerate(views):
        b = v.get("buffer")
        if type(b) is not int or not 0 <= b < len(buffers):
            raise AuditFail(f"bufferViews[{i}].buffer out of range")
        off = _int(v.get("byteOffset", 0), f"bufferViews[{i}].byteOffset")
        n = _int(v.get("byteLength"), f"bufferViews[{i}].byteLength", 1)
        if off + n > buffers[b]["byteLength"]:
            raise AuditFail(f"bufferViews[{i}] exceeds its buffer")
        if "byteStride" in v:
            s = _int(v["byteStride"], f"bufferViews[{i}].byteStride", 4)
            if s > 252 or s % 4:
                raise AuditFail(f"bufferViews[{i}].byteStride invalid")

    def view_bytes(i: int, where: str) -> bytes:
        if type(i) is not int or not 0 <= i < len(views):
            raise AuditFail(f"{where} bufferView out of range")
        v = views[i]
        data = buffer_data.get(v["buffer"])
        if data is None:
            raise AuditUnsupported(f"{where} uses a non-embedded buffer")
        off = v.get("byteOffset", 0)
        return data[off:off + v["byteLength"]]

    acc_plan: dict[int, tuple] = {}
    for i, a in enumerate(accessors):
        where = f"accessors[{i}]"
        if a.get("componentType") not in _COMPONENTS or a.get("type") not in _WIDTHS:
            raise AuditFail(f"{where} componentType/type invalid")
        count = _int(a.get("count"), f"{where}.count", 1)
        if "sparse" in a:
            acc_plan[i] = ("unsupported", f"{where} is sparse")
            continue
        if "bufferView" not in a:
            acc_plan[i] = ("unsupported", f"{where} has no bufferView (implicit zeros)")
            continue
        bv = a["bufferView"]
        if type(bv) is not int or not 0 <= bv < len(views):
            raise AuditFail(f"{where}.bufferView out of range")
        dtype = np.dtype(_COMPONENTS[a["componentType"]]).newbyteorder("<")
        width = _WIDTHS[a["type"]]
        off = _int(a.get("byteOffset", 0), f"{where}.byteOffset")
        if off % dtype.itemsize or views[bv].get("byteOffset", 0) % dtype.itemsize:
            raise AuditFail(f"{where} is misaligned")
        elem = dtype.itemsize * width
        stride = views[bv].get("byteStride", 0) or elem
        if stride < elem or off + stride * (count - 1) + elem > views[bv]["byteLength"]:
            raise AuditFail(f"{where} exceeds its bufferView")
        acc_plan[i] = ("ok", bv, dtype, width, count, off, stride)

    def accessor(index: Any, where: str) -> np.ndarray:
        if type(index) is not int or index not in acc_plan:
            raise AuditFail(f"{where} accessor out of range")
        plan = acc_plan[index]
        if plan[0] != "ok":
            raise AuditUnsupported(plan[1])
        _, bv, dtype, width, count, off, stride = plan
        data = view_bytes(bv, where)
        rows = np.lib.stride_tricks.as_strided(np.frombuffer(data, np.uint8, offset=off),
                                               shape=(count, dtype.itemsize * width), strides=(stride, 1))
        return np.ascontiguousarray(rows).view(dtype).reshape(count, width)

    # images: embedded bytes are already inside the GLB bytes (never double counted)
    out["embedded_images"] = []
    for i, img in enumerate(images):
        entry: dict[str, Any] = {"index": i, "mime": img.get("mimeType")}
        if "uri" in img:
            uri = img["uri"]
            if isinstance(uri, str) and uri.startswith("data:"):
                head, _, payload = uri.partition(",")
                if not head.endswith(";base64"):
                    raise AuditFail(f"images[{i}] data URI is not base64")
                try:
                    data = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    raise AuditFail(f"images[{i}] data URI is not valid base64") from None
                entry["source"] = "data_uri_in_json_chunk"
            else:
                out["unsupported"].append(f"images[{i}] references an external URI (not opened)")
                entry["source"] = "external_uri_not_opened"
                entry["width"] = entry["height"] = None
                out["embedded_images"].append(entry)
                continue
        elif "bufferView" in img:
            try:
                data = view_bytes(img["bufferView"], f"images[{i}]")
            except AuditUnsupported as exc:
                out["unsupported"].append(str(exc))
                entry.update(source="non_embedded_buffer", width=None, height=None)
                out["embedded_images"].append(entry)
                continue
            entry["source"] = "buffer_view_in_bin_chunk"
        else:
            raise AuditFail(f"images[{i}] has neither uri nor bufferView")
        dims = image_dims(data)
        entry.update(encoded_bytes=len(data), width=dims[0] if dims else None, height=dims[1] if dims else None)
        out["embedded_images"].append(entry)
    # meshes: per-primitive triangle and degenerate counts (each mesh decoded once)
    mesh_tris: list[int] = []
    mesh_degenerate: list[int] = []
    mesh_prims: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for m, mesh in enumerate(meshes):
        prims = mesh.get("primitives")
        if not isinstance(prims, list) or not prims:
            raise AuditFail(f"meshes[{m}] has no primitives")
        tris = degen = 0
        decoded = []
        for p, prim in enumerate(prims):
            where = f"meshes[{m}].primitives[{p}]"
            if not isinstance(prim, dict):
                raise AuditFail(f"{where} is not an object")
            if prim.get("mode", 4) != 4:
                raise AuditUnsupported(f"{where} is not a triangle-list primitive")
            attrs = prim.get("attributes")
            if not isinstance(attrs, dict) or "POSITION" not in attrs:
                raise AuditFail(f"{where} has no POSITION")
            pacc = attrs["POSITION"]
            if type(pacc) is int and pacc in acc_plan and acc_plan[pacc][0] == "ok" and \
                    (accessors[pacc]["componentType"] != 5126 or accessors[pacc]["type"] != "VEC3"):
                raise AuditUnsupported(f"{where} POSITION is not float32 VEC3")
            pos = accessor(pacc, f"{where}.POSITION").astype(np.float64)
            if not np.isfinite(pos).all():
                raise AuditFail(f"{where} has non-finite vertices")
            if "indices" in prim:
                iacc = prim["indices"]
                if type(iacc) is int and iacc in acc_plan and acc_plan[iacc][0] == "ok" and \
                        (accessors[iacc]["componentType"] not in _INDEX_TYPES or accessors[iacc]["type"] != "SCALAR"):
                    raise AuditFail(f"{where} indices must be unsigned SCALAR")
                idx = accessor(iacc, f"{where}.indices").reshape(-1).astype(np.int64)
            else:
                idx = np.arange(len(pos), dtype=np.int64)
            if len(idx) % 3:
                raise AuditFail(f"{where} index count is not a multiple of 3")
            if len(idx) and (idx.min() < 0 or idx.max() >= len(pos)):
                raise AuditFail(f"{where} index out of range")
            tri = idx.reshape(-1, 3)
            tris += len(tri)
            degen += int(np.count_nonzero((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])))
            decoded.append((pos, tri))
        mesh_tris.append(tris)
        mesh_degenerate.append(degen)
        mesh_prims.append(decoded)
    # nodes: strict trees (no cycles, single parent), valid mesh refs, finite transforms
    parent: dict[int, int] = {}
    local = []
    for n, node in enumerate(nodes):
        if "mesh" in node and (type(node["mesh"]) is not int or not 0 <= node["mesh"] < len(meshes)):
            raise AuditFail(f"nodes[{n}].mesh out of range")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise AuditFail(f"nodes[{n}].children must be a list")
        for c in children:
            if type(c) is not int or not 0 <= c < len(nodes) or c == n:
                raise AuditFail(f"nodes[{n}].children has an invalid index")
            if c in parent:
                raise AuditFail(f"nodes[{c}] has more than one parent")
            parent[c] = n
        local.append(_node_matrix(node, f"nodes[{n}]"))
    for n in range(len(nodes)):
        seen, cur = set(), n
        while cur in parent:
            if cur in seen:
                raise AuditFail("node hierarchy contains a cycle")
            seen.add(cur)
            cur = parent[cur]

    def walk(roots: Sequence[int], where: str) -> tuple[int, int, int, bool]:
        """(triangles, degenerate, instances, finite) over a traversal from ``roots``."""
        tris = degen = inst = 0
        finite = True
        stack = []
        for r in roots:
            if type(r) is not int or not 0 <= r < len(nodes):
                raise AuditFail(f"{where} references an invalid node")
            stack.append((r, np.eye(4)))
        while stack:
            index, world_parent = stack.pop()
            world = world_parent @ local[index]
            stack.extend((c, world) for c in nodes[index].get("children", []))
            if "mesh" not in nodes[index]:
                continue
            inst += 1
            if inst > MAX_INSTANCES:
                raise AuditUnsupported("more mesh instances than the audit bound")
            m = nodes[index]["mesh"]
            tris += mesh_tris[m]
            degen += mesh_degenerate[m]
            for pos, _tri in mesh_prims[m]:
                if not np.isfinite(pos @ world[:3, :3].T + world[:3, 3]).all():
                    finite = False
        return tris, degen, inst, finite
    # glTF scene semantics
    if scenes:
        scene_index = gltf.get("scene", 0)
        if type(scene_index) is not int or not 0 <= scene_index < len(scenes):
            raise AuditFail("glTF scene index out of range")
        scene_roots = scenes[scene_index].get("nodes", [])
        if not isinstance(scene_roots, list):
            raise AuditFail("scene nodes must be a list")
        for r in scene_roots:
            if r in parent:
                raise AuditFail("a scene root is also a child node")
        scene_rule = "scenes[scene].nodes"
    else:
        scene_roots = [n for n in range(len(nodes)) if n not in parent]
        scene_rule = "no scenes: all parentless nodes (implementation-defined)"
    s_tris, _s_deg, s_inst, s_finite = walk(scene_roots, "scene")
    # current producer traversal (workbench_neural.glb_mesh): scenes default [{}], nodes default all nodes
    p_scene = (gltf.get("scenes") or [{}])
    p_index = gltf.get("scene", 0)
    if type(p_index) is not int or not 0 <= p_index < len(p_scene):
        raise AuditFail("producer traversal scene index out of range")
    p_roots = p_scene[p_index].get("nodes", list(range(len(nodes))))
    if not isinstance(p_roots, list):
        raise AuditFail("producer traversal scene nodes must be a list")
    p_tris, p_deg, p_inst, p_finite = walk(p_roots, "producer traversal")
    if not (s_finite and p_finite):
        raise AuditFail("transformed vertices are non-finite")
    if p_tris == 0:
        raise AuditFail("GLB contains no triangle geometry in the producer traversal")
    raw_indexed = sum(mesh_tris)
    out.update({
        "meshes": len(meshes), "nodes": len(nodes), "accessors": len(accessors), "images": len(images),
        "raw_indexed_triangles": raw_indexed, "scene_rule": scene_rule,
        "scene_instanced_triangles": s_tris, "scene_mesh_instances": s_inst,
        "producer_traversal_triangles": p_tris, "producer_mesh_instances": p_inst,
        "producer_degenerate_triangles": p_deg, "producer_filtered_triangles": p_tris - p_deg,
        "target_triangles": max(raw_indexed, s_tris, p_tris),
        "target_triangles_rule": "max(raw indexed, scene-instanced, producer traversal); degenerate faces are "
                                 "never deducted from the budget",
        "geometry_unsupported": geo_unsupported,
        "target_triangles_is_lower_bound": bool(geo_unsupported),
    })
    return out


# --------------------------------------------------------------------------
# artwork

def regenerate_builtin(kind: str, edge: int, expected_sha256: str) -> dict:
    from bench import workbench_geometry as wg
    if kind not in BUILTIN_KINDS:
        raise AuditUnsupported(f"artwork kind {kind!r} has no regeneration locator")
    if type(edge) is not int or not 8 <= edge <= LIMIT_TEXTURE_EDGE:
        raise AuditFail(f"builtin artwork edge {edge!r} is outside 8..{LIMIT_TEXTURE_EDGE}")
    art = wg.builtin_artwork(kind, edge)
    sha = wg.array_sha256(art)
    if art.dtype != np.uint8 or art.shape != (edge, edge, 3) or art.nbytes != edge * edge * 3:
        raise AuditFail("regenerated builtin artwork has an unexpected shape/dtype")
    if sha != expected_sha256:
        raise AuditFail("regenerated builtin artwork hash differs from the recorded artwork hash")
    return {"kind": kind, "edge_px": edge, "generation": BUILTIN_GENERATION, "sha256": sha,
            "shape": list(art.shape), "dtype": str(art.dtype), "nbytes": int(art.nbytes)}


def check_bundle(doc: Any) -> dict:
    """Pure mirror of ``MetricBundle.from_file`` field checks on the already-read JSON
    (no loader, no GLB read here). A prepared artwork control needs a labelled,
    proper, finite seed pose; evaluator-labelled seeds are rejected."""
    from bench import workbench_contract as wc
    from bench import workbench_neural as wn
    if not isinstance(doc, dict) or set(doc) != set(BUNDLE_KEYS) or doc.get("schema") != BUNDLE_SCHEMA:
        raise AuditFail("metric bundle keys/schema differ from workbench_metric_bundle_v1")
    try:
        wc._str(doc["family"], "family", max_len=64)
        wc._str(doc["glb_path"], "glb_path", max_len=1024)
        wc._ident(doc["seed_frame_id"], "seed_frame_id")
        if not isinstance(doc["seed_label"], str):
            raise wc.ContractError("seed_label must be a string")
        wc._str(doc["seed_label"], "seed_label", max_len=1024)
        pose = wc._pose(doc["seed_pose_m"], "seed_pose_m") if doc["seed_pose_m"] is not None else None
    except (wc.ContractError, TypeError, ValueError) as exc:
        raise AuditFail(f"metric bundle: {exc}") from None
    if not _is_sha(doc["glb_sha256"]):
        raise AuditFail("metric bundle glb_sha256 malformed")
    if type(doc["obj_id"]) is not int or not 0 <= doc["obj_id"] <= 2**31 - 1:
        raise AuditFail("metric bundle obj_id must be an integer in 0..2^31-1")
    if doc["principal_point_origin"] not in ("integer_centers", "corner"):
        raise AuditFail("metric bundle must document the principal point origin (integer_centers|corner)")
    if doc["metric_provenance"] not in ("measured_metric_cad", "dataset_metric_cad"):
        raise AuditFail("metric bundle provenance is not metric CAD (relative reconstructions are not bundles)")
    if pose is None:
        raise AuditFail("prepared artwork control needs a labelled seed pose; seed_pose_m is null")
    if not doc["seed_label"].startswith(tuple(wn.SEED_PREFIXES)):
        raise AuditFail("seed must be labelled controlled_init:/onboarding:, never evaluator data")
    return doc


def check_artwork_record(record: Any, *, bundle: Mapping, producer_mesh_faces: int | None) -> dict:
    """Self-hash, bundle bindings, marker/face bounds, producer face count and builtin regeneration."""
    from bench import workbench_contract as wc
    if not isinstance(record, dict) or record.get("schema") != ARTWORK_SCHEMA:
        raise AuditFail("artwork record schema differs")
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    if not _is_sha(record.get("record_sha256")) or wc.canonical_sha256(body) != record["record_sha256"]:
        raise AuditFail("artwork record_sha256 is not the canonical hash of the record")
    for key, want in (("glb_sha256", bundle.get("glb_sha256")), ("anchor_frame_id", bundle.get("seed_frame_id")),
                      ("principal_point_origin", bundle.get("principal_point_origin")),
                      ("seed_label", bundle.get("seed_label")), ("units", "metres")):
        if record.get(key) != want:
            raise AuditFail(f"artwork record {key} differs from the bundle")
    art = record.get("artwork")
    if not isinstance(art, dict) or set(art) != {"kind", "edge_px", "sha256", "generation"} or not _is_sha(art["sha256"]):
        raise AuditFail("artwork record artwork block is malformed")
    try:
        marker = wc.Marker.from_dict(record.get("marker"))
    except (wc.ContractError, TypeError, ValueError) as exc:
        raise AuditFail(f"artwork marker invalid: {exc}") from None
    if marker.artwork_sha256 != art["sha256"]:
        raise AuditFail("marker artwork hash differs from the artwork hash")
    mesh_faces = _int(record.get("mesh_faces"), "artwork mesh_faces", 1)
    patch_faces = _int(record.get("patch_faces"), "artwork patch_faces", 0)
    if patch_faces > mesh_faces or marker.face_id >= mesh_faces:
        raise AuditFail("artwork marker/patch faces exceed the mesh")
    if producer_mesh_faces is not None and mesh_faces != producer_mesh_faces:
        raise AuditFail(f"artwork mesh_faces {mesh_faces} differs from the producer-decoded {producer_mesh_faces}")
    info = {"record_sha256": record["record_sha256"], "record_sha256_valid": True, "mesh_faces": mesh_faces,
            "patch_faces": patch_faces, "marker_face_id": marker.face_id, "recorded": dict(art)}
    if isinstance(art["edge_px"], int) and art["edge_px"] > LIMIT_TEXTURE_EDGE:
        info["edge_over_limit"] = True
    if art["generation"] != BUILTIN_GENERATION or art["kind"] not in BUILTIN_KINDS:
        raise AuditUnsupported("artwork is not fixed_builtin_v1 checkerboard/directional; no locator to authenticate it")
    info["regenerated"] = regenerate_builtin(art["kind"], art["edge_px"], art["sha256"])
    return info


# --------------------------------------------------------------------------
# guards (restored on every path)

class _ImportBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, attempts: list):
        self.attempts = attempts

    def find_spec(self, name, path=None, target=None):
        if any(name == m or name.startswith(m + ".") for m in FORBIDDEN_IMPORTS):
            self.attempts.append(f"import {name}")
            raise AuditViolation(f"forbidden import attempted: {name}")
        return None


@contextmanager
def audit_guards() -> Iterator[list]:
    from bench import workbench_geometry as wg, workbench_neural as wn, workbench_telemetry as wt
    attempts: list[str] = []
    saved: list[tuple[Any, str, Any]] = []
    blocker = _ImportBlocker(attempts)

    def deny(what: str):
        def _deny(*_a, **_k):
            attempts.append(what)
            raise AuditViolation(what)
        return _deny

    def patch(owner, name, value):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    try:
        patch(wt, "gpu_stage", deny("GPU stage entered"))
        patch(wn, "GoTrackMetricBackend", deny("neural backend construction attempted"))
        patch(wn, "NeuralSession", deny("NeuralSession construction attempted"))
        patch(wn, "NeuralArtwork", deny("NeuralArtwork construction attempted"))
        patch(wn, "segment_prompted", deny("segmentation attempted"))
        patch(wg, "render_marker", deny("render attempted"))
        patch(wg, "rasterize", deny("rasterization attempted"))
        sys.meta_path.insert(0, blocker)
        yield attempts
    finally:
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


# --------------------------------------------------------------------------
# rows

def _new_row(kind: str) -> dict:
    return {"kind": kind, "closure_status": None, "closure_reasons": {"fail": [], "unverified": []},
            "bytes_target": None, "triangles_target": None, "artwork_target": None, "mobile_verified": False,
            "available": True, "accounting": None,
            "measured": {"portable_budget_bytes": None, "portable_lower_bound_bytes": None, "target_triangles": None,
                         "texture_edges_px": None, "artwork_verified": False}}


def _stage(row: dict, fn) -> Any:
    try:
        return fn()
    except AuditFail as exc:
        row["closure_reasons"]["fail"].append(str(exc)[:600])
    except AuditUnsupported as exc:
        row["closure_reasons"]["unverified"].append(str(exc)[:600])
    return None


def finalize_row(row: dict) -> dict:
    """Closure status and targets from the measured values; never an optimistic PASS."""
    reasons = row["closure_reasons"]
    if not row["available"]:
        # An unavailable package carries no payload; an invalid unavailable manifest is still a FAIL.
        row["closure_status"] = "FAIL" if reasons["fail"] else "UNAVAILABLE"
        for t in TARGETS:
            row[t] = None
        row["mobile_verified"] = False
        return row
    status = "FAIL" if reasons["fail"] else "UNVERIFIED" if reasons["unverified"] else "VERIFIED"
    row["closure_status"] = status
    m = row["measured"]
    verified = status == "VERIFIED"

    def target(value, lower, limit):
        if lower is not None and lower > limit:
            return "FAIL"
        if value is not None and value > limit:
            return "FAIL"
        return "PASS" if verified and value is not None else None
    row["bytes_target"] = target(m["portable_budget_bytes"], m["portable_lower_bound_bytes"], LIMIT_BYTES)
    row["triangles_target"] = target(m["target_triangles"], None, LIMIT_TRIANGLES)
    edges = m["texture_edges_px"]
    if edges is not None and any(e is not None and e > LIMIT_TEXTURE_EDGE for e in edges):
        row["artwork_target"] = "FAIL"
    elif verified and m["artwork_verified"] and edges is not None and all(e is not None for e in edges):
        row["artwork_target"] = "PASS"
    else:
        row["artwork_target"] = None
    row["mobile_verified"] = False
    return row


def _model_pins() -> dict:
    return {role: {"sha256": sha, "verification": "not_read", "excluded_from_surface_bytes": True}
            for role, sha in (("primary_head", HEAD_SHA256), ("frozen_base", BASE_SHA256))}


def audit_prepared_control(bundle_path: str | Path, artwork_record_path: str | Path, *, root: Path) -> dict:
    """Metric bundle + the GLB it names + the explicit artwork record. Not a SurfacePackage."""
    root = Path(root)
    row = _new_row("prepared_metric_cad_control")
    reads = _Reads(root)
    st: dict[str, Any] = {}
    with audit_guards() as attempts:
        def bundle_stage():
            p = resolve_input(root, bundle_path)
            raw = reads.read(p, "metric_bundle", MAX_JSON_BYTES)
            doc = check_bundle(strict_json(raw, "metric bundle"))
            st.update(bundle=doc, bundle_path=p, bundle_raw=raw)
        _stage(row, bundle_stage)

        def glb_stage():
            p = resolve_input(root, st["bundle"]["glb_path"])
            st["glb_path"] = p
            st["glb_size"] = os.lstat(p).st_size
            raw = reads.read(p, "glb", MAX_GLB_BYTES)
            if sha256_bytes(raw) != st["bundle"]["glb_sha256"]:
                raise AuditFail("GLB physical hash differs from the bundle pin")
            st["glb_raw"] = raw
        if "bundle" in st:
            _stage(row, glb_stage)

        def artwork_read():
            p = resolve_input(root, artwork_record_path)
            raw = reads.read(p, "artwork_record", MAX_JSON_BYTES)
            st.update(artwork_path=p, artwork_raw=raw, artwork=strict_json(raw, "artwork record"))
        _stage(row, artwork_read)

        def geometry_stage():
            g = parse_glb(st["glb_raw"])
            st["geometry"] = g
            for item in g["unsupported"] + g["geometry_unsupported"]:
                row["closure_reasons"]["unverified"].append(item[:600])
            if g["geometry_unsupported"]:
                g["producer_decoded_triangles"] = None      # never decode unsupported semantics with the producer
                return
            from bench import workbench_neural as wn
            try:
                _v, faces = wn.glb_mesh(st["glb_path"], max_faces=PRODUCER_MAX_FACES)
            except Exception as exc:  # noqa: BLE001 - the producer decoder must agree with the audit parser
                raise AuditFail(f"producer GLB decode failed: {type(exc).__name__}: {str(exc)[:300]}") from None
            g["producer_decoded_triangles"] = int(len(faces))
            if len(faces) != g["producer_filtered_triangles"]:
                raise AuditFail(f"producer decoded {len(faces)} triangles; audit parser filtered "
                                f"{g['producer_filtered_triangles']}")
        if "glb_raw" in st:
            _stage(row, geometry_stage)

        def artwork_stage():
            producer = (st.get("geometry") or {}).get("producer_decoded_triangles")
            info = check_artwork_record(st["artwork"], bundle=st["bundle"], producer_mesh_faces=producer)
            st["artwork_info"] = info
        if "artwork" in st and "bundle" in st:
            _stage(row, artwork_stage)
    # accounting (duplicates retained); generated artwork only when regenerated
    refs = []
    if "bundle_path" in st:
        refs.append(_ref("metric_bundle", root, st["bundle_path"], len(st["bundle_raw"])))
    if "glb_path" in st:
        refs.append(_ref("glb", root, st["glb_path"], st["glb_size"]))
    if "artwork_path" in st:
        refs.append(_ref("artwork_record", root, st["artwork_path"], len(st["artwork_raw"])))
    regen = (st.get("artwork_info") or {}).get("regenerated")
    complete = all(k in st for k in ("bundle_path", "glb_path", "artwork_path"))
    acc = account(refs, regen["nbytes"] if regen else None)
    if not complete:
        acc["portable_budget_bytes"] = None
    row["accounting"] = acc
    g = st.get("geometry") or {}
    rec_art = (st.get("artwork") or {}).get("artwork") if isinstance(st.get("artwork"), dict) else None
    rec_edge = rec_art.get("edge_px") if isinstance(rec_art, dict) and type(rec_art.get("edge_px")) is int else None
    edges = None
    if regen:
        edges = [regen["edge_px"]]          # only the regenerated overlay artwork gates artwork_target
    elif rec_edge is not None and rec_edge > LIMIT_TEXTURE_EDGE:
        edges = [rec_edge]
    if "geometry" in st:
        # parse_glb returned, so its image list is complete: zeros here are measured.
        images = g.get("embedded_images") or []
        dims = [max(i["width"], i["height"]) for i in images
                if i.get("width") is not None and i.get("height") is not None]
        row["embedded_texture_diagnostic"] = {
            "inspection_status": "inspected", "images": len(images), "max_edge_px": max(dims) if dims else None,
            "over_1024": sum(1 for d in dims if d > LIMIT_TEXTURE_EDGE), "unknown_dimensions": len(images) - len(dims),
            "gates_artwork_target": False,
            "note": "embedded CAD material images are diagnostic only; their bytes are counted once inside the GLB"}
    else:
        # The GLB was not read or parse_glb stopped (invalid/unsupported) before images were inspected:
        # counts are unknown, not measured absence.
        row["embedded_texture_diagnostic"] = {
            "inspection_status": "not_inspected", "images": None, "max_edge_px": None, "over_1024": None,
            "unknown_dimensions": None, "gates_artwork_target": False,
            "note": "GLB images were not inspected (GLB unread, invalid, or an unsupported feature stopped parsing); "
                    "null means unknown, not zero"}
    row["measured"].update(portable_budget_bytes=acc["portable_budget_bytes"],
                           portable_lower_bound_bytes=acc["portable_lower_bound_bytes"],
                           target_triangles=g.get("target_triangles"), texture_edges_px=edges,
                           artwork_verified=bool(regen))
    row["geometry"] = g or None
    row["artwork"] = st.get("artwork_info")
    row["bundle"] = None if "bundle" not in st else {
        "path": _rel(root, st["bundle_path"]), "sha256": sha256_bytes(st["bundle_raw"]),
        "bytes": len(st["bundle_raw"]), "glb_path": st["bundle"]["glb_path"], "glb_sha256": st["bundle"]["glb_sha256"],
        "seed_frame_id": st["bundle"]["seed_frame_id"], "principal_point_origin": st["bundle"]["principal_point_origin"],
        "seed_label": st["bundle"]["seed_label"], "metric_provenance": st["bundle"]["metric_provenance"],
        "family": st["bundle"]["family"], "obj_id": st["bundle"]["obj_id"],
        "seed_pose_m": "validated as a finite proper 4x4 pose; value not reported or used"}
    row["inputs_read"] = reads.items
    row["guard_attempts"] = list(attempts)
    row["model_pins"] = _model_pins()
    row["not_surface_package"] = "prepared metric CAD control (MetricBundle + GLB + neural-artwork.json); no conversion"
    return finalize_row(row)


def crosscheck_inventory(row: dict, inv: Mapping) -> list[str]:
    """Inventory claims against the audited physical files; returns failure reasons."""
    out = []
    b, g = row.get("bundle") or {}, row.get("geometry") or {}
    refs = {r["role"]: r for r in (row.get("accounting") or {}).get("references", [])}
    glb_bytes = (refs.get("glb") or {}).get("bytes")
    if b.get("sha256") != inv.get("bundle_sha256"):
        out.append("inventory bundle_sha256 differs from the physical bundle")
    if b.get("glb_sha256") != inv.get("glb_sha256"):
        out.append("inventory glb_sha256 differs from the bundle pin")
    try:
        if b.get("glb_path") is not None and str(inv.get("glb", "")).replace("\\", "/") != b["glb_path"].replace("\\", "/"):
            out.append("inventory glb path differs from the bundle glb_path")
    except AttributeError:
        out.append("inventory glb path malformed")
    if glb_bytes is not None and inv.get("glb_bytes") != glb_bytes:
        out.append("inventory glb_bytes differs from the physical GLB")
    if glb_bytes is not None and b.get("bytes") is not None and inv.get("bundle_plus_glb_bytes") != glb_bytes + b["bytes"]:
        out.append("inventory bundle_plus_glb_bytes differs from the physical bytes")
    if g.get("producer_decoded_triangles") is not None and inv.get("triangles") != g["producer_decoded_triangles"]:
        out.append(f"inventory triangles {inv.get('triangles')} differ from the producer-decoded "
                   f"{g['producer_decoded_triangles']}")
    rec = ((row.get("artwork") or {}).get("recorded") or {})
    if rec and inv.get("artwork_texture_edge_px") != rec.get("edge_px"):
        out.append("inventory artwork_texture_edge_px differs from the artwork record")
    return out


def _npy(data: bytes, what: str, dtype, cols: int | None) -> np.ndarray:
    """Header-validated NPY load: version 1/2, no object dtype, exact payload length before allocation."""
    f = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _fortran, dt = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _fortran, dt = np.lib.format.read_array_header_2_0(f)
        else:
            raise AuditUnsupported(f"{what}: NPY version {version} unsupported")
    except AuditUnsupported:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed headers are invalid
        raise AuditFail(f"{what}: malformed NPY header ({type(exc).__name__})") from None
    if dt.hasobject:
        raise AuditFail(f"{what}: object dtype")
    if dt != np.dtype(dtype):
        raise AuditFail(f"{what}: dtype {dt} differs from {np.dtype(dtype)}")
    if not isinstance(shape, tuple) or not 1 <= len(shape) <= 2 or any(type(s) is not int or s < 0 for s in shape):
        raise AuditFail(f"{what}: invalid shape {shape}")
    if shape[0] > MAX_NPY_ROWS or (cols is not None and (len(shape) != 2 or shape[1] != cols)):
        raise AuditFail(f"{what}: shape {shape} invalid")
    expected = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
    if len(data) - f.tell() != expected:
        raise AuditFail(f"{what}: payload length differs from its header")
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise AuditFail(f"{what}: non-finite values")
    return arr


def _fileref(value: Any, what: str) -> dict:
    from bench import workbench_contract as wc
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        raise AuditFail(f"{what} FileRef malformed")
    try:
        wc._relative_path(value["path"], f"{what}.path")
    except wc.ContractError as exc:
        raise AuditFail(str(exc)) from None
    if not _is_sha(value["sha256"]):
        raise AuditFail(f"{what}.sha256 malformed")
    _int(value["bytes"], f"{what}.bytes", 1)
    return value


def audit_surface_package(package_dir: str | Path, *, root: Path) -> dict:
    """surface.json plus every numeric FileRef and camera depth, authenticated physically."""
    from bench import workbench_contract as wc, workbench_reconstruction as wr
    root = Path(root)
    row = _new_row("surface_package")
    reads = _Reads(root)
    st: dict[str, Any] = {"refs": [], "arrays": {}}
    with audit_guards() as attempts:
        def manifest_stage():
            pdir = resolve_input(root, package_dir, kind="dir")
            st["dir"] = pdir
            entries = {}
            for e in os.scandir(pdir):
                s = os.lstat(e.path)
                entries[e.name] = "link" if _is_link(s) else "file" if stat.S_ISREG(s.st_mode) else \
                    "dir" if stat.S_ISDIR(s.st_mode) else "other"
            st["entries"] = entries
            if entries.get("surface.json") != "file":
                raise AuditFail("surface.json is missing or not a regular file")
            raw = reads.read(pdir / "surface.json", "surface_json", MAX_SURFACE_JSON_BYTES)
            st.update(surface_raw=raw, doc=strict_json(raw, "surface.json"))
            if not isinstance(st["doc"], dict) or st["doc"].get("schema") != wc.SURFACE_SCHEMA:
                raise AuditFail("surface.json schema differs")
        _stage(row, manifest_stage)
        doc = st.get("doc") or {}
        if doc and doc.get("status") == "unavailable":
            row["available"] = False
            row["unavailable_reason"] = str(doc.get("unavailable_reason"))[:600]
        elif doc and doc.get("status") != "available":
            row["closure_reasons"]["fail"].append("surface.json status is neither available nor unavailable")

        def contract_stage():
            try:
                wc.SurfacePackage.from_dict(doc)
                row["contract_valid"] = True
            except (wc.ContractError, TypeError, ValueError) as exc:
                row["contract_valid"] = False
                raise AuditFail(f"surface contract: {str(exc)[:400]}") from None
        if doc:
            _stage(row, contract_stage)

        def refs_stage():
            for key in wr.PACKAGE_FILES:
                if doc.get(key) is not None:
                    st["refs"].append((key, _fileref(doc[key], key)))
            cams = doc.get("cameras")
            if not isinstance(cams, list):
                raise AuditFail("cameras must be a list")
            for i, cam in enumerate(cams):
                if isinstance(cam, dict) and cam.get("depth") is not None:
                    st["refs"].append((f"camera[{i}].depth", _fileref(cam["depth"], f"camera[{i}].depth")))
            st["non_runtime"] = {"camera_image_hashes": sum(1 for c in cams if isinstance(c, dict) and "image_sha256" in c),
                                 "camera_mask_hashes": sum(1 for c in cams if isinstance(c, dict) and "mask_sha256" in c),
                                 "reconstruction_keys": sorted((doc.get("reconstruction") or {}).keys())[:40]
                                 if isinstance(doc.get("reconstruction"), dict) else None,
                                 "preview_keys": sorted((doc.get("preview") or {}).keys())[:40]
                                 if isinstance(doc.get("preview"), dict) else None,
                                 "note": "image/mask hashes, reconstruction history and preview provenance are not "
                                         "runtime payload and are not dereferenced; not a reproducibility audit"}
        if row["available"] and doc:
            _stage(row, refs_stage)
        st["authenticated"] = {}
        for name, ref in st["refs"]:
            def file_stage(name=name, ref=ref):
                if "/" in ref["path"]:
                    raise AuditFail(f"{name} is a nested dependency; only top-level package files are supported")
                p = st["dir"] / ref["path"]
                if st["entries"].get(ref["path"]) != "file":
                    raise AuditFail(f"{name} file {ref['path']} is missing, a link or not regular")
                size = os.lstat(p).st_size
                st.setdefault("sizes", {})[ref["path"]] = size
                if size != ref["bytes"]:
                    raise AuditFail(f"{name} size {size} differs from its FileRef {ref['bytes']}")
                data = reads.read(p, name, MAX_PACKAGE_FILE_BYTES)
                if sha256_bytes(data) != ref["sha256"]:
                    raise AuditFail(f"{name} hash differs from its FileRef")
                st["authenticated"][name] = data
                if name.startswith("camera["):
                    raise AuditUnsupported(f"{name} bytes authenticated, but no depth encoding contract exists")
            _stage(row, file_stage)

        def arrays_stage():
            a = st["authenticated"]
            for key, (_fname, dtype, cols) in wr.PACKAGE_FILES.items():
                if key in a:
                    st["arrays"][key] = _npy(a[key], key, dtype, cols)
            arr = st["arrays"]
            if "vertices" not in arr or "faces" not in arr or "face_support_views" not in arr:
                raise AuditFail("vertices/faces/face_support arrays are not all authenticated")
            v, f, s = arr["vertices"], arr["faces"], arr["face_support_views"]
            if len(v) != doc.get("vertex_count") or len(f) != doc.get("face_count"):
                raise AuditFail("array counts differ from surface.json")
            if len(f) and (f.min() < 0 or f.max() >= len(v)):
                raise AuditFail("face index out of range")
            if s.shape not in ((len(f),), (len(f), 1)) or (s.size and s.min() < 0):
                raise AuditFail("face support must hold one non-negative count per face")
            if ("descriptor_points" in arr) != ("descriptor_values" in arr):
                raise AuditFail("descriptor points/values must be present together")
            if "descriptor_points" in arr and len(arr["descriptor_points"]) != len(arr["descriptor_values"]):
                raise AuditFail("descriptor point/value row counts differ")
            from bench import workbench_geometry as wg
            try:
                wg.validate_mesh(v, f, max_faces=max(1, len(f)))
            except wg.GeometryError as exc:
                raise AuditFail(f"mesh invalid: {exc}") from None
            st["triangles"] = int(len(f))
        if row["available"] and st["authenticated"]:
            _stage(row, arrays_stage)

        def artwork_stage():
            art = doc.get("artwork")
            if art is None:
                st["no_artwork"] = True       # no artwork dependency; the texture target stays null
                return
            if not isinstance(art, dict) or not _is_sha(art.get("sha256")):
                raise AuditFail("package artwork malformed")
            if art.get("generation") != BUILTIN_GENERATION or art.get("kind") not in BUILTIN_KINDS:
                if isinstance(art.get("edge_px"), int) and art["edge_px"] > LIMIT_TEXTURE_EDGE:
                    st["edge_over_limit"] = art["edge_px"]
                raise AuditUnsupported("imported/unknown artwork has no locator; not authenticated")
            st["regenerated"] = regenerate_builtin(art["kind"], art["edge_px"], art["sha256"])
        if row["available"] and doc:
            _stage(row, artwork_stage)
        unknown = sorted(n for n, kind in (st.get("entries") or {}).items()
                         if n != "surface.json" and n not in {r["path"] for _n, r in st["refs"]})
        links = sorted(n for n, kind in (st.get("entries") or {}).items() if kind == "link")
        if row["available"] and (unknown or links):
            row["closure_reasons"]["unverified"].append(
                f"package directory holds unreferenced or linked entries {(unknown + links)[:20]}; not standalone")
    refs = []
    if "surface_raw" in st:
        refs.append(_ref("surface_json", root, st["dir"] / "surface.json", len(st["surface_raw"])))
    for name, ref in st["refs"]:
        size = (st.get("sizes") or {}).get(ref["path"])
        if size is not None:
            refs.append(_ref(name, root, st["dir"] / ref["path"], size))
    regen = st.get("regenerated")
    has_art = doc.get("artwork") is not None if doc else False
    complete = "surface_raw" in st and len(refs) == 1 + len(st["refs"])
    acc = account(refs, regen["nbytes"] if regen else (0 if not has_art else None))
    if not complete:
        acc["portable_budget_bytes"] = None
    row["accounting"] = acc if row["available"] else {**acc, "note": "unavailable package: no runtime payload"}
    edges = [regen["edge_px"]] if regen else ([st["edge_over_limit"]] if "edge_over_limit" in st else None)
    row["measured"].update(portable_budget_bytes=acc["portable_budget_bytes"],
                           portable_lower_bound_bytes=acc["portable_lower_bound_bytes"],
                           target_triangles=st.get("triangles"), texture_edges_px=edges, artwork_verified=bool(regen))
    row["package"] = None if "dir" not in st else {
        "dir": _rel(root, st["dir"]), "package_id": doc.get("package_id"), "status": doc.get("status"),
        "entries": st.get("entries"), "file_refs": [{"name": n, **r} for n, r in st["refs"]],
        "non_runtime_references": st.get("non_runtime"),
        "descriptor_bank_present": "descriptor_points" in st["arrays"],
        "runtime_usability": "unverified: a structurally VERIFIED package is byte/triangle accounting only; "
                             "no tracker admission or runtime usability is implied (with or without a descriptor bank)"}
    row["artwork"] = regen
    row["inputs_read"] = reads.items
    row["guard_attempts"] = list(attempts)
    row["model_pins"] = _model_pins()
    return finalize_row(row)


# --------------------------------------------------------------------------
# closure, output, CLI

def source_pins() -> dict:
    out = {"tools/workbench_portability_audit.py": sha256_file(Path(__file__))}
    for name in SOURCE_MODULES:
        p = ROOT / "bench" / f"{name}.py"
        out[f"bench/{name}.py"] = sha256_file(p) if p.is_file() else None
    return out


def runtime_pins() -> dict:
    return {"python_executable": sys.executable, "python_version": platform.python_version(),
            "numpy_version": np.__version__, "platform": platform.platform()}


def exit_code(rows: Sequence[Mapping]) -> int:
    if not rows:
        return 3
    if any(r["closure_status"] == "FAIL" or any(r.get(t) == "FAIL" for t in TARGETS) for r in rows):
        return 1
    if all(r["closure_status"] == "VERIFIED" and all(r.get(t) == "PASS" for t in TARGETS) for r in rows):
        return 0
    return 3


def _rehash(root: Path, items: Sequence[Mapping]) -> list[str]:
    changed = []
    for item in items:
        p = Path(root) / item["path"]
        try:
            if p.is_symlink() or sha256_file(p) != item["sha256"]:
                changed.append(item["path"])
        except OSError:
            changed.append(item["path"])
    return changed


# Counted trees mirror workbench_cli.Workspace.budget()["derived_bytes"] (derived, scratch, runs,
# masks, suite) under the fixed workspace root; known constant roots, no workspace config is read.
DERIVED_TREES = ("derived", "scratch", "runs", "masks", "suite")
OUTPUT_RESERVE_BYTES = 2 * MAX_OUTPUT_BYTES     # output cap + one same-size temp (peak before the hard link)


def _tree_stat_bytes(path: Path) -> tuple[int, int]:
    """(bytes, links) of a tree by lstat only: regular-file sizes are counted, links/junctions are
    counted by their own lstat size and never followed; directory metadata is not counted."""
    total = links = 0
    if not os.path.lexists(path):
        return 0, 0
    stack = [Path(path)]
    while stack:
        cur = stack.pop()
        st = os.lstat(cur)
        if _is_link(st):
            total += st.st_size
            links += 1
            continue
        if stat.S_ISDIR(st.st_mode):
            with os.scandir(cur) as it:
                stack.extend(Path(e.path) for e in it)
        elif stat.S_ISREG(st.st_mode):
            total += st.st_size
    return total, links


def output_budget(out_target: Path, *, workspace: Path | None = None) -> dict:
    """Stat-only derived-usage and free-disk check before the output is written. Reads no file
    contents (no clip records): the audit acquires no new inputs, so the 250 MB new-input budget is
    not revalidated here."""
    from bench import workbench_contract as wc
    ws = Path(workspace) if workspace is not None else ROOT / WORKSPACE
    trees, derived, links = {}, 0, 0
    for name in DERIVED_TREES:
        size, n = _tree_stat_bytes(ws / name)
        trees[name] = size
        derived += size
        links += n
    free = shutil.disk_usage(Path(out_target).parent).free
    ok_derived = derived + OUTPUT_RESERVE_BYTES <= wc.DERIVED_BUDGET_BYTES
    ok_free = free - OUTPUT_RESERVE_BYTES >= wc.FREE_RESERVE_BYTES
    return {"workspace": _rel(ROOT, ws) if Path(os.path.abspath(ws)).is_relative_to(ROOT) else str(ws),
            "counted_trees": trees, "links_not_followed": links, "derived_bytes": derived,
            "output_reserve_bytes": OUTPUT_RESERVE_BYTES, "derived_limit": wc.DERIVED_BUDGET_BYTES,
            "free_bytes": free, "free_reserve": wc.FREE_RESERVE_BYTES, "within_budget": ok_derived and ok_free,
            "new_input_bytes_added": 0,
            "new_input_budget": "not revalidated: the audit acquires no new inputs (parent-verified separately)",
            "method": "lstat sizes only; links not followed; directory metadata not counted"}


def validate_output(out: Path, root: Path = ROOT) -> Path:
    target = Path(os.path.abspath(out))
    root_abs = Path(os.path.abspath(root))
    if target.suffix != ".json" or target.name.startswith("."):
        raise UsageError("--out must be a visible .json file name")
    try:
        rel = target.relative_to(root_abs)
    except ValueError:
        raise UsageError("--out must be inside the repository") from None
    if os.path.lexists(target):
        raise UsageError("--out already exists; outputs are never overwritten")
    cur = root_abs
    for part in rel.parts[:-1]:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            raise UsageError("--out parent directory does not exist") from None
        if _is_link(st) or not stat.S_ISDIR(st.st_mode):
            raise UsageError("--out parent path has a link or non-directory")
    return target


def _parse_artwork_map(values: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values or []:
        case, sep, path = v.partition("=")
        if not sep or not CASE_RE.match(case) or not path:
            raise UsageError(f"--artwork must be CASE=PATH (got {v!r})")
        if case in out:
            raise UsageError(f"--artwork given twice for case {case!r}")
        out[case] = path
    return out


def _cli_path(value: str) -> str | Path:
    """Absolute CLI paths stay Paths (containment-checked); relative ones are repository-root-relative."""
    return Path(os.path.abspath(value)) if os.path.isabs(value) else value


def _load_inventory(path: str | Path) -> tuple[dict, list[dict], dict]:
    p = resolve_input(ROOT, path)
    if os.lstat(p).st_size > MAX_JSON_BYTES:
        raise UsageError("inventory exceeds the metadata cap")
    raw = p.read_bytes()
    doc = strict_json(raw, "inventory")
    if not isinstance(doc, dict) or doc.get("schema") != INVENTORY_SCHEMA or not isinstance(doc.get("rows"), list):
        raise UsageError("inventory schema differs")
    rows, seen = [], set()
    for r in doc["rows"]:
        if not isinstance(r, dict) or not isinstance(r.get("case"), str) or not CASE_RE.match(r["case"]) \
                or r["case"] in seen or not isinstance(r.get("bundle"), str):
            raise UsageError("inventory rows need unique case names and bundle paths")
        seen.add(r["case"])
        rows.append(r)
    prov = {"path": _rel(ROOT, p), "sha256": sha256_bytes(raw), "bytes": len(raw),
            "note": "provenance only; the inventory is not runtime payload and its boolean claims are not trusted"}
    return prov, rows, doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline portability audit (accounting only).")
    parser.add_argument("--prepared-inventory")
    parser.add_argument("--artwork", action="append", default=[], metavar="CASE=PATH")
    parser.add_argument("--surface-package", action="append", default=[], metavar="DIR")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        artworks = _parse_artwork_map(args.artwork)
        if not args.prepared_inventory and not args.surface_package:
            raise UsageError("request --prepared-inventory and/or --surface-package")
        if artworks and not args.prepared_inventory:
            raise UsageError("--artwork requires --prepared-inventory")
        packages = [os.path.normcase(os.path.abspath(p)) for p in args.surface_package]
        if len(set(packages)) != len(packages):
            raise UsageError("--surface-package given twice")
        target = validate_output(Path(args.out))
        preloaded = sorted(m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_IMPORTS))
        if preloaded:
            raise UsageError(f"model runtime already imported: {preloaded[:5]}")
        inventory = inv_rows = None
        if args.prepared_inventory:
            try:
                inventory, inv_rows, _doc = _load_inventory(_cli_path(args.prepared_inventory))
            except AuditFail as exc:
                raise UsageError(f"inventory: {exc}") from None
            if set(artworks) != {r["case"] for r in inv_rows}:
                raise UsageError(f"--artwork cases {sorted(artworks)} must equal the inventory cases "
                                 f"{sorted(r['case'] for r in inv_rows)} exactly")
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    before = {"sources": source_pins(), "runtime": runtime_pins()}
    rows: list[dict] = []
    try:
        for inv in inv_rows or []:
            row = audit_prepared_control(inv["bundle"], _cli_path(artworks[inv["case"]]), root=ROOT)
            row["case"] = inv["case"]
            row["closure_reasons"]["fail"].extend(crosscheck_inventory(row, inv))
            rows.append(finalize_row(row))
        for pkg in args.surface_package:
            rows.append(audit_surface_package(_cli_path(pkg), root=ROOT))
    except AuditViolation as exc:
        print(f"error: guarded route attempted ({exc}); no audit output written", file=sys.stderr)
        return 2
    after = {"sources": source_pins(), "runtime": runtime_pins()}
    inputs = [i for r in rows for i in r.get("inputs_read", [])]
    if inventory:
        inputs.append({"role": "inventory", **{k: inventory[k] for k in ("path", "sha256", "bytes")}})
    changed = _rehash(ROOT, inputs)
    drift = canonical(before) != canonical(after)
    if changed or drift:
        for r in rows:
            r["closure_reasons"]["fail"].append(
                f"inputs/sources changed during the audit (inputs {changed[:5]}, sources/runtime drift {drift})")
            finalize_row(r)
    code = exit_code(rows)
    try:
        budget = output_budget(target)        # stat-only; no file contents read
    except OSError as exc:
        budget = {"within_budget": False, "error": f"{type(exc).__name__}: {exc}"}
    doc = {"schema": SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "mobile_verified": False,
           "limits": {"bytes": LIMIT_BYTES, "triangles": LIMIT_TRIANGLES, "texture_edge_px": LIMIT_TEXTURE_EDGE},
           "inventory": inventory, "artwork_map": artworks, "surface_packages": list(args.surface_package),
           "rows": rows, "exit_code": code,
           "closure": {"before": before, "after_matches": not drift, "inputs_changed": changed,
                       "inputs_rehashed": len(inputs)},
           "output_budget": budget,
           "model_pins": _model_pins(), "full_mobile_runtime": "unverified",
           "audit_wall_ms": (time.perf_counter() - started) * 1000.0, "claims": CLAIMS}
    data = canonical(doc)
    # Everything reported after publication is prepared before it; nothing fallible follows the commit.
    summary_text = json.dumps({
        "out": _rel(ROOT, target), "sha256": sha256_bytes(data), "exit_code": code,
        "rows": [{"case": r.get("case") or (r.get("package") or {}).get("dir"), "kind": r["kind"],
                  "closure_status": r["closure_status"], **{t: r[t] for t in TARGETS}} for r in rows]}, indent=2)
    try:
        if len(data) >= MAX_OUTPUT_BYTES:
            raise UsageError(f"audit output is {len(data)} bytes, not below {MAX_OUTPUT_BYTES}")
        if not budget.get("within_budget"):
            raise UsageError("1 GiB derived budget or 1 GiB free-disk reserve would be violated (stat-only check)")
        from bench import workbench_image_io as wio
        wio.publish_new_bytes(target, data)
    except Exception as exc:  # noqa: BLE001 - keep a terminal record; nothing is promoted
        sys.stderr.write("audit output not written; terminal record:\n" + data.decode("ascii") + "\n")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(summary_text)
    return code


if __name__ == "__main__":
    sys.exit(main())
