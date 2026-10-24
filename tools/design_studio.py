"""Bounded, immutable local artwork storage. No workbench/model imports or execution."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import threading
import uuid
from datetime import datetime, timezone
import zlib

MAX_DESIGNS = 100
MAX_HANDOFFS = 100
MAX_STORAGE_BYTES = 32 * 1024 * 1024
MAX_PNG_BYTES = 2 * 1024 * 1024
MAX_EDGE = 1024
FREE_FLOOR_BYTES = 1024 * 1024 * 1024
MAX_JSON_BYTES = 3 * 1024 * 1024
ID_RE = re.compile(r"[0-9a-f]{32}\Z")
CLIP_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
PATTERNS = {"solid", "stripes", "checker", "dots", "waves", "grid"}


class StudioError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def loads_strict(data):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError("nonfinite number")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise StudioError("invalid_json", "Provide valid UTF-8 JSON without duplicate keys or nonfinite numbers.") from exc


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise StudioError("invalid_request", "Request fields do not match the artwork contract.")


def _string(value, minimum, maximum, label):
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise StudioError("invalid_request", f"{label} must contain {minimum} to {maximum} characters.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise StudioError("invalid_request", f"{label} contains invalid Unicode.") from exc
    return value


def validate_recipe(recipe):
    _keys(recipe, ("pattern", "colors", "text", "scale", "rotation_deg", "seed", "width", "height"))
    if type(recipe["pattern"]) is not str or recipe["pattern"] not in PATTERNS:
        raise StudioError("invalid_request", "Choose a supported local pattern.")
    colors = recipe["colors"]
    if type(colors) is not list or not 2 <= len(colors) <= 5 or any(
        type(c) is not str or not re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in colors
    ):
        raise StudioError("invalid_request", "Provide two to five six-digit hex colors.")
    _string(recipe["text"], 0, 80, "Text")
    for key, low, high in (("scale", .25, 4), ("rotation_deg", -180, 180)):
        v = recipe[key]
        if type(v) not in (int, float) or not low <= v <= high or not math.isfinite(v):
            raise StudioError("invalid_request", f"{key} is outside its supported range.")
    for key, low, high in (("seed", 0, 4294967295), ("width", 256, 1024), ("height", 256, 1024)):
        if type(recipe[key]) is not int or not low <= recipe[key] <= high:
            raise StudioError("invalid_request", f"{key} must be an integer within {low}..{high}.")
    return recipe


def validate_handoff_request(design_id, clip_id):
    if type(design_id) is not str or not ID_RE.fullmatch(design_id):
        raise StudioError("invalid_request", "Design ID is invalid.")
    if clip_id is not None and (type(clip_id) is not str or not CLIP_RE.fullmatch(clip_id)):
        raise StudioError("invalid_request", "Clip ID is invalid.")


def validate_png(data, recipe):
    """Check all chunk CRCs, then fully decode at most 1024 squared pixels."""
    if len(data) > MAX_PNG_BYTES:
        raise StudioError("png_too_large", "PNG exceeds the 2 MiB limit.", 413)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise StudioError("invalid_png", "Provide a complete PNG image.")
    offset, index, ended = 8, 0, False
    try:
        while offset < len(data):
            length, kind = struct.unpack_from(">I4s", data, offset)
            end = offset + 12 + length
            if end > len(data) or length > MAX_PNG_BYTES:
                raise ValueError("truncated chunk")
            payload = data[offset + 8:offset + 8 + length]
            crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
            if zlib.crc32(kind + payload) & 0xffffffff != crc:
                raise ValueError("invalid chunk CRC")
            if index == 0:
                if kind != b"IHDR" or length != 13:
                    raise ValueError("missing header")
                width, height = struct.unpack_from(">II", payload)
                if not (256 <= width <= MAX_EDGE and 256 <= height <= MAX_EDGE):
                    raise ValueError("dimensions exceed decode limit")
                if (width, height) != (recipe["width"], recipe["height"]):
                    raise ValueError("dimensions differ from recipe")
            elif kind == b"IHDR":
                raise ValueError("duplicate header")
            if kind == b"acTL":
                raise ValueError("animated PNG is not supported")
            offset, index = end, index + 1
            if kind == b"IEND":
                if length or offset != len(data):
                    raise ValueError("invalid image ending")
                ended = True
                break
        if not ended:
            raise ValueError("missing ending")
        from PIL import Image
        with Image.open(io.BytesIO(data)) as picture:
            if picture.format != "PNG" or picture.size != (recipe["width"], recipe["height"]):
                raise ValueError("invalid decoded image")
            picture.verify()
        with Image.open(io.BytesIO(data)) as picture:
            picture.load()
    except ImportError as exc:
        raise StudioError("decoder_unavailable", "The existing PNG decoder is unavailable; use the documented runtime.", 503) from exc
    except (ValueError, OSError, SyntaxError, struct.error, zlib.error) as exc:
        raise StudioError("invalid_png", "PNG is corrupt, truncated, unsupported, or has incorrect dimensions.") from exc


def _no_links(path):
    """Reject symlinks and Windows junction/reparse points on every existing component."""
    path = Path(os.path.abspath(path))
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise StudioError("unsafe_storage", "Storage contains an unsupported link or reparse point.", 409)
    return path


class StudioStore:
    def __init__(self, root, *, free_floor_bytes=FREE_FLOOR_BYTES):
        self.root = _no_links(root)
        self.free_floor_bytes = free_floor_bytes
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("designs", "handoffs"):
            path = _no_links(self.root / name)
            path.mkdir(exist_ok=True)
        self._usage()

    def _usage(self):
        _no_links(self.root)
        total = 0
        for current, dirs, files in os.walk(self.root, followlinks=False):
            _no_links(current)
            for name in dirs + files:
                path = _no_links(Path(current) / name)
                if path.is_file():
                    total += path.stat().st_size
        return total

    def _path(self, collection, item_id):
        if type(item_id) is not str or not ID_RE.fullmatch(item_id):
            raise StudioError("not_found", "Artwork record was not found.", 404)
        path = _no_links(self.root / collection / item_id)
        if not path.is_dir():
            raise StudioError("not_found", "Artwork record was not found.", 404)
        return path

    def _read(self, path, limit):
        _no_links(path)
        try:
            if not path.is_file() or path.stat().st_size > limit:
                raise OSError("record size invalid")
            with path.open("rb") as stream:
                result = stream.read(limit + 1)
            if len(result) > limit:
                raise OSError("record size invalid")
            return result
        except OSError as exc:
            raise StudioError("corrupt_record", "Saved artwork failed its integrity check.", 409) from exc

    def _publish(self, collection, item_id, files, count_limit):
        folder = _no_links(self.root / collection)
        usage = self._usage()
        count = len([p for p in folder.iterdir() if not p.name.startswith(".pending-")])
        output_size = sum(len(data) for data in files.values())
        if count >= count_limit or usage + output_size > MAX_STORAGE_BYTES:
            raise StudioError("storage_full", "Local artwork storage limit reached.", 507)
        if shutil.disk_usage(self.root).free < self.free_floor_bytes + output_size:
            raise StudioError("disk_reserve", "Saving would use the reserved free disk space.", 507)
        pending = folder / (".pending-" + uuid.uuid4().hex)
        destination = folder / item_id
        try:
            pending.mkdir()
            for name, data in files.items():
                with (pending / name).open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            if destination.exists():
                raise StudioError("conflict", "Artwork ID already exists.", 409)
            os.rename(pending, destination)
        finally:
            if pending.exists():
                for child in pending.iterdir():
                    child.unlink()
                pending.rmdir()

    def create_design(self, payload):
        _keys(payload, ("schema", "name", "prompt", "recipe", "png_base64"))
        if payload["schema"] != "visualizeit_design_v1":
            raise StudioError("invalid_request", "Unsupported artwork schema.")
        _string(payload["name"], 1, 80, "Name")
        _string(payload["prompt"], 0, 500, "Prompt")
        recipe = validate_recipe(payload["recipe"])
        encoded = payload["png_base64"]
        if type(encoded) is not str or len(encoded) > ((MAX_PNG_BYTES + 2) // 3) * 4:
            raise StudioError("invalid_png", "Provide a base64 PNG no larger than 2 MiB.")
        try:
            png = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise StudioError("invalid_png", "PNG base64 is invalid.") from exc
        validate_png(png, recipe)
        item_id = uuid.uuid4().hex
        metadata = {"id": item_id, "schema": "visualizeit_design_v1",
                    "created_at": datetime.now(timezone.utc).isoformat(), "name": payload["name"],
                    "prompt": payload["prompt"], "recipe": recipe,
                    "png_sha256": hashlib.sha256(png).hexdigest(), "png_bytes": len(png),
                    "image_url": f"/api/studio/designs/{item_id}/image.png"}
        with self.lock:
            self._publish("designs", item_id, {"metadata.json": json_bytes(metadata), "image.png": png}, MAX_DESIGNS)
        return metadata

    def _design(self, item_id):
        path = self._path("designs", item_id)
        try:
            metadata = loads_strict(self._read(path / "metadata.json", 16384))
            _keys(metadata, ("id", "schema", "created_at", "name", "prompt", "recipe", "png_sha256", "png_bytes", "image_url"))
            _string(metadata["name"], 1, 80, "Name")
            _string(metadata["prompt"], 0, 500, "Prompt")
            _string(metadata["created_at"], 1, 80, "Creation time")
            datetime.fromisoformat(metadata["created_at"])
            validate_recipe(metadata["recipe"])
            png = self._read(path / "image.png", MAX_PNG_BYTES)
            if (metadata["id"] != item_id or metadata["schema"] != "visualizeit_design_v1"
                    or metadata["image_url"] != f"/api/studio/designs/{item_id}/image.png"
                    or type(metadata["png_bytes"]) is not int or metadata["png_bytes"] != len(png)
                    or metadata["png_sha256"] != hashlib.sha256(png).hexdigest()):
                raise ValueError("metadata mismatch")
            validate_png(png, metadata["recipe"])
            return metadata, png
        except (StudioError, ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, StudioError) and exc.code in ("unsafe_storage", "decoder_unavailable"):
                raise
            raise StudioError("corrupt_record", "Saved artwork failed its integrity check.", 409) from exc

    def get_design(self, item_id):
        with self.lock:
            return self._design(item_id)[0]

    def get_image(self, item_id):
        with self.lock:
            return self._design(item_id)[1]

    def list_designs(self):
        with self.lock:
            self._usage()
            return [self.get_design(p.name) for p in sorted((self.root / "designs").iterdir())
                    if not p.name.startswith(".pending-")]

    @staticmethod
    def _manifest(item_id, design, clip_id):
        return {"schema": "visualizeit_artwork_handoff_v1", "id": item_id, "design_id": design["id"],
                "clip_id": clip_id, "png_sha256": design["png_sha256"], "png_bytes": design["png_bytes"],
                "image_url": design["image_url"], "consumer": "mobile/Assets/VisualizeIt/Runtime/DesignImporter.cs",
                "import_method": "Download PNG and select it with the native image picker",
                "state": "ready_for_manual_import", "tracking_applied": False, "native_import_verified": False,
                "artwork": {"kind": "imported", "edge_px": max(design["recipe"]["width"], design["recipe"]["height"]),
                            "sha256": design["png_sha256"], "generation": "visualizeit_local_design_v1"}}

    def create_handoff(self, design_id, clip_id=None, *, available_clip_ids=()):
        validate_handoff_request(design_id, clip_id)
        if clip_id is not None and clip_id not in available_clip_ids:
            raise StudioError("clip_not_found", "Choose a clip from the connected workbench.", 404)
        with self.lock:
            design = self.get_design(design_id)
            item_id = uuid.uuid4().hex
            manifest = self._manifest(item_id, design, clip_id)
            self._publish("handoffs", item_id, {"manifest.json": json_bytes(manifest)}, MAX_HANDOFFS)
            return manifest

    def get_handoff(self, item_id):
        with self.lock:
            path = self._path("handoffs", item_id)
            try:
                manifest = loads_strict(self._read(path / "manifest.json", 16384))
                clip = manifest["clip_id"]
                if clip is not None and (type(clip) is not str or not CLIP_RE.fullmatch(clip)):
                    raise ValueError("invalid clip")
                expected = self._manifest(item_id, self.get_design(manifest["design_id"]), clip)
                if manifest != expected or json_bytes(manifest) != json_bytes(expected):
                    raise ValueError("manifest mismatch")
                return manifest
            except (StudioError, KeyError, TypeError, ValueError) as exc:
                raise StudioError("corrupt_record", "Saved handoff failed its integrity check.", 409) from exc
