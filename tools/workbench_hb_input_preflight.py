"""HB cached-input feasibility preflight: metadata plan, CAD header and opening preview (CPU only).

Question: can ONE truly distinct, calibrated, already-cached RGB source (HB ``val_primesense``
scene 000002, 120 PNG members cached as raw ZIP-member payloads) later become a prepared-metric
development control? This module checks only the *input* side. It makes no head, phone,
tracking, automatic-geometry or quality claim. The object/instance binding and the opening seed
stay UNVERIFIED, nothing is admitted, and no geometry is verified.

Integrity and consistency checks only: the only expectations come from the LOCALLY CAPTURED acquisition
index (central-directory CRC-32 plus exact compressed and expanded sizes). That index is not an
independently trusted source, and it pins no SHA-256 for payloads. The SHA-256 values here are
observed, not independently expected (``cryptographic_source_identity_verified: false``). A later
model control needs a separately reviewed, independently sealed SHA-only allowlist; this module
does not look for one.

Stages (each runs entirely inside the guard; explicit inputs; deterministic outputs):

``plan``
    Metadata only. Parses the explicit acquisition-plan resource index and calibration registry,
    and stats the cached payloads. Opens ZERO payload bytes and no GT. Freezes the future split
    (opening 0..9, planned seed index 9, scoring 10..119) before anything is viewed.
``inspect-mesh``
    Opens ONLY one required candidate CAD payload, after the plan is verified and the
    prospective budget check passes. A bounded stream checks it (raw DEFLATE or stored, exact
    sizes, CRC-32, observed SHA-256). Only the PLY header (<= 64 KiB) is parsed. A feasible header
    is not verified geometry.
``preview-opening``
    Run by the parent ONLY after review. Frame indices are validated (0..9, at most 10) before any
    file is opened. Writes ONE labelled contact sheet (<= 1024 px wide) plus a receipt.

Exit codes: 0 verified data preflight only; 1 failure; 2 usage; 3 incomplete or unsupported.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import stat
import struct
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOL_VERSION = "2"
SCHEMA_PLAN = "workbench_hb_input_preflight_plan_v2"
SCHEMA_MESH = "workbench_hb_mesh_header_v2"
SCHEMA_PREVIEW = "workbench_hb_opening_preview_v2"
SCHEMA_FAILURE = "workbench_hb_preflight_failure_v2"

CORPUS = "HB"
DATASET_REVISION = "985ebaef7b1bd4bffb2039f18dd81faa647fd060"
RGB_ARCHIVE = "hb_val_primesense.zip"
MODEL_ARCHIVE = "hb_models.zip"
ALLOWED_SCENES = {"000002": {"rgb_frames": 120, "rgb_encoded_bytes": 46_130_618, "rgb_expanded_bytes": 110_835_240}}
DEFAULT_SCENE = "000002"
OPENING = tuple(range(0, 10))
PLANNED_SEED_INDEX = 9
SCORING = tuple(range(10, 120))
REQUIRED_MODEL_IDS = tuple(range(1, 11))     # an extra id never substitutes for a missing required id
MODEL_MEMBER_RE = re.compile(r"^models/obj_(\d{6})\.ply$")
IMAGE_SIZE = (640, 480)
EXPECTED_K = {"fx": 537.4799, "fy": 536.1447, "cx": 318.8965, "cy": 238.3781}
K_ABS_TOL = 1e-3                      # the expected values are quoted to 4 decimals
FACE_CAP = 200_000
HEADER_CAP = 64 << 10
CHUNK = 1 << 20
MAX_RGB_EXPANDED = 4 << 20
MAX_MODEL_EXPANDED = 64 << 20
MAX_ACQUISITION_PLAN_BYTES = 8 << 20
MAX_CALIBRATION_BYTES = 256 << 10
MAX_JSON_OUTPUT = 256 << 10
MAX_PREVIEW_AGGREGATE = 2 << 20
MAX_FAILURE_RECEIPT = 64 << 10
DIR_ENTRY_RESERVE = 64 << 10
JSON_STAGE_ALLOWANCE = 2 * MAX_JSON_OUTPUT + DIR_ENTRY_RESERVE            # owned temp + published file
PREVIEW_STAGE_ALLOWANCE = MAX_PREVIEW_AGGREGATE + MAX_FAILURE_RECEIPT + DIR_ENTRY_RESERVE
SHEET_MAX_WIDTH = 1024
THUMB = (192, 144)                    # exact 0.3 scale of 640x480
SHEET_COLUMNS = 5
DERIVED_LIMIT = 1 << 30
FREE_RESERVE = 1 << 30
NEW_INPUT_LIMIT = 250_000_000
FORBIDDEN_MODULES = frozenset({"torch", "torchvision", "sam2", "segment_anything", "OpenGL", "pyrender", "trimesh",
                               "open3d", "pytorch3d", "nvdiffrast", "kaolin"})
FORBIDDEN_EVENTS = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                              "os.startfile", "socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                              "socket.gethostbyaddr", "socket.bind", "urllib.Request", "http.client.connect",
                              "webbrowser.open"})
DENY_COMPONENTS = frozenset({"mask", "masks", "mask_visib", "depth", "checkpoints", "models_eval"})
DENY_NAMES = frozenset({"scene_gt.json", "scene_gt_info.json", "scene_camera.json", "evaluator.json",
                        "inference.json", "results.jsonl", "resources.json", "selection.json", "models_info.json"})
DENY_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".glb", ".gltf", ".ply", ".obj"})
CODE_SUFFIXES = frozenset({".py", ".pyc", ".pyd", ".so", ".dll"})
HELPERS = {"bench": "bench/__init__.py", "bench.vision": "bench/vision.py",
           "bench.workbench_geometry": "bench/workbench_geometry.py"}
FLAGS = {"data_preflight_only": True, "head_or_phone_claim": False, "geometry_quality_claim": False,
         "gt_read": False, "neural_executed": False, "neural_admitted": False, "geometry_verified": False,
         "source_object_binding": "unverified", "instance_binding": "unverified", "opening_seed": "unverified",
         "physical_calibration": False, "calibration_kind": "dataset_reference (not a phone calibration)",
         "clock": "FRAME_INDEX", "physical_timestamp": None, "fps": None,
         "cryptographic_source_identity_verified": False,
         "payload_sha256_independently_expected": False,
         "integrity_check": ("limited local consistency/integrity check: central-directory CRC-32 and exact sizes "
                             "from the locally captured acquisition index; SHA-256 observed, not independently expected"),
         "acquisition_index_trust": "locally captured metadata; not an independently trusted source",
         "future_model_control_requires": "a separately reviewed, independently sealed SHA-only allowlist (not searched for here)"}


class PreflightError(Exception):
    """Failure (exit 1)."""


class UsageError(PreflightError):
    """Usage error (exit 2)."""


class Unsupported(PreflightError):
    """Incomplete or unsupported input (exit 3)."""


class GuardError(PreflightError):
    """A forbidden import, read, process or network action was attempted inside a stage."""


@dataclass(frozen=True)
class Context:
    """Explicit roots so synthetic fixtures never touch the real workspace."""
    root: Path = ROOT
    workspace: Path = ROOT / ".cache/video-workbench"

    @property
    def derived(self) -> Path:
        return self.workspace / "derived"


# --------------------------------------------------------------------------
# guard (process-wide audit hook, active only inside a stage)

_GUARD: dict[str, Any] = {"active": False, "roots": (), "payload_root": None, "allowed": frozenset(), "opened": []}
_HOOK_INSTALLED = False


def _norm(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


def denied_path(path: str | os.PathLike) -> bool:
    """GT / label / mask / depth / checkpoint / raw-mesh paths, denied in ANY open mode."""
    p = Path(_norm(path))
    suffix = p.suffix.lower()
    if suffix in CODE_SUFFIXES:
        return False
    name = p.name.lower()
    return bool({x.lower() for x in p.parts} & DENY_COMPONENTS) or name in DENY_NAMES or name.startswith("scene_gt") \
        or suffix in DENY_SUFFIXES


def _is_payload(path: str, g: Mapping) -> bool:
    root = g["payload_root"]
    if root and (path == root or path.startswith(root + os.sep)):
        return True
    p = Path(path)
    return p.parent.name.lower() == "payloads" and p.suffix.lower() == ".bin"


def _audit(event: str, args: tuple) -> None:
    g = _GUARD
    if not g["active"]:
        return
    if event in FORBIDDEN_EVENTS:
        raise GuardError(f"forbidden action inside a preflight stage: {event}")
    if event == "import" and args and isinstance(args[0], str) and args[0].split(".")[0] in FORBIDDEN_MODULES:
        raise GuardError(f"forbidden import inside a preflight stage: {args[0]}")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = _norm(args[0])
        if _is_payload(path, g):                       # any mode
            if path not in g["allowed"]:
                raise GuardError(f"payload not selected for this stage: {path}")
            g["opened"].append(path)
            return
        if any(path == r or path.startswith(r + os.sep) for r in g["roots"]) and denied_path(path):   # any mode
            raise GuardError(f"forbidden GT/label/model path inside a preflight stage: {path}")


@contextmanager
def guarded_stage(ctx: Context) -> Iterator[list]:
    """Whole-stage guard. Starts with an EMPTY payload allowlist; refuses an already active guard."""
    global _HOOK_INSTALLED
    if _GUARD["active"]:
        raise GuardError("a preflight guard is already active; nested or reentrant stages are refused")
    _preimport()
    present = sorted(m for m in FORBIDDEN_MODULES if m in sys.modules)
    if present:
        raise GuardError(f"forbidden modules already imported: {present}")
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    opened: list[str] = []
    _GUARD.update(active=True, roots=tuple(sorted({_norm(ctx.root), _norm(ROOT)})), payload_root=None,
                  allowed=frozenset(), opened=opened)
    try:
        yield opened
    finally:
        _GUARD.update(active=False, roots=(), payload_root=None, allowed=frozenset(), opened=[])
    present = sorted(m for m in FORBIDDEN_MODULES if m in sys.modules)
    if present:
        raise GuardError(f"forbidden modules imported during the stage: {present}")


def allow_payloads(payload_root: Path, paths: Iterable[Path] = ()) -> None:
    """Narrowly extend the active guard's allowlist with plan-verified selected payloads only."""
    if not _GUARD["active"]:
        raise GuardError("payloads can only be allowed inside an active guard")
    root = _norm(payload_root)
    if _GUARD["payload_root"] not in (None, root):
        raise GuardError("payload root differs within one stage")
    allowed = {_norm(p) for p in paths}
    if any(not (p.startswith(root + os.sep)) for p in allowed):
        raise GuardError("allowed payload lies outside the payload root")
    _GUARD.update(payload_root=root, allowed=_GUARD["allowed"] | frozenset(allowed))


def _preimport() -> None:
    from bench import workbench_geometry  # noqa: F401  numpy-only
    from bench.vision import cv2  # noqa: F401  CPU OpenCV


# --------------------------------------------------------------------------
# small IO helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(raw: bytes, what: str) -> Any:
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise PreflightError(f"{what}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def reject(token):
        raise PreflightError(f"{what}: non-finite JSON value {token}")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{what}: invalid JSON ({exc})") from exc


def dump_json(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _no_links(base: Path, path: Path) -> None:
    current = base
    for part in path.relative_to(base).parts:
        current = current / part
        if current.is_symlink():
            raise PreflightError(f"path traverses a link: {current}")


def contained_input(ctx: Context, value: str | os.PathLike, what: str) -> Path:
    """An existing regular, non-link, non-forbidden file inside the repository root (no URIs or
    traversal). The forbidden-path check runs before any open."""
    text = os.fspath(value)
    if not text or "://" in text or text.lower().startswith("file:") or "\x00" in text:
        raise UsageError(f"{what}: not a local path")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = ctx.root / candidate
    if ".." in candidate.parts:
        raise UsageError(f"{what}: traversal is not allowed")
    candidate = Path(os.path.abspath(candidate))
    if denied_path(candidate):
        raise UsageError(f"{what}: forbidden GT/label/model path {candidate.name}")
    root = Path(os.path.abspath(ctx.root))
    if root not in candidate.parents:
        raise UsageError(f"{what}: must be inside {root}")
    _no_links(root, candidate)
    try:
        st = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise PreflightError(f"{what}: missing ({candidate})") from exc
    if not stat.S_ISREG(st.st_mode):
        raise PreflightError(f"{what}: not a regular file")
    return candidate


def repo_rel(ctx: Context, path: Path) -> str:
    return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(ctx.root))).as_posix()


def read_bounded(path: Path, limit: int, what: str) -> bytes:
    if os.lstat(path).st_size > limit:
        raise PreflightError(f"{what}: exceeds {limit} bytes")
    data = Path(path).read_bytes()
    if len(data) > limit:
        raise PreflightError(f"{what}: grew beyond {limit} bytes")
    return data


def safe_member_path(base: Path, relative: Any, what: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or relative.startswith("/"):
        raise PreflightError(f"{what}: unsafe location {relative!r}")
    parts = PurePosixPath(relative).parts
    if any(p in ("", ".", "..") for p in relative.split("/")):
        raise PreflightError(f"{what}: unsafe location {relative!r}")
    path = Path(os.path.abspath(base.joinpath(*parts)))
    if Path(os.path.abspath(base)) not in path.parents:
        raise PreflightError(f"{what}: location escapes the acquisition directory")
    _no_links(Path(os.path.abspath(base)), path)
    return path


def strict_nonnegative_int(text: str) -> int:
    """argparse type: decimal digits only (no sign, spaces, underscores or exponents)."""
    if not isinstance(text, str) or not re.fullmatch(r"[0-9]{1,15}", text):
        raise argparse.ArgumentTypeError("must be a non-negative decimal integer")
    return int(text)


# --------------------------------------------------------------------------
# sources, runtime, budget


def source_closure() -> dict[str, str]:
    """Explicit helper sources only (this tool + three bench helpers)."""
    import importlib
    out = {}
    for name, rel in HELPERS.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (ROOT / rel).resolve():
            raise PreflightError(f"helper {name} imported from {module.__file__}, not {rel}")
        out[rel] = sha256_file(ROOT / rel)
    out[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha256_file(Path(__file__))
    return dict(sorted(out.items()))


def runtime_identity() -> dict:
    from bench.vision import cv2
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "opencv": cv2.__version__,
            "opencv_module": str(Path(cv2.__file__).resolve())}


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


def workspace_budget(ctx: Context, planned: int) -> dict:
    """Stat-only: derived + scratch + runs + masks + suite + planned <= 1 GiB and >= 1 GiB free
    after the planned bytes. No clip JSON or other file contents are read."""
    used = sum(_tree_bytes(ctx.workspace / d) for d in ("derived", "scratch", "runs", "masks", "suite"))
    if not ctx.workspace.is_dir():
        raise PreflightError(f"workspace {ctx.workspace} does not exist")
    free = shutil.disk_usage(ctx.workspace).free
    if used + planned > DERIVED_LIMIT:
        raise PreflightError(f"workspace derived budget would be exceeded ({used} + {planned} > {DERIVED_LIMIT})")
    if free - planned < FREE_RESERVE:
        raise PreflightError(f"free-disk reserve would be violated ({free} - {planned} < {FREE_RESERVE})")
    return {"derived_bytes": used, "planned_bytes": planned, "free_bytes": free}


def input_budget(candidate_bytes: int, existing_input_bytes: int | None) -> dict:
    """Enforces the 250 MB new-input limit for the candidate (encoded RGB + candidate CAD). The
    combined check with existing workspace inputs passes only with an explicit parent figure."""
    if existing_input_bytes is not None and (type(existing_input_bytes) is not int or existing_input_bytes < 0):
        raise UsageError("existing input bytes must be an integer >= 0")
    if candidate_bytes > NEW_INPUT_LIMIT:
        raise Unsupported(f"candidate inputs {candidate_bytes} exceed the {NEW_INPUT_LIMIT} byte new-input limit")
    out = {"candidate_bytes": candidate_bytes, "new_input_limit_bytes": NEW_INPUT_LIMIT, "candidate_within_limit": True,
           "existing_input_bytes": existing_input_bytes, "placement": "referenced in place; never copied"}
    if existing_input_bytes is None:
        out.update(combined_bytes=None, combined_compliance="unverified_parent_required",
                   note="existing workspace inputs are not estimated or read; the parent must supply them")
    else:
        combined = existing_input_bytes + candidate_bytes
        if combined > NEW_INPUT_LIMIT:
            raise Unsupported(f"existing {existing_input_bytes} + candidate {candidate_bytes} exceed {NEW_INPUT_LIMIT}")
        out.update(combined_bytes=combined, combined_compliance="within_limit_with_parent_declared_existing")
    return out


# --------------------------------------------------------------------------
# outputs (exclusive, verified, never overwrite)


def check_output(ctx: Context, out: str | os.PathLike, what: str) -> Path:
    text = os.fspath(out)
    if "://" in text or text.lower().startswith("file:"):
        raise UsageError(f"{what}: not a local path")
    path = Path(text)
    if not path.is_absolute():
        path = ctx.root / path
    if ".." in path.parts:
        raise UsageError(f"{what}: traversal is not allowed")
    path = Path(os.path.abspath(path))
    if denied_path(path) or path.name.lower() == "payloads" or path.parent.name.lower() == "payloads":
        raise UsageError(f"{what}: output names may not resemble GT/label/model/payload paths")
    derived = Path(os.path.abspath(ctx.derived))
    if derived not in path.parents:
        raise UsageError(f"{what}: must be a new path inside {derived}")
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise UsageError(f"{what}: path traverses a link")
    if os.path.lexists(path):
        raise UsageError(f"{what}: {path} already exists; outputs are never overwritten")
    return path


def publish_file(ctx: Context, out: Path, data: bytes) -> None:
    if len(data) > MAX_JSON_OUTPUT:
        raise PreflightError(f"output {len(data)} bytes exceeds {MAX_JSON_OUTPUT}")
    workspace_budget(ctx, 2 * len(data))      # remaining growth recheck: owned temp + published file
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp-{secrets.token_hex(6)}"
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644),
                       "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(tmp) != sha256_bytes(data):
            raise PreflightError("staged output does not verify")
        try:
            os.link(tmp, out)              # atomic and exclusive
        except FileExistsError as exc:
            raise PreflightError(f"{out} appeared during publication") from exc
        except OSError:
            if os.path.lexists(out):
                raise PreflightError(f"{out} appeared during publication")
            os.rename(tmp, out)
    finally:
        if os.path.lexists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------
# bounded ZIP-member stream (raw DEFLATE method 8 or stored method 0)


def validate_member_row(row: Any, member: str, archive: str, role: str, max_expanded: int) -> dict:
    """Locally captured central-directory metadata for one cached payload; no file access."""
    if not isinstance(row, Mapping):
        raise PreflightError(f"{member}: resource row missing")
    ints = ("sourceMethod", "sourceFlags", "sourceCompressedBytes", "expandedBytes", "crc32", "method", "compressedBytes")
    if any(type(row.get(k)) is not int or row.get(k) < 0 for k in ints):
        raise PreflightError(f"{member}: ZIP metadata must be non-negative integers")
    location = row.get("location")
    if (row.get("datasetRevision") != DATASET_REVISION or row.get("archive") != archive
            or row.get("sourceMember") != member or not isinstance(location, Mapping)
            or location.get("kind") != "payload" or role not in (row.get("allowedRoles") or [])):
        raise PreflightError(f"{member}: revision/archive/member/location/role differ from the expected source")
    if (row["method"] not in (0, 8) or row["method"] != row["sourceMethod"]
            or row["compressedBytes"] != row["sourceCompressedBytes"] or row["sourceFlags"] & ~2048
            or row["crc32"] > 0xFFFFFFFF or row["expandedBytes"] > max_expanded
            or row["compressedBytes"] > max_expanded + (1 << 20)
            or (row["method"] == 0 and row["compressedBytes"] != row["expandedBytes"])):
        raise PreflightError(f"{member}: unsupported or inconsistent ZIP method/flags/sizes")
    token = hashlib.sha256(f"{archive}:{member}".encode("utf-8")).hexdigest()
    if location.get("path") != f"payloads/{token}.bin":
        raise PreflightError(f"{member}: payload location is not the expected member token")
    return {"member": member, "archive": archive, "method": row["method"], "flags": row["sourceFlags"],
            "compressed_bytes": row["compressedBytes"], "expanded_bytes": row["expandedBytes"],
            "crc32": row["crc32"], "payload_path": location["path"]}


def verified_member_stream(path: Path, entry: Mapping, sink: Callable[[bytes], None] | None = None) -> dict:
    """Fully consume one payload with bounded chunk sizes and check its integrity against the local index:
    exact compressed size, exact expansion (no over-expansion, truncation or trailing bytes) and CRC-32.
    Compressed and expanded SHA-256 are OBSERVED (not independently expected). ``sink`` receives
    expanded chunks of at most CHUNK + 1 bytes; nothing is written to disk."""
    method, size, compressed, expected_crc = entry["method"], entry["expanded_bytes"], entry["compressed_bytes"], entry["crc32"]
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_size != compressed:
        raise PreflightError(f"payload {path.name}: not a regular file of the indexed compressed size")
    chash, uhash, crc, count, consumed = hashlib.sha256(), hashlib.sha256(), 0, 0, 0
    decoder = zlib.decompressobj(-15) if method == 8 else None

    def emit(expanded: bytes) -> None:
        nonlocal count, crc
        count += len(expanded)
        if count > size:
            raise PreflightError(f"payload {path.name}: expansion exceeds the indexed size")
        uhash.update(expanded)
        crc = binascii.crc32(expanded, crc)
        if sink is not None and expanded:
            sink(expanded)

    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(CHUNK)
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > compressed:
                raise PreflightError(f"payload {path.name}: longer than its indexed compressed size")
            chash.update(chunk)
            pending = chunk
            while pending:
                if decoder is None:
                    expanded, pending = pending, b""
                else:
                    if decoder.eof:
                        raise PreflightError(f"payload {path.name}: trailing bytes after the DEFLATE stream")
                    expanded = decoder.decompress(pending, min(CHUNK, size - count + 1))
                    pending = decoder.unconsumed_tail
                    if decoder.unused_data:
                        raise PreflightError(f"payload {path.name}: trailing bytes after the DEFLATE stream")
                emit(expanded)
    if decoder is not None:
        while not decoder.eof:
            tail = decoder.decompress(b"", min(CHUNK, size - count + 1))
            if not tail:
                break
            emit(tail)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise PreflightError(f"payload {path.name}: truncated or trailing DEFLATE stream")
    if consumed != compressed or count != size or (crc & 0xFFFFFFFF) != expected_crc:
        raise PreflightError(f"payload {path.name}: size/CRC-32 mismatch")
    return {"compressed_sha256_observed": chash.hexdigest(), "expanded_sha256_observed": uhash.hexdigest(),
            "crc32": crc & 0xFFFFFFFF, "compressed_bytes": consumed, "expanded_bytes": count,
            "crc32_and_sizes_consistent_with_local_index": True, "cryptographic_source_identity_verified": False}


# --------------------------------------------------------------------------
# stage 1: metadata-only plan


def _parse_K(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 3):
        arr = arr.reshape(9)
    if arr.shape != (9,) or not np.isfinite(arr).all():
        raise PreflightError("calibration K must be a finite 3x3 matrix")
    return [float(x) for x in arr]


def select_calibration(registry: Any) -> dict:
    """The dataset-reference 640x480 K (constant across matching rows); identity fields are ignored."""
    if not isinstance(registry, Mapping) or registry.get("schemaVersion") != 1 or not isinstance(registry.get("calibrations"), list):
        raise PreflightError("calibration registry must be schemaVersion 1 with a calibrations list")
    matches = []
    for row in registry["calibrations"]:
        if (isinstance(row, Mapping) and row.get("kind") == "dataset_reference"
                and row.get("width") == IMAGE_SIZE[0] and row.get("height") == IMAGE_SIZE[1]):
            if ("K" in row) == ("cam_K" in row):
                raise PreflightError("calibration row must carry exactly one of K/cam_K")
            matches.append(_parse_K(row.get("K", row.get("cam_K"))))
    if not matches:
        raise Unsupported("no dataset_reference 640x480 calibration in the registry")
    if any(m != matches[0] for m in matches):
        raise PreflightError("dataset_reference 640x480 calibrations disagree")
    K = matches[0]
    observed = {"fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5]}
    if (any(abs(observed[k] - v) > K_ABS_TOL for k, v in EXPECTED_K.items()) or K[1] != 0.0 or K[3] != 0.0
            or K[6:] != [0.0, 0.0, 1.0]):
        raise PreflightError(f"registry K {observed} differs from the expected dataset-reference K")
    return {"kind": "dataset_reference", "width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1], "K": K,
            "matching_rows": len(matches), "phone_calibration": False,
            "note": "dataset reference intrinsics; registry identity fields are not read into this plan"}


def model_inventory(models: Sequence[Mapping]) -> dict:
    """Required ids 1..10 must ALL be cached; extras are reported separately and never substitute."""
    listed = sorted(m["object_id"] for m in models)
    cached = sorted(m["object_id"] for m in models if m["cached"])
    required = list(REQUIRED_MODEL_IDS)
    missing = [i for i in required if i not in cached]
    extra = [i for i in cached if i not in required]
    return {"required_ids": required, "listed_ids": listed, "cached_ids": cached,
            "missing_required_ids": missing, "extra_cached_ids": extra,
            "listed_not_cached_ids": [i for i in listed if i not in cached],
            "inventory_complete": not missing,
            "identity_source": "member names in the local acquisition index; never derived from scene GT"}


def build_plan(ctx: Context, acquisition_plan: str | os.PathLike, calibrations: str | os.PathLike,
               scene: str = DEFAULT_SCENE, explicit_object_id: int | None = None,
               existing_input_bytes: int | None = None) -> tuple[dict, bool]:
    """Metadata only (zero payload bytes). Returns (plan, model_inventory_complete)."""
    if scene not in ALLOWED_SCENES:
        raise UsageError(f"scene {scene!r} is not an allowed candidate ({sorted(ALLOWED_SCENES)})")
    if explicit_object_id is not None and (type(explicit_object_id) is not int or explicit_object_id not in REQUIRED_MODEL_IDS):
        raise UsageError(f"object id must be one of the required candidate ids {REQUIRED_MODEL_IDS[0]}..{REQUIRED_MODEL_IDS[-1]}")
    if existing_input_bytes is not None and (type(existing_input_bytes) is not int or existing_input_bytes < 0):
        raise UsageError("existing input bytes must be an integer >= 0")
    acq = contained_input(ctx, acquisition_plan, "acquisition plan")
    cal = contained_input(ctx, calibrations, "calibration registry")
    acq_raw = read_bounded(acq, MAX_ACQUISITION_PLAN_BYTES, "acquisition plan")
    cal_raw = read_bounded(cal, MAX_CALIBRATION_BYTES, "calibration registry")
    doc = strict_json(acq_raw, "acquisition plan")
    if (not isinstance(doc, Mapping) or doc.get("schemaVersion") != 1 or doc.get("corpus") != CORPUS
            or doc.get("datasetRevision") != DATASET_REVISION or not isinstance(doc.get("resources"), Mapping)):
        raise PreflightError("acquisition plan is not the expected HB revision resource index")
    resources = doc["resources"]
    acq_dir = acq.parent
    payload_root = acq_dir / "payloads"
    pins = ALLOWED_SCENES[scene]
    prefix = f"val_primesense/{scene}/rgb/"
    names = [f"{prefix}{i:06d}.png" for i in range(pins["rgb_frames"])]
    present = [m for m in resources if isinstance(m, str) and m.startswith(prefix)]
    if sorted(present) != names:
        raise Unsupported(f"scene {scene}: cached RGB members are not exactly {names[0]}..{names[-1]}")
    split = {i: ("opening_seed_planned" if i == PLANNED_SEED_INDEX else "opening") if i in OPENING else "scoring"
             for i in range(pins["rgb_frames"])}
    rgb = []
    for index, member in enumerate(names):
        entry = validate_member_row(resources[member], member, RGB_ARCHIVE, "rgb", MAX_RGB_EXPANDED)
        path = safe_member_path(acq_dir, entry["payload_path"], member)
        st = os.lstat(path) if os.path.lexists(path) else None
        if st is None or not stat.S_ISREG(st.st_mode) or st.st_size != entry["compressed_bytes"]:
            raise Unsupported(f"{member}: cached payload missing or not of the indexed size")
        rgb.append({"index": index, "frame_id": f"s{index:06d}", "split": split[index], **entry, "stat_bytes": st.st_size})
    totals = {"rgb_encoded_bytes": sum(r["compressed_bytes"] for r in rgb),
              "rgb_expanded_bytes": sum(r["expanded_bytes"] for r in rgb)}
    if totals["rgb_encoded_bytes"] != pins["rgb_encoded_bytes"] or totals["rgb_expanded_bytes"] != pins["rgb_expanded_bytes"]:
        raise PreflightError(f"scene {scene}: RGB byte totals {totals} differ from the expected totals")
    models = []
    for member in sorted(m for m in resources if isinstance(m, str) and MODEL_MEMBER_RE.match(m)):
        entry = validate_member_row(resources[member], member, MODEL_ARCHIVE, "model", MAX_MODEL_EXPANDED)
        path = safe_member_path(acq_dir, entry["payload_path"], member)
        st = os.lstat(path) if os.path.lexists(path) else None
        cached = st is not None and stat.S_ISREG(st.st_mode) and st.st_size == entry["compressed_bytes"]
        object_id = int(MODEL_MEMBER_RE.match(member).group(1))
        models.append({"object_id": object_id, **entry, "cached": cached, "required": object_id in REQUIRED_MODEL_IDS,
                       "stat_bytes": st.st_size if st is not None else None})
    inventory = model_inventory(models)
    if explicit_object_id is not None and explicit_object_id not in inventory["cached_ids"]:
        raise Unsupported(f"explicit object id {explicit_object_id} has no cached model payload")
    candidate_models = [m for m in models if m["cached"] and m["required"]
                        and (explicit_object_id is None or m["object_id"] == explicit_object_id)]
    budget = input_budget(totals["rgb_encoded_bytes"] + sum(m["compressed_bytes"] for m in candidate_models),
                          existing_input_bytes)
    budget["candidate_cad_ids"] = sorted(m["object_id"] for m in candidate_models)
    calibration = select_calibration(strict_json(cal_raw, "calibration registry"))
    plan = {
        "schema": SCHEMA_PLAN, "tool_version": TOOL_VERSION, "flags": FLAGS,
        "sources": source_closure(), "runtime": runtime_identity(),
        "inputs": {"acquisition_plan": {"path": repo_rel(ctx, acq), "sha256": sha256_bytes(acq_raw), "bytes": len(acq_raw)},
                   "calibrations": {"path": repo_rel(ctx, cal), "sha256": sha256_bytes(cal_raw), "bytes": len(cal_raw)},
                   "payload_root": repo_rel(ctx, payload_root)},
        "source": {"corpus": CORPUS, "dataset_revision": DATASET_REVISION, "rgb_archive": RGB_ARCHIVE,
                   "model_archive": MODEL_ARCHIVE, "scene": scene,
                   "acquisition_status_declared": doc.get("status") if isinstance(doc.get("status"), str) else None,
                   "acquisition_status_trusted": False, "compressed_content_verified": False,
                   "cryptographic_source_identity_verified": False},
        "selection": {"opening": list(OPENING), "planned_seed_index": PLANNED_SEED_INDEX, "scoring": list(SCORING),
                      "scoring_count": len(SCORING), "fixed_before_viewing": True, "seed": "none",
                      "clock": "FRAME_INDEX", "physical_timestamp": None, "fps": None},
        "rgb": rgb, "rgb_totals": totals,
        "models": {"listed": models, **inventory},
        "explicit_object_id": explicit_object_id,
        "calibration": calibration,
        "input_budget": budget,
        "status": {"compressed_content_verified": False, "cryptographic_source_identity_verified": False,
                   "seed": "none", "instance_binding": "unverified", "neural_admitted": False,
                   "geometry_verified": False},
    }
    return plan, inventory["inventory_complete"]


def plan_stage(ctx: Context, acquisition_plan: str, calibrations: str, scene: str, out: str,
               explicit_object_id: int | None = None, existing_input_bytes: int | None = None) -> dict:
    with guarded_stage(ctx) as opened:
        out_path = check_output(ctx, out, "--out")
        workspace_budget(ctx, JSON_STAGE_ALLOWANCE)                       # prospective, before any read
        acq = contained_input(ctx, acquisition_plan, "acquisition plan")
        allow_payloads(acq.parent / "payloads", ())                        # zero payload opens are allowed
        before = sha256_file(acq), sha256_file(contained_input(ctx, calibrations, "calibration registry"))
        plan, complete = build_plan(ctx, acquisition_plan, calibrations, scene, explicit_object_id, existing_input_bytes)
        if (plan["inputs"]["acquisition_plan"]["sha256"], plan["inputs"]["calibrations"]["sha256"]) != before:
            raise PreflightError("inputs changed while planning")
        if opened:
            raise GuardError("plan stage opened payload bytes")
        data = dump_json(plan)
        publish_file(ctx, out_path, data)
    return {"plan": out_path.as_posix(), "plan_sha256": sha256_bytes(data), "payload_opens": 0,
            "model_inventory_complete": complete, "exit": 0 if complete else 3}


# --------------------------------------------------------------------------
# later stages: plan and closure consistency


def load_plan(ctx: Context, plan_path: str | os.PathLike, expected_sha256: str) -> tuple[dict, Path]:
    """External SHA plus exact equality with a fresh metadata rebuild (no payload bytes)."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise UsageError("--expected-plan-sha256 must be 64 lowercase hex characters")
    path = contained_input(ctx, plan_path, "plan")
    raw = read_bounded(path, MAX_JSON_OUTPUT, "plan")
    if sha256_bytes(raw) != expected_sha256:
        raise PreflightError("plan differs from the expected SHA-256")
    plan = strict_json(raw, "plan")
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA_PLAN or plan.get("tool_version") != TOOL_VERSION:
        raise PreflightError("plan schema/version differs")
    try:
        rebuilt, _complete = build_plan(ctx, plan["inputs"]["acquisition_plan"]["path"], plan["inputs"]["calibrations"]["path"],
                                        plan["source"]["scene"], plan["explicit_object_id"],
                                        plan["input_budget"]["existing_input_bytes"])
    except (KeyError, TypeError) as exc:
        raise PreflightError(f"plan is malformed ({exc})") from exc
    if json.loads(dump_json(rebuilt)) != plan:
        raise PreflightError("plan differs from a fresh metadata rebuild (pins, inputs, sources, runtime or payload stats)")
    return plan, path


def closure(ctx: Context, plan: Mapping, plan_path: Path, payloads: Sequence[Mapping]) -> dict:
    acq_dir = contained_input(ctx, plan["inputs"]["acquisition_plan"]["path"], "acquisition plan").parent
    stats = {}
    for entry in payloads:
        path = safe_member_path(acq_dir, entry["payload_path"], entry["member"])
        st = os.lstat(path)
        stats[entry["payload_path"]] = [stat.S_ISREG(st.st_mode), st.st_size]
    return {"plan": sha256_file(plan_path),
            "acquisition_plan": sha256_file(contained_input(ctx, plan["inputs"]["acquisition_plan"]["path"], "acquisition plan")),
            "calibrations": sha256_file(contained_input(ctx, plan["inputs"]["calibrations"]["path"], "calibrations")),
            "sources": source_closure(), "runtime": runtime_identity(), "payload_stats": stats}


def _payload_root(ctx: Context, plan: Mapping) -> Path:
    return contained_input(ctx, plan["inputs"]["acquisition_plan"]["path"], "acquisition plan").parent / "payloads"


# --------------------------------------------------------------------------
# stage 2: PLY header only

PLY_SCALARS = {"char", "uchar", "short", "ushort", "int", "uint", "float", "double", "int8", "uint8", "int16",
               "uint16", "int32", "uint32", "float32", "float64"}
PLY_FLOATS = {"float", "double", "float32", "float64"}
PLY_INTEGERS = PLY_SCALARS - PLY_FLOATS
COUNT_RE = re.compile(r"^[0-9]{1,10}$")
UNIT_RE = re.compile(r"\bunits?\b\s*[:=]?\s*([A-Za-z]+)", re.IGNORECASE)
UNIT_WORDS = {"mm": "millimetres", "millimeter": "millimetres", "millimeters": "millimetres", "millimetre": "millimetres",
              "millimetres": "millimetres", "m": "metres", "meter": "metres", "meters": "metres", "metre": "metres",
              "metres": "metres", "cm": "centimetres", "centimeter": "centimetres", "centimeters": "centimetres",
              "centimetre": "centimetres", "centimetres": "centimetres"}


def parse_ply_header(prefix: bytes) -> dict:
    """Bounded ASCII header parse (<= 64 KiB). Raises PreflightError on a malformed header."""
    data = bytes(prefix[:HEADER_CAP])
    if not (data.startswith(b"ply\n") or data.startswith(b"ply\r\n")):
        raise PreflightError("not a PLY file (missing 'ply' magic)")
    match = re.search(rb"(?:^|\n)end_header\r?\n", data)
    if match is None:
        raise PreflightError(f"no end_header within the first {HEADER_CAP} bytes")
    try:
        text = data[:match.end()].decode("ascii")
    except UnicodeDecodeError as exc:
        raise PreflightError("PLY header is not ASCII") from exc
    lines = [line.rstrip("\r") for line in text.split("\n")[:-1]]
    if lines[0] != "ply" or lines[-1] != "end_header" or len(lines) < 3:
        raise PreflightError("malformed PLY header framing")
    fmt = lines[1].split()
    if len(fmt) != 3 or fmt[0] != "format":
        raise PreflightError("PLY format line must directly follow the magic")
    encoding, version = fmt[1], fmt[2]
    elements: list[dict] = []
    comments: list[str] = []
    texture_files: list[str] = []
    for line in lines[2:-1]:
        tokens = line.split()
        if not tokens:
            raise PreflightError("empty PLY header line")
        key = tokens[0]
        if key in ("comment", "obj_info"):
            body = line[len(key):].strip()
            if len(comments) < 64:
                comments.append(body[:200])
            if len(tokens) >= 3 and tokens[1].lower() in ("texturefile", "texture_file", "texture"):
                texture_files.append(" ".join(tokens[2:])[:200])
        elif key == "element":
            if len(tokens) != 3 or not COUNT_RE.match(tokens[2]):
                raise PreflightError(f"malformed element line {line!r} (count must be a finite non-negative integer)")
            if any(e["name"] == tokens[1] for e in elements):
                raise PreflightError(f"duplicate element {tokens[1]!r}")
            elements.append({"name": tokens[1], "count": int(tokens[2]), "properties": []})
        elif key == "property":
            if not elements:
                raise PreflightError("property before any element")
            props = elements[-1]["properties"]
            if len(tokens) == 5 and tokens[1] == "list":
                if tokens[2] not in PLY_INTEGERS or tokens[3] not in PLY_SCALARS:
                    raise PreflightError(f"malformed list property {line!r}")
                prop = {"name": tokens[4], "list": [tokens[2], tokens[3]]}
            elif len(tokens) == 3 and tokens[1] in PLY_SCALARS:
                prop = {"name": tokens[2], "type": tokens[1]}
            else:
                raise PreflightError(f"malformed property {line!r}")
            if any(p["name"] == prop["name"] for p in props):
                raise PreflightError(f"duplicate property {prop['name']!r}")
            props.append(prop)
        else:
            raise PreflightError(f"unknown PLY header keyword {key!r}")
    return {"encoding": encoding, "version": version, "elements": elements, "comments": comments,
            "texture_files": texture_files, "header_bytes": match.end()}


def evaluate_header(header: Mapping) -> dict:
    """Header-only feasibility for a FUTURE reviewed conversion. Never admission or geometry proof."""
    reasons = []
    if header["encoding"] not in ("ascii", "binary_little_endian"):
        reasons.append(f"unsupported_encoding:{header['encoding']}")
    if header["version"] != "1.0":
        reasons.append(f"unsupported_version:{header['version']}")
    by_name = {e["name"]: e for e in header["elements"]}
    extra = sorted(set(by_name) - {"vertex", "face"})
    if extra:
        reasons.append(f"unsupported_elements:{','.join(extra)}")
    vertex, face = by_name.get("vertex"), by_name.get("face")
    vprops = {p["name"]: p for p in (vertex or {}).get("properties", [])}
    if vertex is None or any(vprops.get(a, {}).get("type") not in PLY_FLOATS for a in ("x", "y", "z")):
        reasons.append("missing_float_vertex_positions")
    if face is None:
        reasons.append("missing_face_element")           # the face cap cannot be bypassed by omitting faces
    else:
        fprops = {p["name"]: p for p in face["properties"]}
        idx = fprops.get("vertex_indices") or fprops.get("vertex_index")
        if idx is None or "list" not in idx or idx["list"][1] not in PLY_INTEGERS:
            reasons.append("missing_face_vertex_indices")
        if face["count"] > FACE_CAP:
            reasons.append(f"face_cap_exceeded:{face['count']}>{FACE_CAP}")
    if (vertex or {}).get("count", 0) == 0 or (face or {}).get("count", 0) == 0:
        reasons.append("empty_geometry")
    colour = all(vprops.get(c, {}).get("type") in ("uchar", "uint8") for c in ("red", "green", "blue"))
    texcoords = (any(a in vprops for a in ("texture_u", "u", "s")) and any(a in vprops for a in ("texture_v", "v", "t"))) or (
        face is not None and any(p["name"] in ("texcoord", "texture_coordinates") for p in face["properties"]))
    if colour:
        appearance = "vertex_colour"
    elif texcoords or header["texture_files"]:
        appearance = "texture_reference"
        reasons.append("texture_image_not_in_cached_inventory")
    else:
        appearance = "none"
        reasons.append("missing_appearance")
    declared = [UNIT_RE.search(c) for c in header["comments"]]
    declared = [m.group(1).lower() for m in declared if m]
    if declared:
        consistent = all(d in UNIT_WORDS for d in declared) and len({UNIT_WORDS.get(d) for d in declared}) == 1
        units = {"status": "declared_in_header" if consistent else "unknown",
                 "value": UNIT_WORDS.get(declared[0]) if consistent else None, "evidence": "header comment"}
    else:
        units = {"status": "assumed_dataset_convention", "value": "millimetres",
                 "evidence": "BOP/HB model distribution convention for the hb_models.zip revision; not stated in "
                             "the header and not verified here; nothing converted"}
    if units["status"] == "unknown":
        reasons.append("unknown_units")
    return {"header_feasible_for_future_reviewed_conversion": not reasons, "reasons": reasons,
            "appearance": appearance, "vertex_colour": colour, "texture_coordinates": bool(texcoords),
            "units": units, "vertex_count": (vertex or {}).get("count"), "face_count": (face or {}).get("count"),
            "face_cap": FACE_CAP, "geometry_verified": False, "admitted": False}


def inspect_mesh(ctx: Context, plan_path: str, expected_sha256: str, object_id: int, out: str) -> dict:
    if type(object_id) is not int or object_id not in REQUIRED_MODEL_IDS:
        raise UsageError(f"--object-id must be one of the required candidate ids {REQUIRED_MODEL_IDS[0]}..{REQUIRED_MODEL_IDS[-1]}")
    with guarded_stage(ctx) as opened:                                     # empty payload allowlist
        out_path = check_output(ctx, out, "--out")
        plan, plan_file = load_plan(ctx, plan_path, expected_sha256)
        if plan["explicit_object_id"] is not None and plan["explicit_object_id"] != object_id:
            raise UsageError("--object-id differs from the explicit object id frozen in the plan")
        matches = [m for m in plan["models"]["listed"] if m["object_id"] == object_id and m["cached"] and m["required"]]
        if len(matches) != 1:
            raise Unsupported(f"object {object_id}: no cached required model payload in the plan")
        entry = matches[0]
        payload_root = _payload_root(ctx, plan)
        path = safe_member_path(payload_root.parent, entry["payload_path"], entry["member"])
        before = closure(ctx, plan, plan_file, [entry])
        workspace_budget(ctx, JSON_STAGE_ALLOWANCE)                       # prospective, before the payload open
        allow_payloads(payload_root, [path])
        prefix = bytearray()

        def keep(chunk: bytes) -> None:
            room = HEADER_CAP - len(prefix)
            if room > 0:
                prefix.extend(chunk[:room])
        checked = verified_member_stream(path, entry, keep)
        try:
            header = parse_ply_header(bytes(prefix))
            feasibility = evaluate_header(header)
        except PreflightError as exc:
            header, feasibility = None, {"header_feasible_for_future_reviewed_conversion": False,
                                         "reasons": [f"malformed_header:{exc}"], "geometry_verified": False,
                                         "admitted": False}
        if closure(ctx, plan, plan_file, [entry]) != before:
            raise PreflightError("plan, inputs, sources, runtime or payload changed during inspection")
        document = {"schema": SCHEMA_MESH, "tool_version": TOOL_VERSION, "flags": FLAGS, "plan_sha256": expected_sha256,
                    "object_id": object_id, "object_id_source": "explicit operator argument (not recovered from GT)",
                    "member": entry["member"], "archive": entry["archive"], "payload_path": entry["payload_path"],
                    "payload_integrity": {**checked, "method": entry["method"]},
                    "memory": {"bounded_stream": True, "read_chunk_bytes": CHUNK, "max_expanded_chunk_bytes": CHUNK + 1,
                               "header_buffer_bytes": HEADER_CAP, "expanded_file_on_disk": False,
                               "payload_buffer_budget_estimate_bytes": 6 * (CHUNK + 1) + HEADER_CAP + (256 << 10),
                               "estimate_label": ("conservative bookkeeping ESTIMATE for stream chunks, header buffer "
                                                  "and a zlib allowance; NOT a certified absolute peak or RSS "
                                                  "measurement; excludes the Python interpreter, runtime and other "
                                                  "stage memory"),
                               "absolute_peak_measured": False},
                    "header": header, "feasibility": feasibility,
                    "payload_opens": [repo_rel(ctx, Path(p)) for p in opened],
                    "not_done": ["vertex/face arrays", "GLB", "decimation", "unit conversion", "geometry verification",
                                 "cryptographic source identity", "registration", "admission"]}
        data = dump_json(document)
        publish_file(ctx, out_path, data)                                  # final publication inside the guard
    return {"out": out_path.as_posix(), "sha256": sha256_bytes(data),
            "exit": 0 if feasibility["header_feasible_for_future_reviewed_conversion"] else 3}


# --------------------------------------------------------------------------
# stage 3: opening preview (parent-run only after review)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def validate_frames(frames: Sequence[Any]) -> list[int]:
    """Opening indices only (0..9), at most 10, unique. Runs before any file is opened."""
    if not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= len(OPENING):
        raise UsageError(f"--frames must list 1..{len(OPENING)} opening indices")
    if any(type(f) is not int or f not in OPENING for f in frames) or len(set(frames)) != len(frames):
        raise UsageError(f"--frames must be unique indices within {OPENING[0]}..{OPENING[-1]}")
    return sorted(frames)


def check_png_structure(raw: bytes) -> dict:
    """Strict IHDR: 640x480, 8-bit truecolour (colour type 2), standard compression/filter, CRC ok, IEND."""
    if len(raw) < 8 + 25 + 12 or raw[:8] != PNG_SIGNATURE or raw[8:16] != b"\x00\x00\x00\rIHDR":
        raise PreflightError("not a PNG with a leading IHDR chunk")
    ihdr = raw[16:29]
    if struct.unpack(">I", raw[29:33])[0] != zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF:
        raise PreflightError("PNG IHDR CRC mismatch")
    width, height, depth, colour, compression, filt, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (width, height) != IMAGE_SIZE or depth != 8 or colour != 2 or compression != 0 or filt != 0 or interlace not in (0, 1):
        raise PreflightError(f"PNG must be {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} 8-bit RGB (got {width}x{height} depth {depth} colour {colour})")
    if not raw.endswith(PNG_IEND):
        raise PreflightError("PNG does not end with IEND")
    return {"width": width, "height": height, "bit_depth": depth, "colour_type": colour, "interlace": interlace}


def decode_rgb(raw: bytes, decode: Callable[[bytes], np.ndarray] | None = None) -> np.ndarray:
    """cv2.imdecode(IMREAD_UNCHANGED) -> exact (480, 640, 3) uint8 BGR -> contiguous RGB."""
    if decode is None:
        from bench.vision import cv2
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        bgr = decode(raw)
    if bgr is None or bgr.dtype != np.uint8 or bgr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise PreflightError("decoded frame is not an exact 640x480x3 uint8 image")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def build_contact_sheet(frames: Sequence[tuple[int, np.ndarray]], scene: str) -> tuple[np.ndarray, list[dict]]:
    """ONE labelled RGB contact sheet (<= 1024 px wide) and per-frame source->thumbnail geometry."""
    from bench.vision import cv2
    margin, label_h, banner_h = 8, 18, 64
    tw, th = THUMB
    cols = min(SHEET_COLUMNS, len(frames))
    rows = math.ceil(len(frames) / SHEET_COLUMNS)
    width = margin + cols * (tw + margin)
    height = banner_h + rows * (label_h + th + margin) + margin
    if width > SHEET_MAX_WIDTH:
        raise PreflightError("contact sheet would exceed 1024 px")
    sheet = np.full((height, width, 3), 24, np.uint8)
    banner = [f"HB val_primesense scene {scene} - opening frames - CANDIDATE INPUT ONLY (not reviewed)",
              "clock FRAME_INDEX; physical time unknown; neural unavailable; no GT/masks/poses/object identity",
              f"* = planned seed index {PLANNED_SEED_INDEX} (unverified); dataset-reference K, not a phone calibration"]
    for i, text in enumerate(banner):
        cv2.putText(sheet, text, (margin, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    sx, sy = tw / IMAGE_SIZE[0], th / IMAGE_SIZE[1]
    geometry = []
    for k, (index, rgb) in enumerate(frames):
        r, c = divmod(k, SHEET_COLUMNS)
        x0 = margin + c * (tw + margin)
        y0 = banner_h + r * (label_h + th + margin) + label_h
        sheet[y0:y0 + th, x0:x0 + tw] = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        label = f"frame {index:06d}" + (" *" if index == PLANNED_SEED_INDEX else "")
        cv2.putText(sheet, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 120), 1, cv2.LINE_AA)
        geometry.append({"index": index, "tile_origin_px": [x0, y0], "tile_size_px": [tw, th], "scale": [sx, sy],
                         "resize": "cv2.INTER_AREA",
                         "mapping": "x_t = x0 + (u + 0.5) * sx - 0.5; y_t = y0 + (v + 0.5) * sy - 0.5 (pixel centres)"})
    return sheet, geometry


def _write_exclusive(path: Path, blob: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644), "wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    if sha256_file(path) != sha256_bytes(blob):
        raise PreflightError(f"staged {path.name} does not verify")


def preview_opening(ctx: Context, plan_path: str, expected_sha256: str, frames: Sequence[Any], out: str,
                    decode: Callable[[bytes], np.ndarray] | None = None) -> dict:
    indices = validate_frames(frames)                       # before ANY file is opened
    staging: Path | None = None
    out_dir: Path | None = None
    stage = "plan"
    try:
        with guarded_stage(ctx) as opened:                  # whole stage; empty payload allowlist
            out_dir = check_output(ctx, out, "--out")
            staging = out_dir.parent / f".{out_dir.name}.staging-{secrets.token_hex(6)}"
            plan, plan_file = load_plan(ctx, plan_path, expected_sha256)
            rows = {r["index"]: r for r in plan["rgb"]}
            entries = [rows[i] for i in indices]
            if any(e["split"] not in ("opening", "opening_seed_planned") for e in entries):
                raise UsageError("preview may only touch opening frames")
            payload_root = _payload_root(ctx, plan)
            paths = [safe_member_path(payload_root.parent, e["payload_path"], e["member"]) for e in entries]
            before = closure(ctx, plan, plan_file, entries)
            workspace_budget(ctx, PREVIEW_STAGE_ALLOWANCE)   # prospective, before any payload open
            allow_payloads(payload_root, paths)
            stage = "decode"
            from bench.vision import cv2
            from bench.workbench_geometry import array_sha256
            cv2.ocl.setUseOpenCL(False)
            records, decoded = [], []
            for entry, path in zip(entries, paths):
                buffer = bytearray()

                def collect(chunk: bytes, _b=buffer, _cap=entry["expanded_bytes"]) -> None:
                    if len(_b) + len(chunk) > min(_cap, MAX_RGB_EXPANDED):
                        raise PreflightError("expanded PNG exceeds its bound")
                    _b.extend(chunk)
                checked = verified_member_stream(path, entry, collect)
                raw = bytes(buffer)
                structure = check_png_structure(raw)
                rgb = decode_rgb(raw, decode)
                records.append({"index": entry["index"], "frame_id": entry["frame_id"], "member": entry["member"],
                                "payload_path": entry["payload_path"],
                                "payload_integrity": {**checked, "method": entry["method"]}, "png": structure,
                                "decoded": {"shape": list(rgb.shape), "dtype": "uint8",
                                            "colour_convention": "cv2 BGR decode -> RGB",
                                            "rgb_array_sha256": array_sha256(rgb)}})
                decoded.append((entry["index"], rgb))
                del buffer, raw
            stage = "contact_sheet"
            sheet, geometry = build_contact_sheet(decoded, plan["source"]["scene"])
            ok, encoded = cv2.imencode(".png", np.ascontiguousarray(sheet[:, :, ::-1]))
            if not ok:
                raise PreflightError("contact sheet encoding failed")
            sheet_png = encoded.tobytes()
            if closure(ctx, plan, plan_file, entries) != before:
                raise PreflightError("plan, inputs, sources, runtime or payloads changed during the preview")
            receipt = {"schema": SCHEMA_PREVIEW, "tool_version": TOOL_VERSION, "status": "completed_data_preflight_only",
                       "flags": FLAGS, "plan_sha256": expected_sha256, "scene": plan["source"]["scene"], "frames": records,
                       "contact_sheet": {"file": "opening-contact-sheet.png", "sha256": sha256_bytes(sheet_png),
                                         "width": int(sheet.shape[1]), "height": int(sheet.shape[0]), "tiles": geometry},
                       "payload_opens": [repo_rel(ctx, Path(p)) for p in opened],
                       "labels": ["candidate input only; not an independent annotation",
                                  "integrity is local CRC-32/size consistency; SHA-256 observed; no cryptographic source identity",
                                  "no automatic reviewed selection or object identification; parent reviews qualitatively",
                                  "physical clock unknown; neural unavailable"]}
            receipt_raw = dump_json(receipt)
            stage = "publish"
            if len(sheet_png) + len(receipt_raw) > MAX_PREVIEW_AGGREGATE:
                raise PreflightError("preview outputs exceed the 2 MiB aggregate")
            workspace_budget(ctx, len(sheet_png) + len(receipt_raw) + MAX_FAILURE_RECEIPT + DIR_ENTRY_RESERVE)
            staging.mkdir(parents=False)
            _write_exclusive(staging / "opening-contact-sheet.png", sheet_png)
            _write_exclusive(staging / "preview-receipt.json", receipt_raw)   # receipt last
            if os.path.lexists(out_dir):
                raise PreflightError("output appeared during publication")
            os.rename(staging, out_dir)
        return {"out": out_dir.as_posix(), "receipt_sha256": sha256_bytes(receipt_raw),
                "contact_sheet_sha256": sha256_bytes(sheet_png), "exit": 0}
    except BaseException as exc:
        if staging is not None and out_dir is not None and staging.is_dir():
            failure = {"schema": SCHEMA_FAILURE, "status": "failed", "completed": False, "stage": stage,
                       "error": f"{type(exc).__name__}: {exc}"[:2000],
                       "staged_files": sorted(p.name for p in staging.iterdir())[:16]}
            blob = dump_json(failure)
            try:
                if len(blob) <= MAX_FAILURE_RECEIPT:
                    _write_exclusive(staging / "failure.json", blob)
                    os.rename(staging, out_dir.parent / f"{out_dir.name}.failed-{secrets.token_hex(4)}")
            except (OSError, PreflightError):
                pass                       # the partial staging directory is retained as evidence
        raise


# --------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None, ctx: Context | None = None) -> int:
    ctx = ctx or Context()
    p = argparse.ArgumentParser(description="HB cached-input preflight (metadata plan / PLY header / opening preview).")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--acquisition-plan", required=True)
    s.add_argument("--calibrations", required=True)
    s.add_argument("--scene", default=DEFAULT_SCENE)
    s.add_argument("--object-id", type=strict_nonnegative_int, default=None,
                   help="explicit operator-supplied required id 1..10 (never from GT)")
    s.add_argument("--existing-input-bytes", type=strict_nonnegative_int, default=None,
                   help="parent-declared existing workspace new-input bytes (otherwise combined compliance is unverified)")
    s.add_argument("--out", required=True)
    s = sub.add_parser("inspect-mesh")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--object-id", type=strict_nonnegative_int, required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("preview-opening", help="parent-run only after review")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--frames", type=strict_nonnegative_int, nargs="+", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan_stage(ctx, args.acquisition_plan, args.calibrations, args.scene, args.out, args.object_id,
                                args.existing_input_bytes)
        elif args.command == "inspect-mesh":
            result = inspect_mesh(ctx, args.plan, args.expected_plan_sha256, args.object_id, args.out)
        else:
            result = preview_opening(ctx, args.plan, args.expected_plan_sha256, args.frames, args.out)
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(result.get("exit", 0))
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except Unsupported as exc:
        print(f"incomplete/unsupported: {exc}", file=sys.stderr)
        return 3
    except (PreflightError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
