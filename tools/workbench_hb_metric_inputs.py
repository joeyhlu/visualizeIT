"""Trust-gated CPU input preparation for a FUTURE prepared-metric HB control (scene 000002).

Scope: inputs only. A later prepared-geometry control for the main neural head
(27e06a51...) would use these inputs. This is NOT an automatic-geometry
experiment, and automatic geometry continues to forbid any GT. Nothing here
runs a model, the GPU or inference. Nothing registers a clip, mutates an
admission or admits a family. HB keeps its own honest family
``hb_primesense_640x480``; it is never renamed to YCB and never inherits a YCB
admission.

Trust: ``historical_local_integrity``. The acquisition receipt, the resources
index and the prior integrity record were pinned (by SHA-256) and verified
historically. That is local integrity, not upstream authentication
(``upstream_authenticated: false``). No hash computed now can manufacture
upstream trust.

The caller's expected allowlist hash must identify a PARENT-ACCEPTED
``filter-allowlist`` output. Full re-validation of a re-hashed, structurally
plausible allowlist does not prove that it was derived from the historical index.

Stages (each runs inside a process-wide guard; explicit inputs only; fresh output directory;
caps enforced before every disk write; receipt published last; never overwrites):

``filter-allowlist``  METADATA ONLY. Verifies the three externally expected anchor SHA-256
                      values BEFORE parsing anything, then copies only the rows for scene 000002
                      RGB 0..119 and the explicitly selected CAD object ids (1..10) into a new
                      SHA-only allowlist. Zero payload bytes are opened. The index may name
                      forbidden files; they are never opened or copied.
``prepare-model``     One explicit allowlisted PLY (vertex colours REQUIRED) -> metre GLB
                      (strict streaming ASCII PLY; mm * 0.001 exactly once; unlit white material).
``prepare-video``     120 explicit allowlisted RGB PNGs -> one fragmented-MP4 mpeg4 derivative,
                      produced by the pinned local FFmpeg through pipes only (presentation 30 fps
                      fixed beforehand; FRAME_INDEX clock), then reopened and fully decoded.
``assemble``          Only with an isolated opening seed (separate module) AND a reviewed opening
                      placement proof; independently re-validates every prerequisite and writes
                      an UNADMITTED, UNREGISTERED bundle + clip proposal.

Later stages read ONLY the allowlist and its pinned payloads, never the resources index, broad
manifests or GT. This module has no GT argument; the opening seed lives in
``tools/workbench_hb_opening_seed.py``, which this module never imports.

Exit codes: 0 completed; 1 failure; 2 usage; 3 blocked / unsupported / missing prerequisite.
"""
from __future__ import annotations

import argparse
import binascii
import collections
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
import subprocess
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOL_VERSION = "3"
TOOL_ID = "visualizeit.workbench_hb_metric_inputs"
SCHEMA_ALLOWLIST = "workbench_hb_sha_allowlist_v1"
SCHEMA_MODEL = "workbench_hb_model_preparation_v1"
SCHEMA_VIDEO = "workbench_hb_video_preparation_v1"
SCHEMA_ASSEMBLY = "workbench_hb_metric_control_assembly_v1"
SCHEMA_FAILURE = "workbench_hb_metric_inputs_failure_v1"
SCHEMA_SEED = "workbench_hb_opening_seed_v1"                       # produced by the separate seed module
SCHEMA_SEED_RECEIPT = "workbench_hb_opening_seed_receipt_v1"
SCHEMA_SELECTION_REVIEW = "workbench_hb_opening_selection_review_v1"
SCHEMA_PLACEMENT_REVIEW = "workbench_hb_opening_placement_review_v1"
BUNDLE_SCHEMA = "workbench_metric_bundle_v1"
SEED_MODULE = "tools/workbench_hb_opening_seed.py"

CORPUS = "HB"
FAMILY = "hb_primesense_640x480"
DATASET_REVISION = "985ebaef7b1bd4bffb2039f18dd81faa647fd060"
RGB_ARCHIVE = "hb_val_primesense.zip"
MODEL_ARCHIVE = "hb_models.zip"
SCENE = "000002"
FRAME_COUNT = 120
OPENING = tuple(range(0, 10))
SEED_INDEX = 9
SCORING = tuple(range(10, 120))
MODEL_IDS = tuple(range(1, 11))
RGB_TOTALS = {"encoded_bytes": 46_130_618, "expanded_bytes": 110_835_240}   # fixed by the closed preflight
IMAGE_SIZE = (640, 480)
EXPECTED_K = (537.4799, 0.0, 318.8965, 0.0, 536.1447, 238.3781, 0.0, 0.0, 1.0)
K_ABS_TOL = 1e-3

# Trust anchors (historical local integrity; the caller must also supply each expected hash).
ANCHORS = {
    "acquisition_receipt": "064d3c2115f3def6291d5ca5f302cff7f3f19081af2d4f4fc57a7cbc710f4d47",
    "resources_index": "a05b65a617f2b4b5b9dc037c2c0510892380b6881e7052f0499c4c3464e78bac",
    "integrity_record": "0ef84395a48b841a7f723fdc27494b933b2f5f422e187f03cffa48cd8c1e7b5a",
}
INTEGRITY_MEMBERS = 2569
TRUST = {"trust_level": "historical_local_integrity", "upstream_authenticated": False,
         "expectations_established_before_current_conversion": True,
         "basis": ("parent-pinned SHA-256 of the acquisition receipt, the resources index and the prior "
                   "integrity record; docs/prototype.md records the historical (Oct 11) verification of 2569 "
                   "members by compressed/expanded SHA-256 and CRC-32. Local integrity only; not independently "
                   "upstream-authenticated."),
         "resources_index_role": "metadata index of ZIP members (storage provenance), not poses",
         "allowlist_derivation": ("the caller's expected allowlist hash must identify a parent-accepted filter "
                                  "output; re-validation alone does not prove derivation from the index")}
BOP_FORMAT_URL = "https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md"
MM_TO_M = 0.001

# Pinned local CPU FFmpeg (no installation; exact bytes verified before every use).
FFMPEG = {"path": ".cache/quality-windows/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe",
          "bytes": 64_654_336, "sha256": "404fdd541eecc577d8f24bf9a976f1e4209f4292022a41c01f5cc2f1b0d375ac",
          "version": "4.2.2", "build": "gcc 9.2.1 (GCC) 20200122",
          "version_source": "parent-verified pin bound by the executable SHA-256 (not re-executed)"}
PRESENTATION_FPS = 30
FFMPEG_ARGS = ("-hide_banner", "-loglevel", "error", "-protocol_whitelist", "pipe", "-threads", "1",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
               "-framerate", str(PRESENTATION_FPS), "-i", "pipe:0",
               "-an", "-sn", "-dn", "-map_metadata", "-1",
               "-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "5", "-g", "30", "-bf", "0", "-threads", "1",
               "-flags:v", "+bitexact", "-fflags", "+bitexact", "-protocol_whitelist", "pipe",
               "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1")
VIDEO_PROFILE = {"encoder": "mpeg4 (software)", "input": "rawvideo rgb24 640x480 on stdin",
                 "encoder_pixel_format": "yuv420p", "quality": "q:v 5", "gop": 30, "b_frames": 0, "threads": 1,
                 "container": "fragmented MP4 (frag_keyframe+empty_moov+default_base_moof) on stdout",
                 "protocols": "pipe only", "hardware": "none", "network": "none",
                 "presentation_fps": PRESENTATION_FPS,
                 "presentation_fps_kind": "synthetic presentation rate; NOT a physical capture rate",
                 "lossy": True, "external_codec_installed": False, "argv_without_executable": list(FFMPEG_ARGS)}
FFMPEG_DEADLINE_S = 900
STDERR_TAIL = 8 << 10
PIPE_CHUNK = 64 << 10
PREFIX_NOTE = "neither container duration nor PTS attests a 10-second prefix"
DECODE_NOTE = "lossy derivative: decoded pixels are distinct from the source PNG pixels"
VIDEO_CLOCK = {"kind": "FRAME_INDEX", "physical_fps": None, "physical_timestamps": None,
               "presentation_fps": PRESENTATION_FPS, "prefix_attestation": PREFIX_NOTE}
VIDEO_GEOMETRY = {"width": 640, "height": 480, "crop": None, "rotation": None, "K_changed": False}

CHUNK = 1 << 20
MAX_ANCHOR_BYTES = {"acquisition_receipt": 64 << 10, "resources_index": 8 << 20, "integrity_record": 64 << 10}
MAX_RGB_EXPANDED = 4 << 20
MAX_MODEL_EXPANDED = 64 << 20
MAX_JSON_OUTPUT = 512 << 10           # read bound for any JSON input/output
MAX_REVIEW_BYTES = 256 << 10
MAX_CALIBRATION_BYTES = 256 << 10
VIDEO_CAP = 6 << 20
GLB_CAP = 5_767_168                   # 5.5 MiB
METADATA_AGGREGATE_CAP = 512 << 10    # all JSON across every preparation stage
FINAL_OUTPUTS_CAP = 12 << 20          # all preparation outputs (GLB + video + metadata)
FAILURE_RESERVE = 16 << 10            # reserved inside each stage's metadata budget for failure.json
STAGE_CAPS = {                        # per-stage share of the aggregate caps (sums: 12 MiB / 0.5 MiB)
    "filter-allowlist": {"total": 192 << 10, "metadata": 192 << 10},
    "prepare-model": {"total": GLB_CAP + (32 << 10), "metadata": 32 << 10},
    "prepare-video": {"total": VIDEO_CAP + (160 << 10), "metadata": 160 << 10},
    "extract-seed": {"total": 32 << 10, "metadata": 32 << 10},
    "assemble": {"total": 96 << 10, "metadata": 96 << 10},
}
assert sum(c["metadata"] for c in STAGE_CAPS.values()) == METADATA_AGGREGATE_CAP
assert sum(c["total"] for c in STAGE_CAPS.values()) == FINAL_OUTPUTS_CAP
DIR_ENTRY_RESERVE = 64 << 10
STAGE_ALLOWANCE = {k: 2 * v["total"] + DIR_ENTRY_RESERVE for k, v in STAGE_CAPS.items()}  # staging + published
FACE_CAP = 200_000                    # execution cap (actual triangles)
PORTABLE_TRIANGLES = 20_000           # portability target, reported separately
PORTABLE_BYTES = 20_000_000
MAX_VERTICES = 2_000_000
HEADER_CAP = 64 << 10
DERIVED_LIMIT = 1 << 30
FREE_RESERVE = 1 << 30
NEW_INPUT_LIMIT = 250_000_000
FORBIDDEN_MODULES = frozenset({"torch", "torchvision", "sam2", "segment_anything", "OpenGL", "pyrender", "trimesh",
                               "open3d", "pytorch3d", "nvdiffrast", "kaolin", "pycolmap"})
FORBIDDEN_FULL_MODULES = frozenset({"bench.quality_gotrack", "bench.quality_gotrack_inference", "bench.workbench_render",
                                    "tools.workbench_hb_opening_seed"})
FORBIDDEN_EVENTS = frozenset({"subprocess.Popen", "_winapi.CreateProcess", "os.system", "os.exec", "os.posix_spawn",
                              "os.spawn", "os.startfile", "socket.connect", "socket.getaddrinfo",
                              "socket.gethostbyname", "socket.gethostbyaddr", "socket.bind", "urllib.Request",
                              "http.client.connect", "webbrowser.open"})
MUTATING_EVENTS = frozenset({"os.remove", "os.rename", "os.link", "os.symlink", "os.truncate", "os.chmod",
                             "os.rmdir", "os.mkdir", "os.utime", "os.chown", "shutil.copyfile", "shutil.copymode",
                             "shutil.copystat", "shutil.copytree", "shutil.move", "shutil.rmtree"})
DENY_COMPONENTS = frozenset({"mask", "masks", "mask_visib", "depth", "checkpoints", "models_eval",
                             "hb-scene-metadata-v1", "hb-results-v1", "hb-results-v2", "results-v1"})
DENY_NAMES = frozenset({"scene_gt.json", "scene_gt_info.json", "scene_camera.json", "evaluator.json",
                        "inference.json", "selection.json", "results.jsonl", "resources.json", "models_info.json",
                        "acquisition-receipt.json", "acquisition-plan.json", "hb-integrity-v1.json"})
DENY_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors", ".onnx"})
CODE_SUFFIXES = frozenset({".py", ".pyc", ".pyd", ".so", ".dll"})
HELPERS = {"bench": "bench/__init__.py", "bench.vision": "bench/vision.py",
           "bench.workbench_geometry": "bench/workbench_geometry.py",
           "bench.workbench_contract": "bench/workbench_contract.py",
           "bench.workbench_neural": "bench/workbench_neural.py"}
FLAGS = {"inputs_only": True, "neural_executed": False, "gpu_used": False, "admitted": False, "registered": False,
         "automatic_geometry": False, "gt_read_by_this_module": False, "mobile_verified": False,
         "head_sha256": "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311",
         "family": FAMILY, "renamed_to_ycb": False, "inherited_admission": None}
APPEARANCE = "vertex_colours_unlit_white_material"


class InputsError(Exception):
    """Failure (exit 1)."""


class UsageError(InputsError):
    """Usage error (exit 2)."""


class Blocked(InputsError):
    """Missing prerequisite / unsupported / over a cap (exit 3)."""


class GuardError(InputsError):
    """A forbidden import, read, write, process or network action inside a stage."""


@dataclass(frozen=True)
class Context:
    """Explicit roots so manufactured fixtures never touch the real workspace. ``preparation`` is
    the ONE explicit preparation scope (inside ``derived``) that every stage of one control writes
    into and that the 12 MiB / 0.5 MiB aggregate accounting covers; it defaults to ``derived``."""
    root: Path = ROOT
    workspace: Path = ROOT / ".cache/video-workbench"
    preparation: Path | None = None

    @property
    def derived(self) -> Path:
        return self.workspace / "derived"

    @property
    def scope(self) -> Path:
        return Path(os.path.abspath(self.preparation if self.preparation is not None else self.derived))


def preparation_scope(ctx: Context) -> Path:
    """The validated preparation scope: ``derived`` itself or a directory strictly inside it, with no
    symlink/junction on its path."""
    scope, derived = ctx.scope, Path(os.path.abspath(ctx.derived))
    if scope != derived and derived not in scope.parents:
        raise UsageError(f"--preparation-root must be inside {derived}")
    if ".." in scope.parts or denied_path(scope):
        raise UsageError("--preparation-root is not a safe output scope")
    root = Path(os.path.abspath(ctx.root))
    if root in scope.parents:
        existing = scope
        while not existing.exists() and existing != root:
            existing = existing.parent
        _no_links(root, existing)
    return scope


# --------------------------------------------------------------------------
# guard (process-wide audit hook; active only inside a stage)

_GUARD: dict[str, Any] = {"active": False, "roots": (), "payload_root": None, "allowed": frozenset(),
                          "exceptions": frozenset(), "opened": [], "grant": None, "create_pending": False}
_HOOK_INSTALLED = False


def _norm(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


def denied_path(path: str | os.PathLike) -> bool:
    """GT / label / mask / depth / evaluator / index / checkpoint paths (any mode)."""
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


def _write_mode(args: tuple) -> bool:
    if len(args) > 1 and isinstance(args[1], str):
        return any(c in args[1] for c in "wax+")
    if len(args) > 2 and isinstance(args[2], int):
        return bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    return False


def _audit(event: str, args: tuple) -> None:
    g = _GUARD
    if not g["active"]:
        return
    if event == "subprocess.Popen":
        grant, g["grant"] = g["grant"], None                       # single use, consumed on ANY attempt
        argv = args[1] if len(args) > 1 else None
        if isinstance(argv, (list, tuple)):
            argv_ok = grant is not None and tuple(argv) == grant[1]
        else:                                                      # Windows audits list2cmdline(argv)
            argv_ok = grant is not None and isinstance(argv, str) and argv == subprocess.list2cmdline(grant[1])
        if grant is None or not args or args[0] is None or _norm(args[0]) != grant[0] or not argv_ok:
            raise GuardError("process creation outside the exact single-use FFmpeg grant")
        g["create_pending"] = True
        return
    if event == "_winapi.CreateProcess":
        if not g["create_pending"]:
            raise GuardError("CreateProcess without a granted Popen")
        g["create_pending"] = False
        return
    if event in FORBIDDEN_EVENTS:
        raise GuardError(f"forbidden action inside an input-preparation stage: {event}")
    if event == "import" and args and isinstance(args[0], str):
        if args[0].split(".")[0] in FORBIDDEN_MODULES or args[0] in FORBIDDEN_FULL_MODULES:
            raise GuardError(f"forbidden import inside an input-preparation stage: {args[0]}")
        return
    if event in MUTATING_EVENTS:
        for value in args[:2]:
            if isinstance(value, (str, bytes, os.PathLike)):
                path = _norm(value)
                if path in g["exceptions"]:
                    raise GuardError(f"read exceptions are read-only: {event} {path}")
                if _is_payload(path, g) or denied_path(path):
                    raise GuardError(f"forbidden mutation of a payload/GT/index path: {event} {path}")
        return
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = _norm(args[0])
        if path in g["exceptions"]:
            if _write_mode(args):
                raise GuardError(f"read exceptions are read-only: {path}")
            return
        if _is_payload(path, g):
            if path not in g["allowed"] or _write_mode(args):
                raise GuardError(f"payload not selected for this stage (or opened for writing): {path}")
            g["opened"].append(path)
            return
        if any(path == r or path.startswith(r + os.sep) for r in g["roots"]) and denied_path(path):
            raise GuardError(f"forbidden GT/label/index/model path inside an input-preparation stage: {path}")


def _preimport() -> None:
    from bench import workbench_geometry, workbench_contract, workbench_neural  # noqa: F401  numpy-only
    from bench.vision import cv2  # noqa: F401  CPU OpenCV
    platform.platform()          # warm the stdlib cache BEFORE the guard (Windows spawns `ver` on first call)


@contextmanager
def guarded_stage(ctx: Context, *, read_exceptions: Iterable[Path] = ()) -> Iterator[list]:
    """Whole-stage guard with an EMPTY payload allowlist. ``read_exceptions`` are exact READ-ONLY
    paths (the filter passes its three metadata anchors; no preparation stage passes any). The
    guard is reset on every exit path, including exceptions."""
    global _HOOK_INSTALLED
    if _GUARD["active"]:
        raise GuardError("an input-preparation guard is already active; nested stages are refused")
    _preimport()
    present = sorted(m for m in FORBIDDEN_MODULES if m in sys.modules)
    if present:
        raise GuardError(f"forbidden modules already imported: {present}")
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    opened: list[str] = []
    _GUARD.update(active=True, roots=tuple(sorted({_norm(ctx.root), _norm(ROOT)})), payload_root=None,
                  allowed=frozenset(), exceptions=frozenset(_norm(p) for p in read_exceptions), opened=opened,
                  grant=None, create_pending=False)
    try:
        yield opened
    finally:
        _GUARD.update(active=False, roots=(), payload_root=None, allowed=frozenset(), exceptions=frozenset(),
                      opened=[], grant=None, create_pending=False)
    present = sorted(m for m in FORBIDDEN_MODULES if m in sys.modules)
    if present:
        raise GuardError(f"forbidden modules imported during the stage: {present}")


def allow_payloads(payload_root: Path, paths: Iterable[Path]) -> None:
    """Widen the active guard to exact allowlisted payloads (only after trust + budget checks)."""
    if not _GUARD["active"]:
        raise GuardError("payloads can only be allowed inside an active guard")
    root = _norm(payload_root)
    if _GUARD["payload_root"] not in (None, root):
        raise GuardError("payload root differs within one stage")
    allowed = {_norm(p) for p in paths}
    if any(not p.startswith(root + os.sep) for p in allowed):
        raise GuardError("allowed payload lies outside the payload root")
    _GUARD.update(payload_root=root, allowed=_GUARD["allowed"] | frozenset(allowed))


def _set_payload_root(payload_root: Path) -> None:
    """Declare the payload root with ZERO allowed payloads (any payload open then fails)."""
    allow_payloads(payload_root, ())


def add_read_exception(path: Path) -> None:
    """Widen the ACTIVE guard by one exact READ-ONLY path. No stage of this module calls it; the
    separate opening-seed module uses it for its single pinned file after all its prerequisites."""
    if not _GUARD["active"]:
        raise GuardError("read exceptions can only be added inside an active guard")
    _GUARD.update(exceptions=_GUARD["exceptions"] | frozenset({_norm(path)}))


def _grant_subprocess(executable: Path, argv: Sequence[str]) -> None:
    """One exact executable + argv; consumed by the next process-creation attempt."""
    if not _GUARD["active"]:
        raise GuardError("subprocess grants exist only inside an active guard")
    _GUARD.update(grant=(_norm(executable), tuple(argv)), create_pending=False)


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
    """Strict JSON: UTF-8; duplicate keys, NaN/Infinity tokens, numeric float overflow (e.g. 1e400)
    and unbounded integers are rejected wherever they occur."""
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise InputsError(f"{what}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def reject(token):
        raise InputsError(f"{what}: non-finite JSON value {token}")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise InputsError(f"{what}: numeric value {text[:40]} overflows or is non-finite")
        return value

    def bounded_int(text):
        if len(text) > 64:
            raise InputsError(f"{what}: integer literal too long")
        return int(text)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject,
                          parse_float=finite_float, parse_int=bounded_int)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise InputsError(f"{what}: invalid JSON ({exc})") from None


def dump_json(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def check_sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise UsageError(f"{what} must be 64 lowercase hex characters")
    return value


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_count(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _is_link(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def _no_links(base: Path, path: Path) -> None:
    current = base
    for part in path.relative_to(base).parts:
        current = current / part
        if _is_link(current):
            raise InputsError(f"path traverses a symlink/junction: {current}")


def contained_input(ctx: Context, value: str | os.PathLike, what: str, *, allow_denied: bool = False) -> Path:
    """Existing regular non-link file inside the repository root (no URI, stream, traversal)."""
    text = os.fspath(value)
    if not text or "://" in text or text.lower().startswith("file:") or "\x00" in text:
        raise UsageError(f"{what}: not a local path")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = ctx.root / candidate
    if ".." in candidate.parts:
        raise UsageError(f"{what}: traversal is not allowed")
    candidate = Path(os.path.abspath(candidate))
    if ":" in candidate.name:
        raise UsageError(f"{what}: alternate data streams are not allowed")
    if not allow_denied and denied_path(candidate):
        raise UsageError(f"{what}: forbidden GT/label/index/model path {candidate.name}")
    root = Path(os.path.abspath(ctx.root))
    if root not in candidate.parents:
        raise UsageError(f"{what}: must be inside {root}")
    _no_links(root, candidate)
    try:
        st = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise InputsError(f"{what}: missing ({candidate})") from exc
    if not stat.S_ISREG(st.st_mode):
        raise InputsError(f"{what}: not a regular file")
    real = os.path.normcase(os.path.realpath(candidate))
    if real != os.path.normcase(str(candidate)) and os.path.normcase(os.path.realpath(root)) not in Path(real).parents:
        raise InputsError(f"{what}: resolves outside the repository")
    return candidate


def repo_rel(ctx: Context, path: Path) -> str:
    return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(ctx.root))).as_posix()


def read_bounded(path: Path, limit: int, what: str) -> bytes:
    if os.lstat(path).st_size > limit:
        raise InputsError(f"{what}: exceeds {limit} bytes")
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise InputsError(f"{what}: grew beyond {limit} bytes")
    return data


def read_pinned(ctx: Context, value: str | os.PathLike, expected_sha256: str, limit: int, what: str, *,
                allow_denied: bool = False) -> tuple[bytes, Path]:
    """Hash-verify BEFORE parsing: the bytes are returned only when they match the caller's expectation."""
    expected_sha256 = check_sha(expected_sha256, f"expected {what} SHA-256")
    path = contained_input(ctx, value, what, allow_denied=allow_denied)
    raw = read_bounded(path, limit, what)
    if sha256_bytes(raw) != expected_sha256:
        raise InputsError(f"{what}: SHA-256 differs from the caller's expected value")
    return raw, path


def safe_member_path(base: Path, relative: Any, what: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or relative.startswith("/"):
        raise InputsError(f"{what}: unsafe location {relative!r}")
    if any(p in ("", ".", "..") for p in relative.split("/")):
        raise InputsError(f"{what}: unsafe location {relative!r}")
    path = Path(os.path.abspath(base.joinpath(*PurePosixPath(relative).parts)))
    if Path(os.path.abspath(base)) not in path.parents:
        raise InputsError(f"{what}: location escapes its root")
    _no_links(Path(os.path.abspath(base)), path)
    return path


# --------------------------------------------------------------------------
# closures and budget


def source_closure() -> dict[str, str]:
    import importlib
    out = {}
    for name, rel in HELPERS.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (ROOT / rel).resolve():
            raise InputsError(f"helper {name} imported from {module.__file__}, not {rel}")
        out[rel] = sha256_file(ROOT / rel)
    out[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha256_file(Path(__file__))
    return dict(sorted(out.items()))


def runtime_identity() -> dict:
    from bench.vision import cv2
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "opencv": cv2.__version__,
            "opencv_module": str(Path(cv2.__file__).resolve()),
            "opencv_build_information_sha256": sha256_bytes(cv2.getBuildInformation().encode("utf-8")),
            "dont_write_bytecode": bool(sys.dont_write_bytecode)}


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return 0
    for base, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not _is_link(Path(base) / d)]
        for name in files:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


def workspace_budget(ctx: Context, planned: int) -> dict:
    """Stat-only: derived+scratch+runs+masks+suite + planned <= 1 GiB, and >= 1 GiB free after it."""
    if not ctx.workspace.is_dir():
        raise InputsError(f"workspace {ctx.workspace} does not exist")
    used = sum(_tree_bytes(ctx.workspace / d) for d in ("derived", "scratch", "runs", "masks", "suite"))
    free = shutil.disk_usage(ctx.workspace).free
    if used + planned > DERIVED_LIMIT:
        raise Blocked(f"derived budget would be exceeded ({used} + {planned} > {DERIVED_LIMIT})")
    if free - planned < FREE_RESERVE:
        raise Blocked(f"free-disk reserve would be violated ({free} - {planned} < {FREE_RESERVE})")
    return {"derived_bytes": used, "planned_bytes": planned, "free_bytes": free,
            "derived_limit": DERIVED_LIMIT, "free_reserve": FREE_RESERVE,
            "method": "lstat sizes; links not followed; logical bytes (no hard-link deduction)"}


def input_budget(existing_input_bytes: Any, candidate_bytes: int, new_derived_bytes: int) -> dict:
    """Parent-declared existing inputs + allowlisted candidate payloads + new derived outputs <= 250 MB."""
    if type(existing_input_bytes) is not int or existing_input_bytes < 0:
        raise UsageError("--existing-input-bytes must be a parent-declared integer >= 0 (never estimated here)")
    total = existing_input_bytes + candidate_bytes + new_derived_bytes
    if total > NEW_INPUT_LIMIT:
        raise Blocked(f"inputs {existing_input_bytes} + {candidate_bytes} + new derived {new_derived_bytes} "
                      f"exceed {NEW_INPUT_LIMIT}")
    return {"existing_input_bytes_parent_declared": existing_input_bytes, "candidate_payload_bytes": candidate_bytes,
            "new_derived_bytes_counted": new_derived_bytes, "total": total, "limit": NEW_INPUT_LIMIT,
            "placement": "payloads referenced in place; never copied"}


# --------------------------------------------------------------------------
# outputs (fresh directory, ledgered caps, exclusive publication, receipt last)


def check_output_dir(ctx: Context, out: str | os.PathLike, what: str = "--out") -> Path:
    text = os.fspath(out)
    if "://" in text or text.lower().startswith("file:") or "\x00" in text:
        raise UsageError(f"{what}: not a local path")
    path = Path(text)
    if not path.is_absolute():
        path = ctx.root / path
    if ".." in path.parts:
        raise UsageError(f"{what}: traversal is not allowed")
    path = Path(os.path.abspath(path))
    if denied_path(path) or path.name.lower() == "payloads":
        raise UsageError(f"{what}: output names may not resemble GT/label/index/payload paths")
    scope = preparation_scope(ctx)
    if scope not in path.parents:
        raise UsageError(f"{what}: must be a NEW directory inside the preparation scope {scope}")
    _no_links(Path(os.path.abspath(ctx.root)), path.parent if path.parent.exists() else scope if scope.exists()
              else Path(os.path.abspath(ctx.derived)))
    if os.path.lexists(path):
        raise UsageError(f"{what}: {path} already exists; outputs are never overwritten")
    return path


def outside_scope_input(ctx: Context, path: Path, what: str) -> Path:
    """Source inputs (anchors, reviews, calibrations, scene annotation) stay in the INPUT budget:
    they may not live inside the preparation scope, where they would be counted as outputs."""
    scope = ctx.scope
    if Path(os.path.abspath(path)) == scope or scope in Path(os.path.abspath(path)).parents:
        raise UsageError(f"{what}: source inputs must live outside the preparation scope {scope}")
    return path


def inside_scope(ctx: Context, path: Path, what: str) -> Path:
    """Prior-stage OUTPUTS must lie inside the same preparation scope (no detached attempts)."""
    if ctx.scope not in Path(os.path.abspath(path)).parents:
        raise Blocked(f"{what} lies outside the bound preparation scope {ctx.scope}")
    return path


def _make_fresh_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(path)                      # fails if it appeared meanwhile


def scope_usage(scope: Path) -> dict:
    """Every regular file now present in the preparation scope (retained failed/partial siblings,
    failure metadata, staging partials and published outputs): lstat logical bytes, links never
    followed (a link inside the scope is refused)."""
    total = metadata = files = 0
    if not os.path.lexists(scope):
        return {"total": 0, "metadata": 0, "files": 0}
    stack = [Path(scope)]
    while stack:
        current = stack.pop()
        if _is_link(current):
            raise Blocked(f"link inside the preparation scope: {current}")
        st = os.lstat(current)
        if stat.S_ISDIR(st.st_mode):
            with os.scandir(current) as entries:
                stack.extend(Path(e.path) for e in entries)
        elif stat.S_ISREG(st.st_mode):
            total += st.st_size
            files += 1
            if current.suffix.lower() == ".json":
                metadata += st.st_size
        else:
            raise Blocked(f"non-regular entry inside the preparation scope: {current}")
    return {"total": total, "metadata": metadata, "files": files}


class OutputLedger:
    """Aggregate accounting over the WHOLE preparation scope: every retained attempt, partial and
    failure record already on disk plus the proposed bytes must stay <= 12 MiB in total and
    <= 0.5 MiB of JSON metadata. Checked BEFORE every encoded disk write; normal writes keep
    FAILURE_RESERVE free so a failure record can still be published. The global 1 GiB derived
    budget and the 1 GiB free-disk reserve are separate (``workspace_budget``)."""

    def __init__(self, ctx: Context, out_dir: Path, stage: str):
        if stage not in STAGE_CAPS:
            raise InputsError(f"unknown stage {stage!r}")
        self.ctx, self.dir, self.stage = ctx, out_dir, stage
        self.scope = ctx.scope
        self.total = 0
        self.metadata = 0
        self.files: dict[str, int] = {}

    def check(self, nbytes: int, *, metadata: bool, failure: bool = False) -> dict:
        if type(nbytes) is not int or nbytes < 0:
            raise InputsError("invalid write size")
        usage = scope_usage(self.scope)
        reserve = 0 if failure else FAILURE_RESERVE
        if usage["total"] + nbytes > FINAL_OUTPUTS_CAP - reserve:
            raise Blocked(f"{self.stage}: preparation scope holds {usage['total']} bytes; +{nbytes} would exceed "
                          f"the {FINAL_OUTPUTS_CAP} byte aggregate (retained attempts included)")
        if metadata and usage["metadata"] + nbytes > METADATA_AGGREGATE_CAP - reserve:
            raise Blocked(f"{self.stage}: preparation scope holds {usage['metadata']} metadata bytes; +{nbytes} "
                          f"would exceed the {METADATA_AGGREGATE_CAP} byte aggregate metadata cap")
        if shutil.disk_usage(self.ctx.workspace).free - 2 * nbytes < FREE_RESERVE:
            raise Blocked("the 1 GiB free-disk reserve would be violated")
        return usage

    def record(self, name: str, nbytes: int, *, metadata: bool) -> None:
        self.files[name] = self.files.get(name, 0) + nbytes
        self.total += nbytes
        if metadata:
            self.metadata += nbytes

    def summary(self) -> dict:
        return {"stage": self.stage, "stage_written_bytes": self.total, "stage_metadata_bytes": self.metadata,
                "files": dict(self.files), "scope_usage_now": scope_usage(self.scope),
                "aggregate_caps": {"total": FINAL_OUTPUTS_CAP, "metadata": METADATA_AGGREGATE_CAP,
                                   "failure_reserve": FAILURE_RESERVE}}


def publish_bytes(ledger: OutputLedger, name: str, data: bytes, *, metadata: bool = True,
                  failure: bool = False) -> dict:
    """Cap + prospective growth check BEFORE writing; exclusive owned temp (fsync) -> verify ->
    os.link to a fresh name; the temp is removed."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise InputsError(f"invalid output name {name!r}")
    ledger.check(len(data), metadata=metadata, failure=failure)
    if not failure:
        workspace_budget(ledger.ctx, 2 * len(data))           # remaining growth: owned temp + published name
    final = ledger.dir / name
    if os.path.lexists(final):
        raise InputsError(f"{final} already exists")
    tmp = ledger.dir / f".{name}.tmp-{secrets.token_hex(6)}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(tmp) != sha256_bytes(data):
            raise InputsError(f"{name}: staged bytes do not verify")
        os.link(tmp, final)
        ledger.record(name, len(data), metadata=metadata)
        if sha256_file(final) != sha256_bytes(data):
            raise InputsError(f"{name}: published bytes do not verify")
    finally:
        if os.path.lexists(tmp):
            os.unlink(tmp)
    return {"name": name, "sha256": sha256_bytes(data), "bytes": len(data)}


def _failure(ledger: OutputLedger, stage: str, exc: BaseException, extra: Mapping | None = None) -> None:
    retained = {}
    if ledger.dir.is_dir():
        for p in ledger.dir.iterdir():
            try:
                retained[p.name] = os.lstat(p).st_size
            except OSError:
                pass
    try:
        ledger_summary = ledger.summary()
    except Exception as summary_exc:  # noqa: BLE001 - never mask the original failure
        ledger_summary = {"error": f"{type(summary_exc).__name__}: {str(summary_exc)[:200]}"}
    body = {"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage, "status": "failed",
            "reason": f"{type(exc).__name__}: {str(exc)[:1500]}", "completed_receipt_published": False,
            "retained_files": retained, "ledger": ledger_summary, **(dict(extra) if extra else {})}
    data = dump_json(body)
    if len(data) > FAILURE_RESERVE:
        body["retained_files"] = {"count": len(retained), "bytes": sum(retained.values())}
        data = dump_json(body)[:FAILURE_RESERVE]
    try:
        publish_bytes(ledger, "failure.json", data, failure=True)
    except Exception:  # noqa: BLE001 - keep a terminal record
        sys.stderr.write(data.decode("ascii", "replace"))


# --------------------------------------------------------------------------
# bounded ZIP-member payload stream (raw DEFLATE method 8 or stored method 0)


def verified_payload_stream(path: Path, row: Mapping, sink: Callable[[bytes], None] | None = None) -> dict:
    """Consume one allowlisted payload with bounded chunks; check exact compressed size + SHA-256,
    exact expansion (no over-expansion, truncation, trailing bytes or padding), CRC-32 and expanded
    SHA-256 against the allowlist. ``sink`` receives expanded chunks; nothing is written to disk."""
    method, size, compressed = row["method"], row["expanded_bytes"], row["compressed_bytes"]
    if method not in (0, 8) or not _is_count(size) or not _is_count(compressed):
        raise InputsError(f"payload {Path(path).name}: unsupported method or malformed sizes")
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_size != compressed:
        raise InputsError(f"payload {path.name}: not a regular file of the allowlisted compressed size")
    chash, uhash, crc, count, consumed = hashlib.sha256(), hashlib.sha256(), 0, 0, 0
    decoder = zlib.decompressobj(-15) if method == 8 else None

    def emit(expanded: bytes) -> None:
        nonlocal count, crc
        if count + len(expanded) > size:
            raise InputsError(f"payload {path.name}: expansion exceeds the allowlisted size")
        count += len(expanded)
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
                raise InputsError(f"payload {path.name}: longer than its allowlisted compressed size")
            chash.update(chunk)
            pending = chunk
            while pending:
                if decoder is None:
                    expanded, pending = pending, b""
                else:
                    if decoder.eof:
                        raise InputsError(f"payload {path.name}: trailing bytes after the DEFLATE stream")
                    expanded = decoder.decompress(pending, min(CHUNK, size - count + 1))
                    pending = decoder.unconsumed_tail
                    if decoder.unused_data:
                        raise InputsError(f"payload {path.name}: trailing bytes after the DEFLATE stream")
                emit(expanded)
    if decoder is not None:
        while not decoder.eof:
            tail = decoder.decompress(b"", min(CHUNK, size - count + 1))
            if not tail:
                break
            emit(tail)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise InputsError(f"payload {path.name}: truncated or trailing DEFLATE stream")
    if (consumed != compressed or count != size or (crc & 0xFFFFFFFF) != row["crc32"]
            or chash.hexdigest() != row["compressed_sha256"] or uhash.hexdigest() != row["uncompressed_sha256"]):
        raise InputsError(f"payload {path.name}: size/CRC-32/SHA-256 differs from the allowlist")
    return {"compressed_sha256": chash.hexdigest(), "uncompressed_sha256": uhash.hexdigest(),
            "crc32": crc & 0xFFFFFFFF, "compressed_bytes": consumed, "expanded_bytes": count, "verified": True}


# --------------------------------------------------------------------------
# stage 1: metadata-only SHA allowlist filter + full allowlist re-validation


ROW_KEYS = frozenset({"member", "archive", "dataset_revision", "method", "flags", "source_offset", "compressed_bytes",
                      "expanded_bytes", "crc32", "compressed_sha256", "uncompressed_sha256", "location_kind",
                      "payload_path", "roles"})
RGB_ROW_KEYS = ROW_KEYS | {"index", "frame_id", "split"}
MODEL_ROW_KEYS = ROW_KEYS | {"object_id", "identity_note"}
ALLOWLIST_KEYS = frozenset({"schema", "tool_version", "trust", "flags", "anchors", "source", "source_seals",
                            "frozen_split", "rgb", "rgb_totals", "models", "excluded", "sources", "runtime", "budget",
                            "ledger", "payload_opens", "closure", "preparation_scope"})
SOURCE_KEYS = frozenset({"corpus", "dataset_revision", "scene", "rgb_archive", "model_archive", "selection_digest",
                         "payload_root"})
FROZEN_SPLIT = {"opening": list(OPENING), "seed_index": SEED_INDEX, "scoring": list(SCORING),
                "fixed_before_viewing": True}
IDENTITY_NOTE = "CAD identity by member name + pinned hashes; NOT a visible-object binding"
EXCLUDED_NOTE = ("every other index row (depth, masks, scene metadata, GT, other scenes, unselected models) is "
                 "excluded and never opened or copied")


def _member_token(archive: str, member: str) -> str:
    return hashlib.sha256(f"{archive}:{member}".encode("utf-8")).hexdigest()


def _split(index: int) -> str:
    return "opening_seed" if index == SEED_INDEX else "opening" if index in OPENING else "scoring"


def validate_index_row(row: Any, member: str, archive: str, role: str, max_expanded: int) -> dict:
    """Pure check of one resources-index row; returns the allowlist row (no file access)."""
    if not isinstance(row, Mapping):
        raise InputsError(f"{member}: index row missing")
    ints = ("sourceMethod", "sourceFlags", "sourceCompressedBytes", "expandedBytes", "crc32", "method",
            "compressedBytes", "sourceOffset")
    if any(type(row.get(k)) is not int or row.get(k) < 0 for k in ints):
        raise InputsError(f"{member}: ZIP metadata must be non-negative integers (booleans are invalid)")
    location = row.get("location")
    if (row.get("datasetRevision") != DATASET_REVISION or row.get("archive") != archive
            or row.get("sourceMember") != member or not isinstance(location, Mapping)
            or location.get("kind") != "payload" or row.get("allowedRoles") != [role]):
        raise InputsError(f"{member}: revision/archive/member/location/role differ from the expected source")
    if (row["method"] not in (0, 8) or row["method"] != row["sourceMethod"]
            or row["compressedBytes"] != row["sourceCompressedBytes"] or row["sourceFlags"] & ~2048
            or row["crc32"] > 0xFFFFFFFF or row["expandedBytes"] > max_expanded or row["expandedBytes"] == 0
            or row["compressedBytes"] == 0 or row["compressedBytes"] > max_expanded + (1 << 20)
            or (row["method"] == 0 and row["compressedBytes"] != row["expandedBytes"])):
        raise InputsError(f"{member}: unsupported or inconsistent ZIP method/flags/sizes")
    for key in ("compressedSHA256", "uncompressedSHA256"):
        if not _is_sha(row.get(key)):
            raise InputsError(f"{member}: {key} missing or malformed")
    if location.get("path") != f"payloads/{_member_token(archive, member)}.bin":
        raise InputsError(f"{member}: payload location is not the expected member token")
    return {"member": member, "archive": archive, "dataset_revision": DATASET_REVISION,
            "method": row["method"], "flags": row["sourceFlags"], "source_offset": row["sourceOffset"],
            "compressed_bytes": row["compressedBytes"], "expanded_bytes": row["expandedBytes"], "crc32": row["crc32"],
            "compressed_sha256": row["compressedSHA256"], "uncompressed_sha256": row["uncompressedSHA256"],
            "location_kind": "payload", "payload_path": location["path"], "roles": [role]}


def _revalidate_row(row: Any, member: str, archive: str, role: str, max_expanded: int, keys: frozenset) -> dict:
    """An allowlist row must be EXACTLY what validate_index_row would produce (no missing, unknown or
    conflicting fields; exact types)."""
    if not isinstance(row, Mapping) or set(row) != keys:
        raise InputsError(f"{member}: allowlist row has missing or unknown fields")
    index_form = {"datasetRevision": row["dataset_revision"], "archive": row["archive"], "sourceMember": row["member"],
                  "sourceMethod": row["method"], "sourceFlags": row["flags"], "sourceOffset": row["source_offset"],
                  "sourceCompressedBytes": row["compressed_bytes"], "expandedBytes": row["expanded_bytes"],
                  "crc32": row["crc32"], "method": row["method"], "compressedBytes": row["compressed_bytes"],
                  "allowedRoles": row["roles"], "location": {"kind": row["location_kind"], "path": row["payload_path"]},
                  "compressedSHA256": row["compressed_sha256"], "uncompressedSHA256": row["uncompressed_sha256"]}
    rebuilt = validate_index_row(index_form, member, archive, role, max_expanded)
    if any(type(rebuilt[k]) is not type(row[k]) or rebuilt[k] != row[k] for k in rebuilt):
        raise InputsError(f"{member}: allowlist row conflicts with its own revalidation")
    return rebuilt


def validate_allowlist_doc(doc: Any) -> dict:
    """Full pure re-validation of a filter output (run BEFORE any payload permission)."""
    if not isinstance(doc, dict) or set(doc) != ALLOWLIST_KEYS:
        raise InputsError("allowlist has missing or unknown top-level fields")
    if doc["schema"] != SCHEMA_ALLOWLIST or doc["tool_version"] != TOOL_VERSION:
        raise InputsError("allowlist schema/version differs")
    if doc["trust"] != TRUST or doc["flags"] != FLAGS or doc["frozen_split"] != FROZEN_SPLIT \
            or doc["excluded"] != EXCLUDED_NOTE or doc["payload_opens"] != 0 or type(doc["payload_opens"]) is not int:
        raise InputsError("allowlist trust/flags/split/exclusion/open-count differ")
    anchors = doc["anchors"]
    if not isinstance(anchors, Mapping) or set(anchors) != set(ANCHORS):
        raise InputsError("allowlist must carry exactly the three pinned historical anchors")
    for key, expected in ANCHORS.items():
        a = anchors[key]
        if (not isinstance(a, Mapping) or set(a) != {"path", "sha256", "bytes", "verified_before_parsing"}
                or a["sha256"] != expected or not isinstance(a["path"], str) or not _is_count(a["bytes"], 1)
                or a["verified_before_parsing"] is not True):
            raise InputsError(f"allowlist anchor {key} is not the pinned, verified-before-parsing record")
    src = doc["source"]
    if (not isinstance(src, Mapping) or set(src) != SOURCE_KEYS or src["corpus"] != CORPUS
            or src["dataset_revision"] != DATASET_REVISION or src["scene"] != SCENE or src["rgb_archive"] != RGB_ARCHIVE
            or src["model_archive"] != MODEL_ARCHIVE or not _is_sha(src["selection_digest"])
            or not isinstance(src["payload_root"], str)):
        raise InputsError("allowlist source identity differs")
    if PurePosixPath(src["payload_root"]) != PurePosixPath(anchors["resources_index"]["path"]).parent / "payloads":
        raise InputsError("allowlist payload root is not beside the anchored resources index")
    seals = doc["source_seals"]
    acq = seals.get("acquisition_source_hashes") if isinstance(seals, Mapping) else None
    if (not isinstance(seals, Mapping) or set(seals) != {"acquisition_source_hashes", "index_input_hashes"}
            or not isinstance(acq, Mapping) or not acq or not all(isinstance(k, str) and _is_sha(v) for k, v in acq.items())
            or not (seals["index_input_hashes"] is None or (isinstance(seals["index_input_hashes"], Mapping) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in seals["index_input_hashes"].items())))):
        raise InputsError("allowlist source seals are missing or malformed")
    rgb = doc["rgb"]
    if not isinstance(rgb, list) or len(rgb) != FRAME_COUNT:
        raise InputsError("allowlist must hold exactly 120 RGB rows")
    for i, row in enumerate(rgb):
        member = f"val_primesense/{SCENE}/rgb/{i:06d}.png"
        _revalidate_row(row, member, RGB_ARCHIVE, "rgb", MAX_RGB_EXPANDED, RGB_ROW_KEYS)
        if type(row["index"]) is not int or row["index"] != i or row["frame_id"] != f"s{i:06d}" or row["split"] != _split(i):
            raise InputsError(f"allowlist RGB row {i} index/frame/split differ")
    totals = {"encoded_bytes": sum(r["compressed_bytes"] for r in rgb), "expanded_bytes": sum(r["expanded_bytes"] for r in rgb)}
    if totals != RGB_TOTALS or doc["rgb_totals"] != totals:
        raise InputsError(f"allowlist RGB totals {totals} differ from the expected {RGB_TOTALS}")
    models = doc["models"]
    if not isinstance(models, list) or not models:
        raise InputsError("allowlist must hold at least one explicit model row")
    ids = []
    for row in models:
        if not isinstance(row, Mapping) or type(row.get("object_id")) is not int or row["object_id"] not in MODEL_IDS:
            raise InputsError("allowlist model row has an invalid object id")
        ids.append(row["object_id"])
        _revalidate_row(row, f"models/obj_{row['object_id']:06d}.ply", MODEL_ARCHIVE, "model", MAX_MODEL_EXPANDED,
                        MODEL_ROW_KEYS)
        if row["identity_note"] != IDENTITY_NOTE:
            raise InputsError("allowlist model identity note differs")
    if ids != sorted(set(ids)):
        raise InputsError("allowlist model ids must be unique and sorted")
    if not isinstance(doc["sources"], Mapping) or not all(_is_sha(v) for v in doc["sources"].values()):
        raise InputsError("allowlist source closure malformed")
    if not isinstance(doc["preparation_scope"], str) or not doc["preparation_scope"]:
        raise InputsError("allowlist does not bind a preparation scope")
    closure = doc["closure"]
    if not isinstance(closure, Mapping) or closure.get("after_matches") is not True:
        raise InputsError("allowlist closure was not confirmed")
    return doc


def build_allowlist(receipt: Any, index: Any, integrity: Any, object_ids: Sequence[int], *,
                    anchors: Mapping[str, Mapping]) -> dict:
    """Pure metadata filter over already hash-verified anchors."""
    ids = sorted(set(object_ids))
    if not ids or len(ids) != len(object_ids) or any(type(i) is not int or i not in MODEL_IDS for i in ids):
        raise UsageError(f"object ids must be explicit, unique and within {MODEL_IDS[0]}..{MODEL_IDS[-1]}")
    if not all(isinstance(x, Mapping) for x in (receipt, index, integrity)):
        raise InputsError("anchors must be JSON objects")
    selection = receipt.get("selectionDigest")
    if (receipt.get("status") != "complete" or receipt.get("datasetRevision") != DATASET_REVISION
            or receipt.get("indexSHA256") != ANCHORS["resources_index"] or not _is_sha(selection)):
        raise InputsError("acquisition receipt is not the complete pinned acquisition of this revision/index")
    if (integrity.get("status") != "verified" or integrity.get("datasetRevision") != DATASET_REVISION
            or integrity.get("indexSHA256") != ANCHORS["resources_index"] or integrity.get("selectionDigest") != selection
            or integrity.get("verifiedMembers") != INTEGRITY_MEMBERS):
        raise InputsError("integrity record does not bind the pinned index/revision/selection")
    if (index.get("schemaVersion") != 1 or index.get("corpus") != CORPUS or index.get("datasetRevision") != DATASET_REVISION
            or index.get("storageKind") != "original_zip_members" or index.get("selectionDigest") != selection
            or not isinstance(index.get("resources"), Mapping) or len(index["resources"]) != INTEGRITY_MEMBERS):
        raise InputsError("resources index identity differs from the receipt/integrity bindings")
    if index.get("acquisitionSourceHashes") != receipt.get("sourceHashes"):
        raise InputsError("acquisition source seals differ between the index and the receipt")
    resources = index["resources"]
    rgb = []
    for i in range(FRAME_COUNT):
        member = f"val_primesense/{SCENE}/rgb/{i:06d}.png"
        row = validate_index_row(resources.get(member), member, RGB_ARCHIVE, "rgb", MAX_RGB_EXPANDED)
        rgb.append({"index": i, "frame_id": f"s{i:06d}", "split": _split(i), **row})
    totals = {"encoded_bytes": sum(r["compressed_bytes"] for r in rgb),
              "expanded_bytes": sum(r["expanded_bytes"] for r in rgb)}
    if totals != RGB_TOTALS:
        raise InputsError(f"scene {SCENE} RGB totals {totals} differ from the expected {RGB_TOTALS}")
    bindings = {b.get("objectId"): b for b in index.get("selectedMeshBindings") or [] if isinstance(b, Mapping)}
    models = []
    for object_id in ids:
        member = f"models/obj_{object_id:06d}.ply"
        row = validate_index_row(resources.get(member), member, MODEL_ARCHIVE, "model", MAX_MODEL_EXPANDED)
        binding = bindings.get(object_id)
        if (binding is None or binding.get("meshMember") != member or binding.get("meshSHA256") != row["uncompressed_sha256"]
                or binding.get("meshExpandedBytes") != row["expanded_bytes"]
                or binding.get("meshCRC32") != f"{row['crc32']:08x}"):
            raise InputsError(f"object {object_id}: index mesh binding disagrees with its member row")
        models.append({"object_id": object_id, **row, "identity_note": IDENTITY_NOTE})
    return {
        "schema": SCHEMA_ALLOWLIST, "tool_version": TOOL_VERSION, "trust": dict(TRUST), "flags": dict(FLAGS),
        "anchors": {k: dict(v) for k, v in anchors.items()},
        "source": {"corpus": CORPUS, "dataset_revision": DATASET_REVISION, "scene": SCENE,
                   "rgb_archive": RGB_ARCHIVE, "model_archive": MODEL_ARCHIVE, "selection_digest": selection},
        "source_seals": {"acquisition_source_hashes": receipt.get("sourceHashes"),
                         "index_input_hashes": index.get("inputHashes")},
        "frozen_split": dict(FROZEN_SPLIT), "rgb": rgb, "rgb_totals": totals, "models": models,
        "excluded": EXCLUDED_NOTE,
    }


def filter_allowlist(ctx: Context, *, receipt_path: str, expected_receipt_sha256: str, index_path: str,
                     expected_index_sha256: str, integrity_path: str, expected_integrity_sha256: str,
                     object_ids: Sequence[int], out: str) -> dict:
    expected = {"acquisition_receipt": check_sha(expected_receipt_sha256, "--expected-receipt-sha256"),
                "resources_index": check_sha(expected_index_sha256, "--expected-index-sha256"),
                "integrity_record": check_sha(expected_integrity_sha256, "--expected-integrity-sha256")}
    for key, value in expected.items():
        if value != ANCHORS[key]:
            raise Blocked(f"{key}: expected SHA-256 is not the parent-pinned trust anchor; no trust can be derived")
    paths = {"acquisition_receipt": contained_input(ctx, receipt_path, "acquisition receipt", allow_denied=True),
             "resources_index": contained_input(ctx, index_path, "resources index", allow_denied=True),
             "integrity_record": contained_input(ctx, integrity_path, "integrity record", allow_denied=True)}
    out_dir = check_output_dir(ctx, out)
    for key, path in paths.items():
        outside_scope_input(ctx, path, key)
    with guarded_stage(ctx, read_exceptions=paths.values()) as opened:
        before = {"sources": source_closure(), "runtime": runtime_identity()}
        budget = workspace_budget(ctx, STAGE_ALLOWANCE["filter-allowlist"])
        _set_payload_root(paths["resources_index"].parent / "payloads")      # zero payload opens allowed
        raws = {}
        for key, path in paths.items():
            raw = read_bounded(path, MAX_ANCHOR_BYTES[key], key)
            if sha256_bytes(raw) != expected[key]:                          # BEFORE any parsing
                raise InputsError(f"{key}: SHA-256 differs from the expected anchor")
            raws[key] = raw
        anchors = {k: {"path": repo_rel(ctx, paths[k]), "sha256": expected[k], "bytes": len(raws[k]),
                       "verified_before_parsing": True} for k in paths}
        doc = build_allowlist(strict_json(raws["acquisition_receipt"], "acquisition receipt"),
                              strict_json(raws["resources_index"], "resources index"),
                              strict_json(raws["integrity_record"], "integrity record"), list(object_ids),
                              anchors=anchors)
        payload_root = paths["resources_index"].parent
        doc["source"]["payload_root"] = repo_rel(ctx, payload_root / "payloads")
        for row in doc["rgb"] + doc["models"]:
            p = safe_member_path(payload_root, row["payload_path"], row["member"])
            st = os.lstat(p) if os.path.lexists(p) else None        # stat only; never opened
            if st is None or not stat.S_ISREG(st.st_mode) or st.st_size != row["compressed_bytes"]:
                raise Blocked(f"{row['member']}: cached payload missing or not of the allowlisted size")
        if opened:
            raise GuardError("the metadata filter opened payload bytes")
        ledger = OutputLedger(ctx, out_dir, "filter-allowlist")
        doc.update(sources=before["sources"], runtime=before["runtime"], budget=budget, payload_opens=0,
                   closure={"before": "sources + runtime at stage start", "after_matches": True},
                   ledger={"stage": "filter-allowlist", "aggregate_total_cap": FINAL_OUTPUTS_CAP,
                           "aggregate_metadata_cap": METADATA_AGGREGATE_CAP},
                   preparation_scope=repo_rel(ctx, ctx.scope))
        validate_allowlist_doc(doc)
        data = dump_json(doc)
        for key, path in paths.items():                                      # closure immediately before publication
            if sha256_file(path) != expected[key]:
                raise InputsError(f"{key}: changed during filtering")
        if {"sources": source_closure(), "runtime": runtime_identity()} != before:
            raise InputsError("sources/runtime changed during filtering")
        ledger.check(len(data), metadata=True)                              # before any directory or write
        workspace_budget(ctx, 2 * len(data))
        _make_fresh_dir(out_dir)
        published = publish_bytes(ledger, "allowlist.json", data)
    return {"allowlist": repo_rel(ctx, out_dir / "allowlist.json"), "allowlist_sha256": published["sha256"],
            "rgb_members": FRAME_COUNT, "model_ids": sorted(object_ids), "payload_opens": 0, "exit": 0}


def load_allowlist(ctx: Context, path: str | os.PathLike, expected_sha256: str) -> tuple[dict, Path, Path]:
    """Caller-expected hash first (it must name a parent-accepted filter output), then strict parse
    and FULL re-validation of every field. No index is read and no payload permission is granted."""
    raw, allow_path = read_pinned(ctx, path, expected_sha256, MAX_JSON_OUTPUT, "allowlist")
    doc = validate_allowlist_doc(strict_json(raw, "allowlist"))          # structure first
    if doc["preparation_scope"] != repo_rel(ctx, preparation_scope(ctx)):
        raise InputsError("allowlist was produced for a different preparation scope")
    inside_scope(ctx, allow_path, "allowlist")
    payload_root = Path(os.path.abspath(ctx.root / doc["source"]["payload_root"]))
    if payload_root.name != "payloads" or Path(os.path.abspath(ctx.root)) not in payload_root.parents:
        raise InputsError("allowlist payload root is not a contained payloads directory")
    _no_links(Path(os.path.abspath(ctx.root)), payload_root)
    outside_scope_input(ctx, payload_root, "payload root")
    return doc, allow_path, payload_root


def _candidate_bytes(doc: Mapping) -> int:
    return sum(r["compressed_bytes"] for r in doc["rgb"]) + sum(r["compressed_bytes"] for r in doc.get("models") or [])


def _payload_path(payload_root: Path, row: Mapping) -> Path:
    return safe_member_path(payload_root.parent, row["payload_path"], row["member"])


# --------------------------------------------------------------------------
# strict streaming ASCII PLY


PLY_FLOAT_TYPES = {"float", "float32", "double", "float64"}
PLY_UCHAR_TYPES = {"uchar", "uint8"}
PLY_INDEX_COUNT_TYPES = {"uchar", "uint8"}
PLY_INDEX_TYPES = {"int", "int32", "uint", "uint32"}
VERTEX_PROPS = {"x": PLY_FLOAT_TYPES, "y": PLY_FLOAT_TYPES, "z": PLY_FLOAT_TYPES,
                "nx": PLY_FLOAT_TYPES, "ny": PLY_FLOAT_TYPES, "nz": PLY_FLOAT_TYPES,
                "red": PLY_UCHAR_TYPES, "green": PLY_UCHAR_TYPES, "blue": PLY_UCHAR_TYPES, "alpha": PLY_UCHAR_TYPES}
UNIT_RE = re.compile(r"\bunits?\b\s*[:=]?\s*([A-Za-z]+)", re.IGNORECASE)
MM_WORDS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}


def parse_ply_header(data: bytes) -> dict:
    """Strict ASCII 1.0 header: exactly vertex then face; known typed vertex properties; one
    triangle-list face property. Raises InputsError/Blocked."""
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise InputsError("PLY header is not ASCII") from None
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if len(lines) < 3 or lines[0] != "ply" or lines[-1] != "end_header":
        raise InputsError("malformed PLY header framing")
    if lines[1].split() != ["format", "ascii", "1.0"]:
        raise Blocked(f"only 'format ascii 1.0' PLY is supported (got {lines[1]!r})")
    elements, comments = [], []
    for line in lines[2:-1]:
        tokens = line.split()
        if not tokens:
            raise InputsError("empty PLY header line")
        if tokens[0] in ("comment", "obj_info"):
            comments.append(line[len(tokens[0]):].strip()[:200])
        elif tokens[0] == "element":
            if len(tokens) != 3 or not re.fullmatch(r"[0-9]{1,10}", tokens[2]):
                raise InputsError(f"malformed element line {line!r}")
            elements.append({"name": tokens[1], "count": int(tokens[2]), "props": []})
        elif tokens[0] == "property":
            if not elements:
                raise InputsError("property before any element")
            elements[-1]["props"].append(tokens[1:])
        else:
            raise InputsError(f"unknown PLY header keyword {tokens[0]!r}")
    if [e["name"] for e in elements] != ["vertex", "face"]:
        raise Blocked("PLY must contain exactly a vertex element followed by a face element")
    vertex, face = elements
    names = []
    for prop in vertex["props"]:
        if len(prop) != 2 or prop[1] not in VERTEX_PROPS or prop[0] not in VERTEX_PROPS[prop[1]] or prop[1] in names:
            raise Blocked(f"unsupported vertex property {' '.join(prop)!r}")
        names.append(prop[1])
    if not {"x", "y", "z"} <= set(names):
        raise InputsError("PLY lacks x/y/z")
    normals = {"nx", "ny", "nz"} & set(names)
    if normals and normals != {"nx", "ny", "nz"}:
        raise InputsError("partial normal properties")
    colours = {"red", "green", "blue"} & set(names)
    if colours and colours != {"red", "green", "blue"} or ("alpha" in names and not colours):
        raise InputsError("partial colour properties")
    if len(face["props"]) != 1 or len(face["props"][0]) != 4 or face["props"][0][0] != "list" \
            or face["props"][0][1] not in PLY_INDEX_COUNT_TYPES or face["props"][0][2] not in PLY_INDEX_TYPES \
            or face["props"][0][3] not in ("vertex_indices", "vertex_index"):
        raise Blocked(f"unsupported face property {face['props']}")
    if not 3 <= vertex["count"] <= MAX_VERTICES:
        raise Blocked(f"vertex count {vertex['count']} outside 3..{MAX_VERTICES}")
    if not 1 <= face["count"] <= FACE_CAP:
        raise Blocked(f"face count {face['count']} outside 1..{FACE_CAP} (execution cap)")
    declared = [m.group(1).lower() for m in (UNIT_RE.search(c) for c in comments) if m]
    if any(d not in MM_WORDS for d in declared):
        raise Blocked(f"PLY header declares non-millimetre units {declared}; refusing a silent unit change")
    return {"vertex_count": vertex["count"], "face_count": face["count"], "vertex_properties": names,
            "face_index_type": face["props"][0][2], "comments": comments[:16],
            "units_declared_in_header": declared[0] if declared else None}


class AsciiPlyStream:
    """Bounded streaming ASCII PLY parser fed with expanded payload chunks. Arrays are allocated
    once from the header counts (after the caps); every line is checked for its exact token count."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.header: dict | None = None
        self.vertex_done = 0
        self.face_done = 0
        self.values: np.ndarray | None = None
        self.faces: np.ndarray | None = None

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk
        if self.header is None:
            match = re.search(rb"(?:^|\n)end_header\r?\n", bytes(self.buffer[:HEADER_CAP + 2]))
            if match is None:
                if len(self.buffer) > HEADER_CAP:
                    raise InputsError(f"no end_header within {HEADER_CAP} bytes")
                return
            self.header = parse_ply_header(bytes(self.buffer[:match.end()]))
            del self.buffer[:match.end()]
            self.values = np.empty((self.header["vertex_count"], len(self.header["vertex_properties"])), np.float64)
            self.faces = np.empty((self.header["face_count"], 3), np.int64)
        cut = self.buffer.rfind(b"\n")
        if cut >= 0:
            block = bytes(self.buffer[:cut + 1])
            del self.buffer[:cut + 1]
            self._lines(block.split(b"\n")[:-1])

    def _lines(self, raw_lines: list[bytes]) -> None:
        try:
            lines = [line.decode("ascii").rstrip("\r") for line in raw_lines]
        except UnicodeDecodeError:
            raise InputsError("PLY body is not ASCII") from None
        h = self.header
        nprop = len(h["vertex_properties"])
        pos = 0
        if self.vertex_done < h["vertex_count"]:
            take = min(len(lines), h["vertex_count"] - self.vertex_done)
            rows = [line.split() for line in lines[:take]]
            if any(len(r) != nprop for r in rows):
                raise InputsError("PLY vertex line has the wrong number of values")
            try:
                block = np.asarray(rows, dtype=np.float64).reshape(take, nprop) if take else np.empty((0, nprop))
            except ValueError:
                raise InputsError("PLY vertex values are not numeric") from None
            if not np.isfinite(block).all():
                raise InputsError("PLY contains non-finite vertex values")
            self.values[self.vertex_done:self.vertex_done + take] = block
            self.vertex_done += take
            pos = take
        if pos < len(lines) and self.face_done < h["face_count"]:
            take = min(len(lines) - pos, h["face_count"] - self.face_done)
            rows = [line.split() for line in lines[pos:pos + take]]
            if any(len(r) != 4 or r[0] != "3" for r in rows):
                raise InputsError("every PLY face must list exactly 3 vertex indices")
            try:
                block = np.asarray([r[1:] for r in rows], dtype=np.int64).reshape(take, 3)
            except ValueError:
                raise InputsError("PLY face indices are not integers") from None
            if take and (block.min() < 0 or block.max() >= h["vertex_count"]):
                raise InputsError("PLY face index out of range")
            self.faces[self.face_done:self.face_done + take] = block
            self.face_done += take
            pos += take
        if any(line.strip() for line in lines[pos:]):
            raise InputsError("PLY has data after the declared elements")

    def finish(self) -> dict:
        if self.header is None:
            raise InputsError("PLY ended before end_header")
        if self.buffer:
            tail = bytes(self.buffer)
            self.buffer.clear()
            self._lines([tail])
        h = self.header
        if self.vertex_done != h["vertex_count"] or self.face_done != h["face_count"]:
            raise InputsError(f"PLY body is truncated ({self.vertex_done}/{h['vertex_count']} vertices, "
                              f"{self.face_done}/{h['face_count']} faces)")
        column = {name: self.values[:, i] for i, name in enumerate(h["vertex_properties"])}
        colours = None
        if "red" in column:
            chans = [column[c] for c in ("red", "green", "blue")] + ([column["alpha"]] if "alpha" in column else [])
            colours = np.stack(chans, axis=1)
            if (colours < 0).any() or (colours > 255).any() or (colours != np.round(colours)).any():
                raise InputsError("PLY vertex colours must be integers in [0, 255]")
            if colours.shape[1] == 3:
                colours = np.concatenate([colours, np.full((len(colours), 1), 255.0)], axis=1)
            colours = colours.astype(np.uint8)
        return {"positions_mm": np.stack([column["x"], column["y"], column["z"]], axis=1),
                "normals": np.stack([column[c] for c in ("nx", "ny", "nz")], axis=1) if "nx" in column else None,
                "colours": colours, "triangles": self.faces.astype(np.uint32), "header": h}


# --------------------------------------------------------------------------
# GLB (metres; own minimal writer + independent reader)


_FLOAT, _UINT, _UBYTE = 5126, 5125, 5121
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_DTYPES = {_FLOAT: np.dtype("<f4"), _UINT: np.dtype("<u4"), _UBYTE: np.dtype("u1")}
_WIDTH = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}


def metre_positions(positions_mm: Any) -> np.ndarray:
    """mm * 0.001 exactly once (float64), then float32; non-finite results are refused BEFORE serialisation."""
    p = np.asarray(positions_mm, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise InputsError("source positions must be finite Nx3")
    with np.errstate(over="ignore", invalid="ignore"):
        metres = (p * MM_TO_M).astype("<f4")
    if not np.isfinite(metres).all():
        raise InputsError("metre positions overflow float32; refusing to serialise")
    return metres


def unit_normals(normals: Any) -> tuple[np.ndarray, str]:
    """Overflow-safe normalisation (scale by the largest component first). Zero-length normals are
    refused: nothing is fabricated or silently omitted. Output is finite unit float32."""
    n = np.asarray(normals, np.float64)
    if n.ndim != 2 or n.shape[1] != 3 or not np.isfinite(n).all():
        raise InputsError("source normals must be finite Nx3")
    largest = np.max(np.abs(n), axis=1)
    if (largest == 0).any():
        raise Blocked(f"{int((largest == 0).sum())} zero-length source normals; none are fabricated or omitted")
    scaled = n / largest[:, None]
    length = np.sqrt((scaled * scaled).sum(axis=1))
    unit = (scaled / length[:, None]).astype("<f4")
    check = np.linalg.norm(unit.astype(np.float64), axis=1)
    if not np.isfinite(unit).all() or float(np.abs(check - 1.0).max()) > 1e-6:
        raise InputsError("normals did not normalise to finite unit float32 vectors")
    with np.errstate(over="ignore"):
        source_len = largest * length
    deviation = float(np.abs(source_len - 1.0).max()) if np.isfinite(source_len).all() else None
    note = ("explicit source normals renormalised to unit length (overflow-safe); directions preserved; "
            f"max |source length - 1| = {'overflow (> float64 range)' if deviation is None else f'{deviation:.3g}'}")
    return unit, note


def build_glb(mesh: Mapping, *, name: str, source_sha256: str) -> tuple[bytes, dict]:
    """Positions: mm * 0.001 applied EXACTLY ONCE (float64), then stored as float32 metres. Origin,
    basis, winding and vertex colours are preserved; no UV or texture is fabricated. Vertex colours
    are REQUIRED (this is a vertex-colour control; there is no white-appearance fallback)."""
    if mesh.get("colours") is None:
        raise Blocked("source vertex colours are required for this control; no white-appearance fallback")
    positions_m = metre_positions(mesh["positions_mm"])
    triangles = np.asarray(mesh["triangles"], np.uint32)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not 1 <= len(triangles) <= FACE_CAP \
            or int(triangles.max()) >= len(positions_m):
        raise InputsError("triangles must be Mx3 valid indices within the execution cap")
    colours = np.asarray(mesh["colours"], np.uint8)
    if colours.shape != (len(positions_m), 4):
        raise InputsError("vertex colours must be Nx4 uint8")
    normals_out, normals_note = (None, "absent in source (none fabricated)")
    if mesh.get("normals") is not None:
        normals_out, normals_note = unit_normals(mesh["normals"])
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add(array: np.ndarray, component: int, kind: str, target: int, *, normalized=False, bounds=False) -> int:
        data = np.ascontiguousarray(array.astype(_DTYPES[component])).tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        blob.extend(b"\x00" * (-len(blob) % 4))
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": int(array.shape[0]), "type": kind}
        if normalized:
            acc["normalized"] = True
        if bounds:
            flat = array.astype(_DTYPES[component]).reshape(len(array), -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    attributes = {"POSITION": add(positions_m, _FLOAT, "VEC3", _ARRAY_BUFFER, bounds=True)}
    if normals_out is not None:
        attributes["NORMAL"] = add(normals_out, _FLOAT, "VEC3", _ARRAY_BUFFER)
    attributes["COLOR_0"] = add(colours, _UBYTE, "VEC4", _ARRAY_BUFFER, normalized=True)
    indices = add(triangles.reshape(-1), _UINT, "SCALAR", _ELEMENT_ARRAY_BUFFER)
    gltf = {
        "asset": {"version": "2.0", "generator": f"{TOOL_ID}/{TOOL_VERSION}",
                  "extras": {"units": "metres", "unit_change": "BOP millimetres * 0.001 (applied once)",
                             "basis": "original BOP model frame (right-handed; origin/basis/winding unchanged)",
                             "source_ply_sha256": source_sha256}},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attributes, "indices": indices, "mode": 4,
                                                  "material": 0}]}],
        "materials": [{"name": f"{name}-white-unlit",
                       "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                                "roughnessFactor": 1.0},
                       "extensions": {"KHR_materials_unlit": {}}}],
        "extensionsUsed": ["KHR_materials_unlit"],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    json_chunk = json.dumps(gltf, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = bytes(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
           + struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk)
    return glb, {"normals": normals_note, "normals_out": normals_out, "appearance": APPEARANCE,
                 "uv": "none (no texture fabricated)"}


def read_glb_document(glb: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Independent reader for this writer's layout (bounds-checked): (glTF JSON, primitive arrays)."""
    if len(glb) < 28:
        raise InputsError("GLB too short")
    magic, version, length = struct.unpack_from("<4sII", glb, 0)
    if magic != b"glTF" or version != 2 or length != len(glb):
        raise InputsError("GLB header invalid")
    jlen, jtype = struct.unpack_from("<I4s", glb, 12)
    if jtype != b"JSON" or 20 + jlen + 8 > len(glb):
        raise InputsError("GLB JSON chunk invalid")
    gltf = strict_json(glb[20:20 + jlen], "GLB JSON")
    blen, btype = struct.unpack_from("<I4s", glb, 20 + jlen)
    binary = glb[28 + jlen:28 + jlen + blen]
    if btype != b"BIN\x00" or len(binary) != blen or 28 + jlen + blen != len(glb):
        raise InputsError("GLB BIN chunk invalid")

    def accessor(index: int) -> np.ndarray:
        acc = gltf["accessors"][index]
        view = gltf["bufferViews"][acc["bufferView"]]
        start, size = view["byteOffset"], view["byteLength"]
        if start + size > len(binary):
            raise InputsError("GLB view exceeds the buffer")
        width = _WIDTH[acc["type"]]
        arr = np.frombuffer(binary[start:start + size], dtype=_DTYPES[acc["componentType"]])
        if arr.size != acc["count"] * width:
            raise InputsError("GLB accessor length mismatch")
        return arr.reshape(acc["count"], width) if width > 1 else arr

    try:
        prim = gltf["meshes"][0]["primitives"][0]
        out = {name: accessor(i) for name, i in prim["attributes"].items()}
        out["indices"] = accessor(prim["indices"])
    except (KeyError, IndexError, TypeError) as exc:
        raise InputsError(f"GLB layout is not this writer's single-primitive layout ({exc})") from None
    return gltf, out


def read_glb_arrays(glb: bytes) -> dict[str, np.ndarray]:
    return read_glb_document(glb)[1]


def verify_glb(glb_path: Path, glb: bytes, mesh: Mapping, normals_out: np.ndarray | None) -> dict:
    """Strict round trip with the independent reader AND the producer decoder (workbench_neural.glb_mesh)."""
    from bench import workbench_neural as wn
    expected = metre_positions(mesh["positions_mm"])
    tri = np.asarray(mesh["triangles"], np.uint32)
    arrays = read_glb_arrays(glb)
    if not np.array_equal(arrays["POSITION"], expected):
        raise InputsError("round trip: positions differ")
    if not np.array_equal(arrays["indices"].reshape(-1, 3), tri):
        raise InputsError("round trip: triangle indices/winding differ")
    if "COLOR_0" not in arrays or not np.array_equal(arrays["COLOR_0"], mesh["colours"]):
        raise InputsError("round trip: vertex colours differ")
    if (normals_out is None) != ("NORMAL" not in arrays) or (normals_out is not None and not np.array_equal(arrays["NORMAL"], normals_out)):
        raise InputsError("round trip: normals differ")
    error_mm = float(np.abs(arrays["POSITION"].astype(np.float64) / MM_TO_M - mesh["positions_mm"]).max())
    v, f = wn.glb_mesh(glb_path, max_faces=FACE_CAP)
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    if not np.array_equal(v, expected.astype(np.float64)):
        raise InputsError("producer decode: positions differ")
    if not np.array_equal(f, tri[keep].astype(np.int64)):
        raise InputsError("producer decode: triangles differ from the source minus its degenerate faces")
    return {"positions_bitwise_equal_float32": True, "indices_and_winding_equal": True,
            "vertex_colours_equal": True, "normals_equal": normals_out is not None,
            "max_position_error_mm_vs_source": error_mm,
            "producer_glb_mesh_positions_equal": True, "producer_glb_mesh_triangles": int(len(f)),
            "source_degenerate_faces_dropped_by_producer": int((~keep).sum()),
            "decimated": False}


def prepare_model(ctx: Context, *, allowlist: str, expected_allowlist_sha256: str, object_id: int,
                  existing_input_bytes: int, out: str) -> dict:
    if type(object_id) is not int or object_id not in MODEL_IDS:
        raise UsageError(f"--object-id must be within {MODEL_IDS[0]}..{MODEL_IDS[-1]}")
    out_dir = check_output_dir(ctx, out)
    with guarded_stage(ctx) as opened:
        before = {"allowlist": check_sha(expected_allowlist_sha256, "--expected-allowlist-sha256"),
                  "sources": source_closure(), "runtime": runtime_identity()}
        doc, allow_path, payload_root = load_allowlist(ctx, allowlist, expected_allowlist_sha256)
        _set_payload_root(payload_root)
        rows = [m for m in doc["models"] if m["object_id"] == object_id]
        if len(rows) != 1:
            raise Blocked(f"object {object_id} is not explicitly allowlisted")
        row = rows[0]
        path = _payload_path(payload_root, row)
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size != row["compressed_bytes"]:
            raise Blocked(f"{row['member']}: payload missing or not of the allowlisted size")
        budget = workspace_budget(ctx, STAGE_ALLOWANCE["prepare-model"])                 # prospective
        inputs = input_budget(existing_input_bytes, _candidate_bytes(doc), FINAL_OUTPUTS_CAP)
        allow_payloads(payload_root, [path])                       # widened only after trust + budget
        _make_fresh_dir(out_dir)
        ledger = OutputLedger(ctx, out_dir, "prepare-model")
        try:
            parser = AsciiPlyStream()
            verified = verified_payload_stream(path, row, parser.feed)
            mesh = parser.finish()
            if mesh["colours"] is None:
                raise Blocked("source PLY has no vertex colours; this vertex-colour control has no white fallback")
            name = f"hb_obj_{object_id:06d}"
            glb, info = build_glb(mesh, name=name, source_sha256=row["uncompressed_sha256"])
            if len(glb) > GLB_CAP:
                raise Blocked(f"GLB is {len(glb)} bytes, above the {GLB_CAP} byte (5.5 MiB) cap; nothing decimated")
            glb_meta = publish_bytes(ledger, f"{name}.glb", glb, metadata=False)
            roundtrip = verify_glb(out_dir / f"{name}.glb", glb, mesh, info.pop("normals_out"))
            faces = mesh["header"]["face_count"]
            receipt = {
                "schema": SCHEMA_MODEL, "tool_version": TOOL_VERSION, "status": "prepared", "flags": dict(FLAGS),
                "trust": dict(TRUST), "object_id": object_id, "family": FAMILY,
                "allowlist": {"path": repo_rel(ctx, allow_path), "sha256": before["allowlist"]},
                "source": {"member": row["member"], "archive": row["archive"], "dataset_revision": DATASET_REVISION,
                           "compressed_sha256": verified["compressed_sha256"],
                           "uncompressed_sha256": verified["uncompressed_sha256"], "crc32": verified["crc32"],
                           "compressed_bytes": verified["compressed_bytes"], "expanded_bytes": verified["expanded_bytes"],
                           "header": mesh["header"]},
                "units": {"source": "millimetres", "source_reference": BOP_FORMAT_URL,
                          "source_evidence": ("BOP dataset format convention (models and translations in mm); "
                                              "header units " + (mesh["header"]["units_declared_in_header"] or "not declared")),
                          "output": "metres", "factor": MM_TO_M, "applied": "exactly once"},
                "glb": {"path": repo_rel(ctx, out_dir / glb_meta["name"]), "sha256": glb_meta["sha256"],
                        "bytes": glb_meta["bytes"], **info},
                "geometry": {"vertices": mesh["header"]["vertex_count"], "triangles": faces,
                             "execution_cap_triangles": FACE_CAP, "roundtrip": roundtrip,
                             "geometry_verified": "parsed and round-tripped (not header-only); no quality claim"},
                "portable_targets": {"triangles_target": PORTABLE_TRIANGLES,
                                     "triangles": "PASS" if faces <= PORTABLE_TRIANGLES else "FAIL",
                                     "bytes_target": PORTABLE_BYTES,
                                     "note": "portability targets are separate from the execution cap; a FAIL is "
                                             "not geometry failure and nothing is decimated"},
                "budget": budget, "input_budget": inputs, "ledger": ledger.summary(),
                "closure": {"before": before, "after_matches": True}, "mobile_verified": False,
                "preparation_scope": repo_rel(ctx, ctx.scope),
            }
            after = {"allowlist": sha256_file(allow_path), "sources": source_closure(), "runtime": runtime_identity()}
            if after != before or sha256_file(out_dir / glb_meta["name"]) != glb_meta["sha256"]:
                raise InputsError("allowlist/sources/runtime/GLB changed during preparation")
            published = publish_bytes(ledger, "model-receipt.json", dump_json(receipt))
        except BaseException as exc:
            _failure(ledger, "prepare-model", exc)
            raise
        if [p for p in opened if p != _norm(path)]:
            raise GuardError("unexpected payload opens")
    return {"receipt": repo_rel(ctx, out_dir / "model-receipt.json"), "receipt_sha256": published["sha256"],
            "glb_sha256": glb_meta["sha256"], "triangles": faces,
            "portable_triangles": receipt["portable_targets"]["triangles"], "exit": 0}


# --------------------------------------------------------------------------
# RGB stream -> derivative video (pinned FFmpeg, pipes only)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_rgb8_header(raw: bytes) -> tuple[int, int]:
    """Native 8-bit truecolour RGB (colour type 2) PNG; returns (width, height)."""
    if raw[:8] != PNG_SIGNATURE or raw[12:16] != b"IHDR" or len(raw) < 33:
        raise InputsError("not a PNG with a leading IHDR")
    width, height, depth, colour = struct.unpack(">IIBB", raw[16:26])
    if (width, height) != IMAGE_SIZE or depth != 8 or colour != 2:
        raise InputsError(f"PNG is {width}x{height} depth {depth} colour type {colour}; expected 640x480 8-bit RGB")
    if not raw.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
        raise InputsError("PNG does not end with IEND")
    return width, height


def decode_png_rgb(raw: bytes) -> np.ndarray:
    from bench.vision import cv2
    png_rgb8_header(raw)
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint8 or bgr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise InputsError("PNG did not decode to 480x640x3 uint8")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def codec_settings() -> dict:
    """Frozen BEFORE any outcome (profile + pinned executable identity)."""
    return {"profile": dict(VIDEO_PROFILE), "ffmpeg": dict(FFMPEG)}


def ffmpeg_executable(ctx: Context) -> Path:
    """The pinned local FFmpeg: exact size and SHA-256 verified before every use (never installed)."""
    path = Path(os.path.abspath(ctx.root / FFMPEG["path"]))
    if not path.is_file() or _is_link(path):
        path = Path(os.path.abspath(ROOT / FFMPEG["path"]))
    if not path.is_file() or _is_link(path) or os.lstat(path).st_size != FFMPEG["bytes"] or sha256_file(path) != FFMPEG["sha256"]:
        raise Blocked("pinned FFmpeg executable missing or its bytes differ from the pin")
    return path


class FfmpegPipeEncoder:
    """Exactly one granted FFmpeg process: frames on stdin, fragmented MP4 from stdout into ONE
    staging file with the cap checked BEFORE every write, stderr drained to a bounded tail. A
    watchdog kills the process at the deadline; every exit path kills/reaps and joins/closes."""

    def __init__(self, ctx: Context, ledger: OutputLedger, staging: Path):
        self.ctx, self.ledger, self.staging = ctx, ledger, staging
        self.exe = ffmpeg_executable(ctx)
        self.argv = [str(self.exe), *FFMPEG_ARGS]
        self.proc: subprocess.Popen | None = None
        self.written = 0
        self.error: BaseException | None = None
        self.stderr_tail: collections.deque = collections.deque()
        self.stderr_bytes = 0
        self.threads: list[threading.Thread] = []
        self.fd: int | None = None
        self.timer: threading.Timer | None = None

    def __enter__(self) -> "FfmpegPipeEncoder":
        self.fd = os.open(self.staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
        try:
            self.deadline_at = time.monotonic() + FFMPEG_DEADLINE_S
            _grant_subprocess(self.exe, self.argv)
            self.proc = subprocess.Popen(self.argv, executable=str(self.exe), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, shell=False,
                                         close_fds=True, cwd=str(self.staging.parent),
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.timer = threading.Timer(FFMPEG_DEADLINE_S, self._deadline)
            self.timer.daemon = True
            self.timer.start()
            for name, target in (("ffmpeg-stdout-drain", self._drain_stdout),
                                 ("ffmpeg-stderr-drain", self._drain_stderr)):
                t = threading.Thread(target=target, name=name, daemon=True)
                t.start()
                self.threads.append(t)
        except BaseException:
            self.__exit__(None, None, None)                        # kill/reap, close the staging handle
            raise
        return self

    def _deadline(self) -> None:
        self.error = self.error or Blocked(f"FFmpeg exceeded its {FFMPEG_DEADLINE_S} s deadline")
        self._kill()

    def _kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass

    def _drain_stdout(self) -> None:
        try:
            while True:
                chunk = self.proc.stdout.read(PIPE_CHUNK)
                if not chunk:
                    return
                if self.written + len(chunk) > VIDEO_CAP:                  # cap BEFORE the write
                    raise Blocked(f"derivative video would exceed the {VIDEO_CAP} byte cap")
                self.ledger.check(len(chunk), metadata=False)
                view = memoryview(chunk)
                while view:
                    n = os.write(self.fd, view)
                    if type(n) is not int or n <= 0 or n > len(view):
                        raise InputsError(f"staging write reported invalid progress {n!r}")
                    view = view[n:]
                    self.written += n
        except BaseException as exc:  # noqa: BLE001 - surfaced by the main thread
            self.error = self.error or exc
            self._kill()

    def _drain_stderr(self) -> None:
        try:
            while True:
                chunk = self.proc.stderr.read(4096)
                if not chunk:
                    return
                self.stderr_tail.append(chunk)
                self.stderr_bytes += len(chunk)
                while sum(len(c) for c in self.stderr_tail) > STDERR_TAIL:
                    self.stderr_tail.popleft()
        except BaseException:  # noqa: BLE001
            pass

    def write_frame(self, rgb: np.ndarray) -> None:
        """Deliver ALL bytes of one frame: raw pipe writes may consume fewer bytes than offered, so
        the remainder is re-offered until done. Zero/None/invalid progress is refused, and the
        drain error, the watchdog and the deadline are checked between writes."""
        data = np.ascontiguousarray(rgb, np.uint8).tobytes()
        view = memoryview(data)
        deadline = getattr(self, "deadline_at", None)
        while len(view):
            if self.error is not None:
                raise self.error
            if deadline is not None and time.monotonic() > deadline:
                self._kill()
                raise Blocked(f"FFmpeg exceeded its {FFMPEG_DEADLINE_S} s deadline while receiving frames")
            try:
                n = self.proc.stdin.write(view)
            except (BrokenPipeError, OSError) as exc:
                raise (self.error or InputsError(
                    f"FFmpeg stdin closed after {len(data) - len(view)} of {len(data)} frame bytes: {exc}; "
                    f"stderr tail: {self._tail()[-512:]}")) from None
            if type(n) is not int or n <= 0 or n > len(view):
                raise InputsError(f"FFmpeg stdin reported invalid write progress {n!r} "
                                  f"after {len(data) - len(view)} of {len(data)} frame bytes")
            view = view[n:]

    def finish(self) -> dict:
        self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=max(1.0, getattr(self, "deadline_at", time.monotonic() + FFMPEG_DEADLINE_S)
                                              - time.monotonic()))
        except subprocess.TimeoutExpired:
            self._kill()
            raise Blocked("FFmpeg did not exit before its deadline") from None
        for t in self.threads:
            t.join(timeout=60)
        alive = [t.name for t in self.threads if t.is_alive()]
        if alive:
            raise InputsError(f"FFmpeg pipe readers did not finish: {alive}")
        if self.error is not None:
            raise self.error
        if code != 0:
            raise InputsError(f"FFmpeg exited {code}: {self._tail()}")
        os.fsync(self.fd)
        return {"exit_code": code, "stdout_bytes": self.written, "stderr_bytes": self.stderr_bytes,
                "stderr_tail": self._tail()}

    def _tail(self) -> str:
        tail = getattr(self, "stderr_tail", ())
        return b"".join(tail).decode("utf-8", "replace")[-STDERR_TAIL:]

    def __exit__(self, *exc) -> None:
        """Kill, reap, close the pipes (which also unblocks a stuck reader), join the drain threads,
        and close the staging handle ONLY when the stdout drain thread no longer uses it. Incomplete
        cleanup is reported in ``self.cleanup`` and raised when no other error is in flight."""
        report = {"reaped": None, "returncode": None, "threads_alive": [], "staging_fd_closed": False,
                  "pipes_closed": None}
        if self.timer is not None:
            self.timer.cancel()
        self._kill()
        if self.proc is not None:
            try:
                if self.proc.stdin is not None and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=30)
                report["reaped"] = True
            except subprocess.TimeoutExpired:
                report["reaped"] = False
            for t in self.threads:
                t.join(timeout=30)
            for pipe in (self.proc.stdout, self.proc.stderr):
                try:
                    if pipe is not None and not pipe.closed:
                        pipe.close()
                except OSError:
                    pass
            for t in self.threads:
                t.join(timeout=5)
            report["returncode"] = self.proc.returncode
            report["pipes_closed"] = all(p is None or p.closed for p in (self.proc.stdin, self.proc.stdout,
                                                                          self.proc.stderr))
        report["threads_alive"] = [t.name for t in self.threads if t.is_alive()]
        stdout_reader_alive = bool(self.threads) and self.threads[0].is_alive()
        if self.fd is not None and not stdout_reader_alive:            # never close a handle still in use
            os.close(self.fd)
            self.fd = None
            report["staging_fd_closed"] = True
        if self.written:
            self.ledger.record("video.mp4", self.written, metadata=False)
        _GUARD.update(grant=None, create_pending=False)
        self.cleanup = report
        incomplete = (self.proc is not None and (report["reaped"] is False or not report["pipes_closed"])) \
            or report["threads_alive"] or self.fd is not None
        if incomplete and exc and exc[0] is None:
            raise InputsError(f"FFmpeg cleanup incomplete: {report}")


def prepare_video(ctx: Context, *, allowlist: str, expected_allowlist_sha256: str, existing_input_bytes: int,
                  out: str) -> dict:
    from bench import workbench_geometry as wg
    out_dir = check_output_dir(ctx, out)
    settings = codec_settings()
    with guarded_stage(ctx) as opened:
        before = {"allowlist": check_sha(expected_allowlist_sha256, "--expected-allowlist-sha256"),
                  "sources": source_closure(), "runtime": runtime_identity()}
        doc, allow_path, payload_root = load_allowlist(ctx, allowlist, expected_allowlist_sha256)
        _set_payload_root(payload_root)
        paths = []
        for row in doc["rgb"]:
            p = _payload_path(payload_root, row)
            st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode) or st.st_size != row["compressed_bytes"]:
                raise Blocked(f"{row['member']}: payload missing or not of the allowlisted size")
            paths.append(p)
        ffmpeg_executable(ctx)                                      # pinned bytes verified before any payload
        budget = workspace_budget(ctx, STAGE_ALLOWANCE["prepare-video"])
        inputs = input_budget(existing_input_bytes, _candidate_bytes(doc), FINAL_OUTPUTS_CAP)
        allow_payloads(payload_root, paths)
        _make_fresh_dir(out_dir)
        ledger = OutputLedger(ctx, out_dir, "prepare-video")
        staging = out_dir / f".staging-{secrets.token_hex(6)}.mp4"
        encoder = None
        try:
            frames = []
            with FfmpegPipeEncoder(ctx, ledger, staging) as encoder:
                for row, p in zip(doc["rgb"], paths):
                    chunks: list[bytes] = []
                    verified = verified_payload_stream(p, row, chunks.append)
                    rgb = decode_png_rgb(b"".join(chunks))
                    encoder.write_frame(rgb)
                    frames.append({"index": row["index"], "frame_id": row["frame_id"], "split": row["split"],
                                   "member": row["member"], "png_sha256": verified["uncompressed_sha256"],
                                   "source_pixel_sha256": wg.array_sha256(rgb)})
                    del chunks, rgb
                process = encoder.finish()
            size = os.lstat(staging).st_size
            if size > VIDEO_CAP or size != process["stdout_bytes"]:
                raise InputsError(f"derivative video size {size} differs from the drained stream or exceeds the cap")
            decoded = reopen_and_decode(staging)
            for frame, digest in zip(frames, decoded["pixel_sha256"]):
                frame["derivative_decoded_pixel_sha256"] = digest
                frame["decoded_identical_to_source"] = digest == frame["source_pixel_sha256"]
            video_sha = sha256_file(staging)
            final = out_dir / "video.mp4"
            os.link(staging, final)
            os.unlink(staging)                         # the owned staging alias never coexists with later checks
            if sha256_file(final) != video_sha:
                raise InputsError("published video does not verify")
            receipt = {
                "schema": SCHEMA_VIDEO, "tool_version": TOOL_VERSION, "status": "prepared", "flags": dict(FLAGS),
                "trust": dict(TRUST), "allowlist": {"path": repo_rel(ctx, allow_path), "sha256": before["allowlist"]},
                "video": {"path": repo_rel(ctx, final), "workspace_relative": Path(os.path.abspath(final)).relative_to(
                    Path(os.path.abspath(ctx.workspace))).as_posix(), "sha256": video_sha, "bytes": size,
                    "cap_bytes": VIDEO_CAP},
                "codec": settings, "process": process, "clock": dict(VIDEO_CLOCK), "geometry": dict(VIDEO_GEOMETRY),
                "split": dict(FROZEN_SPLIT),
                "decode": {"sequential_frames": decoded["frames"], "end_of_video_after": FRAME_COUNT,
                           "dimensions": decoded["dimensions"], "backend": decoded["backend"],
                           "frame_reader_seek_check": decoded["seek_check"], "pixel_identity_claimed": False,
                           "note": DECODE_NOTE},
                "frames": frames, "budget": budget, "input_budget": inputs,
                "runtime": {**before["runtime"], "ffmpeg": dict(FFMPEG)}, "mobile_verified": False,
                "closure": {"before": before, "after_matches": True},
                "preparation_scope": repo_rel(ctx, ctx.scope), "encoder_cleanup": encoder.cleanup,
            }
            after = {"allowlist": sha256_file(allow_path), "sources": source_closure(), "runtime": runtime_identity()}
            if after != before or sha256_file(final) != video_sha:
                raise InputsError("allowlist/sources/runtime/video changed during preparation")
            receipt["ledger"] = ledger.summary()
            published = publish_bytes(ledger, "video-receipt.json", dump_json(receipt))
        except BaseException as exc:
            _failure(ledger, "prepare-video", exc,
                     {"staging_retained_bytes": os.lstat(staging).st_size if os.path.lexists(staging) else 0,
                      "encoder_cleanup": getattr(encoder, "cleanup", None)})
            raise
        if set(opened) - {_norm(p) for p in paths}:
            raise GuardError("unexpected payload opens")
    return {"receipt": repo_rel(ctx, out_dir / "video-receipt.json"), "receipt_sha256": published["sha256"],
            "video_sha256": video_sha, "video_bytes": size, "exit": 0}


def reopen_and_decode(path: Path) -> dict:
    """Sequential decode of the whole derivative (exact 120, then end of video, 640x480; no frame
    dumps) plus the runtime FrameReader access pattern: a first read at s000010 must seek exactly."""
    from bench.vision import cv2
    from bench import workbench_geometry as wg
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputsError("cannot reopen the derivative video")
    try:
        dims = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else None
        hashes = []
        while len(hashes) <= FRAME_COUNT:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if bgr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) or bgr.dtype != np.uint8:
                raise InputsError(f"decoded frame {len(hashes)} is {bgr.shape} {bgr.dtype}")
            hashes.append(wg.array_sha256(np.ascontiguousarray(bgr[:, :, ::-1])))
    finally:
        cap.release()
    if dims != list(IMAGE_SIZE):
        raise InputsError(f"derivative dimensions {dims} differ from 640x480")
    if len(hashes) != FRAME_COUNT:
        raise InputsError(f"derivative decodes to {len(hashes)} frames, expected exactly {FRAME_COUNT}")
    target = SEED_INDEX + 1
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, bgr = cap.read()
        position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    finally:
        cap.release()
    exact = bool(ok) and position == target + 1 and wg.array_sha256(np.ascontiguousarray(bgr[:, :, ::-1])) == hashes[target]
    if not exact:
        raise Blocked("the runtime FrameReader seek to s000010 is not frame-exact on this derivative")
    return {"frames": len(hashes), "pixel_sha256": hashes, "dimensions": dims, "backend": backend,
            "seek_check": {"frame_id": f"s{target:06d}", "exact": True}}


# --------------------------------------------------------------------------
# assemble (UNADMITTED, UNREGISTERED) with independent prerequisite re-validation


def select_calibration(registry: Any) -> list[float]:
    if not isinstance(registry, Mapping) or registry.get("schemaVersion") != 1 or not isinstance(registry.get("calibrations"), list):
        raise InputsError("calibration registry must be schemaVersion 1 with a calibrations list")
    matches = []
    for row in registry["calibrations"]:
        if isinstance(row, Mapping) and row.get("kind") == "dataset_reference" and \
                row.get("width") == IMAGE_SIZE[0] and row.get("height") == IMAGE_SIZE[1]:
            if ("K" in row) == ("cam_K" in row):
                raise InputsError("calibration row must carry exactly one of K/cam_K")
            K = row.get("K", row.get("cam_K"))
            if not isinstance(K, list) or len(K) != 9 or any(type(v) not in (int, float) or not math.isfinite(v) for v in K):
                raise InputsError("calibration K must be 9 finite numbers")
            matches.append([float(x) for x in K])
    if not matches or any(m != matches[0] for m in matches):
        raise Blocked("no single consistent dataset_reference 640x480 calibration")
    K = matches[0]
    if any(abs(a - b) > K_ABS_TOL for a, b in zip(K, EXPECTED_K)) or K[1] != 0 or K[3] != 0 or K[6:] != [0.0, 0.0, 1.0]:
        raise InputsError("registry K differs from the expected HB dataset-reference K")
    return K


def validate_selection_review(review: Any, allow: Mapping) -> dict:
    """Frozen opening-only RGB selection review (parent-authored)."""
    req = {"schema", "scene", "frame_index", "frame_id", "rgb_member", "rgb_source_sha256", "object_id",
           "instance_index", "positive", "negative", "reviewed", "reviewed_by", "attestation", "opening_only",
           "derived_from_scored_frames"}
    if not isinstance(review, Mapping) or set(review) - (req | {"roi"}) or not req <= set(review):
        raise Blocked("selection review is missing fields or has unknown fields")
    seed_row = allow["rgb"][SEED_INDEX]
    if (review["schema"] != SCHEMA_SELECTION_REVIEW or review["scene"] != SCENE or type(review["frame_index"]) is not int
            or review["frame_index"] != SEED_INDEX or review["frame_id"] != f"s{SEED_INDEX:06d}"
            or review["rgb_member"] != seed_row["member"] or review["rgb_source_sha256"] != seed_row["uncompressed_sha256"]):
        raise Blocked("selection review does not bind frame 9 and its allowlisted RGB source SHA-256")
    if (type(review["object_id"]) is not int or review["object_id"] not in MODEL_IDS
            or type(review["instance_index"]) is not int or review["instance_index"] < 0):
        raise Blocked("selection review must bind an explicit object id and instance index")
    if review["reviewed"] is not True or review["opening_only"] is not True or review["derived_from_scored_frames"] is not False:
        raise Blocked("selection review must be a reviewed opening-only attestation not derived from scored frames")
    if not isinstance(review["attestation"], str) or not review["attestation"].strip() \
            or not isinstance(review["reviewed_by"], str) or not review["reviewed_by"].strip():
        raise Blocked("selection review needs a qualitative attestation and a reviewer")
    for key in ("positive", "negative"):
        pts = review[key]
        if not isinstance(pts, list) or (key == "positive" and not pts) or any(
                not isinstance(p, list) or len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
                or not (-0.5 <= p[0] < IMAGE_SIZE[0] - 0.5 and -0.5 <= p[1] < IMAGE_SIZE[1] - 0.5) for p in pts):
            raise Blocked(f"selection review {key} points must be finite native 640x480 pixels")
    if "roi" in review:
        roi = review["roi"]
        if not isinstance(roi, list) or len(roi) != 4 or any(type(v) not in (int, float) for v in roi) \
                or not (0 <= roi[0] < roi[2] <= IMAGE_SIZE[0] and 0 <= roi[1] < roi[3] <= IMAGE_SIZE[1]):
            raise Blocked("selection review ROI must be [x0, y0, x1, y1] inside 640x480")
    return dict(review)


def validate_seed(seed: Any, review_sha: str, review: Mapping, allow: Mapping, allow_sha: str) -> dict:
    """The isolated seed produced by the separate module; only its selected pose is ever consumed."""
    from bench import workbench_contract as wc
    from bench import workbench_neural as wn
    if not isinstance(seed, Mapping) or seed.get("schema") != SCHEMA_SEED:
        raise Blocked("seed is not a workbench_hb_opening_seed_v1 record")
    if (seed.get("scene") != SCENE or seed.get("frame_index") != SEED_INDEX or seed.get("frame_id") != f"s{SEED_INDEX:06d}"
            or seed.get("object_id") != review["object_id"] or seed.get("instance_index") != review["instance_index"]
            or seed.get("selection_review_sha256") != review_sha or seed.get("rgb_source_sha256") != review["rgb_source_sha256"]
            or seed.get("allowlist_sha256") != allow_sha or seed.get("units") != "metres"
            or seed.get("dataset_revision") != DATASET_REVISION):
        raise Blocked("seed does not bind the same frame/object/instance/review/RGB/allowlist identities")
    if seed.get("applies_once_to_frame_id") != f"s{SEED_INDEX + 1:06d}" or seed.get("reuse_on_recovery") is not False \
            or seed.get("other_frames_materialized") is not False:
        raise Blocked("seed must apply once to s000010, never on recovery, and materialize no other frame")
    gt = seed.get("scene_gt")
    if (not isinstance(gt, Mapping) or not _is_sha(gt.get("sha256")) or gt.get("scan_sha256") != gt.get("sha256")
            or gt.get("expected_sha256_matched_before_parsing") is not True):
        raise Blocked("seed must record a matching full GT pre-hash and scan hash")
    label = seed.get("seed_label")
    if not isinstance(label, str) or not label.startswith(tuple(wn.SEED_PREFIXES)) or \
            not label.startswith("controlled_init:dataset_onboarding:HB:") or not label.endswith(f"s{SEED_INDEX:06d}"):
        raise Blocked("seed label must be controlled_init:dataset_onboarding:HB:...s000009")
    try:
        pose = wc._pose(seed.get("seed_pose_m"), "seed_pose_m")
    except (wc.ContractError, TypeError, ValueError) as exc:
        raise Blocked(f"seed pose invalid: {exc}") from None
    if not all(math.isfinite(v) for row in pose for v in row):
        raise Blocked("seed pose is non-finite")
    return {"pose": [list(r) for r in pose], "label": label}


def validate_placement_review(review: Any, *, seed_sha: str, glb_sha: str, video_sha: str, selection_sha: str,
                              K: Sequence[float], anchor: Sequence[float]) -> dict:
    req = {"schema", "seed_sha256", "glb_sha256", "video_sha256", "selection_review_sha256", "K",
           "principal_point_origin", "principal_point_evidence", "anchor_frame_id", "anchor_pixel",
           "qualitative_review", "reviewed_by", "physical_proof"}
    if not isinstance(review, Mapping) or set(review) != req or review["schema"] != SCHEMA_PLACEMENT_REVIEW:
        raise Blocked("placement review fields/schema differ")
    if (review["seed_sha256"] != seed_sha or review["glb_sha256"] != glb_sha or review["video_sha256"] != video_sha
            or review["selection_review_sha256"] != selection_sha):
        raise Blocked("placement review does not bind the same seed/GLB/video/selection identities")
    if not isinstance(review["K"], list) or len(review["K"]) != 9 or \
            any(type(v) not in (int, float) for v in review["K"]) or [float(v) for v in review["K"]] != list(K):
        raise Blocked("placement review K differs from the dataset-reference K")
    if not isinstance(review["anchor_pixel"], list):
        raise Blocked("placement anchor pixel must be a list")
    if review["principal_point_origin"] not in ("integer_centers", "corner"):
        raise Blocked("placement review must state the principal point origin (integer_centers|corner)")
    if review["principal_point_evidence"] not in ("reviewed_assumption", "documented_dataset_convention"):
        raise Blocked("principal point evidence must be reviewed_assumption or documented_dataset_convention")
    if review["anchor_frame_id"] != f"s{SEED_INDEX:06d}" or list(review["anchor_pixel"]) != list(anchor):
        raise Blocked("placement anchor must be the reviewed first positive point on s000009")
    if review["physical_proof"] is not False:
        raise Blocked("a reviewed placement is not physical proof; physical_proof must be false")
    if not isinstance(review["qualitative_review"], str) or not review["qualitative_review"].strip() \
            or not isinstance(review["reviewed_by"], str) or not review["reviewed_by"].strip():
        raise Blocked("placement review needs a qualitative review and a reviewer")
    return dict(review)


def _producer_matches(receipt: Mapping, sources: Mapping, runtime: Mapping) -> None:
    before = (receipt.get("closure") or {}).get("before") or {}
    if receipt.get("tool_version") != TOOL_VERSION or before.get("sources") != sources or before.get("runtime") != runtime \
            or (receipt.get("closure") or {}).get("after_matches") is not True:
        raise Blocked(f"{receipt.get('schema')}: producer version/source closure/runtime differ from this assembly")


MODEL_RECEIPT_KEYS = frozenset({"schema", "tool_version", "status", "flags", "trust", "object_id", "family", "allowlist",
                                "source", "units", "glb", "geometry", "portable_targets", "budget", "input_budget",
                                "ledger", "closure", "mobile_verified", "preparation_scope"})
VIDEO_RECEIPT_KEYS = frozenset({"schema", "tool_version", "status", "flags", "trust", "allowlist", "video", "codec",
                                "process", "clock", "geometry", "split", "decode", "frames", "budget", "input_budget",
                                "runtime", "mobile_verified", "closure", "ledger", "preparation_scope",
                                "encoder_cleanup"})
MODEL_SOURCE_KEYS = frozenset({"member", "archive", "dataset_revision", "compressed_sha256", "uncompressed_sha256",
                               "crc32", "compressed_bytes", "expanded_bytes", "header"})
HEADER_KEYS = frozenset({"vertex_count", "face_count", "vertex_properties", "face_index_type", "comments",
                         "units_declared_in_header"})
ROUNDTRIP_KEYS = frozenset({"positions_bitwise_equal_float32", "indices_and_winding_equal", "vertex_colours_equal",
                            "normals_equal", "max_position_error_mm_vs_source", "producer_glb_mesh_positions_equal",
                            "producer_glb_mesh_triangles", "source_degenerate_faces_dropped_by_producer", "decimated"})
DECODE_KEYS = frozenset({"sequential_frames", "end_of_video_after", "dimensions", "backend", "frame_reader_seek_check",
                         "pixel_identity_claimed", "note"})
FRAME_KEYS = frozenset({"index", "frame_id", "split", "member", "png_sha256", "source_pixel_sha256",
                        "derivative_decoded_pixel_sha256", "decoded_identical_to_source"})


def _exact(value: Any, keys: frozenset, what: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise Blocked(f"{what}: missing or unknown fields")
    return value


def _scope_bound(ctx: Context, receipt: Mapping, what: str) -> None:
    if receipt.get("preparation_scope") != repo_rel(ctx, preparation_scope(ctx)):
        raise Blocked(f"{what} was produced for a different preparation scope")


def validate_model_receipt(ctx: Context, model: Any, allow: Mapping, allow_sha: str, review: Mapping,
                           sources: Mapping, runtime: Mapping) -> dict:
    """Nested structural coherence FIRST (fields, units, header/geometry/roundtrip/appearance), then
    the ACTUAL GLB: bytes, document (attributes, normalised COLOR_0, white unlit material, no UV or
    texture), finite unit normals, counts and index bounds, plus a producer re-decode. The PLY is
    not re-opened; the parent-accepted expected receipt hash remains the trust boundary."""
    from bench import workbench_neural as wn
    if not isinstance(model, Mapping) or set(model) != MODEL_RECEIPT_KEYS or model["schema"] != SCHEMA_MODEL \
            or model["status"] != "prepared" or model["flags"] != FLAGS or model["trust"] != TRUST \
            or model["family"] != FAMILY or model["mobile_verified"] is not False:
        raise Blocked("model receipt fields/schema/status/flags/trust/family differ")
    _producer_matches(model, sources, runtime)
    oid = review["object_id"]
    rows = [m for m in allow["models"] if m["object_id"] == oid]
    if model["object_id"] != oid or len(rows) != 1 or not isinstance(model["allowlist"], Mapping) \
            or model["allowlist"].get("sha256") != allow_sha:
        raise Blocked("model receipt object/allowlist differ from the reviewed object binding")
    row, src = rows[0], _exact(model["source"], MODEL_SOURCE_KEYS, "model source")
    if any(src[k] != row[k] for k in ("member", "archive", "compressed_sha256", "uncompressed_sha256", "crc32",
                                      "compressed_bytes", "expanded_bytes")) or src["dataset_revision"] != DATASET_REVISION:
        raise Blocked("model receipt source differs from the allowlisted member")
    header = _exact(src["header"], HEADER_KEYS, "model source header")
    props = header["vertex_properties"]
    if not isinstance(props, list) or not {"x", "y", "z", "red", "green", "blue"} <= set(props):
        raise Blocked("model source header lacks positions or vertex colours")
    has_normals = {"nx", "ny", "nz"} <= set(props)
    units = model["units"]
    if not isinstance(units, Mapping) or set(units) != {"source", "source_reference", "source_evidence", "output",
                                                        "factor", "applied"} \
            or units["source"] != "millimetres" or units["source_reference"] != BOP_FORMAT_URL \
            or units["output"] != "metres" or units["factor"] != MM_TO_M or units["applied"] != "exactly once":
        raise Blocked("model receipt units differ")
    geometry = _exact(model["geometry"], frozenset({"vertices", "triangles", "execution_cap_triangles", "roundtrip",
                                                     "geometry_verified"}), "model geometry")
    vertices, triangles = geometry["vertices"], geometry["triangles"]
    if (not _is_count(vertices, 3) or not _is_count(triangles, 1) or triangles > FACE_CAP
            or geometry["execution_cap_triangles"] != FACE_CAP or header["vertex_count"] != vertices
            or header["face_count"] != triangles):
        raise Blocked("model geometry counts contradict the source header or the execution cap")
    rt = _exact(geometry["roundtrip"], ROUNDTRIP_KEYS, "model roundtrip")
    if (rt["positions_bitwise_equal_float32"] is not True or rt["indices_and_winding_equal"] is not True
            or rt["vertex_colours_equal"] is not True or rt["normals_equal"] is not has_normals
            or rt["producer_glb_mesh_positions_equal"] is not True or rt["decimated"] is not False
            or not _is_count(rt["source_degenerate_faces_dropped_by_producer"])
            or rt["producer_glb_mesh_triangles"] != triangles - rt["source_degenerate_faces_dropped_by_producer"]
            or type(rt["max_position_error_mm_vs_source"]) not in (int, float)
            or not math.isfinite(rt["max_position_error_mm_vs_source"]) or rt["max_position_error_mm_vs_source"] < 0):
        raise Blocked("model roundtrip record is not a successful, normals-consistent round trip")
    portable = _exact(model["portable_targets"], frozenset({"triangles_target", "triangles", "bytes_target", "note"}),
                      "model portable targets")
    if (portable["triangles_target"] != PORTABLE_TRIANGLES or portable["bytes_target"] != PORTABLE_BYTES
            or portable["triangles"] != ("PASS" if triangles <= PORTABLE_TRIANGLES else "FAIL")):
        raise Blocked("model receipt portable result contradicts its triangle count")
    glb = _exact(model["glb"], frozenset({"path", "sha256", "bytes", "normals", "appearance", "uv"}), "model glb")
    if (glb["appearance"] != APPEARANCE or glb["uv"] != "none (no texture fabricated)"
            or not isinstance(glb["normals"], str)
            or glb["normals"].startswith("explicit source normals renormalised") is not has_normals):
        raise Blocked("model GLB appearance/UV/normal notes contradict the source")
    _scope_bound(ctx, model, "model receipt")
    # actual artifact
    path = inside_scope(ctx, contained_input(ctx, glb["path"], "GLB"), "GLB")
    raw = read_bounded(path, GLB_CAP, "GLB")
    if sha256_bytes(raw) != glb["sha256"] or len(raw) != glb["bytes"]:
        raise Blocked("actual GLB bytes/size contradict the model receipt")
    gltf, arrays = read_glb_document(raw)
    prim = gltf["meshes"][0]["primitives"][0]
    expected_attrs = {"POSITION", "COLOR_0"} | ({"NORMAL"} if has_normals else set())
    if set(prim["attributes"]) != expected_attrs or prim.get("mode") != 4 or prim.get("material") != 0:
        raise Blocked("actual GLB attributes/mode/material contradict the receipt (no UVs may be fabricated)")
    if any(k in gltf for k in ("images", "textures", "samplers")):
        raise Blocked("actual GLB carries a fabricated texture")
    materials = gltf.get("materials")
    pbr = (materials[0].get("pbrMetallicRoughness") or {}) if isinstance(materials, list) and len(materials) == 1 else {}
    if (len(materials or []) != 1 or pbr.get("baseColorFactor") != [1.0, 1.0, 1.0, 1.0] or "baseColorTexture" in pbr
            or "KHR_materials_unlit" not in (materials[0].get("extensions") or {})):
        raise Blocked("actual GLB material is not the single white unlit material")
    colour_acc = gltf["accessors"][prim["attributes"]["COLOR_0"]]
    if colour_acc.get("componentType") != _UBYTE or colour_acc.get("type") != "VEC4" or colour_acc.get("normalized") is not True:
        raise Blocked("actual GLB COLOR_0 is not normalised RGBA8")
    pos, idx = arrays["POSITION"], arrays["indices"]
    if (pos.shape != (vertices, 3) or not np.isfinite(pos).all() or arrays["COLOR_0"].shape != (vertices, 4)
            or idx.shape != (3 * triangles,) or (len(idx) and int(idx.max()) >= vertices)):
        raise Blocked("actual GLB positions/colours/indices contradict the receipt counts or bounds")
    if has_normals:
        normals = arrays["NORMAL"].astype(np.float64)
        if normals.shape != (vertices, 3) or not np.isfinite(normals).all() \
                or float(np.abs(np.linalg.norm(normals, axis=1) - 1.0).max()) > 1e-6:
            raise Blocked("actual GLB normals are not finite unit vectors for every vertex")
    _v, faces = wn.glb_mesh(path, max_faces=FACE_CAP)
    if rt["producer_glb_mesh_triangles"] != len(faces) or len(_v) != vertices:
        raise Blocked("producer re-decode contradicts the model receipt round trip")
    return {"path": path, "sha256": glb["sha256"], "bytes": len(raw), "dir": path.parent}


def validate_video_receipt(ctx: Context, video: Any, allow: Mapping, allow_sha: str, sources: Mapping,
                           runtime: Mapping) -> dict:
    """Nested structural coherence FIRST (fixed honest clock/geometry/decode profile, codec, process
    exit 0, exact frame mapping, encoder cleanup), then the ACTUAL video: bytes, size == drained
    stdout, workspace path, a full re-decode (120 then EOF) and the frame-exact s000010 seek."""
    if not isinstance(video, Mapping) or set(video) != VIDEO_RECEIPT_KEYS or video["schema"] != SCHEMA_VIDEO \
            or video["status"] != "prepared" or video["flags"] != FLAGS or video["trust"] != TRUST \
            or video["mobile_verified"] is not False:
        raise Blocked("video receipt fields/schema/status/flags/trust differ")
    _producer_matches(video, sources, runtime)
    if not isinstance(video["allowlist"], Mapping) or video["allowlist"].get("sha256") != allow_sha \
            or video["codec"] != codec_settings() or video["runtime"] != {**runtime, "ffmpeg": dict(FFMPEG)} \
            or video["split"] != FROZEN_SPLIT:
        raise Blocked("video receipt allowlist/codec/runtime/split differ")
    if video["clock"] != VIDEO_CLOCK:
        raise Blocked("video clock must be FRAME_INDEX, presentation 30 fps, null physical fps/timestamps and no "
                      "physical prefix attestation")
    if video["geometry"] != VIDEO_GEOMETRY:
        raise Blocked("video geometry must be native 640x480 with no crop, rotation or K change")
    decode = _exact(video["decode"], DECODE_KEYS, "video decode")
    if (decode["sequential_frames"] != FRAME_COUNT or decode["end_of_video_after"] != FRAME_COUNT
            or decode["dimensions"] != list(IMAGE_SIZE) or decode["pixel_identity_claimed"] is not False
            or decode["note"] != DECODE_NOTE or not isinstance(decode["backend"], str)
            or decode["frame_reader_seek_check"] != {"frame_id": f"s{SEED_INDEX + 1:06d}", "exact": True}):
        raise Blocked("video decode record is not exactly 120 frames, EOF, 640x480 and an exact s000010 seek")
    process = _exact(video["process"], frozenset({"exit_code", "stdout_bytes", "stderr_bytes", "stderr_tail"}),
                     "video process")
    if (type(process["exit_code"]) is not int or process["exit_code"] != 0 or not _is_count(process["stdout_bytes"], 1)
            or not _is_count(process["stderr_bytes"]) or not isinstance(process["stderr_tail"], str)
            or len(process["stderr_tail"]) > STDERR_TAIL):
        raise Blocked("video process record is not a bounded successful FFmpeg exit")
    cleanup = video["encoder_cleanup"]
    if (not isinstance(cleanup, Mapping) or cleanup.get("reaped") is not True or cleanup.get("returncode") != 0
            or cleanup.get("threads_alive") != [] or cleanup.get("staging_fd_closed") is not True
            or cleanup.get("pipes_closed") is not True):
        raise Blocked("video encoder cleanup record is incomplete")
    meta = _exact(video["video"], frozenset({"path", "workspace_relative", "sha256", "bytes", "cap_bytes"}), "video file")
    if meta["cap_bytes"] != VIDEO_CAP or not _is_count(meta["bytes"], 1) or meta["bytes"] > VIDEO_CAP \
            or meta["bytes"] != process["stdout_bytes"]:
        raise Blocked("video size contradicts the drained FFmpeg stdout or the cap")
    frames = video["frames"]
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise Blocked("video receipt must list exactly 120 frames")
    for i, (frame, row) in enumerate(zip(frames, allow["rgb"])):
        if (not isinstance(frame, Mapping) or set(frame) != FRAME_KEYS or frame["index"] != i
                or frame["frame_id"] != f"s{i:06d}" or frame["split"] != _split(i) or frame["member"] != row["member"]
                or frame["png_sha256"] != row["uncompressed_sha256"] or not _is_sha(frame["source_pixel_sha256"])
                or not _is_sha(frame["derivative_decoded_pixel_sha256"])
                or frame["decoded_identical_to_source"] is not (
                    frame["derivative_decoded_pixel_sha256"] == frame["source_pixel_sha256"])):
            raise Blocked(f"video frame {i} mapping contradicts the allowlist")
    _scope_bound(ctx, video, "video receipt")
    # actual artifact
    path = inside_scope(ctx, contained_input(ctx, meta["path"], "video"), "video")
    size = os.lstat(path).st_size
    if size != meta["bytes"] or sha256_file(path) != meta["sha256"]:
        raise Blocked("actual video bytes/size contradict the video receipt")
    if Path(os.path.abspath(ctx.workspace / meta["workspace_relative"])) != Path(os.path.abspath(path)):
        raise Blocked("video is not inside the workbench workspace root at its recorded path")
    decoded = reopen_and_decode(path)
    if decoded["pixel_sha256"] != [f["derivative_decoded_pixel_sha256"] for f in frames]:
        raise Blocked("actual decoded frames contradict the video receipt")
    return {"path": path, "sha256": meta["sha256"], "bytes": size, "workspace_relative": meta["workspace_relative"],
            "dir": path.parent}


def assemble(ctx: Context, *, allowlist: str, expected_allowlist_sha256: str, model_receipt: str,
             expected_model_receipt_sha256: str, video_receipt: str, expected_video_receipt_sha256: str,
             seed: str | None, expected_seed_sha256: str | None, selection_review: str | None,
             expected_selection_review_sha256: str | None, placement_review: str | None,
             expected_placement_review_sha256: str | None, calibrations: str, expected_calibrations_sha256: str,
             out: str) -> dict:
    missing = [n for n, v in (("seed", seed), ("selection review", selection_review),
                              ("placement review", placement_review)) if not v]
    if missing:
        raise Blocked(f"assembly refused: missing prerequisite(s) {missing}; no bundle is written")
    from bench import workbench_contract as wc
    from bench import workbench_neural as wn
    out_dir = check_output_dir(ctx, out)
    with guarded_stage(ctx) as opened:
        sources, runtime = source_closure(), runtime_identity()
        allow_sha = check_sha(expected_allowlist_sha256, "--expected-allowlist-sha256")
        allow, allow_path, payload_root = load_allowlist(ctx, allowlist, allow_sha)
        _set_payload_root(payload_root)                                    # zero payload opens
        pinned = {}
        for key, (value, expected, limit) in {
                "model_receipt": (model_receipt, expected_model_receipt_sha256, MAX_JSON_OUTPUT),
                "video_receipt": (video_receipt, expected_video_receipt_sha256, MAX_JSON_OUTPUT),
                "selection_review": (selection_review, expected_selection_review_sha256, MAX_REVIEW_BYTES),
                "seed": (seed, expected_seed_sha256, MAX_REVIEW_BYTES),
                "placement_review": (placement_review, expected_placement_review_sha256, MAX_REVIEW_BYTES),
                "calibrations": (calibrations, expected_calibrations_sha256, MAX_CALIBRATION_BYTES)}.items():
            raw, path = read_pinned(ctx, value, expected, limit, key.replace("_", " "))
            pinned[key] = {"raw": raw, "path": path, "sha256": sha256_bytes(raw)}
        review = validate_selection_review(strict_json(pinned["selection_review"]["raw"], "selection review"), allow)
        model = validate_model_receipt(ctx, strict_json(pinned["model_receipt"]["raw"], "model receipt"), allow,
                                       allow_sha, review, sources, runtime)
        video_doc = strict_json(pinned["video_receipt"]["raw"], "video receipt")
        video = validate_video_receipt(ctx, video_doc, allow, allow_sha, sources, runtime)
        if video_doc["frames"][SEED_INDEX]["png_sha256"] != review["rgb_source_sha256"]:
            raise Blocked("video frame 9 source differs from the reviewed RGB frame")
        seed_doc = validate_seed(strict_json(pinned["seed"]["raw"], "seed"), pinned["selection_review"]["sha256"],
                                 review, allow, allow_sha)
        seed_dir = pinned["seed"]["path"].parent
        seed_receipt_raw = read_bounded(contained_input(ctx, seed_dir / "seed-receipt.json", "seed receipt"),
                                        MAX_JSON_OUTPUT, "seed receipt")
        seed_receipt = strict_json(seed_receipt_raw, "seed receipt")
        seed_module_sha = sha256_file(ROOT / SEED_MODULE)
        if (not isinstance(seed_receipt, Mapping) or seed_receipt.get("schema") != SCHEMA_SEED_RECEIPT
                or seed_receipt.get("status") != "extracted"
                or (seed_receipt.get("seed") or {}).get("sha256") != pinned["seed"]["sha256"]
                or (seed_receipt.get("closure") or {}).get("after_matches") is not True
                or (seed_receipt.get("closure") or {}).get("after_publication_matches") is not True
                or ((seed_receipt.get("closure") or {}).get("before") or {}).get("sources") != sources
                or ((seed_receipt.get("closure") or {}).get("before") or {}).get("runtime") != runtime
                or ((seed_receipt.get("closure") or {}).get("before") or {}).get("seed_module") != seed_module_sha):
            raise Blocked("seed receipt does not bind this seed, the current sources/runtime and seed module")
        _scope_bound(ctx, seed_receipt, "seed receipt")
        K = select_calibration(strict_json(pinned["calibrations"]["raw"], "calibrations"))
        placement = validate_placement_review(strict_json(pinned["placement_review"]["raw"], "placement review"),
                                              seed_sha=pinned["seed"]["sha256"], glb_sha=model["sha256"],
                                              video_sha=video["sha256"], selection_sha=pinned["selection_review"]["sha256"],
                                              K=K, anchor=review["positive"][0])
        # locations AFTER structural validation: outputs inside the bound scope, inputs outside it
        if pinned["model_receipt"]["path"].parent != model["dir"] or pinned["video_receipt"]["path"].parent != video["dir"]:
            raise Blocked("model/video receipts must sit beside the artifacts they describe")
        inside_scope(ctx, pinned["seed"]["path"], "seed")
        for key in ("selection_review", "placement_review", "calibrations"):
            outside_scope_input(ctx, pinned[key]["path"], key.replace("_", " "))
        prior_dirs = {allow_path.parent, model["dir"], video["dir"], seed_dir}
        if len(prior_dirs) != 4 or ctx.scope in prior_dirs:
            raise Blocked("allowlist, model, video and seed outputs must be distinct stage directories in the scope")
        budget = workspace_budget(ctx, STAGE_ALLOWANCE["assemble"])
        ledger = OutputLedger(ctx, out_dir, "assemble")
        accounting = ledger.check(0, metadata=True)          # whole scope incl. retained failed siblings
        bundle = {"schema": BUNDLE_SCHEMA, "family": FAMILY, "glb_path": repo_rel(ctx, model["path"]),
                  "glb_sha256": model["sha256"], "obj_id": review["object_id"],
                  "principal_point_origin": placement["principal_point_origin"],
                  "seed_pose_m": seed_doc["pose"], "seed_label": seed_doc["label"],
                  "seed_frame_id": f"s{SEED_INDEX:06d}", "metric_provenance": "dataset_metric_cad"}
        frames = [{"frame_id": f"s{i:06d}", "source_frame_index": i, "timestamp_s": None, "media_time_s": None}
                  for i in range(FRAME_COUNT)]
        clip = {"schema": wc.CLIP_SCHEMA, "clip_id": f"hb-val-primesense-{SCENE}",
                "title": f"HB val_primesense {SCENE} (lossy mpeg4 derivative of 120 RGB PNGs)",
                "source_kind": "video_file", "clock_kind": "frame_index", "clock_source": None,
                "width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1],
                "media": {"root_id": "workbench", "path": video["workspace_relative"], "sha256": video["sha256"],
                          "bytes": video["bytes"]},
                "frames": frames,
                "intrinsics": {"status": "dataset_reference", "K": K,
                               "source": f"HB dataset reference K (calibrations sha256 {pinned['calibrations']['sha256'][:16]}); "
                                         "not a phone calibration"},
                "onboarding": {"keyframe_ids": [f"s{i:06d}" for i in OPENING],
                               "selection": {"frame_id": f"s{SEED_INDEX:06d}", "positive": review["positive"],
                                             "negative": review["negative"], "reviewed": True,
                                             "reviewed_by": review["reviewed_by"]}},
                "scoring_frame_ids": [f"s{i:06d}" for i in SCORING],
                "evidence_role": "dataset_prepared_metric_control_development",
                "evaluator_inputs_read": False, "excluded_source_fields": [],
                "provenance": {"family": FAMILY, "trust_level": TRUST["trust_level"], "upstream_authenticated": False,
                               "presentation_fps": PRESENTATION_FPS, "physical_fps": None, "lossy_derivative": True,
                               "principal_point_evidence": placement["principal_point_evidence"],
                               "registered": False, "admitted": False}}
        wc.ClipSpec.from_dict(clip)                                         # core validation, nothing registered

        def closure_now() -> dict:
            files = {"allowlist": sha256_file(allow_path), "glb": sha256_file(model["path"]),
                     "video": sha256_file(video["path"]), "seed_receipt": sha256_bytes(read_bounded(
                         seed_dir / "seed-receipt.json", MAX_JSON_OUTPUT, "seed receipt")),
                     **{k: sha256_file(v["path"]) for k, v in pinned.items()}}
            return {"inputs": files, "sources": source_closure(), "runtime": runtime_identity(),
                    "seed_module": sha256_file(ROOT / SEED_MODULE)}
        before = {"inputs": {"allowlist": allow_sha, "glb": model["sha256"], "video": video["sha256"],
                             "seed_receipt": sha256_bytes(seed_receipt_raw),
                             **{k: v["sha256"] for k, v in pinned.items()}},
                  "sources": sources, "runtime": runtime, "seed_module": seed_module_sha}
        if closure_now() != before:
            raise InputsError("an assembly input, source or runtime changed during validation")
        bundle_bytes, clip_bytes = dump_json(bundle), dump_json(clip)
        ledger.check(len(bundle_bytes) + len(clip_bytes), metadata=True)
        _make_fresh_dir(out_dir)
        try:
            bundle_meta = publish_bytes(ledger, "metric-bundle.json", bundle_bytes)
            parsed = wn.MetricBundle.from_file(out_dir / "metric-bundle.json", root=ctx.root)   # core loader check
            if parsed.family != FAMILY or parsed.seed_frame_id != f"s{SEED_INDEX:06d}":
                raise InputsError("core MetricBundle parse differs")
            clip_meta = publish_bytes(ledger, "clip-proposal.json", clip_bytes)
            receipt = {"schema": SCHEMA_ASSEMBLY, "tool_version": TOOL_VERSION, "status": "assembled_unadmitted_unregistered",
                       "flags": dict(FLAGS), "trust": dict(TRUST), "inputs": before["inputs"],
                       "outputs": {"metric_bundle": bundle_meta, "clip_proposal": clip_meta},
                       "admission": {"admitted": False, "family": FAMILY,
                                     "note": "no existing admission covers this family; a fresh reviewed admission is "
                                             "required before any neural execution; nothing was mutated"},
                       "registration": {"registered": False, "note": "clip proposal only; the workspace clip registry "
                                                                     "was not touched"},
                       "seed_usage": {"applies_once_to_frame_id": f"s{SEED_INDEX + 1:06d}", "reuse_on_recovery": False},
                       "principal_point": {"origin": placement["principal_point_origin"],
                                           "evidence": placement["principal_point_evidence"], "physical_proof": False},
                       "aggregate_accounting": {"scope_usage_before_assembly": accounting,
                                                "assembly_ledger": ledger.summary(),
                                                "preparation_scope": repo_rel(ctx, ctx.scope)},
                       "preparation_scope": repo_rel(ctx, ctx.scope),
                       "budget": budget, "closure": {"before": before, "after_matches": True},
                       "mobile_verified": False}
            if closure_now() != before:
                raise InputsError("an assembly input, source or runtime changed before publication")
            published = publish_bytes(ledger, "assembly-receipt.json", dump_json(receipt))
        except BaseException as exc:
            _failure(ledger, "assemble", exc)
            raise
        if opened:
            raise GuardError("assembly opened payload bytes")
    return {"receipt": repo_rel(ctx, out_dir / "assembly-receipt.json"), "receipt_sha256": published["sha256"],
            "status": "assembled_unadmitted_unregistered", "exit": 0}


# --------------------------------------------------------------------------
# CLI


def _object_id(text: str) -> int:
    if not re.fullmatch(r"[0-9]{1,2}", text) or int(text) not in MODEL_IDS:
        raise argparse.ArgumentTypeError(f"object id must be {MODEL_IDS[0]}..{MODEL_IDS[-1]}")
    return int(text)


def _count(text: str) -> int:
    if not re.fullmatch(r"[0-9]{1,15}", text):
        raise argparse.ArgumentTypeError("must be a non-negative decimal integer")
    return int(text)


def apply_preparation_root(ctx: Context, value: str | None) -> Context:
    """Bind the ONE explicit preparation scope (root-relative or absolute, inside ``derived``)."""
    if value is None:
        return ctx
    path = Path(value)
    if not path.is_absolute():
        path = ctx.root / path
    if ".." in path.parts:
        raise UsageError("--preparation-root: traversal is not allowed")
    bound = Context(root=ctx.root, workspace=ctx.workspace, preparation=Path(os.path.abspath(path)))
    preparation_scope(bound)
    return bound


def main(argv: Sequence[str] | None = None, ctx: Context | None = None) -> int:
    ctx = ctx or Context()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("filter-allowlist")
    s.add_argument("--acquisition-receipt", required=True)
    s.add_argument("--expected-receipt-sha256", required=True)
    s.add_argument("--resources-index", required=True)
    s.add_argument("--expected-index-sha256", required=True)
    s.add_argument("--integrity-record", required=True)
    s.add_argument("--expected-integrity-sha256", required=True)
    s.add_argument("--object-id", type=_object_id, action="append", required=True)
    s.add_argument("--preparation-root")
    s.add_argument("--out", required=True)
    for name in ("prepare-model", "prepare-video"):
        s = sub.add_parser(name)
        s.add_argument("--allowlist", required=True)
        s.add_argument("--expected-allowlist-sha256", required=True)
        s.add_argument("--existing-input-bytes", type=_count, required=True)
        s.add_argument("--preparation-root")
        s.add_argument("--out", required=True)
        if name == "prepare-model":
            s.add_argument("--object-id", type=_object_id, required=True)
    s = sub.add_parser("assemble")
    for name in ("allowlist", "model-receipt", "video-receipt", "seed", "selection-review", "placement-review",
                 "calibrations"):
        required = name in ("allowlist", "model-receipt", "video-receipt", "calibrations")
        s.add_argument(f"--{name}", required=required)
        s.add_argument(f"--expected-{name}-sha256", required=required)
    s.add_argument("--preparation-root")
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        ctx = apply_preparation_root(ctx, args.preparation_root)
        if args.stage == "filter-allowlist":
            result = filter_allowlist(ctx, receipt_path=args.acquisition_receipt,
                                      expected_receipt_sha256=args.expected_receipt_sha256,
                                      index_path=args.resources_index, expected_index_sha256=args.expected_index_sha256,
                                      integrity_path=args.integrity_record,
                                      expected_integrity_sha256=args.expected_integrity_sha256,
                                      object_ids=args.object_id, out=args.out)
        elif args.stage == "prepare-model":
            result = prepare_model(ctx, allowlist=args.allowlist, expected_allowlist_sha256=args.expected_allowlist_sha256,
                                   object_id=args.object_id, existing_input_bytes=args.existing_input_bytes, out=args.out)
        elif args.stage == "prepare-video":
            result = prepare_video(ctx, allowlist=args.allowlist, expected_allowlist_sha256=args.expected_allowlist_sha256,
                                   existing_input_bytes=args.existing_input_bytes, out=args.out)
        else:
            result = assemble(ctx, allowlist=args.allowlist, expected_allowlist_sha256=args.expected_allowlist_sha256,
                              model_receipt=args.model_receipt, expected_model_receipt_sha256=args.expected_model_receipt_sha256,
                              video_receipt=args.video_receipt, expected_video_receipt_sha256=args.expected_video_receipt_sha256,
                              seed=args.seed, expected_seed_sha256=args.expected_seed_sha256,
                              selection_review=args.selection_review,
                              expected_selection_review_sha256=args.expected_selection_review_sha256,
                              placement_review=args.placement_review,
                              expected_placement_review_sha256=args.expected_placement_review_sha256,
                              calibrations=args.calibrations, expected_calibrations_sha256=args.expected_calibrations_sha256,
                              out=args.out)
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except Blocked as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 3
    except InputsError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("exit", 0))


if __name__ == "__main__":
    if not sys.dont_write_bytecode:
        raise SystemExit("Use python -B (for example: python -B -S tools/prototype_bootstrap.py "
                         "tools.workbench_hb_metric_inputs ...)")
    raise SystemExit(main())
