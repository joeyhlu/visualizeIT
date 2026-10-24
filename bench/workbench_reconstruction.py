"""Real installed-COLMAP surface reconstruction for the workbench.

Only a fixed allowlist of COLMAP commands is run, with flags verified against
the installed executable's own ``--help`` output. Missing COLMAP, unsupported
flags, failed stages or failed admission gates produce an explicit
``unavailable`` SurfacePackage plus an original image/mask preview; no mesh,
curvature or pose is ever invented.

Inputs are only the causal onboarding prefix (<= 24 frames; <= 10 s when the
clip clock attests it), their observed masks and intrinsics. Evaluator data is
never accepted.

Conventions: COLMAP principal points and keypoints use a corner origin; the
workbench uses integer pixel centres. Supplying K adds 0.5 into COLMAP;
parsing subtracts 0.5. Dense depths are interpreted with the *undistorted*
cameras written by ``image_undistorter``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import workbench_contract as wc
from . import workbench_geometry as wg

COMMANDS = ("feature_extractor", "exhaustive_matcher", "mapper", "model_converter", "image_undistorter",
            "patch_match_stereo", "stereo_fusion", "poisson_mesher")
REQUIRED_FLAGS = {
    "feature_extractor": ("--database_path", "--image_path", "--ImageReader.camera_model",
                          "--ImageReader.camera_params", "--ImageReader.single_camera", "--ImageReader.mask_path"),
    "exhaustive_matcher": ("--database_path",),
    "mapper": ("--database_path", "--image_path", "--output_path"),
    "model_converter": ("--input_path", "--output_path", "--output_type"),
    "image_undistorter": ("--image_path", "--input_path", "--output_path"),
    "patch_match_stereo": ("--workspace_path",),
    "stereo_fusion": ("--workspace_path", "--output_path"),
    "poisson_mesher": ("--input_path", "--output_path"),
}
ALTERNATIVE_FLAGS = {
    # newer COLMAP uses FeatureExtraction.*; older releases SiftExtraction.*
    "feature_extractor_gpu": ("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"),
    "matcher_gpu": ("--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"),
    # COLMAP 3.11.1 exposes --SiftMatching.num_threads; the newer name is used
    # only when the installed executable's own help lists it.
    "matcher_threads": ("--SiftMatching.num_threads", "--FeatureMatching.num_threads"),
}
# Pinned CPU matcher resource policy. Measured evidence: the default-thread CPU
# exhaustive matcher aborted natively (Windows 0xc0000409) in run
# run-20261021T011149-6654274b, while the same COLMAP 3.11.1 executable, copied
# database and features completed with use_gpu 0 + num_threads 1
# (derived/diagnostics/matcher-cpu-one-thread-v1). No memory-cause or quality
# claim is made; this only pins the configuration that actually completed.
MATCHER_RESOURCE_POLICY = {"name": "cpu_exhaustive_matcher_one_thread_v1", "use_gpu": 0, "num_threads": 1,
                           "evidence": "derived/diagnostics/matcher-cpu-one-thread-v1/diagnostic.json"}
FLAG_RE = re.compile(r"--[A-Za-z_][A-Za-z0-9_.]*")
VERSION_RE = re.compile(r"COLMAP\s+(\d+\.\d+(?:\.\d+)?(?:-dev)?)")


class ReconstructionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    max_frames: int = wc.MAX_KEYFRAMES
    stage_timeout_s: float = 1800.0
    workspace_bytes: int = wc.DERIVED_BUDGET_BYTES
    free_reserve_bytes: int = wc.FREE_RESERVE_BYTES
    max_log_bytes: int = 8 * 1024 * 1024
    poll_s: float = 0.5
    min_face_support: int = wc.ADMISSION_GATES["min_face_support_views"]
    depth_tolerance: float = 0.02
    marker_size: float = 0.2
    baseline_bytes: int = 0  # derived/scratch/cache bytes already used elsewhere in the 1 GiB budget

    def to_dict(self) -> dict:
        return asdict(self)


def budget_violation(workspace: Path, limits: "Limits", *, extra: int = 0) -> str | None:
    """Return a reason if baseline + workspace bytes + extra would exceed the budget
    or the free-disk reserve; ``None`` when within limits."""
    used = limits.baseline_bytes + _tree_size(workspace) + extra
    if used > limits.workspace_bytes:
        return f"workspace byte budget exceeded ({used} > {limits.workspace_bytes})"
    probe = workspace if workspace.exists() else workspace.parent
    if shutil.disk_usage(probe).free - extra < limits.free_reserve_bytes:
        return "free-disk reserve would be violated"
    return None


# --------------------------------------------------------------------------
# preflight

def _run_capture(argv: Sequence[str], timeout: float = 30.0) -> tuple[int | None, str]:
    try:
        proc = subprocess.run(list(argv), capture_output=True, timeout=timeout, check=False,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    text = (proc.stdout or b"").decode("utf-8", "replace") + (proc.stderr or b"").decode("utf-8", "replace")
    return proc.returncode, text[-200_000:]


def find_colmap(explicit: str | os.PathLike | None = None) -> Path | None:
    candidates = [explicit, os.environ.get("VISUALIZEIT_COLMAP")]
    for candidate in candidates:
        if candidate:
            p = Path(candidate)
            return p.resolve() if p.is_file() else None
    for name in ("colmap", "colmap.exe", "colmap.bat", "COLMAP.bat"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    return None


def preflight(executable: str | os.PathLike | None = None, *, runner: Callable = _run_capture) -> dict:
    """Inspect the installed executable's version and per-command flags (help only)."""
    exe = find_colmap(executable)
    if exe is None:
        return {"available": False, "executable": None, "version": None, "commands": {},
                "reason": "colmap_not_installed: no executable on PATH or VISUALIZEIT_COLMAP; nothing installed",
                "action": "install COLMAP separately (operator) and pass --colmap PATH; the workbench never installs it"}
    code, text = runner([str(exe), "help"])
    version_match = VERSION_RE.search(text or "")
    commands, missing = {}, []
    for cmd in COMMANDS:
        c_code, c_text = runner([str(exe), cmd, "--help"])
        flags = sorted(set(FLAG_RE.findall(c_text or "")))
        need = REQUIRED_FLAGS[cmd]
        lacking = [f for f in need if f not in flags]
        commands[cmd] = {"returncode": c_code, "flags": flags, "missing_flags": lacking}
        if c_code is None or lacking:
            missing.append(cmd)
    gpu_ext = next((f for f in ALTERNATIVE_FLAGS["feature_extractor_gpu"] if f in commands["feature_extractor"]["flags"]), None)
    gpu_match = next((f for f in ALTERNATIVE_FLAGS["matcher_gpu"] if f in commands["exhaustive_matcher"]["flags"]), None)
    match_threads = next((f for f in ALTERNATIVE_FLAGS["matcher_threads"]
                          if f in commands["exhaustive_matcher"]["flags"]), None)
    type_flag = "--FeatureExtraction.type" if "--FeatureExtraction.type" in commands["feature_extractor"]["flags"] else None
    if gpu_ext is None:
        missing.append("feature_extractor:use_gpu")
    # The pinned matcher policy needs both controls; no silent fallback to the
    # executable's default GPU use or unbounded thread count.
    if gpu_match is None:
        missing.append("exhaustive_matcher:use_gpu")
    if match_threads is None:
        missing.append("exhaustive_matcher:num_threads")
    available = not missing and version_match is not None
    return {"available": available, "executable": str(exe), "version": version_match.group(1) if version_match else None,
            "help_returncode": code, "commands": commands, "feature_gpu_flag": gpu_ext, "matcher_gpu_flag": gpu_match,
            "matcher_threads_flag": match_threads, "feature_type_flag": type_flag,
            "matcher_resource_policy": dict(MATCHER_RESOURCE_POLICY),
            "reason": None if available else f"unsupported_colmap: missing {missing or ['version string']}",
            "action": None if available else "use a COLMAP build exposing the listed flags"}


# --------------------------------------------------------------------------
# process supervision

def _tree_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30,
                           check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


EXCERPT_CHARS = 2048
METADATA_STRING_CHARS = 4096


def safe_text(value: Any, limit: int = EXCERPT_CHARS) -> tuple[str, bool]:
    """Control-free escaped text, truncated AFTER escaping to ``limit`` chars.

    Newlines/tabs/other control characters become visible escapes, so the
    result always satisfies the contract's bounded, control-free string rule.
    """
    text = value if isinstance(value, str) else str(value)
    escaped = "".join(ch if (ord(ch) >= 32 and ch != "\x7f") else
                      {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, f"\\x{ord(ch):02x}") for ch in text)
    if len(escaped) <= limit:
        return escaped, False
    marker = "...[truncated]"
    return escaped[: max(0, limit - len(marker))] + marker, True


def sanitize_metadata(value: Any, depth: int = 0) -> Any:
    """Recursively make metadata contract-safe: control-free strings <= 4096
    chars, finite numbers (non-finite -> None), bounded nesting."""
    if depth > 8:
        return safe_text(value, 256)[0]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return safe_text(value, METADATA_STRING_CHARS)[0]
    if isinstance(value, Mapping):
        return {safe_text(k, 128)[0]: sanitize_metadata(v, depth + 1) for k, v in list(value.items())[:256]}
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(v, depth + 1) for v in list(value)[:4096]]
    if isinstance(value, Path):
        return safe_text(str(value), METADATA_STRING_CHARS)[0]
    if hasattr(value, "tolist"):
        return sanitize_metadata(value.tolist(), depth + 1)
    return safe_text(repr(value), 512)[0]


def _tail(path: Path, limit: int = 16_384) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _log_ref(path: Path, workspace: Path) -> dict:
    """Reference to the full bounded raw log (kept unchanged) plus a safe excerpt of its tail."""
    ref: dict[str, Any] = {"path": None, "bytes": None, "sha256": None, "excerpt": "", "excerpt_truncated": False}
    try:
        try:
            ref["path"] = Path(path).resolve().relative_to(Path(workspace).resolve()).as_posix()
        except ValueError:
            ref["path"] = Path(path).name
        if Path(path).is_file():
            ref["bytes"] = Path(path).stat().st_size
            ref["sha256"] = _sha_file(Path(path))
            raw = _tail(Path(path))
            ref["excerpt"], truncated = safe_text(raw, EXCERPT_CHARS)
            ref["excerpt_truncated"] = truncated or ref["bytes"] > 16_384
    except OSError as exc:
        ref["error"] = safe_text(f"{type(exc).__name__}: {exc}", 300)[0]
    return ref


def _stage_receipt(name: str, argv: Sequence[str], *, status: str, returncode: int | None, detail: Any,
                   elapsed_s: float, log_dir: Path, workspace: Path, gpu_stage: str, started_at: float) -> dict:
    out_path, err_path = log_dir / f"{name}.stdout.log", log_dir / f"{name}.stderr.log"
    stdout, stderr = _log_ref(out_path, workspace), _log_ref(err_path, workspace)
    receipt = {"stage": safe_text(name, 128)[0], "argv": [safe_text(a, 1024)[0] for a in argv],
               "status": status, "returncode": returncode,
               "detail": None if detail is None else safe_text(detail, 1024)[0],
               "elapsed_s": round(float(elapsed_s), 3), "started_at": started_at,
               "gpu_stage": gpu_stage, "gpu_stage_policy": GPU_STAGE_POLICY,
               "stdout": stdout, "stderr": stderr,
               # Back-compatible keys, now control-free and bounded.
               "stdout_tail": stdout["excerpt"], "stderr_tail": stderr["excerpt"]}
    _write_stage_receipt(log_dir, name, receipt)
    return receipt


def _write_stage_receipt(log_dir: Path, name: str, receipt: dict) -> None:
    """Promptly persist a small fsynced receipt, independent of package serialization.
    Never overwrites: a repeated stage name gets a numbered sibling."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        data = json.dumps(receipt, sort_keys=True, ensure_ascii=True).encode("ascii")
        for n in range(1000):
            path = log_dir / (f"{name}.stage.receipt.json" if n == 0 else f"{name}.stage.receipt.{n}.json")
            try:
                with path.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                receipt["receipt_file"] = path.name
                return
            except FileExistsError:
                continue
    except OSError as exc:
        receipt["receipt_write_error"] = safe_text(f"{type(exc).__name__}: {exc}", 300)[0]


GPU_STAGE_POLICY = ("every COLMAP subprocess (CPU-only feature/matching commands included) runs inside the "
                    "repository GPU stage; a busy stage fails the step without spawning")


def run_stage(name: str, argv: Sequence[str], *, workspace: Path, log_dir: Path, limits: Limits,
              cancel: threading.Event | None, gpu_lock_dir: Path | None = None) -> dict:
    """Run one allowlisted command under the repository GPU stage.

    The stage is acquired before ``Popen`` and held until the process has been
    waited for (and killed on timeout/cancel/budget), then released on every
    path. If another stage holds it, a structured ``failed`` receipt with
    reason ``gpu_stage_busy`` is returned and nothing is spawned; the foreign
    lock is never removed. A spawn failure (``OSError`` from ``Popen``) is a
    structured ``failed`` receipt too. Every outcome writes an fsynced
    ``<stage>.stage.receipt.json`` next to the raw logs. Serializing CPU-only
    commands is intentional and conservative.
    """
    if Path(argv[0]).name.lower().split(".")[0] != "colmap" or argv[1] not in COMMANDS:
        raise ReconstructionUnavailable("only allowlisted COLMAP commands may run")
    from contextlib import ExitStack
    from . import workbench_telemetry as wt
    started_at = time.time()
    stack = ExitStack()
    try:
        stack.enter_context(wt.gpu_stage(f"colmap_{name}", gpu_lock_dir))
    except RuntimeError as exc:
        return _stage_receipt(name, argv, status="failed", returncode=None,
                              detail=f"gpu_stage_busy: not started ({exc})", elapsed_s=0.0, log_dir=log_dir,
                              workspace=workspace, gpu_stage="busy", started_at=started_at)
    with stack:
        status, returncode, detail, elapsed = _run_stage_unlocked(name, argv, workspace=workspace, log_dir=log_dir,
                                                                  limits=limits, cancel=cancel)
    return _stage_receipt(name, argv, status=status, returncode=returncode, detail=detail, elapsed_s=elapsed,
                          log_dir=log_dir, workspace=workspace, gpu_stage="held", started_at=started_at)


def _run_stage_unlocked(name: str, argv: Sequence[str], *, workspace: Path, log_dir: Path, limits: Limits,
                        cancel: threading.Event | None) -> tuple[str, int | None, str | None, float]:
    """Process supervision body; callers must hold the GPU stage (see ``run_stage``).
    Returns (status, returncode, detail, elapsed_s); raw logs stay on disk unchanged."""
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path, err_path = log_dir / f"{name}.stdout.log", log_dir / f"{name}.stderr.log"
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    started = time.monotonic()
    status, detail = "completed", None
    before = budget_violation(workspace, limits)
    if before is not None:
        return "failed", None, f"not started: {before}", 0.0
    with out_path.open("wb") as out, err_path.open("wb") as err:
        try:
            proc = subprocess.Popen(list(argv), stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                    cwd=str(workspace), **kwargs)
        except OSError as exc:
            return "failed", None, f"spawn_failed: {type(exc).__name__}: {exc}", round(time.monotonic() - started, 3)
        last_budget = 0.0
        while proc.poll() is None:
            time.sleep(limits.poll_s)
            now = time.monotonic()
            if cancel is not None and cancel.is_set():
                status, detail = "cancelled", "cancel requested"
            elif now - started > limits.stage_timeout_s:
                status, detail = "failed", f"timeout after {limits.stage_timeout_s}s"
            elif out_path.stat().st_size + err_path.stat().st_size > limits.max_log_bytes:
                status, detail = "failed", "log bound exceeded"
            elif now - last_budget > 2.0:
                last_budget = now
                violation = budget_violation(workspace, limits)
                if violation is not None:
                    status, detail = "failed", violation
            if detail is not None:
                _kill_tree(proc)
                break
        returncode = proc.wait()
    if status == "completed" and returncode != 0:
        status, detail = "failed", f"exit code {returncode}"
    if status == "completed":
        # A fast process can finish between polls; re-check every bound after exit.
        if out_path.stat().st_size + err_path.stat().st_size > limits.max_log_bytes:
            status, detail = "failed", "log bound exceeded"
        else:
            violation = budget_violation(workspace, limits)
            if violation is not None:
                status, detail = "failed", f"after exit: {violation}"
    return status, returncode, detail, round(time.monotonic() - started, 3)


# --------------------------------------------------------------------------
# COLMAP parsers (TXT sparse, dense .bin depth, PLY)

def _quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _data_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            yield line


def read_cameras_txt(path: Path) -> dict[int, dict]:
    cams = {}
    for line in _data_lines(path):
        parts = line.split()
        cid, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        if model != "PINHOLE" or len(params) != 4:
            raise ReconstructionUnavailable(f"unsupported COLMAP camera model {model}; only PINHOLE is parsed")
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]])  # corner -> integer centres
        cams[cid] = {"width": w, "height": h, "K": K}
    return cams


def read_images_txt(path: Path) -> dict[str, dict]:
    lines = list(_data_lines_keep_empty(path))
    images = {}
    for i in range(0, len(lines) - 1, 2):
        head, obs = lines[i].split(), lines[i + 1].split()
        image_id = int(head[0])
        qw, qx, qy, qz, tx, ty, tz = map(float, head[1:8])
        name = " ".join(head[9:])
        pose = np.eye(4)
        pose[:3, :3] = _quat_to_R(qw, qx, qy, qz)
        pose[:3, 3] = (tx, ty, tz)
        arr = np.array(obs, dtype=np.float64).reshape(-1, 3) if obs else np.zeros((0, 3))
        xy = arr[:, :2] - 0.5
        pids = arr[:, 2].astype(np.int64)
        images[name] = {"image_id": image_id, "camera_id": int(head[8]), "pose": pose, "xy": xy, "point_ids": pids}
    return images


def _data_lines_keep_empty(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            yield line


def read_points3d_txt(path: Path) -> dict[int, dict]:
    points = {}
    for line in _data_lines(path):
        parts = line.split()
        pid = int(parts[0])
        xyz = np.array(list(map(float, parts[1:4])))
        track = np.array(parts[8:], dtype=np.int64).reshape(-1, 2)
        points[pid] = {"xyz": xyz, "error": float(parts[7]), "track": track}
    return points


def read_colmap_array(path: Path) -> np.ndarray:
    """COLMAP dense ``.bin`` array: ASCII ``w&h&c&`` header then column-major float32."""
    raw = path.read_bytes()
    amp, idx = 0, 0
    while amp < 3:
        if idx >= min(len(raw), 64):
            raise ReconstructionUnavailable("malformed COLMAP array header")
        if raw[idx:idx + 1] == b"&":
            amp += 1
        idx += 1
    w, h, c = (int(x) for x in raw[:idx].decode("ascii").split("&")[:3])
    data = np.frombuffer(raw[idx:], dtype="<f4")
    if data.size != w * h * c:
        raise ReconstructionUnavailable("COLMAP array size mismatch")
    arr = data.reshape((w, h, c), order="F").transpose(1, 0, 2)
    return arr[..., 0] if c == 1 else arr


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply(path: Path, *, max_faces: int = 5_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Parse ascii / binary_little_endian PLY vertices (x,y,z) and triangle faces."""
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    if not raw.startswith(b"ply") or end < 0:
        raise ReconstructionUnavailable("not a PLY file")
    nl = raw.index(b"\n", end)
    header = raw[:nl].decode("ascii", "replace").splitlines()
    body = raw[nl + 1:]
    fmt, elements = None, []
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
        elif parts[0] == "property":
            if parts[1] == "list":
                elements[-1]["props"].append(("list", parts[2], parts[3], parts[4]))
            else:
                elements[-1]["props"].append(("scalar", parts[1], parts[2]))
    if fmt not in ("ascii", "binary_little_endian"):
        raise ReconstructionUnavailable(f"unsupported PLY format {fmt}")
    vertices = faces = None
    if fmt == "ascii":
        tokens = body.decode("ascii").split()
        pos = 0
        for el in elements:
            if el["name"] == "vertex":
                names = [p[2] for p in el["props"]]
                if any(p[0] == "list" for p in el["props"]):
                    raise ReconstructionUnavailable("vertex list properties unsupported")
                n = len(names)
                arr = np.array(tokens[pos:pos + n * el["count"]], dtype=np.float64).reshape(el["count"], n)
                pos += n * el["count"]
                vertices = arr[:, [names.index("x"), names.index("y"), names.index("z")]]
            elif el["name"] == "face":
                rows = []
                for _ in range(el["count"]):
                    k = int(tokens[pos])
                    rows.append([int(t) for t in tokens[pos + 1:pos + 1 + k]])
                    pos += 1 + k
                    if len(el["props"]) != 1:
                        raise ReconstructionUnavailable("extra face properties unsupported in ascii PLY")
                faces = _triangulate(rows)
            else:
                raise ReconstructionUnavailable(f"unsupported PLY element {el['name']}")
    else:
        offset = 0
        for el in elements:
            if el["name"] == "vertex":
                if any(p[0] == "list" for p in el["props"]):
                    raise ReconstructionUnavailable("vertex list properties unsupported")
                dtype = np.dtype([(p[2], "<" + _PLY_TYPES[p[1]]) for p in el["props"]])
                arr = np.frombuffer(body, dtype=dtype, count=el["count"], offset=offset)
                offset += dtype.itemsize * el["count"]
                vertices = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
            elif el["name"] == "face":
                if len(el["props"]) != 1 or el["props"][0][0] != "list":
                    raise ReconstructionUnavailable("only a single face index list is supported")
                _, ctype, itype, _ = el["props"][0]
                cdt, idt = np.dtype("<" + _PLY_TYPES[ctype]), np.dtype("<" + _PLY_TYPES[itype])
                if el["count"] > max_faces:
                    raise ReconstructionUnavailable("PLY face count exceeds bound")
                # Fast path: all triangles.
                rec = np.dtype([("n", cdt), ("i", idt, (3,))])
                try:
                    arr = np.frombuffer(body, dtype=rec, count=el["count"], offset=offset)
                    if np.all(arr["n"] == 3):
                        faces = arr["i"].astype(np.int64)
                        offset += rec.itemsize * el["count"]
                        continue
                except ValueError:
                    pass
                rows = []
                for _ in range(el["count"]):
                    k = int(np.frombuffer(body, dtype=cdt, count=1, offset=offset)[0])
                    offset += cdt.itemsize
                    rows.append(np.frombuffer(body, dtype=idt, count=k, offset=offset).tolist())
                    offset += idt.itemsize * k
                faces = _triangulate(rows)
            else:
                raise ReconstructionUnavailable(f"unsupported PLY element {el['name']}")
    if vertices is None:
        raise ReconstructionUnavailable("PLY has no vertices")
    if faces is None:
        faces = np.zeros((0, 3), dtype=np.int64)
    return vertices, faces


def _triangulate(rows) -> np.ndarray:
    out = []
    for r in rows:
        if len(r) < 3:
            continue
        out.extend([r[0], r[k], r[k + 1]] for k in range(1, len(r) - 1))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


# --------------------------------------------------------------------------
# admission gates and mesh support

def sparse_gates(images: Mapping[str, dict], cameras: Mapping[int, dict], points: Mapping[int, dict],
                 registered_names: set[str]) -> dict:
    by_id = {img["image_id"]: (name, img) for name, img in images.items() if name in registered_names}
    centers = {iid: -img["pose"][:3, :3].T @ img["pose"][:3, 3] for iid, (_, img) in by_id.items()}
    angles, reproj, strong = [], [], 0
    for pid, pt in points.items():
        ids = [int(i) for i in pt["track"][:, 0] if int(i) in by_id]
        if len(set(ids)) >= 3:
            strong += 1
            rays = [pt["xyz"] - centers[i] for i in set(ids)]
            rays = [r / np.linalg.norm(r) for r in rays]
            best = 0.0
            for a in range(len(rays)):
                for b in range(a + 1, len(rays)):
                    best = max(best, math.degrees(math.acos(max(-1.0, min(1.0, float(rays[a] @ rays[b]))))))
            angles.append(best)
        for image_id, p2d in pt["track"]:
            if int(image_id) not in by_id:
                continue
            name, img = by_id[int(image_id)]
            K = cameras[img["camera_id"]]["K"]
            uv, z = wg.project(pt["xyz"][None], K, img["pose"])
            if z[0] > 0 and 0 <= p2d < len(img["xy"]):
                reproj.append(float(np.linalg.norm(uv[0] - img["xy"][p2d])))
    gates = wc.ADMISSION_GATES
    metrics = {"registered_views": len(by_id), "sparse_points_3plus_views": strong,
               "median_triangulation_angle_deg": float(np.median(angles)) if angles else None,
               "median_reprojection_px": float(np.median(reproj)) if reproj else None}
    failures = []
    if metrics["registered_views"] < gates["min_registered_views"]:
        failures.append("registered_views")
    if strong < gates["min_sparse_points_3plus_views"]:
        failures.append("sparse_points_3plus_views")
    if metrics["median_triangulation_angle_deg"] is None or metrics["median_triangulation_angle_deg"] < gates["min_median_triangulation_angle_deg"]:
        failures.append("median_triangulation_angle")
    if metrics["median_reprojection_px"] is None or metrics["median_reprojection_px"] > gates["max_median_reprojection_px"]:
        failures.append("median_reprojection")
    return {"metrics": metrics, "gates": dict(gates), "failures": failures, "passed": not failures,
            "note": "development heuristics, not accuracy proof"}


def face_support(vertices: np.ndarray, faces: np.ndarray, views: Sequence[dict], *, tolerance: float) -> np.ndarray:
    """Count views where all vertices + centroid project inside the observed
    mask with positive depth and the centroid agrees with the dense depth."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    counts = np.zeros(len(f), dtype=np.int32)
    centroids = v[f].mean(axis=1)
    for view in views:
        K, pose, mask, depth = view["K"], view["pose"], view["mask"], view["depth"]
        h, w = mask.shape
        uv_v, z_v = wg.project(v, K, pose)
        uv_c, z_c = wg.project(centroids, K, pose)

        def inside(uv):
            iu = np.round(uv[..., 0])
            iv = np.round(uv[..., 1])
            ok = np.isfinite(uv).all(axis=-1) & (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
            iu = np.where(ok, iu, 0).astype(int)
            iv = np.where(ok, iv, 0).astype(int)
            return ok & mask[iv, iu], iu, iv
        v_ok, vu, vv = inside(uv_v)
        c_ok, cu, cv = inside(uv_c)
        ok = v_ok[f].all(axis=1) & (z_v[f] > 0).all(axis=1) & c_ok & (z_c > 0)

        def depth_ok(z, iu, iv):
            d = depth[iv, iu]
            return np.isfinite(d) & (d > 0) & (np.abs(z - d) <= tolerance * np.where(d > 0, d, 1))
        # All three vertices and the centroid must agree with the dense depth
        # (a centroid-only test can accept faces with occluded vertices).
        vd = depth_ok(z_v, vu, vv)
        ok &= vd[f].all(axis=1) & depth_ok(z_c, cu, cv)
        counts += ok
    return counts


def compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering on a uniform grid until the face count fits."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) <= max_faces:
        return v, f
    extent = float(np.max(np.ptp(v, axis=0)))
    cell = extent / 512
    for _ in range(12):
        keys = np.floor((v - v.min(axis=0)) / cell).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        newv = np.zeros((inverse.max() + 1, 3))
        np.add.at(newv, inverse, v)
        newv /= np.bincount(inverse)[:, None]
        nf = inverse[f]
        nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
        key = np.sort(nf, axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
        nf = nf[np.sort(first)]
        if len(nf) <= max_faces:
            return compact_mesh(newv, nf)
        cell *= 1.4
    raise ReconstructionUnavailable("mesh could not be decimated within the triangle budget")


# --------------------------------------------------------------------------
# package persistence

PACKAGE_FILES = {"vertices": ("vertices.npy", np.float32, 3), "faces": ("faces.npy", np.int32, 3),
                 "face_support_views": ("face_support.npy", np.int32, None),
                 "descriptor_points": ("bank_points.npy", np.float32, 3),
                 "descriptor_values": ("bank_descriptors.npy", np.uint8, 32)}


def _save_npy(directory: Path, name: str, array: np.ndarray) -> wc.FileRef:
    path = directory / name
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    raw = path.read_bytes()
    return wc.FileRef(name, hashlib.sha256(raw).hexdigest(), len(raw))


def save_package(directory: Path, package: dict, arrays: Mapping[str, np.ndarray] | None = None) -> wc.SurfacePackage:
    """Write an immutable package (exclusive create) and validate it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    body = dict(package)
    for key, (filename, dtype, cols) in PACKAGE_FILES.items():
        if arrays and key in arrays and arrays[key] is not None:
            arr = np.ascontiguousarray(arrays[key], dtype=dtype)
            body[key] = _save_npy(directory, filename, arr).to_dict()
        else:
            body.setdefault(key, None)
    pkg = wc.SurfacePackage.from_dict(body)
    with (directory / "surface.json").open("xb") as stream:
        stream.write(wc.canonical_json(pkg.to_dict()))
    return pkg


def load_package(directory: Path) -> tuple[wc.SurfacePackage, dict[str, np.ndarray]]:
    """Load an immutable package; containment and the 20 MB aggregate are checked
    from ``stat`` before any component is read."""
    import io
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise wc.ContractError("package directory missing or a symlink")
    base = directory.resolve()

    def component(relative: str) -> Path:
        wc._relative_path(relative, "package component")
        if "/" in relative:
            raise wc.ContractError("package components must be top-level files")
        p = directory / relative
        if p.is_symlink() or not p.is_file():
            raise wc.ContractError(f"package file {relative} missing or a symlink")
        if p.resolve().parent != base:
            raise wc.ContractError(f"package file {relative} escapes the package directory")
        return p
    surface = component("surface.json")
    if surface.stat().st_size > 1_000_000:
        raise wc.ContractError("surface.json too large")
    pkg = wc.SurfacePackage.from_dict(wc.loads_strict(surface.read_bytes()))
    refs = {key: getattr(pkg, key) for key in PACKAGE_FILES if getattr(pkg, key) is not None}
    paths, total = {}, surface.stat().st_size
    for key, ref in refs.items():
        p = component(ref.path)
        size = p.stat().st_size
        if size != ref.bytes:
            raise wc.ContractError(f"package file {ref.path} size differs")
        total += size
        paths[key] = p
    if total > wc.MAX_PACKAGE_BYTES:
        raise wc.ContractError("package exceeds 20 MB before reading")
    arrays = {}
    for key, ref in refs.items():
        _filename, dtype, cols = PACKAGE_FILES[key]
        data = paths[key].read_bytes()
        if len(data) != ref.bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise wc.ContractError(f"package file {ref.path} hash mismatch")
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        if arr.dtype != dtype or (cols is not None and (arr.ndim != 2 or arr.shape[1] != cols)):
            raise wc.ContractError(f"package array {key} dtype/shape invalid")
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise wc.ContractError(f"package array {key} non-finite")
        arrays[key] = arr
    if pkg.status == "available":
        wg.validate_mesh(arrays["vertices"], arrays["faces"])
        if len(arrays["vertices"]) != pkg.vertex_count or len(arrays["faces"]) != pkg.face_count:
            raise wc.ContractError("package counts differ from arrays")
        if arrays["face_support_views"].shape != (pkg.face_count,) and arrays["face_support_views"].shape != (pkg.face_count, 1):
            raise wc.ContractError("face support must have one count per face")
    return pkg, arrays


def unavailable_package(*, package_id: str, clip_id: str, reason: str, prefix_frames: Sequence[Mapping],
                        masks: Mapping[str, Mapping], reconstruction: dict) -> dict:
    reason = safe_text(reason, 2000)[0]
    reconstruction = sanitize_metadata(reconstruction)
    preview = {"kind": "original_image_and_observed_mask",
               "frames": [{"frame_id": f["frame_id"], "image_sha256": f["image_sha256"],
                           "mask_sha256": (masks.get(f["frame_id"]) or {}).get("sha256")} for f in prefix_frames]}
    return {"schema": wc.SURFACE_SCHEMA, "package_id": package_id, "clip_id": clip_id, "status": "unavailable",
            "unavailable_reason": reason[:2000], "units": "object_unit", "metric_scale_available": False,
            "normalization": None, "vertices": None, "faces": None, "face_support_views": None,
            "vertex_count": 0, "face_count": 0, "cameras": [], "descriptor_points": None, "descriptor_values": None,
            "admission": {"passed": False}, "marker": None, "artwork": None, "frozen": True,
            "frozen_after_frame_id": prefix_frames[-1]["frame_id"] if prefix_frames else None,
            "prefix_frame_ids": [f["frame_id"] for f in prefix_frames], "reconstruction": reconstruction,
            "preview": preview, "evaluator_inputs_read": False}


# --------------------------------------------------------------------------
# reconstruction entry point

def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_gray(path: Path) -> np.ndarray:
    from .vision import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ReconstructionUnavailable(f"cannot read image {path.name}")
    return img


def _read_mask(path: Path) -> np.ndarray:
    from .vision import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ReconstructionUnavailable(f"cannot read mask {path.name}")
    return m > 127


def reconstruct(prefix_frames: Sequence[Mapping[str, Any]], observed_masks: Mapping[str, Mapping[str, Any]],
                intrinsics: Any, *, output_dir: Path, limits: Limits | None = None,
                cancel: threading.Event | None = None, colmap: str | os.PathLike | None = None,
                clip_id: str = "clip", package_id: str | None = None, selection: wc.Selection | None = None,
                artwork_kind: str = "checkerboard", prefix_attestation: Mapping | None = None,
                preflight_result: Mapping | None = None) -> dict:
    """Run the fixed COLMAP pipeline on the causal prefix; always return a package
    record (available or unavailable) plus per-stage receipts."""
    limits = limits or Limits()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    package_id = package_id or f"{clip_id}-surface"
    receipts: list[dict] = []
    recon = {"backend": "colmap", "limits": limits.to_dict(), "stages": receipts,
             "prefix_attestation": dict(prefix_attestation or {}), "evaluator_inputs_read": False}

    def fail(reason: str) -> dict:
        # Persist the cause first so a later serialization problem cannot lose it.
        safe_reason = safe_text(reason, 2000)[0]
        receipt_path = output_dir / "receipt.json"
        if not receipt_path.exists():
            receipt_path.write_bytes(wc.canonical_json({"status": "unavailable", "reason": safe_reason,
                                                        "reconstruction": sanitize_metadata(recon)}))
        record = unavailable_package(package_id=package_id, clip_id=clip_id, reason=safe_reason,
                                     prefix_frames=prefix_frames, masks=observed_masks, reconstruction=recon)
        pkg = save_package(output_dir / "package", record)
        return {"status": "unavailable", "reason": safe_reason, "package": pkg,
                "package_dir": str(output_dir / "package")}

    # ---- input validation (identity, causal order, attestation, dimensions) before any tool work
    if not prefix_frames:
        return fail("no_prefix_frames")
    if len(prefix_frames) > limits.max_frames:
        return fail(f"prefix exceeds {limits.max_frames} frames")
    att = dict(prefix_attestation or {})
    if "prefix_bound_attested" not in att or "first_scored_source_frame_index" not in att:
        return fail("prefix_attestation_missing: a causal onboarding attestation is required")
    ids = [f.get("frame_id") for f in prefix_frames]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not wc.ID_RE.fullmatch(i) for i in ids):
        return fail("prefix frame ids must be unique valid identifiers")
    indices = [f.get("source_frame_index") for f in prefix_frames]
    if any(type(i) is not int for i in indices) or any(b <= a for a, b in zip(indices, indices[1:])):
        return fail("prefix frames must have strictly increasing integer source_frame_index")
    first_scored = att["first_scored_source_frame_index"]
    if type(first_scored) is not int or indices[-1] >= first_scored:
        return fail("prefix includes a frame at or after the first scored frame")
    stamps = [f.get("timestamp_s") for f in prefix_frames]
    if att["prefix_bound_attested"] is True:
        if any(s is None for s in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            return fail("attested prefix requires increasing true timestamps")
        if stamps[-1] - stamps[0] > wc.MAX_PREFIX_SECONDS + 1e-9:
            return fail("prefix exceeds ten seconds of true time")
    elif any(s is not None for s in stamps):
        return fail("unattested (frame-index) prefix must not carry timestamps")
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3) if intrinsics is not None else None
    if K is None:
        return fail("calibration_unavailable: measured intrinsics are required for reconstruction")
    wg.check_K(K)
    dims = set()
    for frame in prefix_frames:
        img = Path(frame["image_path"])
        if not img.is_file() or _sha_file(img) != frame["image_sha256"]:
            return fail(f"prefix image identity mismatch for {frame['frame_id']}")
        mask = observed_masks.get(frame["frame_id"])
        if mask is None:
            return fail(f"observed_mask_unavailable for prefix frame {frame['frame_id']}")
        mp = Path(mask["path"])
        if not mp.is_file() or _sha_file(mp) != mask["sha256"]:
            return fail(f"observed mask identity mismatch for {frame['frame_id']}")
        if (mask["width"], mask["height"]) != (frame["width"], frame["height"]):
            return fail(f"mask dimensions differ for {frame['frame_id']}")
        dims.add((frame["width"], frame["height"]))
    if len(dims) != 1:
        return fail("prefix frames have differing dimensions")
    width, height = dims.pop()
    pre = dict(preflight_result) if preflight_result is not None else preflight(colmap)
    recon["preflight"] = {k: pre.get(k) for k in ("available", "executable", "version", "reason",
                                                    "feature_gpu_flag", "matcher_gpu_flag", "matcher_threads_flag",
                                                    "feature_type_flag")}
    recon["matcher_resource_policy"] = dict(MATCHER_RESOURCE_POLICY)
    if not pre.get("available"):
        return fail(pre.get("reason") or "colmap_unavailable")
    if not pre.get("matcher_gpu_flag") or not pre.get("matcher_threads_flag"):
        return fail("unsupported_colmap: the pinned CPU matcher policy needs help-verified use_gpu and "
                    "num_threads flags; no unbounded default fallback")
    exe = pre["executable"]

    # ---- workspace (ordinal file names; ids never become paths)
    copy_bytes = sum(Path(f["image_path"]).stat().st_size + Path(observed_masks[f["frame_id"]]["path"]).stat().st_size
                     for f in prefix_frames)
    violation = budget_violation(output_dir, limits, extra=copy_bytes)
    if violation is not None:
        return fail(f"budget_blocked before copies: {violation}")
    work = output_dir / "colmap"
    images_dir, masks_dir = work / "images", work / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()
    names = {}
    for ordinal, frame in enumerate(prefix_frames):
        name = f"p{ordinal:03d}.png"
        shutil.copyfile(frame["image_path"], images_dir / name)
        shutil.copyfile(observed_masks[frame["frame_id"]]["path"], masks_dir / f"{name}.png")
        names[name] = frame
    recon["image_names"] = {name: frame["frame_id"] for name, frame in names.items()}
    cam_params = f"{K[0, 0]!r},{K[1, 1]!r},{K[0, 2] + 0.5!r},{K[1, 2] + 0.5!r}"
    db = work / "database.db"
    sparse, txt = work / "sparse", work / "sparse_txt"
    dense = work / "dense"
    for d in (sparse, txt):
        d.mkdir()
    fe = [exe, "feature_extractor", "--database_path", str(db), "--image_path", str(images_dir),
          "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", cam_params,
          "--ImageReader.single_camera", "1", "--ImageReader.mask_path", str(masks_dir),
          pre["feature_gpu_flag"], "0"]
    if pre.get("feature_type_flag"):
        fe += [pre["feature_type_flag"], "SIFT"]
    em = [exe, "exhaustive_matcher", "--database_path", str(db),
          pre["matcher_gpu_flag"], str(MATCHER_RESOURCE_POLICY["use_gpu"]),
          pre["matcher_threads_flag"], str(MATCHER_RESOURCE_POLICY["num_threads"])]
    plan = [("feature_extractor", fe), ("exhaustive_matcher", em),
            ("mapper", [exe, "mapper", "--database_path", str(db), "--image_path", str(images_dir),
                        "--output_path", str(sparse)])]
    logs = output_dir / "logs"
    for name, argv in plan:
        r = run_stage(name, argv, workspace=output_dir, log_dir=logs, limits=limits, cancel=cancel)
        receipts.append(r)
        if r["status"] != "completed":
            return fail(f"colmap_{name}_{r['status']}: {r['detail']}")
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        return fail("colmap_mapper_produced_no_model")
    # choose the model with the most registered images (deterministic tie-break by name)
    best_model, best_count = None, -1
    for model in models:
        out = txt / model.name
        out.mkdir()
        r = run_stage(f"model_converter_{model.name}", [exe, "model_converter", "--input_path", str(model),
                                                        "--output_path", str(out), "--output_type", "TXT"],
                      workspace=output_dir, log_dir=logs, limits=limits, cancel=cancel)
        receipts.append(r)
        if r["status"] != "completed":
            return fail(f"colmap_model_converter_{r['status']}: {r['detail']}")
        count = len(read_images_txt(out / "images.txt"))
        if count > best_count:
            best_model, best_count = model, count
    model_txt = txt / best_model.name
    try:
        cameras = read_cameras_txt(model_txt / "cameras.txt")
        images = read_images_txt(model_txt / "images.txt")
        points = read_points3d_txt(model_txt / "points3D.txt")
    except (ReconstructionUnavailable, ValueError, IndexError) as exc:
        return fail(f"sparse_model_unparseable: {exc}")
    registered = {n for n in images if n in names}
    gates = sparse_gates(images, cameras, points, registered)
    recon["sparse_gates"] = gates
    if not gates["passed"]:
        return fail(f"admission_gates_failed: {gates['failures']}")

    # ---- dense
    dense_plan = [
        ("image_undistorter", [exe, "image_undistorter", "--image_path", str(images_dir), "--input_path", str(best_model),
                               "--output_path", str(dense), "--output_type", "COLMAP"]),
        ("patch_match_stereo", [exe, "patch_match_stereo", "--workspace_path", str(dense),
                                "--PatchMatchStereo.geom_consistency", "true"]),
        ("stereo_fusion", [exe, "stereo_fusion", "--workspace_path", str(dense), "--input_type", "geometric",
                           "--output_path", str(dense / "fused.ply")]),
        ("poisson_mesher", [exe, "poisson_mesher", "--input_path", str(dense / "fused.ply"),
                            "--output_path", str(dense / "meshed-poisson.ply")]),
    ]
    for name, argv in dense_plan:
        flags = set(pre["commands"][name]["flags"])
        unsupported = [a for a in argv[2:] if a.startswith("--") and a not in flags]
        if unsupported:
            return fail(f"unsupported_flags_{name}: {unsupported}")
        r = run_stage(name, argv, workspace=output_dir, log_dir=logs, limits=limits, cancel=cancel)
        receipts.append(r)
        if r["status"] != "completed":
            return fail(f"colmap_{name}_{r['status']}: {r['detail']}")
    dense_txt = work / "dense_sparse_txt"
    dense_txt.mkdir()
    r = run_stage("model_converter_dense", [exe, "model_converter", "--input_path", str(dense / "sparse"),
                                            "--output_path", str(dense_txt), "--output_type", "TXT"],
                  workspace=output_dir, log_dir=logs, limits=limits, cancel=cancel)
    receipts.append(r)
    if r["status"] != "completed":
        return fail(f"colmap_model_converter_dense_{r['status']}: {r['detail']}")
    try:
        ucams = read_cameras_txt(dense_txt / "cameras.txt")
        uimages = read_images_txt(dense_txt / "images.txt")
        mesh_v, mesh_f = read_ply(dense / "meshed-poisson.ply")
    except (ReconstructionUnavailable, ValueError, IndexError, KeyError) as exc:
        return fail(f"dense_outputs_unparseable: {exc}")
    views = []
    for name in sorted(registered):
        if name not in uimages:
            continue
        ui = uimages[name]
        uc = ucams[ui["camera_id"]]
        if (uc["width"], uc["height"]) != (width, height) or not np.allclose(uc["K"], K, atol=1e-6):
            return fail("undistorted camera geometry differs from the observed-mask geometry; mask binding unavailable")
        depth_path = dense / "stereo" / "depth_maps" / f"{name}.geometric.bin"
        if not depth_path.is_file():
            continue
        depth = read_colmap_array(depth_path)
        if depth.shape != (height, width):
            return fail(f"dense depth dimensions differ for {name}")
        frame = names[name]
        views.append({"frame_id": frame["frame_id"], "name": name, "K": uc["K"], "pose": ui["pose"],
                      "mask": _read_mask(Path(observed_masks[frame["frame_id"]]["path"])), "depth": depth,
                      "depth_sha256": _sha_file(depth_path), "frame": frame})
    if len(views) < wc.ADMISSION_GATES["min_registered_views"]:
        return fail("insufficient dense depth views for face support")
    if not len(mesh_f):
        return fail("poisson mesh is empty")
    support = face_support(mesh_v, mesh_f, views, tolerance=limits.depth_tolerance)
    keep = support >= limits.min_face_support
    if not keep.any():
        return fail("no Poisson faces have observed multi-view support")
    v, f = compact_mesh(mesh_v, mesh_f[keep])
    try:
        v, f = decimate(v, f, wc.MAX_MESH_TRIANGLES)
        support = face_support(v, f, views, tolerance=limits.depth_tolerance)
        keep = support >= limits.min_face_support
        v, f = compact_mesh(v, f[keep])
        support = support[keep]
        f = wg.orient_faces(f.astype(np.int32))
        nv, norm = wg.normalize_geometry(v)
        mesh_info = wg.validate_mesh(nv, f.astype(np.int32))
    except (ReconstructionUnavailable, wg.GeometryError, ValueError) as exc:
        return fail(f"supported_mesh_invalid: {exc}")
    cams, bank_views = [], []
    for view in views:
        npose = wg.normalize_pose(view["pose"], norm)
        frame = view["frame"]
        cams.append({"frame_id": frame["frame_id"], "image_sha256": frame["image_sha256"],
                     "mask_sha256": observed_masks[frame["frame_id"]]["sha256"], "width": width, "height": height,
                     "K": view["K"].ravel().tolist(), "pose": npose.tolist(), "depth": None})
        bank_views.append({"frame_id": frame["frame_id"], "gray": _read_gray(Path(frame["image_path"])),
                           "mask": view["mask"], "K": view["K"], "pose": npose})
    bank = wg.build_descriptor_bank(bank_views, nv, f.astype(np.int32))
    art = wg.builtin_artwork(artwork_kind if artwork_kind in ("checkerboard", "directional") else "checkerboard")
    art_sha = wg.array_sha256(art)
    marker = None
    if selection is not None:
        cam = next((c for c in cams if c["frame_id"] == selection.frame_id), None)
        if cam is not None:
            try:
                u, vv = selection.positive[0]
                hit = wg.anchor_from_pixel(nv, f, np.array(cam["K"]).reshape(3, 3), np.array(cam["pose"]), u, vv,
                                           width=width, height=height,
                                           observed_mask=_read_mask(Path(observed_masks[selection.frame_id]["path"])))
                marker = wg.make_marker(nv, f, hit, np.array(cam["pose"]), limits.marker_size, art_sha)
                recon["marker_anchor"] = {"frame_id": selection.frame_id, "pixel": [u, vv]}
            except wg.GeometryError as exc:
                recon["marker_anchor"] = {"status": "unavailable", "reason": str(exc)}
        else:
            recon["marker_anchor"] = {"status": "unavailable", "reason": "selection frame was not registered"}
    recon.update({"colmap_version": pre.get("version"), "model": best_model.name, "mesh": mesh_info,
                  "depth_bindings": {view["frame_id"]: view["depth_sha256"] for view in views},
                  "bank_points": int(len(bank["points"]))})
    record = {"schema": wc.SURFACE_SCHEMA, "package_id": package_id, "clip_id": clip_id, "status": "available",
              "unavailable_reason": None, "units": "object_unit", "metric_scale_available": False,
              "normalization": norm.to_dict(), "vertex_count": int(len(nv)), "face_count": int(len(f)),
              "cameras": cams, "admission": {**gates, "face_support_min_views": limits.min_face_support},
              "marker": marker, "artwork": {"kind": artwork_kind if artwork_kind in ("checkerboard", "directional") else "checkerboard",
                                            "edge_px": int(art.shape[0]), "sha256": art_sha,
                                            "generation": "fixed_builtin_v1"},
              "frozen": True, "frozen_after_frame_id": prefix_frames[-1]["frame_id"],
              "prefix_frame_ids": [fr["frame_id"] for fr in prefix_frames], "reconstruction": recon,
              "preview": {"kind": "none"}, "evaluator_inputs_read": False}
    package_arrays = {"vertices": nv, "faces": f.astype(np.int32), "face_support_views": support.astype(np.int32),
                      "descriptor_points": bank["points"], "descriptor_values": bank["descriptors"]}
    estimate = sum(int(np.asarray(a).nbytes) + 256 for a in package_arrays.values()) + 1_000_000
    if estimate > wc.MAX_PACKAGE_BYTES:
        return fail("package would exceed 20 MB")
    violation = budget_violation(output_dir, limits, extra=estimate)
    if violation is not None:
        return fail(f"budget_blocked before package write: {violation}")
    record["reconstruction"] = sanitize_metadata(record["reconstruction"])
    record["admission"] = sanitize_metadata(record["admission"])
    try:
        pkg = save_package(output_dir / "package", record, package_arrays)
    except (wc.ContractError, OSError) as exc:
        shutil.rmtree(output_dir / "package", ignore_errors=True)
        return fail(f"package_rejected: {exc}")
    (output_dir / "receipt.json").write_bytes(wc.canonical_json({"status": "available", "reconstruction": recon}))
    return {"status": "available", "reason": None, "package": pkg, "package_dir": str(output_dir / "package")}
