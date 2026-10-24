"""Latency/memory telemetry and the one-GPU-stage lock for the workbench.

Only measured values are recorded. Fields a platform cannot measure stay
``None``. Torch is never imported here: GPU statistics are read only when
torch is already loaded *and* CUDA was actually initialized by real work.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from typing import Any, Iterator, Mapping

TELEMETRY_SCHEMA = "workbench_telemetry_v1"
_GPU_THREAD_LOCK = threading.Lock()
_GPU_OWNER: int | None = None  # thread ident of the current gpu_stage holder
# One canonical, repository-scoped lock root shared by every workspace.
CANONICAL_LOCK_ROOT = Path(__file__).resolve().parents[1] / ".cache" / "gpu-stage-lock"


# --------------------------------------------------------------------------
# memory

def process_rss_bytes() -> int | None:
    """Current resident set size, or None when unsupported."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = PMC()
            counters.cb = ctypes.sizeof(PMC)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        with open("/proc/self/statm", encoding="ascii") as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return None


def gpu_memory_snapshot() -> dict:
    """Best-effort CUDA memory state; a broken CUDA context yields nulls plus the error."""
    out = {"gpu_allocated_bytes": None, "gpu_reserved_bytes": None, "gpu_free_bytes": None,
           "gpu_total_bytes": None, "gpu_error": None}
    torch = _cuda_torch()
    if torch is None:
        return out
    try:
        out["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated())
        out["gpu_reserved_bytes"] = int(torch.cuda.memory_reserved())
        free, total = torch.cuda.mem_get_info()
        out["gpu_free_bytes"], out["gpu_total_bytes"] = int(free), int(total)
    except Exception as exc:  # noqa: BLE001 - e.g. after a CUDA OOM/abort
        out["gpu_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return out


def resource_sample(label: str, *, disk_path: Path | None = None, extra: Mapping[str, Any] | None = None) -> dict:
    """One measured resource sample (no estimates); every field may be null."""
    import shutil
    sample = {"label": label, "time": time.time(), "process_rss_bytes": process_rss_bytes(),
              "process_peak_rss_bytes": process_peak_rss_bytes(), **gpu_memory_snapshot(), "disk_free_bytes": None}
    if disk_path is not None:
        try:
            sample["disk_free_bytes"] = int(shutil.disk_usage(disk_path).free)
        except OSError:
            pass
    if extra:
        sample.update(dict(extra))
    return sample


def process_peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, or None when unsupported."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = PMC()
            counters.cb = ctypes.sizeof(PMC)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)  # Linux reports KiB
    except Exception:  # noqa: BLE001 - unsupported platform -> None, never an estimate
        return None


def _cuda_torch():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is not None and cuda.is_available() and cuda.is_initialized():
            return torch
    except Exception:  # noqa: BLE001
        return None
    return None


def gpu_synchronize() -> bool:
    torch = _cuda_torch()
    if torch is None:
        return False
    torch.cuda.synchronize()
    return True


def gpu_peak_bytes() -> dict:
    torch = _cuda_torch()
    if torch is None:
        return {"gpu_used": False, "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    return {"gpu_used": True, "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}


# --------------------------------------------------------------------------
# timing

def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median_ms": None, "p95_ms": None, "max_ms": None, "total_ms": None}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"count": len(values), "median_ms": statistics.median(ordered), "p95_ms": p95,
            "max_ms": ordered[-1], "total_ms": sum(ordered)}


class Telemetry:
    """Collects measured stage and per-frame timings for one run or job."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stages: dict[str, list[float]] = {}
        self.frames: list[dict] = []
        self.cache: dict[str, dict] = {}
        self.cached_timings: dict[str, Any] = {}
        self.finished: float | None = None

    @contextmanager
    def measure(self, name: str, *, sync_gpu: bool = False) -> Iterator[dict]:
        """Time a block; GPU work is synchronized before both clock reads when used."""
        if sync_gpu:
            gpu_synchronize()
        holder = {"ms": None}
        start = time.perf_counter()
        try:
            yield holder
        finally:
            if sync_gpu:
                gpu_synchronize()
            holder["ms"] = (time.perf_counter() - start) * 1000.0
            self.stages.setdefault(name, []).append(holder["ms"])

    def frame(self, role: str, frame_id: str, **timings_ms: float | None) -> None:
        self.frames.append({"role": role, "frame_id": frame_id,
                            **{k: (None if v is None else float(v)) for k, v in timings_ms.items()}})

    def cache_event(self, stage: str, status: str, lookup_ms: float | None = None, **detail) -> None:
        self.cache[stage] = {"status": status, "lookup_ms": lookup_ms, **detail}

    def reuse_cached(self, stage: str, original: Any) -> None:
        """Attach the original run's measured timings, explicitly labelled as cached evidence."""
        self.cached_timings[stage] = {"source": "cached_original_run", "timings": original}

    def role_timings(self, role: str) -> dict:
        rows = [f for f in self.frames if f["role"] == role]
        keys = sorted({k for f in rows for k in f if k.endswith("_ms")})
        return {k: _summary([f[k] for f in rows if f.get(k) is not None]) for k in keys}

    def snapshot(self) -> dict:
        wall = ((self.finished or time.perf_counter()) - self.started) * 1000.0
        hits = [k for k, v in self.cache.items() if v["status"] == "cache_hit"]
        misses = [k for k, v in self.cache.items() if v["status"] != "cache_hit"]
        mode = "cold" if not hits else ("cached" if not misses else "partially_cached")
        return {"schema": TELEMETRY_SCHEMA, "run_wall_ms": wall, "run_cache_mode": mode,
                "stages": {k: _summary(v) for k, v in self.stages.items()},
                "per_role_frame_timings": {r: self.role_timings(r) for r in sorted({f["role"] for f in self.frames})},
                "frames": self.frames, "cache": self.cache, "cached_original_timings": self.cached_timings,
                "process_peak_rss_bytes": process_peak_rss_bytes(), **gpu_peak_bytes(),
                "note": "measured values only; unsupported fields are null; cache lookups are not inference latency"}


# --------------------------------------------------------------------------
# dependencies

def dependency_versions() -> dict:
    """Installed versions read from metadata (no imports)."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "torch", "pyrender", "trimesh", "Pillow"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


# --------------------------------------------------------------------------
# one-GPU-stage lock (in-process + cross-process lock file)

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: exists but not ours
        code = wintypes.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_stage_active() -> bool:
    """True only for the thread that currently holds the gpu_stage (explicit ownership)."""
    return _GPU_OWNER is not None and _GPU_OWNER == threading.get_ident()


@contextmanager
def gpu_stage(name: str, lock_dir: Any = None, *, timeout_s: float = 0.0,
              recover_stale: bool = False) -> Iterator[None]:
    """Hold the single neural/GPU stage: segmentation, pose, parity and
    production geometry never overlap.

    The default lock lives in one canonical repository root
    (``CANONICAL_LOCK_ROOT``) shared by every workspace, so two different
    ``--workspace`` values cannot bypass serialization. Pass ``lock_dir`` only
    to override that root (tests). Nested use is an overlap and is refused.

    Any existing foreign lock file is refused (raises ``RuntimeError`` after
    ``timeout_s``). An unreadable/empty/malformed lock is never deleted: its
    owner may still be writing it. Only with ``recover_stale=True`` is a lock
    whose fully written PID/token names a dead process removed. Cleanup removes
    the lock only when it still contains this holder's own token.
    """
    import uuid
    global _GPU_OWNER
    if lock_dir is None:
        lock_dir = CANONICAL_LOCK_ROOT
    acquired = (_GPU_THREAD_LOCK.acquire(timeout=timeout_s) if timeout_s > 0
                else _GPU_THREAD_LOCK.acquire(blocking=False))
    if not acquired:
        raise RuntimeError(f"gpu_stage_busy: another neural stage is active in this process ({name})")
    path, token, owned = None, uuid.uuid4().hex, False
    try:
        if lock_dir is not None:
            Path(lock_dir).mkdir(parents=True, exist_ok=True)
            path = Path(lock_dir) / "gpu-stage.lock"
            deadline = time.monotonic() + max(timeout_s, 0.0)
            while True:
                try:
                    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    holder = _parse_lock(path)
                    if (recover_stale and holder is not None and holder[0] != os.getpid()
                            and not _pid_alive(holder[0])):
                        if _parse_lock(path) == holder:  # unchanged since inspection
                            path.unlink(missing_ok=True)
                            continue
                    if time.monotonic() >= deadline:
                        who = f"process {holder[0]}" if holder else "an unreadable or in-progress lock"
                        raise RuntimeError(f"gpu_stage_busy: held by {who} ({name}); refusing to take it")
                    time.sleep(0.5)
                    continue
                owned = True
                with os.fdopen(fd, "w") as stream:
                    stream.write(f"{os.getpid()} {token} {name}\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                break
        _GPU_OWNER = threading.get_ident()
        yield
    finally:
        if _GPU_OWNER == threading.get_ident():
            _GPU_OWNER = None
        if owned and path is not None:
            current = _parse_lock(path)
            if current is not None and current[1] == token:
                try:
                    path.unlink()
                except OSError:
                    pass
        _GPU_THREAD_LOCK.release()


def _parse_lock(path: Path) -> tuple[int, str] | None:
    """(pid, token) from a fully written lock file, else None (never guessed)."""
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    parts = text.split()
    if not text.endswith("\n") or len(parts) < 2 or not parts[0].isdigit() or len(parts[1]) != 32:
        return None
    return int(parts[0]), parts[1]
