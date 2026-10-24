"""Bounded two-view CUDA parity bootstrap for the admitted Kaggle activation.

Importing this module is safe on a model-free workstation.  Runtime/model
imports happen only inside an explicitly admitted stage after all source,
recipe, exposure, and resource pins have been rechecked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kaggle_activation_request_v1"
PLAN_SCHEMA = "kaggle_activation_plan_v1"
OBSERVER_PLAN_SCHEMA = "kaggle_activation_observer_plan_v1"
FREEZE_SCHEMA = "kaggle_activation_freeze_v1"
STAGE_SCHEMA = "kaggle_activation_stage_v1"
COMPARISON_SCHEMA = "kaggle_activation_comparison_v1"
RENDER_SCHEMA = "kaggle_activation_render_v1"
BANK_SCHEMA = "kaggle_activation_banks_v1"
BRANCH_SCHEMA = "kaggle_activation_branch_v1"
SNAPSHOT_SCHEMA = "gotrack_larger_preparation_parity_snapshot_v1"
SNAPSHOT_RECEIPT_SCHEMA = "kaggle_activation_snapshot_receipt_v1"
POLICY = "kaggle-activation-v1"
MASK_PRODUCER = "bench/quality_kaggle_activation.py"
SOURCE_LOCK_SHA256 = "44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33"
MAX_METADATA_BYTES = 16 * 1024**2
GIB = 1024**3
MAX_CACHE_BYTES = 8 * GIB
DEFAULT_LIMITS = {"wall_seconds": 1800, "rss_bytes": 8 * GIB,
    "storage_bytes": 18 * GIB, "reserve_bytes": GIB}
EXPECTED_TRAIN_OBJECTS = ("1", "6", "10", "13", "17", "18", "20", "21", "23", "29", "31", "32", "33")
EXPECTED_VALIDATION_OBJECTS = ("4",)
EXPECTED_TEST_OBJECTS = ("7",)
SELECTED_ROWS = (("1", 0), ("1", 1))
BRANCHES = ("legacy-v1", "amortized-v1")
_ACTIVATION_ISSUER = object()
_EXECUTION = threading.local()


class ActivationError(RuntimeError):
    """Activation evidence or an execution bound failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _no_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def strict_json_bytes(raw: bytes, name: str = "metadata") -> Any:
    if not isinstance(raw, bytes) or len(raw) > MAX_METADATA_BYTES:
        raise ValueError(f"{name} exceeds the bounded metadata read")

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs,
                       parse_constant=_no_constant)

    def check(item, depth=0):
        if depth > 32:
            raise ValueError(f"{name} exceeds JSON nesting limit")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} contains a non-finite number")
        if isinstance(item, dict):
            for child in item.values():
                check(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                check(child, depth + 1)
    check(value)
    return value


def read_metadata(path: Path, name: str = "metadata") -> tuple[bytes, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError(f"{name} exceeds the bounded metadata read")
    return raw, strict_json_bytes(raw, name)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} has unknown or missing fields")


def _digest(value, name="sha256"):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _absolute(value, name: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or ".." in Path(value).parts:
        raise ValueError(f"{name} requires an absolute non-traversing path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} requires an absolute path")
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{name} traverses a symlink")
    return absolute


def file_pin(path: Path) -> dict[str, Any]:
    path = Path(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"pin target is not a regular nonsymlink file: {path}")
    path = path.resolve(strict=True)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file_pin(value, *, under: Path | None = None, maximum=8 * GIB) -> Path:
    _keys(value, ("path", "bytes", "sha256"), "file pin")
    path = _absolute(value["path"], "pinned file")
    _digest(value["sha256"])
    if type(value["bytes"]) is not int or not 0 < value["bytes"] <= maximum:
        raise ValueError("pinned file byte count is outside its bound")
    if under is not None and not path.is_relative_to(Path(under).resolve()):
        raise ValueError("pinned file escapes its authorized root")
    if not path.is_file() or path.stat().st_size != value["bytes"] or sha256_file(path) != value["sha256"]:
        raise ValueError(f"pinned file changed or is missing: {path}")
    return path


@dataclass(frozen=True, slots=True)
class ActivationContext:
    """Detached request bytes; no caller-provided verified flags or callbacks."""
    request_path: Path
    raw: bytes
    sha256: str

    def document(self) -> dict[str, Any]:
        if hashlib.sha256(self.raw).hexdigest() != self.sha256:
            raise ValueError("immutable activation request bytes changed")
        value = strict_json_bytes(self.raw, "activation request")
        validate_request(value)
        return value


def validate_request(doc: dict[str, Any], *, source_root: Path | None = None,
                     kaggle_paths: bool = True) -> dict[str, Any]:
    fields = ("schema", "source_root", "output_root", "scratch_root", "model_cache",
        "recipe_pin", "exposure_audit_pin", "source_files", "source_lock", "limits", "provenance")
    _keys(doc, fields, "activation request")
    if doc["schema"] != SCHEMA:
        raise ValueError("unsupported activation request schema")
    root = Path(source_root or ROOT).resolve()
    if _absolute(doc["source_root"], "source root") != root:
        raise ValueError("activation source root differs from this imported checkout")
    cache = _absolute(doc["model_cache"], "model cache")
    if cache != root / ".cache/model-quality":
        raise ValueError("model cache must be the exact loader cache under source root")
    output = _absolute(doc["output_root"], "output root")
    scratch = _absolute(doc["scratch_root"], "scratch root")
    if kaggle_paths:
        if (not output.is_relative_to(Path("/kaggle/working"))
                or output == Path("/kaggle/working")
                or not scratch.is_relative_to(Path("/tmp")) or scratch == Path("/tmp")):
            raise ValueError("activation output/scratch roots must be scoped under Kaggle working/tmp")
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for a, b in ((output, scratch), (output, cache), (scratch, cache))):
        raise ValueError("activation output, scratch, and model cache roots must be disjoint")
    # Output exists after metadata-only freeze, so freshness belongs to freeze()
    # rather than request parsing (which runs again in each later stage).
    if output.is_symlink() or scratch.is_symlink():
        raise ValueError("activation roots cannot be symlinks")
    limits = doc["limits"]
    _keys(limits, tuple(DEFAULT_LIMITS), "activation limits")
    ceilings = {"wall_seconds": 1800, "rss_bytes": 8 * GIB, "storage_bytes": 18 * GIB}
    for key, maximum in ceilings.items():
        if type(limits[key]) is not int or not 0 < limits[key] <= maximum:
            raise ValueError(f"activation limit {key} cannot exceed its admitted ceiling")
    if (type(limits["reserve_bytes"]) is not int or limits["reserve_bytes"] < GIB
            or limits["reserve_bytes"] >= limits["storage_bytes"]):
        raise ValueError("activation free-space reserve must be at least one GiB and below storage ceiling")
    if not isinstance(doc["source_files"], dict) or not doc["source_files"]:
        raise ValueError("exact runtime source closure is required")
    for relative, digest in doc["source_files"].items():
        if (not isinstance(relative, str) or "\\" in relative or ".." in PurePosixPath(relative).parts
                or PurePosixPath(relative).is_absolute()):
            raise ValueError("runtime source closure paths must be normalized POSIX relatives")
        _digest(digest, "source digest")
    lock = doc["source_lock"]
    _keys(lock, ("path", "bytes", "sha256"), "immutable source lock")
    expected_lock = root / "bench/runtime/adaptation-sources.json"
    if (_absolute(lock["path"], "source lock") != expected_lock
            or lock["sha256"] != SOURCE_LOCK_SHA256 or type(lock["bytes"]) is not int or lock["bytes"] <= 0):
        raise ValueError("immutable adaptation source lock pin differs")
    for key in ("recipe_pin", "exposure_audit_pin"):
        pin = doc[key]
        _keys(pin, ("path", "bytes", "sha256"), key)
        _absolute(pin["path"], key)
        _digest(pin["sha256"], key + " digest")
        if type(pin["bytes"]) is not int or not 0 < pin["bytes"] <= MAX_METADATA_BYTES:
            raise ValueError(f"{key} exceeds the bounded metadata size")
    provenance = doc["provenance"]
    _keys(provenance, ("notebook_url", "observed_at_utc", "free_resource_evidence",
                       "cuda_device_identity"), "parent provenance")
    url = provenance["notebook_url"]
    if not isinstance(url, str) or not url.startswith("https://www.kaggle.com/code/"):
        raise ValueError("parent-observed Kaggle notebook URL is required")
    if not isinstance(provenance["observed_at_utc"], str) or not provenance["observed_at_utc"].endswith("Z"):
        raise ValueError("parent observation timestamp must be UTC ISO text")
    _keys(provenance["free_resource_evidence"], ("path", "bytes", "sha256"), "free-resource evidence pin")
    _digest(provenance["free_resource_evidence"]["sha256"], "free-resource evidence digest")
    identity = provenance["cuda_device_identity"]
    if not isinstance(identity, (str, dict)) or not identity:
        raise ValueError("parent-observed CUDA device identity is required")
    if isinstance(identity, dict):
        _keys(identity, ("name", "uuid"), "CUDA device identity")
        if not all(isinstance(identity[key], str) and identity[key] for key in identity):
            raise ValueError("CUDA device identity must include nonempty name and UUID")
    elif not isinstance(identity, str) or not identity:
        raise ValueError("CUDA device identity must be a nonempty string")
    return doc


def load_request(path: Path, expected_sha256: str) -> ActivationContext:
    path = _absolute(str(path), "request path")
    raw, doc = read_metadata(path, "activation request")
    expected = _digest(expected_sha256, "request digest")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("activation request bytes differ from caller pin")
    validate_request(doc)
    return ActivationContext(path, raw, expected)


def expected_source_paths() -> frozenset[str]:
    from .quality_adaptation_portable import source_paths
    return frozenset(source_paths() | {
        "bench/quality_kaggle_activation.py", "scripts/kaggle_activation.py"})


def verify_sources(context: ActivationContext, *, upstream: bool = False) -> dict[str, Any]:
    if type(context) is not ActivationContext:
        raise TypeError("exact immutable ActivationContext required")
    doc = context.document()
    root = Path(doc["source_root"])
    expected = expected_source_paths()
    if set(doc["source_files"]) != set(expected):
        raise ValueError("activation runtime source closure differs from portable closure plus exact helper files")
    for relative, digest in doc["source_files"].items():
        path = _absolute((root / relative).as_posix(), "executing source")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("activation executing source changed: " + relative)
    lock_path = verify_file_pin(doc["source_lock"], under=root, maximum=MAX_METADATA_BYTES)
    if sha256_file(lock_path) != SOURCE_LOCK_SHA256:
        raise ValueError("immutable upstream source lock changed")
    result = {"source_lock_sha256": SOURCE_LOCK_SHA256,
        "source_files_sha256": hashlib.sha256(canonical_bytes(doc["source_files"])).hexdigest()}
    if not upstream:
        return result
    lock = strict_json_bytes(lock_path.read_bytes(), "adaptation source lock")
    from .quality_assets import SOURCES
    for kind in ("gotrack", "sam2"):
        entry = lock.get(kind)
        if not isinstance(entry, dict) or entry.get("revision") != SOURCES[kind]:
            raise ValueError("upstream revision differs from immutable source admission")
        for relative, digest in entry["files"].items():
            if not relative.startswith(".cache/model-quality/sources/"):
                continue
            path = _absolute((root / relative).as_posix(), "upstream source")
            if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
                raise ValueError("authenticated upstream source changed: " + relative)
        allowed = {rel for name in ("gotrack", "sam2") for rel in lock[name]["files"]
                   if rel.startswith(".cache/model-quality/sources/")}
        upstream_root = root / ".cache/model-quality/sources"
        for path in upstream_root.rglob("*"):
            if path.suffix in (".py", ".yaml", ".yml") and path.relative_to(root).as_posix() not in allowed:
                raise ValueError("unlisted executable upstream source: " + path.name)
    result["upstream_verified"] = True
    return result


def _require_stage(context: ActivationContext, name: str) -> None:
    if getattr(_EXECUTION, "token", None) != (context.sha256, name):
        raise RuntimeError("activation producer requires its explicit guarded stage")


def _lstat_tree_bytes(roots: list[Path]) -> tuple[int, int]:
    """Bound real physical files without following links or racing renames.

    Files that disappear during an atomic create/replace are skipped for this
    sample; stage writers must reserve their worst-case size before writing.
    Persistent inspection errors and every observed symlink fail closed.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for root in roots:
        try:
            root_info = root.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(root_info.st_mode):
            raise ActivationError("symlink in bounded activation tree: " + root.as_posix())
        if not stat.S_ISDIR(root_info.st_mode):
            raise ActivationError("bounded activation root is not a directory: " + root.as_posix())
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    names = [entry.name for entry in entries]
            except FileNotFoundError:
                # A directory may be atomically replaced between queueing and
                # scanning. Planned writes are charged by resource_check.
                continue
            except OSError as exc:
                raise ActivationError("cannot inspect bounded activation tree") from exc
            for name in names:
                path = directory / name
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise ActivationError("cannot lstat bounded activation entry") from exc
                if stat.S_ISLNK(info.st_mode):
                    raise ActivationError("symlink in bounded activation tree: " + path.as_posix())
                identity = (info.st_dev, info.st_ino)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode) and identity not in seen:
                    total += info.st_size
                    seen.add(identity)
                elif not stat.S_ISREG(info.st_mode):
                    raise ActivationError("unsupported filesystem entry in bounded activation tree")
    return total, len(seen)


def resource_check(context: ActivationContext, reserve: int = 0) -> dict[str, Any]:
    doc = context.document()
    if type(reserve) is not int or reserve < 0:
        raise ValueError("resource reservation must be a nonnegative integer")
    roots = [Path(doc[key]) for key in ("output_root", "scratch_root", "model_cache")]
    used, files = _lstat_tree_bytes(roots)
    emergency = min(MAX_METADATA_BYTES * 2, doc["limits"]["storage_bytes"] // 4)
    if used + reserve + emergency > doc["limits"]["storage_bytes"]:
        raise ActivationError("combined physical activation storage bound exhausted")
    cache_bytes, _ = _lstat_tree_bytes([roots[-1]])
    if cache_bytes > MAX_CACHE_BYTES:
        raise ActivationError("model source/checkpoint cache exceeds eight GiB")
    checked: set[int] = set()
    for root in roots:
        anchor = root
        while True:
            try:
                anchor_info = anchor.stat()
                break
            except FileNotFoundError:
                if anchor.parent == anchor:
                    raise ActivationError("cannot locate physical filesystem for a bounded root")
                anchor = anchor.parent
        if anchor_info.st_dev in checked:
            continue
        checked.add(anchor_info.st_dev)
        if shutil.disk_usage(anchor).free < reserve + doc["limits"]["reserve_bytes"]:
            raise ActivationError("physical filesystem free-space reserve exhausted")
    return {"shared_physical_bytes": used, "physical_files": files,
            "devices": sorted(checked), "quota_isolation": "unknown"}


def _rss_tree() -> int:
    processes: dict[int, tuple[int, int]] = {}
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
            processes[int(path.parent.name)] = (int(fields["PPid"]),
                int(fields.get("VmRSS", "0 kB").split()[0]) * 1024)
        except (OSError, ValueError, KeyError):
            continue
    descendants, pending = set(), [os.getpid()]
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(child for child, (parent, _) in processes.items() if parent == pid)
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:1024]


def _atomic_write_json(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_METADATA_BYTES or path.is_symlink():
        raise ActivationError("bounded activation JSON output is malformed, oversized, or a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        _write_exclusive(path, raw)
    else:
        temporary = path.with_name("." + path.name + ".part-" + uuid.uuid4().hex)
        try:
            _write_exclusive(temporary, raw)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return file_pin(path)


def _read_json(path: Path, name: str = "activation metadata") -> Any:
    target = _absolute(str(Path(path)), name)
    info = target.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METADATA_BYTES:
        raise ValueError(f"{name} is not a bounded regular nonsymlink file")
    return read_metadata(target, name)[1]


def _stage_paths(root: Path, name: str) -> tuple[Path, Path]:
    return root / ("stage-" + name + "-started.json"), root / ("stage-" + name + ".json")


def _stage_elapsed(root: Path) -> float:
    freeze_receipt = _read_json(root / "freeze.json", "activation freeze receipt")
    began = freeze_receipt.get("created_unix_s")
    if type(began) not in (int, float) or not math.isfinite(began) or began > time.time() + 1:
        raise ActivationError("invalid activation cumulative wall-clock origin")
    return max(0., time.time() - began)


@contextmanager
def stage(context: ActivationContext, name: str, *, cloud_execution: bool = False):
    """One-shot CUDA-stage supervisor; only actual Kaggle execution is admitted."""
    from . import quality_adaptation_larger as larger

    doc, _, _ = _frozen(context)
    if name not in ("render", "banks", "legacy", "amortized", "compare"):
        raise ValueError("unknown Kaggle activation stage")
    if (not cloud_execution or platform.system() != "Linux"
            or os.environ.get("VISUALIZEIT_CLOUD") != "1"
            or not Path("/kaggle/input").is_dir()
            or not Path(doc["output_root"]).is_relative_to("/kaggle/working")
            or not Path(doc["scratch_root"]).is_relative_to("/tmp")):
        raise RuntimeError("explicit Kaggle Linux cloud execution and scoped working/tmp roots required")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("activation watchdog requires the main thread")
    root = Path(doc["output_root"])
    dependencies = {"render": (), "banks": ("render",), "legacy": ("render", "banks"),
        "amortized": ("render", "banks", "legacy"), "compare": ("legacy", "amortized")}
    for prerequisite in dependencies[name]:
        _, terminal = _stage_paths(root, prerequisite)
        prior = _read_json(terminal, "activation prerequisite receipt")
        if prior.get("status") != "completed" or prior.get("request_sha256") != context.sha256:
            raise ActivationError("required prior activation stage did not complete: " + prerequisite)
    marker, terminal = _stage_paths(root, name)
    if marker.exists() or terminal.exists():
        raise FileExistsError("activation stage already attempted; preserve original evidence")
    if _stage_elapsed(root) >= doc["limits"]["wall_seconds"]:
        raise ActivationError("cumulative activation wall deadline exhausted")
    # This exclusive marker is the retry barrier. It precedes imports and any
    # model/device work; a crash leaves the attempt visibly incomplete.
    _atomic_write_json(marker, {"schema": STAGE_SCHEMA, "stage": name,
        "status": "started", "request_sha256": context.sha256,
        "attempt_id": uuid.uuid4().hex, "started_unix_s": time.time()})
    started = time.monotonic()
    remaining = doc["limits"]["wall_seconds"] - _stage_elapsed(root)
    stopped, problems = threading.Event(), []
    peak = 0

    def interrupt(signum, frame):
        raise ActivationError(problems[0] if problems else "cumulative activation deadline exhausted")

    def monitor():
        nonlocal peak
        while not stopped.wait(.1):
            try:
                peak = max(peak, _rss_tree())
                if peak > doc["limits"]["rss_bytes"]:
                    raise ActivationError("activation process-tree RSS bound exhausted")
                resource_check(context, MAX_METADATA_BYTES)
            except BaseException as exc:
                if not stopped.is_set():
                    problems.append(_failure(exc))
                    os.kill(os.getpid(), signal.SIGALRM)
                return

    status, error = "failed", None
    token = None
    cleanup_errors: list[str] = []
    with larger.stage_lock():
        old_handler = signal.getsignal(signal.SIGALRM)
        if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
            raise RuntimeError("another execution deadline is active")
        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, max(0.01, remaining))
        token = getattr(_EXECUTION, "token", None)
        _EXECUTION.token = (context.sha256, name)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            resource_check(context, MAX_METADATA_BYTES)
            verify_sources(context, upstream=(name != "freeze"))
            _verify_neural_environment(context)
            yield
            resource_check(context, MAX_METADATA_BYTES)
            status = "completed"
        except BaseException as exc:
            error = _failure(exc)
            raise
        finally:
            stopped.set()
            watcher.join(timeout=2)
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, old_handler)
            if token is None:
                try:
                    del _EXECUTION.token
                except AttributeError:
                    pass
            else:
                _EXECUTION.token = token
            try:
                from .quality_adaptation_portable import _close_neighbors
                _close_neighbors()
            except BaseException as exc:
                cleanup_errors.append(_failure(exc))
                status, error = "failed", _failure(exc)
            try:
                _atomic_write_json(terminal, {"schema": STAGE_SCHEMA, "stage": name,
                    "status": status, "request_sha256": context.sha256,
                    "attempt_id": _read_json(marker)["attempt_id"],
                    "elapsed_seconds": time.monotonic() - started,
                    "cumulative_wall_seconds": _stage_elapsed(root),
                    "process_tree_peak_rss_bytes": peak, "error": error,
                    "cleanup_errors": cleanup_errors, "cleanup_success": not cleanup_errors,
                    "test_payload_reads": 0, "annotation_inputs": []})
            except BaseException as receipt_error:
                if error is None:
                    error = _failure(receipt_error)
                status = "failed"
                raise
            if cleanup_errors:
                raise RuntimeError("activation worker cleanup failed: " + cleanup_errors[0])


def _inference_runtime_versions() -> dict[str, Any]:
    """Check frozen inference dependencies; SAM training remains out of scope."""
    from importlib.metadata import version
    from .quality_adaptation_runtime import runtime_versions
    versions = runtime_versions(sam=False)
    for name, expected in (("hydra-core", "1.3.2"), ("omegaconf", "2.3.0")):
        actual = version(name)
        if actual != expected:
            raise ActivationError(f"expected inference dependency {name}=={expected}, found {actual}")
        versions[name] = actual
    return versions


def _verify_neural_environment(context: ActivationContext) -> dict[str, Any]:
    doc = context.document()
    verify_sources(context, upstream=True)
    root, cache = Path(doc["source_root"]), Path(doc["model_cache"])
    if Path(__import__(__name__, fromlist=["__file__"]).__file__).resolve().parents[1] != root:
        raise ActivationError("activation helper import path differs from admitted checkout")
    for base in (root / "bench", root / "scripts", cache / "sources"):
        if base.exists() and any(base.rglob("*.pyc")):
            raise ActivationError("unverified bytecode exists; use a clean -B source runtime")
    from .quality_assets import CACHE, MODELS
    if CACHE.resolve(strict=True) != cache:
        raise ActivationError("model cache differs from actual authenticated loader cache")
    cache_bytes, _ = _lstat_tree_bytes([cache])
    if cache_bytes > MAX_CACHE_BYTES:
        raise ActivationError("model cache exceeds eight GiB")
    from .quality_adaptation_data import read_json as read_cache_json
    lock = read_cache_json(cache / "lock.json")
    if set(MODELS) != set(lock.get("models", {})):
        raise ActivationError("model cache descriptor inventory differs from loader")
    model_pins = {}
    for name, descriptor in MODELS.items():
        path = cache / "checkpoints" / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise ActivationError("missing or changed pre-existing model checkpoint: " + name)
        if lock["models"].get(name, {}).get("sha256") != descriptor["sha256"]:
            raise ActivationError("model cache lock differs from loader descriptor: " + name)
        model_pins[name] = {"bytes": path.stat().st_size, "sha256": descriptor["sha256"]}
    from .quality_adaptation_runtime import _source_imports
    _source_imports("gotrack")
    _source_imports("sam2")
    versions = _inference_runtime_versions()
    if sys.version_info[:3] != (3, 11, 13):
        raise ActivationError("activation requires the pinned Python 3.11.13 runtime")
    return {"model_cache_bytes": cache_bytes, "models": model_pins,
            "versions": versions, "python": platform.python_version(),
            "source_lock_sha256": SOURCE_LOCK_SHA256}


def _cuda_identity(torch, provenance) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1 or torch.cuda.current_device() != 0:
        raise ActivationError("CUDA device 0 is unavailable; no CPU fallback is permitted")
    properties = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    raw_uuid = getattr(properties, "uuid", None)
    if not raw_uuid:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"],
            check=True, capture_output=True, text=True, timeout=10)
        raw_uuid = completed.stdout.strip().removeprefix("GPU-")
    try:
        # Torch 2.5.1 exposes a _CUuuid object; nvidia-smi adds a GPU- prefix.
        # Both are the same physical identifier, not distinct device identities.
        device_uuid = str(uuid.UUID(str(raw_uuid)))
    except (ValueError, TypeError, AttributeError) as error:
        raise ActivationError("selected CUDA device has no stable physical UUID") from error
    identity = provenance["cuda_device_identity"]
    if isinstance(identity, dict):
        try:
            observed_uuid = str(uuid.UUID(identity["uuid"].removeprefix("GPU-")))
        except (ValueError, TypeError, AttributeError) as error:
            raise ActivationError("parent CUDA device UUID is malformed") from error
        matches = identity["name"] == name and observed_uuid == device_uuid
    else:
        matches = identity in (name, device_uuid, f"GPU-{device_uuid}", f"{name} ({device_uuid})")
    if not matches:
        raise ActivationError("selected CUDA device differs from parent-observed physical identity")
    return {"device": "cuda", "selected_device": "cuda:0", "name": name, "uuid": device_uuid,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": torch.get_num_threads()}


def _physical_state_rows(plan_rows):
    from .quality_adaptation_refinement_states import state_rows
    return state_rows({"rows": [dict(entry["physical"], iterations=[0]) for entry in plan_rows]})


def _branch_entries(context, variant, rows, render_index, bank_doc, mask_rows):
    rendered_by_id = {entry["physical"]["id"]: entry for entry in render_index["rows"]}
    result = []
    for physical, mask in zip(rows, mask_rows):
        rendered = rendered_by_id[physical["id"]]
        entry = {"physical": physical,
            "input": rendered["input"] if rendered["status"] == "available" else None,
            "video": rendered["video"] if rendered["status"] == "available" else None,
            "mesh": rendered["mesh"] if rendered["status"] == "available" else None,
            "bank": bank_doc["pose_bank"] if bank_doc["status"] == "available" else None,
            "mask": mask.get("mask") if mask.get("status") == "available" else None,
            "mask_index": mask.get("mask_index") if mask.get("status") == "available" else None,
            "mask_status": mask.get("status"), "mask_failure": mask.get("failure_reason")}
        result.append(entry)
    return result


def _empty_observations() -> dict[str, Any]:
    return {"detections": [], "candidates": [], "test_payload_reads": 0,
            "annotation_inputs": []}


def _run_capture_rows(context, variant, branch_root, raw_plan, rows, bank_doc, render_index,
                      state_rows, torch, hardware) -> tuple[dict[str, Any], str | None]:
    from . import quality_adaptation_larger as larger
    from . import quality_adaptation_refinement_states as shared
    from .quality_adaptation_larger_optimized import PreparedFoundPoseSearch, Timings, dispose_search
    from .quality_gotrack import GoTrackRefiner, TexturedRenderer, load_network
    from .quality_foundpose import FoundPoseRecovery
    from .quality_contract import Frame
    from .quality_time import PHYSICAL
    from .quality_runner import read_rgb
    from .vision import cv2
    import numpy as np
    doc = context.document()
    root = Path(doc["output_root"])
    capture_root = branch_root / "capture"
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise ActivationError("capture output directory must be the precreated branch directory")
    initial_index = _read_json(capture_root / "capture-index.json", "initial activation capture index")
    if (set(initial_index) != {"rows", "test_payload_reads", "training_ready"}
            or initial_index["test_payload_reads"] != 0 or initial_index["training_ready"] is not False
            or initial_index["rows"] != state_rows):
        raise ActivationError("initial activation capture inventory changed")
    physical_by_id = {row["id"]: row for row in rows}
    rendered_by_id = {entry["physical"]["id"]: entry for entry in render_index["rows"]}
    mask_index = _read_json(branch_root / "mask-index.json", "branch mask index")
    mask_by_id = {entry["physical_id"]: entry for entry in mask_index["rows"]}
    plan_doc = strict_json_bytes(raw_plan, "activation observer plan")
    plan_rows = plan_doc["rows"]
    if len(plan_rows) != 2:
        raise ActivationError("capture plan must retain exactly two prospective physical rows")
    sinks: dict[str, Any] = {}
    capabilities: dict[str, Any] = {}
    view_inputs: dict[str, Any] = {}
    fatal = None
    # All selected metadata and authentic input/mask/bank pins pass the new
    # capability preflight before either view's video or mask pixels are decoded.
    for entry in plan_rows:
        physical = physical_by_id[entry["physical"]["id"]]
        mask = mask_by_id[physical["id"]]
        if mask["status"] != "available":
            failure = mask["failure_reason"] or "observed mask unavailable"
            for state in state_rows:
                if state["id"] == physical["id"] + ":iteration0":
                    state["failure_reason"] = failure
            continue
        rendered = rendered_by_id[physical["id"]]
        bundle, manifest, native_k = _manifest_for(context, physical, rendered)
        callback = lambda: resource_check(context, 32 * 1024**2)
        sink_physical = _activation_physical(physical, rendered, bank_doc, mask)
        sink = shared.RefStateCapture(capture_root, sink_physical, state_rows, {}, native_k,
            max_output_bytes=doc["limits"]["storage_bytes"],
            reserve_bytes=doc["limits"]["reserve_bytes"], resource_check=callback)
        capability = issue_observer(raw_plan, hashlib.sha256(raw_plan).hexdigest(), sink,
            context=context)
        sinks[physical["id"]] = sink
        capabilities[physical["id"]] = capability
        view_inputs[physical["id"]] = (bundle, manifest, native_k, rendered, mask)
    network = None
    search = None
    timings = Timings("cuda", torch)
    candidates = []
    cleanup_errors = []
    try:
        if sinks:
            from .quality_gotrack import load_network
            network = load_network("cuda").eval()
            if variant == "amortized-v1":
                search = PreparedFoundPoseSearch(object_id="1", bank_pin=bank_doc["pose_bank"],
                    asset_sha256=rows[0]["mesh_recipe"]["sha256"], variant=variant,
                    timings=timings, loader=lambda path: torch.load(path, map_location="cpu", weights_only=False),
                    pin_resolver=lambda pin: verify_file_pin(pin, under=root))
        for physical in rows:
            physical_id = physical["id"]
            if physical_id not in sinks:
                _atomic_write_json(capture_root / "capture-index.json",
                    {"rows": state_rows, "test_payload_reads": 0, "training_ready": False}, replace=True)
                continue
            sink, capability = sinks[physical_id], capabilities[physical_id]
            bundle, manifest, native_k, rendered, mask_record = view_inputs[physical_id]
            renderer = seed_refiner = recovery = bank_state = refiner = None
            reason = None
            try:
                resource_check(context, 128 * 1024**2)
                rgb = read_rgb(bundle, manifest, 0)
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                    raise ValueError("normalized RGB decode differs from 640x480 uint8")
                mask_path = verify_file_pin(mask_record["mask"], under=branch_root, maximum=16 * 1024**2)
                observed_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if observed_mask is None or observed_mask.shape != rgb.shape[:2] or not np.any(observed_mask):
                    raise ValueError("actual observed SAM2 mask is unreadable, empty, or wrong-sized")
                frame = Frame(0, rgb, native_k, timestamp_s=0., clock_mode=PHYSICAL)
                import random
                random.seed(0); np.random.seed(0); torch.manual_seed(0)
                renderer = TexturedRenderer(bundle / manifest["asset"], manifest["object_id"],
                    unlit=True, disable_multisampling=True,
                    coordinate_mode="integer_centers_v1")
                seed_refiner = GoTrackRefiner(network, renderer, manifest["object_id"], "cuda",
                    inference_geometry=larger.inference.PROFILE)
                if variant == "legacy-v1":
                    bank_path = verify_file_pin(bank_doc["pose_bank"], under=root, maximum=MAX_CACHE_BYTES)
                    bank_state = torch.load(bank_path, map_location="cpu", weights_only=False)
                    if bank_state.get("template_count") != 798 or bank_state.get("asset_sha256") != physical["mesh_recipe"]["sha256"]:
                        raise ValueError("authenticated 798-template FoundPose bank does not match the selected object")
                    recovery = FoundPoseRecovery(seed_refiner, bank_state,
                        time_policy=manifest["_timing"]["time_policy"])
                else:
                    binding = dict(object_id="1", bank_pin=bank_doc["pose_bank"],
                        asset_sha256=physical["mesh_recipe"]["sha256"], variant=variant)
                    recovery = search.bind(seed_refiner, **binding,
                        time_policy=manifest["_timing"]["time_policy"])
                seeds = recovery.recover(frame, observed_mask, top_k=5)
                candidates.append({"id": physical_id, "poses": [np.asarray(seed).tolist() for seed in seeds]})
                if variant == "legacy-v1":
                    dispose_search(recovery)
                bank_state = recovery = seed_refiner = None
                if not seeds:
                    raise ValueError("unavailable_seed: FoundPose returned no estimated hypothesis")
                with shared.observe_selected_crops([0]) as checks:
                    sink.sampling_checks = checks
                    capability.validate(sink, larger.inference.PROFILE)
                    refiner = GoTrackRefiner(network, renderer, manifest["object_id"], "cuda",
                        inference_geometry=larger.inference.PROFILE,
                        chronological_capture_callback=sink,
                        inference_observer_capability=capability)
                    refiner.set_chronological_capture_context({"physical_id": physical_id})
                    refiner.refine(frame, observed_mask, seeds[0])
            except BaseException as exc:
                reason = _failure(exc)
                if isinstance(exc, (KeyboardInterrupt, SystemExit, ActivationError)):
                    fatal = reason
                    raise
                for state in state_rows:
                    if state["id"] == physical_id + ":iteration0" and state["status"] != "available":
                        state["failure_reason"] = reason
            finally:
                if sink is not None:
                    sink.finish(reason)
                if variant == "legacy-v1" and recovery is not None:
                    try:
                        dispose_search(recovery)
                    except BaseException as exc:
                        cleanup_errors.append(_failure(exc))
                if renderer is not None:
                    try:
                        renderer.close()
                    except BaseException as exc:
                        cleanup_errors.append(_failure(exc))
                renderer = seed_refiner = recovery = bank_state = refiner = None
                _atomic_write_json(capture_root / "capture-index.json",
                    {"rows": state_rows, "test_payload_reads": 0, "training_ready": False}, replace=True)
                if cleanup_errors:
                    fatal = "capture cleanup failed: " + cleanup_errors[0]
                    break
    except BaseException as exc:
        fatal = fatal or _failure(exc)
    finally:
        if search is not None:
            try:
                search.close()
            except BaseException as exc:
                cleanup_errors.append(_failure(exc))
        if network is not None:
            try:
                network.cpu()
            except BaseException as exc:
                cleanup_errors.append(_failure(exc))
        try:
            larger.cleanup_device("cuda", torch)
        except BaseException as exc:
            cleanup_errors.append(_failure(exc))
        if cleanup_errors:
            fatal = fatal or "device/search cleanup failed: " + cleanup_errors[0]
        for state in state_rows:
            if state.get("failure_reason") == "not_processed":
                state["failure_reason"] = fatal or "unrun_after_termination"
        _atomic_write_json(capture_root / "capture-index.json",
            {"rows": state_rows, "test_payload_reads": 0, "training_ready": False}, replace=True)
    return {"rows": state_rows, "candidates": candidates,
        "fatal_error": fatal, "cleanup_errors": cleanup_errors,
        "timings": timings.report(), "hardware": hardware}


def run_branch(context: ActivationContext, variant: str) -> dict[str, Any]:
    """Run one independent authentic CUDA mask/recovery/capture branch."""
    stage_name = {"legacy-v1": "legacy", "amortized-v1": "amortized"}.get(variant)
    if stage_name is None:
        raise ValueError("activation branch variant must be exact legacy-v1 or amortized-v1")
    _require_stage(context, stage_name)
    doc, activation, rows = _frozen(context)
    root = Path(doc["output_root"])
    branch_root = root / "branches" / variant
    branch_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    stage_marker, _ = _stage_paths(root, stage_name)
    stage_start = _read_json(stage_marker, "activation stage start")
    if (stage_start.get("schema") != STAGE_SCHEMA or stage_start.get("stage") != stage_name
            or stage_start.get("status") != "started"
            or stage_start.get("request_sha256") != context.sha256
            or not isinstance(stage_start.get("attempt_id"), str)):
        raise ActivationError("branch is not bound to its exclusive stage attempt")
    stage_attempt_id = stage_start["attempt_id"]
    _atomic_write_json(branch_root / "branch-started.json", {"schema": BRANCH_SCHEMA,
        "status": "started", "request_sha256": context.sha256, "variant": variant,
        "stage_attempt_id": stage_attempt_id, "started_unix_s": started})
    fatal = None
    hardware = None
    observations_path = branch_root / "observations.json"
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_larger_parity import observe_runtime
    render_index = _read_render_index(context)
    banks_receipt = _read_banks(context)
    mask_rows = []
    capture_result = {"rows": _physical_state_rows([{"physical": row} for row in rows]),
        "candidates": [], "fatal_error": None, "cleanup_errors": [], "timings": {}, "hardware": None}
    try:
        for row, rendered in zip(rows, render_index["rows"]):
            if rendered["status"] == "available":
                _manifest_for(context, row, rendered)
        if banks_receipt["status"] == "available" and any(row["status"] == "available" for row in render_index["rows"]):
            configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if configured not in (None, ":4096:8"):
                raise ActivationError("CUDA workspace was configured inconsistently before model imports")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch = larger.configure_torch("cuda", deterministic=True)
            hardware = _cuda_identity(torch, doc["provenance"])
            with observe_runtime(observations_path) as observations:
                if variant == "legacy-v1":
                    mask_rows = _legacy_masks(context, branch_root, rows, render_index, banks_receipt, torch)
                else:
                    mask_rows = _amortized_masks(context, branch_root, rows, render_index, banks_receipt, torch)
                _atomic_write_json(branch_root / "mask-index.json", {"schema": BRANCH_SCHEMA,
                    "variant": variant, "request_sha256": context.sha256,
                    "rows": [{"physical_id": row["id"], **mask} for row, mask in zip(rows, mask_rows)],
                    "test_payload_reads": 0, "annotation_inputs": []})
                branch_rows = _branch_entries(context, variant, rows, render_index, banks_receipt, mask_rows)
                raw_plan = _observer_plan_bytes(context, variant, branch_rows)
                _write_exclusive(branch_root / "capture-plan.json", raw_plan)
                states = _physical_state_rows(branch_rows)
                _atomic_write_json(branch_root / "capture" / "capture-index.json", {
                    "rows": states, "test_payload_reads": 0, "training_ready": False})
                capture_result = _run_capture_rows(context, variant, branch_root, raw_plan,
                    rows, banks_receipt, render_index, states, torch, hardware)
        else:
            mask_rows = [_unavailable_mask(ValueError("render_or_shared_bank_unavailable")) for _ in rows]
            _atomic_write_json(observations_path, _empty_observations())
            _atomic_write_json(branch_root / "mask-index.json", {"schema": BRANCH_SCHEMA,
                "variant": variant, "request_sha256": context.sha256,
                "rows": [{"physical_id": row["id"], **mask} for row, mask in zip(rows, mask_rows)],
                "test_payload_reads": 0, "annotation_inputs": []})
            branch_rows = _branch_entries(context, variant, rows, render_index, banks_receipt, mask_rows)
            raw_plan = _observer_plan_bytes(context, variant, branch_rows)
            _write_exclusive(branch_root / "capture-plan.json", raw_plan)
            states = _physical_state_rows(branch_rows)
            for state, mask in zip(states, mask_rows):
                state["failure_reason"] = mask["failure_reason"]
            (branch_root / "capture").mkdir(parents=True, exist_ok=False)
            _atomic_write_json(branch_root / "capture" / "capture-index.json", {
                "rows": states, "test_payload_reads": 0, "training_ready": False})
            capture_result = {"rows": states, "candidates": [], "fatal_error": None,
                "cleanup_errors": [], "timings": {}}
        if hardware is None and capture_result.get("fatal_error") is None:
            capture_result["fatal_error"] = "no_cuda_execution:render_or_shared_bank_unavailable"
        fatal = capture_result.get("fatal_error")
        snapshot_path = branch_root / "snapshot.json"
        snapshot_pin = None
        if hardware is not None:
            snapshot_doc = _collect_snapshot(context, variant, hardware=hardware, fatal_error=fatal)
            snapshot_pin = _atomic_write_json(snapshot_path, snapshot_doc)
        receipt = {"schema": BRANCH_SCHEMA, "status": "failed" if fatal else "completed",
            "request_sha256": context.sha256, "variant": variant,
            "stage_attempt_id": stage_attempt_id,
            "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
            "branch_root": branch_root.as_posix(),
            "capture_plan": file_pin(branch_root / "capture-plan.json"),
            "mask_index": file_pin(branch_root / "mask-index.json"),
            "observations": file_pin(observations_path),
            "capture_index": file_pin(branch_root / "capture" / "capture-index.json"),
            "snapshot": snapshot_pin,
            "hardware": hardware, "fatal_error": fatal,
            "cleanup_errors": capture_result.get("cleanup_errors", []),
            "cleanup_success": not capture_result.get("cleanup_errors", []),
            "requested_physical_frames": 2, "requested_refinement_states": 2,
            "elapsed_seconds": time.time() - started,
            "test_payload_reads": 0, "annotation_inputs": []}
        _atomic_write_json(branch_root / "branch-receipt.json", receipt)
        if snapshot_pin is not None:
            snapshot_activation(context, variant)
        if fatal:
            raise RuntimeError("activation branch failed: " + str(fatal))
        return receipt
    except BaseException as exc:
        fatal = fatal or _failure(exc)
        if not observations_path.exists():
            _atomic_write_json(observations_path, _empty_observations())
        receipt_path = branch_root / "branch-receipt.json"
        if not receipt_path.exists():
            capture_index_path = branch_root / "capture" / "capture-index.json"
            if not capture_index_path.exists():
                (branch_root / "capture").mkdir(parents=True, exist_ok=True)
                state_rows = _physical_state_rows([{"physical": row} for row in rows])
                for state in state_rows:
                    state["failure_reason"] = fatal
                _atomic_write_json(capture_index_path, {"rows": state_rows,
                    "test_payload_reads": 0, "training_ready": False})
            for required_path, placeholder in ((branch_root / "mask-index.json", {"schema": BRANCH_SCHEMA,
                    "variant": variant, "request_sha256": context.sha256, "rows": [],
                    "test_payload_reads": 0, "annotation_inputs": []}),
                    (branch_root / "capture-plan.json", None)):
                if not required_path.exists() and placeholder is not None:
                    _atomic_write_json(required_path, placeholder)
            if (branch_root / "capture-plan.json").exists() and (branch_root / "mask-index.json").exists():
                _atomic_write_json(receipt_path, {"schema": BRANCH_SCHEMA, "status": "failed",
                    "request_sha256": context.sha256, "variant": variant,
                    "stage_attempt_id": stage_attempt_id,
                    "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
                    "branch_root": branch_root.as_posix(),
                    "capture_plan": file_pin(branch_root / "capture-plan.json"),
                    "mask_index": file_pin(branch_root / "mask-index.json"),
                    "observations": file_pin(observations_path),
                    "capture_index": file_pin(capture_index_path),
                    "snapshot": None,
                    "hardware": hardware, "fatal_error": fatal,
                    "cleanup_errors": capture_result.get("cleanup_errors", []),
                    "cleanup_success": not capture_result.get("cleanup_errors", []),
                    "requested_physical_frames": 2, "requested_refinement_states": 2,
                    "elapsed_seconds": time.time() - started,
                    "test_payload_reads": 0, "annotation_inputs": []})
        raise


def _verify_exposure_audit(audit: dict[str, Any], spec: dict[str, Any], recipe_sha: str,
                           *, verify_evidence: bool, source_root: Path) -> dict[str, Any]:
    if (audit.get("schema") != "kaggle_exposure_audit_v2" or audit.get("complete") is not True
            or audit.get("recipe_sha256") != recipe_sha
            or audit.get("admitted_for_execution") is not True
            or audit.get("local_validation_only") is not False
            or audit.get("whole_catalog_exposure_audit_complete") is not False
            or audit.get("unknown_pretrained_history_claim") is not False
            or audit.get("new_or_unseen_identity_claim") is not False
            or audit.get("sealed_test_payload_acquisition_forbidden") is not True):
        raise ValueError("full exposure audit is not admitted/rebound for this exact recipe")
    if not isinstance(audit.get("assets"), list) or len(audit["assets"]) != len(spec["assets"]):
        raise ValueError("exposure audit inventory must match every full recipe asset")
    by_id = {entry.get("physical_object_id"): entry for entry in audit["assets"]
             if isinstance(entry, dict)}
    if len(by_id) != len(audit["assets"]):
        raise ValueError("exposure audit has duplicate/malformed physical identities")
    for asset in spec["assets"]:
        item = by_id.get(asset["physical_object_id"])
        if item is None or (item.get("mesh_sha256") != asset["mesh"]["sha256"]
                or item.get("aliases") != asset["aliases"] or item.get("eligible") is not True
                or item.get("known_roles_are_exhaustive_history") is not False
                or not isinstance(item.get("historical_roles"), list)
                or not item["historical_roles"] or not isinstance(item.get("role_evidence"), list)
                or not item["role_evidence"]):
            raise ValueError("exposure audit identity/known-role evidence differs from full recipe")
        for evidence in item["role_evidence"]:
            if (not isinstance(evidence, dict) or set(evidence) != {"source", "selector", "supports"}
                    or not all(isinstance(evidence[k], str) and evidence[k] for k in evidence)):
                raise ValueError("exposure audit role evidence is malformed")
    descriptors = audit.get("evidence_descriptors")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("complete exposure audit must pin its source evidence descriptors")
    if verify_evidence:
        for descriptor in descriptors:
            verify_file_pin(descriptor, maximum=MAX_METADATA_BYTES)
    return {"assets": len(audit["assets"]), "evidence_descriptors": len(descriptors),
            "audit_sha256": hashlib.sha256(canonical_bytes(audit)).hexdigest()}


def _check_split(spec, scale_plan, physical_rows) -> dict[str, Any]:
    roles = {}
    for row in physical_rows:
        obj = row.get("object_id")
        split = row.get("split")
        if not isinstance(obj, str) or split not in ("train", "validation", "test"):
            raise ValueError("prospective physical inventory has an invalid split row")
        previous = roles.setdefault(obj, split)
        if previous != split:
            raise ValueError("one physical object crossed prospective split roles")
    if (frozenset(obj for obj, split in roles.items() if split == "train") != frozenset(EXPECTED_TRAIN_OBJECTS)
            or frozenset(obj for obj, split in roles.items() if split == "validation") != frozenset(EXPECTED_VALIDATION_OBJECTS)
            or frozenset(obj for obj, split in roles.items() if split == "test") != frozenset(EXPECTED_TEST_OBJECTS)
            or len(roles) != len(spec["assets"])):
        raise ValueError("full prospective split differs from the fixed 13/1/1 object roles")
    split = scale_plan.get("global_manifest", {}).get("split_manifest")
    if not isinstance(split, dict) or not isinstance(split.get("groups"), list):
        raise ValueError("portable prospective plan lacks its authenticated full split groups")
    selected = [row for row in physical_rows if (row["object_id"], row["view_index"]) in SELECTED_ROWS]
    if ([(row["object_id"], row["view_index"]) for row in selected] != list(SELECTED_ROWS)
            or any(row["split"] != "train" or row["iterations"] != [0] for row in selected)):
        raise ValueError("fixed object 1 TRAIN views 0 then 1 with iteration 0 differ")
    return {"train_objects": list(EXPECTED_TRAIN_OBJECTS),
            "validation_objects": list(EXPECTED_VALIDATION_OBJECTS),
            "sealed_test_objects": list(EXPECTED_TEST_OBJECTS),
            "physical_rows": len(physical_rows), "selected_rows": [row["id"] for row in selected],
            "seed": spec["seed"], "iterations": spec["iterations"],
            "split_manifest_sha256": hashlib.sha256(canonical_bytes(split)).hexdigest()}


def _write_exclusive(path: Path, raw: bytes) -> None:
    path = Path(path)
    if len(raw) > MAX_METADATA_BYTES or path.is_symlink():
        raise ValueError("exclusive metadata output exceeds bound or is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = Path(path)
    if len(raw) > MAX_METADATA_BYTES or path.is_symlink():
        raise ValueError("JSON output exceeds bound or is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        _write_exclusive(path, raw)
        return
    temporary = path.with_name(path.name + ".part-" + uuid.uuid4().hex)
    _write_exclusive(temporary, raw)
    os.replace(temporary, path)


def freeze(context: ActivationContext) -> dict[str, Any]:
    """Freeze full recipe/split metadata and the two prospective TRAIN inputs."""
    doc = context.document()
    verify_sources(context, upstream=False)
    root = Path(doc["source_root"])
    recipe_path = verify_file_pin(doc["recipe_pin"], maximum=MAX_METADATA_BYTES)
    audit_path = verify_file_pin(doc["exposure_audit_pin"], maximum=MAX_METADATA_BYTES)
    recipe_raw, spec = read_metadata(recipe_path, "full activation recipe")
    audit_raw, audit = read_metadata(audit_path, "full activation exposure audit")
    from . import quality_adaptation_portable as portable
    plan = portable.prospective(spec)
    physical = portable.physical_rows(spec, plan)
    audit_check = _verify_exposure_audit(audit, spec, hashlib.sha256(recipe_raw).hexdigest(),
                                         verify_evidence=True, source_root=root)
    inventory = _check_split(spec, plan, physical)
    if spec["iterations"] != [0] or spec["seed"] != 521:
        raise ValueError("full recipe must preserve iteration 0 and seed 521")
    selected = [row for row in physical if (row["object_id"], row["view_index"]) in SELECTED_ROWS]
    split = plan["global_manifest"]["split_manifest"]
    root_out = Path(doc["output_root"])
    scratch_root = Path(doc["scratch_root"])
    if root_out.exists() or root_out.is_symlink():
        raise FileExistsError("activation freeze requires a fresh output directory")
    if scratch_root.exists() or scratch_root.is_symlink():
        raise FileExistsError("activation freeze requires a fresh scratch directory")
    root_out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    marker = root_out / "freeze-started.json"
    _write_json(marker, {"schema": FREEZE_SCHEMA, "status": "started",
        "request_sha256": context.sha256, "started_unix_s": started})
    try:
        _write_exclusive(root_out / "request.json", context.raw)
        _write_exclusive(root_out / "recipe.json", recipe_raw)
        _write_exclusive(root_out / "exposure-audit.json", audit_raw)
        plan_raw = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        split_raw = (json.dumps(split, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(plan_raw) > MAX_METADATA_BYTES or len(split_raw) > MAX_METADATA_BYTES:
            raise ValueError("full prospective split/plan exceeds bounded metadata size")
        _write_exclusive(root_out / "scale-plan.json", plan_raw)
        _write_exclusive(root_out / "split.json", split_raw)
        _write_json(root_out / "physical-index.json", {"rows": physical, "test_payload_reads": 0})
        activation_plan = {
            "schema": PLAN_SCHEMA, "request_sha256": context.sha256,
            "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
            "audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
            "source_files_sha256": hashlib.sha256(canonical_bytes(doc["source_files"])).hexdigest(),
            "source_lock_sha256": SOURCE_LOCK_SHA256,
            "scale_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "split_sha256": hashlib.sha256(split_raw).hexdigest(),
            "inventory": inventory, "audit": audit_check,
            "full_recipe": {"schema": spec["schema"], "asset_count": len(spec["assets"]),
                "iterations": list(spec["iterations"]), "seed": spec["seed"], "targets": list(spec["targets"])},
            "branches": list(BRANCHES), "rows": selected,
            "sealed_test_metadata_only": True, "test_payload_reads": 0,
            "annotation_inputs": [],
        }
        _write_json(root_out / "activation-plan.json", activation_plan)
        receipt = {"schema": FREEZE_SCHEMA, "status": "completed",
            "request_sha256": context.sha256,
            "request": file_pin(root_out / "request.json"),
            "recipe": file_pin(root_out / "recipe.json"),
            "exposure_audit": file_pin(root_out / "exposure-audit.json"),
            "scale_plan": file_pin(root_out / "scale-plan.json"),
            "split": file_pin(root_out / "split.json"),
            "physical_index": file_pin(root_out / "physical-index.json"),
            "activation_plan": file_pin(root_out / "activation-plan.json"),
            "inventory": inventory, "test_payload_reads": 0, "annotation_inputs": [],
            "elapsed_seconds": time.time() - started, "created_unix_s": started}
        _write_json(root_out / "freeze.json", receipt)
        return receipt
    except BaseException as exc:
        _write_json(root_out / "freeze-failed.json", {"schema": FREEZE_SCHEMA,
            "status": "failed", "request_sha256": context.sha256,
            "error": f"{type(exc).__name__}: {exc}"[:1024],
            "test_payload_reads": 0, "annotation_inputs": []})
        raise


def _frozen(context: ActivationContext) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    doc = context.document()
    verify_sources(context, upstream=False)
    root = Path(doc["output_root"])
    if root != Path(doc["output_root"]) or root.is_symlink() or not root.is_dir():
        raise ValueError("activation output root is unavailable or aliased")
    if sha256_file(root / "request.json") != context.sha256:
        raise ValueError("frozen request changed")
    receipt = strict_json_bytes((root / "freeze.json").read_bytes(), "freeze receipt")
    if (set(receipt) != {"schema", "status", "request_sha256", "request", "recipe",
            "exposure_audit", "scale_plan", "split", "physical_index", "activation_plan",
            "inventory", "test_payload_reads", "annotation_inputs", "elapsed_seconds", "created_unix_s"}
            or receipt.get("schema") != FREEZE_SCHEMA or receipt.get("status") != "completed"
            or receipt.get("request_sha256") != context.sha256 or receipt.get("test_payload_reads") != 0
            or receipt.get("annotation_inputs") != [] or (root / "freeze-failed.json").exists()):
        raise ValueError("activation freeze receipt is incomplete or contaminated")
    for key, path_name in (("request", "request.json"), ("recipe", "recipe.json"),
            ("exposure_audit", "exposure-audit.json"), ("scale_plan", "scale-plan.json"),
            ("split", "split.json"), ("physical_index", "physical-index.json"),
            ("activation_plan", "activation-plan.json")):
        verify_file_pin(receipt[key], under=root, maximum=MAX_METADATA_BYTES)
        if Path(receipt[key]["path"]) != root / path_name:
            raise ValueError("freeze receipt artifact path differs")
    activation = strict_json_bytes((root / "activation-plan.json").read_bytes(), "activation plan")
    request = context.document()
    recipe_raw, spec = read_metadata(root / "recipe.json", "frozen full activation recipe")
    audit_raw, audit = read_metadata(root / "exposure-audit.json", "frozen full exposure audit")
    if hashlib.sha256(recipe_raw).hexdigest() != request["recipe_pin"]["sha256"]:
        raise ValueError("frozen full recipe bytes differ from the parent pin")
    if hashlib.sha256(audit_raw).hexdigest() != request["exposure_audit_pin"]["sha256"]:
        raise ValueError("frozen exposure audit bytes differ from the parent pin")
    if spec.get("iterations") != [0] or spec.get("seed") != 521:
        raise ValueError("frozen full recipe lost the iteration-0/seed-521 admission")
    from . import quality_adaptation_portable as portable
    scale_plan = portable.prospective(spec)
    physical = portable.physical_rows(spec, scale_plan)
    split = scale_plan["global_manifest"]["split_manifest"]
    inventory = _check_split(spec, scale_plan, physical)
    audit_check = _verify_exposure_audit(audit, spec, hashlib.sha256(recipe_raw).hexdigest(),
        verify_evidence=True, source_root=Path(request["source_root"]))
    scale_raw = (root / "scale-plan.json").read_bytes()
    split_raw = (root / "split.json").read_bytes()
    physical_raw = _read_json(root / "physical-index.json", "full physical metadata index")
    expected_plan = {
        "schema": PLAN_SCHEMA, "request_sha256": context.sha256,
        "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "source_files_sha256": hashlib.sha256(canonical_bytes(request["source_files"])).hexdigest(),
        "source_lock_sha256": SOURCE_LOCK_SHA256,
        "scale_plan_sha256": hashlib.sha256(scale_raw).hexdigest(),
        "split_sha256": hashlib.sha256(split_raw).hexdigest(),
        "inventory": inventory, "audit": audit_check,
        "full_recipe": {"schema": spec["schema"], "asset_count": len(spec["assets"]),
            "iterations": list(spec["iterations"]), "seed": spec["seed"], "targets": list(spec["targets"])},
        "branches": list(BRANCHES),
        "rows": [row for row in physical if (row["object_id"], row["view_index"]) in SELECTED_ROWS],
        "sealed_test_metadata_only": True, "test_payload_reads": 0, "annotation_inputs": [],
    }
    if strict_json_bytes(scale_raw, "full scale plan") != scale_plan or strict_json_bytes(split_raw, "full split") != split:
        raise ValueError("frozen full prospective recipe or split changed")
    if physical_raw != {"rows": physical, "test_payload_reads": 0}:
        raise ValueError("frozen physical metadata inventory changed")
    if set(activation) != set(expected_plan) or activation != expected_plan:
        raise ValueError("frozen activation plan changed from authenticated full recipe metadata")
    rows = expected_plan["rows"]
    if (len(rows) != 2 or [(row["object_id"], row["view_index"]) for row in rows] != list(SELECTED_ROWS)
            or any(row.get("split") != "train" or row.get("iterations") != [0] for row in rows)):
        raise ValueError("frozen activation rows differ from fixed TRAIN iteration-0 inventory")
    if receipt["inventory"] != inventory:
        raise ValueError("freeze receipt inventory differs from reauthenticated full plan")
    return doc, activation, rows


def _read_completed_stage(context: ActivationContext, name: str) -> dict[str, Any]:
    root = Path(context.document()["output_root"])
    _, path = _stage_paths(root, name)
    value = _read_json(path, "activation stage receipt")
    if (value.get("schema") != STAGE_SCHEMA or value.get("stage") != name
            or value.get("status") != "completed" or value.get("request_sha256") != context.sha256
            or value.get("cleanup_success") is not True or value.get("cleanup_errors") != []
            or value.get("test_payload_reads") != 0 or value.get("annotation_inputs") != []):
        raise ActivationError("activation dependency receipt is incomplete: " + name)
    return value


def _manifest_for(context: ActivationContext, physical: dict[str, Any], rendered: dict[str, Any]):
    """Authenticate only the actual normalized input/video/mesh inference bundle."""
    doc = context.document()
    from .quality_runner import read_input
    for name in ("input", "video", "mesh"):
        pin = rendered.get(name)
        verify_file_pin(pin, under=Path(doc["output_root"]), maximum=64 * 1024**2)
    bundle = Path(rendered["input"]["path"]).parent
    if (Path(rendered["input"]["path"]) != bundle / "input.json"
            or Path(rendered["video"]["path"]) != bundle / "synthetic.mp4"
            or Path(rendered["mesh"]["path"]) != bundle / "object.glb"
            or rendered["mesh"]["sha256"] != physical["mesh_recipe"]["sha256"]):
        raise ActivationError("rendered bundle paths or mesh identity differ from frozen recipe")
    manifest = read_input(bundle)
    allowed = {"schema_version", "object", "object_id", "physical_object_id", "scene_id",
        "sequence_id", "background_id", "background_kind", "provenance_kind", "units",
        "video", "asset", "source_hashes", "intrinsics", "native_resolution",
        "setup_frame_id", "frame_ids", "clock", "timeline", "_timing"}
    if set(manifest) != allowed:
        raise ActivationError("foreign annotation/generation fields in observed inference manifest")
    if (manifest.get("schema_version") != 2 or manifest.get("provenance_kind") != "synthetic"
            or manifest.get("units") != "metres" or manifest.get("asset") != "object.glb"
            or manifest.get("video") != "synthetic.mp4"
            or manifest.get("physical_object_id") != physical["physical_object_id"]
            or manifest.get("object_id") != "synthetic_" + rendered["mesh"]["sha256"][:16]
            or manifest.get("sequence_id") != physical["sequence_id"]
            or manifest.get("scene_id") != physical["capture_group"]
            or manifest.get("background_id") != "procedural:gray"
            or manifest.get("background_kind") != "procedural"
            or manifest.get("native_resolution") != [640, 480]
            or manifest.get("frame_ids") != [0]
            or manifest.get("source_hashes") != {"object.glb": rendered["mesh"]["sha256"],
                "synthetic.mp4": rendered["video"]["sha256"]}
            or manifest.get("_timing", {}).get("clock_mode") != "capture_seconds_v1"
            or len(manifest.get("_timing", {}).get("scored_keys", ())) != 1
            or manifest["_timing"]["scored_keys"][0].timestamp_s != 0.
            or manifest["_timing"]["scored_keys"][0].frame_id != 0):
        raise ActivationError("observed manifest differs from frozen normalized TRAIN row")
    from . import quality_gotrack_inference as inference
    native = inference._validated_native_intrinsics(manifest, {})
    from .quality_adaptation_smoke import legacy_native_intrinsics
    effective = legacy_native_intrinsics(native)
    return bundle, manifest, effective


def _read_render_index(context) -> dict[str, Any]:
    doc, activation, rows = _frozen(context)
    path = Path(doc["output_root"]) / "render-index.json"
    index = _read_json(path, "shared normalized rendering index")
    if (index.get("schema") != RENDER_SCHEMA or index.get("request_sha256") != context.sha256
            or index.get("activation_plan_sha256") != sha256_file(Path(doc["output_root"]) / "activation-plan.json")
            or index.get("test_payload_reads") != 0 or index.get("annotation_inputs") != []
            or len(index.get("rows", [])) != len(rows)):
        raise ActivationError("shared rendering index differs from frozen two-view activation")
    for actual, planned in zip(index["rows"], rows):
        if actual.get("physical") != planned:
            raise ActivationError("shared rendering row identity/order changed")
        if actual.get("status") == "available":
            _manifest_for(context, planned, actual)
        elif actual.get("status") != "unavailable" or not actual.get("failure_reason"):
            raise ActivationError("invalid shared rendering availability/failure")
    return index


def render(context: ActivationContext) -> dict[str, Any]:
    """Render the two already-frozen TRAIN views once for both real branches."""
    _require_stage(context, "render")
    doc, activation, rows = _frozen(context)
    from .quality_ranker_synthetic import _render_bundle
    root = Path(doc["output_root"])
    split = _read_json(root / "split.json", "full prospective split metadata")
    groups = {group["sequence_id"]: group for group in split["groups"]}
    output_rows = []
    for physical in rows:
        entry = {"physical": physical, "status": "pending", "failure_reason": None}
        try:
            resource_check(context, 576 * 1024**2)
            mesh = verify_file_pin(physical["mesh_recipe"], maximum=64 * 1024**2)
            if mesh.stat().st_size != physical["mesh_recipe"]["bytes"]:
                raise ActivationError("selected metric mesh changed before rendering")
            if physical["sequence_id"] not in groups:
                raise ActivationError("selected sequence is absent from full split metadata")
            view_root = root / "shared" / "views" / f"object-{physical['object_id']}-view-{physical['view_index']:02d}"
            _render_bundle(mesh, view_root, groups[physical["sequence_id"]], frames=1,
                width=640, height=480, fps=10., mesh_units="metres",
                observationally_ambiguous=True, cloud_execution=True,
                split_manifest_path=root / "split.json", view_mode="normalized24-v1",
                view_index=physical["view_index"])
            bundle = view_root / "bundle"
            pins = {"input": file_pin(bundle / "input.json"),
                    "video": file_pin(bundle / "synthetic.mp4"),
                    "mesh": file_pin(bundle / "object.glb")}
            pins["input_manifest_sha256"] = pins["input"]["sha256"]
            _manifest_for(context, physical, pins)
            entry.update(status="available", **pins)
        except BaseException as exc:
            entry.update(status="unavailable", failure_reason=_failure(exc))
            output_rows.append(entry)
            _atomic_write_json(root / "render-index.json", {"schema": RENDER_SCHEMA,
                "request_sha256": context.sha256,
                "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
                "rows": output_rows, "test_payload_reads": 0, "annotation_inputs": []}, replace=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit, ActivationError)):
                raise
            continue
        output_rows.append(entry)
        _atomic_write_json(root / "render-index.json", {"schema": RENDER_SCHEMA,
            "request_sha256": context.sha256,
            "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
            "rows": output_rows + [dict(physical=row, status="pending", failure_reason=None)
                for row in rows[len(output_rows):]],
            "test_payload_reads": 0, "annotation_inputs": []}, replace=True)
    result = {"schema": RENDER_SCHEMA, "request_sha256": context.sha256,
        "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
        "rows": output_rows, "test_payload_reads": 0, "annotation_inputs": []}
    _atomic_write_json(root / "render-index.json", result, replace=True)
    return {"rendered": sum(row["status"] == "available" for row in output_rows),
            "requested": len(rows), "test_payload_reads": 0}


def _verify_cnos_bank(path: Path, expected_asset: str) -> None:
    import numpy as np
    with np.load(path, allow_pickle=False) as bank:
        if set(bank.files) != {"descriptors", "asset_sha256", "unlit", "grayscale"}:
            raise ActivationError("CNOS bank has missing/foreign fields")
        descriptors = bank["descriptors"]
        if (descriptors.ndim != 2 or descriptors.shape[0] != 57
                or not np.isfinite(descriptors).all()
                or str(bank["asset_sha256"]) != expected_asset
                or bool(bank["unlit"]) is not True or bool(bank["grayscale"]) is not False):
            raise ActivationError("CNOS bank differs from the exact shared 57-view object bank")


def banks(context: ActivationContext) -> dict[str, Any]:
    """Build one shared authentic CNOS/FoundPose bank from the frozen obj1 mesh."""
    _require_stage(context, "banks")
    doc, activation, rows = _frozen(context)
    render_index = _read_render_index(context)
    root = Path(doc["output_root"])
    available = [entry for entry in render_index["rows"] if entry["status"] == "available"]
    result: dict[str, Any] = {"schema": BANK_SCHEMA, "request_sha256": context.sha256,
        "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
        "status": "unavailable", "failure_reason": None, "cnos_bank": None,
        "pose_bank": None, "asset_sha256": None, "template_count": None,
        "test_payload_reads": 0, "annotation_inputs": []}
    if not available:
        result["failure_reason"] = "no_shared_normalized_render_available"
        _atomic_write_json(root / "banks.json", result)
        return result
    from . import quality_adaptation_larger as larger
    from .quality_cnos import build_descriptors
    from .quality_foundpose import build_bank
    from .quality_gotrack import TexturedRenderer, load_network
    torch = network = renderer = state = None
    errors = []
    cnos_path = root / "shared" / "banks" / "cnos-000001.npz"
    pose_path = root / "shared" / "banks" / "pose-000001.pt"
    try:
        resource_check(context, 6 * GIB)
        if cnos_path.exists() or pose_path.exists() or pose_path.with_suffix(".part").exists():
            raise FileExistsError("shared bank output already exists; preserve attempt")
        # Use this activation's strict manifest validator, without fabricating a
        # PortableContext or borrowing its parity authorization.
        bundle, manifest, _ = _manifest_for(context, rows[0], available[0])
        import random
        import numpy as np
        random.seed(0); np.random.seed(0)
        torch = larger.configure_torch("cuda", deterministic=True)
        torch.manual_seed(0)
        build_descriptors(bundle, "cuda", unlit=True, output=cnos_path)
        larger.cleanup_device("cuda", torch)
        _verify_cnos_bank(cnos_path, rows[0]["mesh_recipe"]["sha256"])
        result["cnos_bank"] = file_pin(cnos_path)
        _atomic_write_json(root / "shared" / "banks" / "cnos-000001.json",
            {"bank": result["cnos_bank"]})
        network = load_network("cuda").eval()
        renderer = TexturedRenderer(bundle / "object.glb", manifest["object_id"], unlit=True)
        state = build_bank(renderer, network.backbone, "cuda", pose_path)
        if state.get("template_count") != 798 or state.get("asset_sha256") != rows[0]["mesh_recipe"]["sha256"]:
            raise ActivationError("FoundPose bank differs from the standard 798-template object bank")
        result.update(status="available", pose_bank=file_pin(pose_path),
            asset_sha256=state["asset_sha256"], template_count=state["template_count"])
    except BaseException as exc:
        result.update(status="unavailable", failure_reason=_failure(exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit, ActivationError)):
            raise
    finally:
        state = None
        for action in (lambda: renderer.close() if renderer is not None else None,
                lambda: network.cpu() if network is not None else None,
                lambda: larger.cleanup_device("cuda", torch) if torch is not None else None):
            try:
                action()
            except BaseException as exc:
                errors.append(_failure(exc))
        if errors:
            result.update(status="unavailable", failure_reason="cleanup failure: " + errors[0])
            raise RuntimeError("shared bank cleanup failed: " + errors[0])
    _atomic_write_json(root / "banks.json", result)
    return result


def _read_banks(context: ActivationContext) -> dict[str, Any]:
    doc, _, rows = _frozen(context)
    root = Path(doc["output_root"])
    result = _read_json(root / "banks.json", "shared activation banks")
    if (result.get("schema") != BANK_SCHEMA or result.get("request_sha256") != context.sha256
            or result.get("activation_plan_sha256") != sha256_file(root / "activation-plan.json")
            or result.get("test_payload_reads") != 0 or result.get("annotation_inputs") != []):
        raise ActivationError("shared bank receipt differs from the frozen activation")
    if result.get("status") == "available":
        if (result.get("asset_sha256") != rows[0]["mesh_recipe"]["sha256"]
                or result.get("template_count") != 798):
            raise ActivationError("shared FoundPose bank identity/count changed")
        verify_file_pin(result["cnos_bank"], under=root, maximum=MAX_CACHE_BYTES)
        verify_file_pin(result["pose_bank"], under=root, maximum=MAX_CACHE_BYTES)
        _verify_cnos_bank(Path(result["cnos_bank"]["path"]), rows[0]["mesh_recipe"]["sha256"])
        sidecar = _read_json(root / "shared" / "banks" / "cnos-000001.json", "shared CNOS bank pin")
        if sidecar != {"bank": result["cnos_bank"]}:
            raise ActivationError("shared CNOS bank loader sidecar changed")
    elif result.get("status") != "unavailable" or not result.get("failure_reason"):
        raise ActivationError("shared bank receipt has invalid availability/failure")
    return result


def _activation_physical(physical, rendered, bank, mask):
    row = dict(physical)
    row["iterations"] = [0]
    if rendered.get("status") == "available":
        row.update(input=rendered["input"], video=rendered["video"], mesh=rendered["mesh"])
    else:
        row.update(input=None, video=None, mesh=None)
    row["bank"] = bank.get("pose_bank") if bank.get("status") == "available" else None
    row["mask"] = mask.get("mask") if mask.get("status") == "available" else None
    row["mask_index"] = mask.get("mask_index") if mask.get("status") == "available" else None
    row["mask_status"] = mask.get("status")
    row["mask_failure"] = mask.get("failure_reason")
    return row


def _mask_index_path(branch_root: Path, physical: dict[str, Any]) -> Path:
    return branch_root / f"view-{physical['view_index']:02d}" / "observed-masks" / "results.json"


def _write_observed_mask(context: ActivationContext, branch_root: Path, physical: dict[str, Any],
                         rendered: dict[str, Any], bank: dict[str, Any], variant: str,
                         selected, diagnostics: dict[str, Any], prompt_box: list[int],
                         proposal_sha256: str | None = None) -> dict[str, Any]:
    from .vision import cv2
    import numpy as np
    if selected.dtype.kind not in "bu" or selected.shape != (480, 640) or not selected.any():
        raise ValueError("actual SAM2 observed mask must be nonempty 640x480 pixels")
    resource_check(context, 2 * 1024**2 + MAX_METADATA_BYTES)
    directory = _mask_index_path(branch_root, physical).parent
    directory.mkdir(parents=True, exist_ok=False)
    mask_path = directory / "0.png"
    if mask_path.exists() or mask_path.is_symlink():
        raise FileExistsError("observed mask destination already exists")
    if not cv2.imwrite(str(mask_path), selected.astype(np.uint8) * 255):
        raise OSError("SAM2 mask PNG write failed")
    mask_pin = file_pin(mask_path)
    row = {"schema": "kaggle_activation_observed_mask_v1", "status": "available",
        "request_sha256": context.sha256, "physical_id": physical["id"], "variant": variant,
        "input": rendered["input"], "video": rendered["video"], "mesh": rendered["mesh"],
        "cnos_bank": bank["cnos_bank"], "pose_bank": bank["pose_bank"],
        "mask": mask_pin, "detector": diagnostics, "prompt_box_xyxy": prompt_box,
        "producer_source": MASK_PRODUCER,
        "producer_source_sha256": sha256_file(ROOT / MASK_PRODUCER),
        "proposal_intermediate_sha256": proposal_sha256,
        "automatic": True, "mask_origin": "observed_CNOS_box_then_SAM2_image_predictor",
        "test_payload_reads": 0, "annotation_inputs": []}
    index_pin = _atomic_write_json(_mask_index_path(branch_root, physical), row)
    return {"status": "available", "mask": mask_pin, "mask_index": index_pin,
        "detector": diagnostics, "prompt_box_xyxy": prompt_box,
        "failure_reason": None}


def _unavailable_mask(exc: BaseException, diagnostics=None, prompt_box=None) -> dict[str, Any]:
    return {"status": "unavailable", "failure_reason": _failure(exc),
        "mask": None, "mask_index": None, "detector": diagnostics,
        "prompt_box_xyxy": prompt_box}


def _legacy_masks(context, branch_root, rows, rendered_index, banks_receipt, torch):
    from . import quality_adaptation_larger as larger
    from .quality_cnos import detect
    from .quality_gotrack import upstream_path
    import numpy as np
    from .vision import cv2
    from .quality_adaptation_runtime import _source_imports
    _source_imports("sam2")
    upstream_path()
    by_id = {r["physical"]["id"]: r for r in rendered_index["rows"]}
    result_rows = []
    for physical in rows:
        rendered = by_id[physical["id"]]
        diagnostics = prompt_box = None
        try:
            if rendered.get("status") != "available" or banks_receipt.get("status") != "available":
                raise ActivationError("render_or_shared_bank_unavailable")
            bundle, manifest, _ = _manifest_for(context, physical, rendered)
            from .quality_runner import read_rgb
            rgb = read_rgb(bundle, manifest, 0)
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ActivationError("decoded normalized RGB differs from frozen 640x480 uint8 input")
            resource_check(context, 16 * 1024**2)
            initial, diagnostics = detect(bundle, rgb, device="cuda",
                bank_path=Path(banks_receipt["cnos_bank"]["path"]))
            larger.cleanup_device("cuda", torch)
            if initial is None or np.asarray(initial).shape != (480, 640) or not np.any(initial):
                raise ValueError("CNOS observed association unavailable; no SAM prompt was invented")
            yy, xx = np.nonzero(initial)
            prompt_box = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
            cache = Path(context.document()["model_cache"])
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.modeling.position_encoding import PositionEmbeddingRandom
            from .quality_sam2 import deterministic_position_grid
            original_grid = PositionEmbeddingRandom.forward
            predictor = model = None
            try:
                PositionEmbeddingRandom.forward = deterministic_position_grid
                model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml",
                    str(cache / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
                    device="cuda", apply_postprocessing=False)
                predictor = SAM2ImagePredictor(model)
                with torch.inference_mode():
                    predictor.set_image(rgb)
                    predictions, scores, _ = predictor.predict(
                        box=np.asarray(prompt_box, np.float32), multimask_output=True)
                selected = np.asarray(predictions[int(np.argmax(scores))], dtype=bool)
            finally:
                predictor = model = None
                PositionEmbeddingRandom.forward = original_grid
                larger.cleanup_device("cuda", torch)
            result_rows.append(_write_observed_mask(context, branch_root, physical, rendered,
                banks_receipt, "legacy-v1", selected, diagnostics, prompt_box))
        except BaseException as exc:
            result_rows.append(_unavailable_mask(exc, diagnostics, prompt_box))
            if isinstance(exc, (KeyboardInterrupt, SystemExit, ActivationError)):
                raise
    return result_rows


def _amortized_masks(context, branch_root, rows, rendered_index, banks_receipt, torch):
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_larger_optimized import NeuralMaskBackend, run_mask_phases
    from .quality_adaptation_data import sha256
    from .quality_runner import read_rgb
    from .vision import cv2
    import numpy as np
    doc = context.document()
    root = Path(doc["output_root"])
    shared_bank_root = root / "shared" / "banks"
    by_id = {entry["physical"]["id"]: entry for entry in rendered_index["rows"]}
    by_sequence = {row["sequence_id"]: row for row in rows}
    by_view = {row["sequence_id"]: row for row in rows}
    mask_rows: dict[str, dict[str, Any]] = {}
    usable = []
    for physical in rows:
        rendered = by_id[physical["id"]]
        if rendered.get("status") == "available" and banks_receipt.get("status") == "available":
            view = dict(physical)
            view["view_index"] = physical["view_index"]
            view["mesh"] = rendered["mesh"]
            usable.append(view)
        else:
            mask_rows[physical["id"]] = _unavailable_mask(
                ValueError("render_or_shared_bank_unavailable"))
    if not usable:
        return [mask_rows[row["id"]] for row in rows]
    for physical in rows:
        if physical["id"] in by_id and by_id[physical["id"]]["status"] == "available":
            by_sequence[physical["sequence_id"]] = physical

    def input_reader(view):
        physical = by_sequence[view["sequence_id"]]
        rendered = by_id[physical["id"]]
        bundle, manifest, _ = _manifest_for(context, physical, rendered)
        return rendered, bundle, manifest, read_rgb(bundle, manifest, 0)

    def mask_writer(view, selected, diagnostic):
        physical = by_sequence[view["sequence_id"]]
        rendered = by_id[physical["id"]]
        result = _write_observed_mask(context, branch_root, physical, rendered,
            banks_receipt, "amortized-v1", selected.astype(bool), diagnostic["detector"],
            diagnostic["prompt_box_xyxy"], diagnostic.get("proposals_sha256"))
        mask_rows[physical["id"]] = result
        return {"mask": result["mask"], "mask_index": result["mask_index"]}

    backend_doc = {"assets": [{"object_id": "1", "mesh": rows[0]["mesh_recipe"]}],
        "source_contract": {"sha256": sha256_file(root / "activation-plan.json")}}
    backend = NeuralMaskBackend(shared_bank_root, backend_doc, "1", "cuda", torch,
        Path(doc["model_cache"]),
        pin_resolver=lambda value: verify_file_pin(value, under=root),
        input_reader=input_reader, mask_writer=mask_writer)
    scratch = Path(doc["scratch_root"])
    scratch.mkdir(parents=True, exist_ok=True)

    def persist(results, summary):
        for result in results:
            physical = by_sequence[result["view"]["sequence_id"]]
            if result["status"] != "available" and physical["id"] not in mask_rows:
                mask_rows[physical["id"]] = _unavailable_mask(
                    ValueError(result.get("failure_reason") or "amortized mask unavailable"),
                    result.get("detector"), result.get("prompt_box_xyxy"))
        _atomic_write_json(branch_root / "mask-phases.json", summary)

    run_mask_phases(usable, scratch / "amortized-proposals",
        backend, reserve=lambda amount: resource_check(context, amount + MAX_METADATA_BYTES),
        persist=persist)
    for physical in rows:
        mask_rows.setdefault(physical["id"], _unavailable_mask(ValueError("unrun_after_termination")))
    return [mask_rows[row["id"]] for row in rows]


@dataclass(frozen=True, slots=True)
class _ActivationWitness:
    issuer: object
    request_sha256: str
    plan_sha256: str
    variant: str
    branch_root: str
    capture_root: str
    sink: object
    resource_callback: object
    physical_sha256: str
    native_k: tuple[tuple[float, ...], ...]
    row_identities: tuple[tuple[Any, ...], ...]
    input_bindings_sha256: str


def _observer_plan_bytes(context, variant, branch_rows) -> bytes:
    doc, activation, _ = _frozen(context)
    root = Path(doc["output_root"])
    branch_root = root / "branches" / variant
    value = {"schema": OBSERVER_PLAN_SCHEMA, "request_sha256": context.sha256,
        "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
        "source_files_sha256": hashlib.sha256(canonical_bytes(doc["source_files"])).hexdigest(),
        "variant": variant, "branch_root": branch_root.as_posix(),
        "capture_root": (branch_root / "capture").as_posix(), "rows": branch_rows,
        "test_payload_reads": 0, "annotation_inputs": []}
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_METADATA_BYTES:
        raise ActivationError("branch observer plan exceeds metadata limit")
    return raw


def _row_identity(row) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in ("id", "object_id", "split", "sequence_id",
        "capture_group", "frame_id", "timestamp_s", "iteration"))


def observer_preflight(raw_plan: bytes, expected_sha256: str, sink,
                       *, context: ActivationContext) -> _ActivationWitness:
    """Authenticate complete branch metadata and every selected input before decode."""
    if type(context) is not ActivationContext:
        raise TypeError("exact activation request context required")
    if type(raw_plan) is not bytes or hashlib.sha256(raw_plan).hexdigest() != _digest(expected_sha256):
        raise ValueError("activation branch plan bytes changed")
    doc, activation, expected_rows = _frozen(context)
    verify_sources(context, upstream=True)
    plan = strict_json_bytes(raw_plan, "activation observer plan")
    keys = {"schema", "request_sha256", "activation_plan_sha256", "source_files_sha256",
        "variant", "branch_root", "capture_root", "rows", "test_payload_reads", "annotation_inputs"}
    _keys(plan, keys, "activation observer plan")
    if plan["schema"] != OBSERVER_PLAN_SCHEMA or plan["request_sha256"] != context.sha256:
        raise ValueError("activation observer plan schema/request binding differs")
    if plan["variant"] not in BRANCHES:
        raise ValueError("activation observer plan has an unknown branch")
    root = Path(doc["output_root"])
    branch_root = root / "branches" / plan["variant"]
    capture_root = branch_root / "capture"
    if (plan["activation_plan_sha256"] != sha256_file(root / "activation-plan.json")
            or plan["source_files_sha256"] != hashlib.sha256(canonical_bytes(doc["source_files"])).hexdigest()
            or plan["branch_root"] != branch_root.as_posix()
            or plan["capture_root"] != capture_root.as_posix()
            or plan["test_payload_reads"] != 0 or plan["annotation_inputs"] != []
            or not isinstance(plan["rows"], list) or len(plan["rows"]) != 2):
        raise ValueError("activation branch plan/root/source/inventory binding differs")
    plan_path = branch_root / "capture-plan.json"
    if (plan_path.is_symlink()
            or read_metadata(plan_path, "activation capture plan")[0] != raw_plan):
        raise ValueError("activation branch plan file differs from the issued bytes")
    # Reject every row identity and the entire sink inventory before opening
    # any video, image, mesh, mask or bank payload (including earlier valid rows).
    early_fields = {"physical", "input", "video", "mesh", "bank", "mask", "mask_index",
                    "mask_status", "mask_failure"}
    for ordinal, (row, physical) in enumerate(zip(plan["rows"], expected_rows)):
        _keys(row, early_fields, "activation branch physical row")
        if (row["physical"] != physical
                or (physical["object_id"], physical["view_index"]) != SELECTED_ROWS[ordinal]):
            raise ValueError("activation branch row identity/order differs from the full frozen recipe")
        if row["mask_status"] not in ("available", "unavailable"):
            raise ValueError("activation branch mask state is not terminal")
    if type(sink) is not __import__("bench.quality_adaptation_refinement_states",
            fromlist=["RefStateCapture"]).RefStateCapture:
        raise TypeError("exact RefStateCapture required for activation observer")
    from . import quality_adaptation_refinement_states as shared
    if len(sink.rows) != 2:
        raise ValueError("activation sink must preserve both frozen refinement states")
    expected_states = shared.state_rows({"rows": [dict(row["physical"], iterations=[0]) for row in plan["rows"]]})
    expected_ids = tuple(_row_identity(state) for state in expected_states)
    actual_ids = tuple(_row_identity(state) for state in sink.rows)
    if actual_ids != expected_ids:
        raise ValueError("activation sink identity/order differs, including the final frozen row")
    # Sink binds one selected physical row while the plan authenticates both.
    sink_physical = sink.physical
    matching = [entry for entry in plan["rows"] if entry["physical"]["id"] == sink_physical.get("id")]
    if len(matching) != 1:
        raise ValueError("activation sink physical row is absent or duplicated")
    expected_sink_physical = _activation_physical(matching[0]["physical"],
        {"status": "available" if matching[0]["input"] is not None else "unavailable",
         "input": matching[0]["input"], "video": matching[0]["video"],
         "mesh": matching[0]["mesh"]},
        {"status": "available" if matching[0]["bank"] is not None else "unavailable",
         "pose_bank": matching[0]["bank"]},
        {"status": matching[0]["mask_status"], "mask": matching[0]["mask"],
         "mask_index": matching[0]["mask_index"], "failure_reason": matching[0]["mask_failure"]})
    if canonical_bytes(sink_physical) != canonical_bytes(expected_sink_physical):
        raise ValueError("activation sink physical input/mask/bank binding changed")
    if (Path(sink.output).resolve() != capture_root.resolve()
            or sink.max_output_bytes != doc["limits"]["storage_bytes"]
            or sink.reserve_bytes != doc["limits"]["reserve_bytes"]
            or not callable(sink.resource_check)):
        raise ValueError("activation capture sink output or bounded storage binding changed")
    render_index = _read_render_index(context)
    bank_doc = _read_banks(context)
    render_by_id = {entry["physical"]["id"]: entry for entry in render_index["rows"]}
    row_fields = {"physical", "input", "video", "mesh", "bank", "mask", "mask_index",
        "mask_status", "mask_failure"}
    validated = []
    for ordinal, (row, physical) in enumerate(zip(plan["rows"], expected_rows)):
        _keys(row, row_fields, "activation branch physical row")
        if row["physical"] != physical or (physical["object_id"], physical["view_index"]) != SELECTED_ROWS[ordinal]:
            raise ValueError("activation branch row identity/order differs from the full frozen recipe")
        rendered = render_by_id[physical["id"]]
        expected_pins = {name: rendered[name] if rendered["status"] == "available" else None
                         for name in ("input", "video", "mesh")}
        expected_bank = bank_doc["pose_bank"] if bank_doc["status"] == "available" else None
        if any(row[name] != expected_pins[name] for name in ("input", "video", "mesh")) or row["bank"] != expected_bank:
            raise ValueError("activation branch common-input/bank pins changed")
        if row["mask_status"] not in ("available", "unavailable"):
            raise ValueError("activation branch mask state is not terminal")
        if rendered["status"] == "available":
            bundle, manifest, native_k = _manifest_for(context, physical, rendered)
            if Path(rendered["input"]["path"]).parent != bundle:
                raise ValueError("activation branch input escaped its normalized bundle")
        else:
            manifest, native_k = None, None
        if row["mask_status"] == "available":
            if bank_doc["status"] != "available" or rendered["status"] != "available":
                raise ValueError("available mask lacks an authenticated common input or bank")
            for key in ("mask", "mask_index"):
                verify_file_pin(row[key], under=branch_root, maximum=MAX_METADATA_BYTES if key == "mask_index" else 16 * 1024**2)
            mask_path = _mask_index_path(branch_root, physical).with_name("0.png")
            if Path(row["mask"]["path"]) != mask_path or Path(row["mask_index"]["path"]) != _mask_index_path(branch_root, physical):
                raise ValueError("activation observed mask escaped its branch/view directory")
            mask_record = _read_json(Path(row["mask_index"]["path"]), "activation observed mask receipt")
            mask_fields = {"schema", "status", "request_sha256", "physical_id", "variant",
                "input", "video", "mesh", "cnos_bank", "pose_bank", "mask", "detector",
                "prompt_box_xyxy", "producer_source", "producer_source_sha256",
                "proposal_intermediate_sha256", "automatic", "mask_origin",
                "test_payload_reads", "annotation_inputs"}
            if set(mask_record) != mask_fields:
                raise ValueError("activation observed mask receipt has unknown or missing fields")
            expected_mask = {"schema": "kaggle_activation_observed_mask_v1", "status": "available",
                "request_sha256": context.sha256, "physical_id": physical["id"],
                "variant": plan["variant"], "input": rendered["input"], "video": rendered["video"],
                "mesh": rendered["mesh"], "cnos_bank": bank_doc["cnos_bank"],
                "pose_bank": bank_doc["pose_bank"], "mask": row["mask"],
                "producer_source": MASK_PRODUCER,
                "producer_source_sha256": doc["source_files"][MASK_PRODUCER],
                "automatic": True, "mask_origin": "observed_CNOS_box_then_SAM2_image_predictor",
                "test_payload_reads": 0, "annotation_inputs": []}
            if any(mask_record.get(key) != value for key, value in expected_mask.items()):
                raise ValueError("activation observed mask producer binding differs")
            if (not isinstance(mask_record.get("detector"), dict)
                    or not isinstance(mask_record.get("prompt_box_xyxy"), list)
                    or len(mask_record["prompt_box_xyxy"]) != 4
                    or any(type(value) is not int for value in mask_record["prompt_box_xyxy"])):
                raise ValueError("activation mask lacks actual CNOS detector/box evidence")
            if (mask_record["detector"].get("bank_sha256") != bank_doc["cnos_bank"]["sha256"]
                    or not (0 <= mask_record["prompt_box_xyxy"][0] <= mask_record["prompt_box_xyxy"][2] < 640
                    and 0 <= mask_record["prompt_box_xyxy"][1] <= mask_record["prompt_box_xyxy"][3] < 480)):
                raise ValueError("activation observed mask detector/bounding-box evidence is unbound")
            if row["mask_failure"] is not None:
                raise ValueError("available activation mask carries a failure")
        elif row["mask"] is not None or row["mask_index"] is not None or not row["mask_failure"]:
            raise ValueError("unavailable activation mask must retain a failure without a fake artifact")
        validated.append({"row": row, "manifest": manifest, "native_k": native_k,
            "mask_record": mask_record if row["mask_status"] == "available" else None})
    native = next(entry["native_k"] for entry in validated
        if entry["row"]["physical"]["id"] == sink_physical["id"])
    if native is None:
        raise ValueError("activation sink has no authenticated native geometry")
    import numpy as np
    if not np.array_equal(np.asarray(sink.native_k), native):
        raise ValueError("activation sink native calibration differs from its authenticated view")
    from .quality_adaptation_data import canonical_hash
    return _ActivationWitness(_ACTIVATION_ISSUER, context.sha256, expected_sha256,
        plan["variant"], branch_root.as_posix(), capture_root.as_posix(), sink,
        sink.resource_check, canonical_hash(expected_sink_physical),
        tuple(tuple(float(value) for value in line) for line in native.tolist()),
        expected_ids, hashlib.sha256(canonical_bytes(plan["rows"])).hexdigest())


def issue_observer(raw_plan: bytes, expected_sha256: str, sink,
                   *, context: ActivationContext):
    """Issue only after strict raw-plan, full inventory, branch and sink preflight."""
    from . import quality_adaptation_larger as larger
    witness = observer_preflight(raw_plan, expected_sha256, sink, context=context)
    cap = larger.LargerObserverCapability(_ACTIVATION_ISSUER, raw_plan, expected_sha256,
        sink, POLICY, context, witness)
    cap.validate(sink, larger.inference.PROFILE)
    return cap


def validate_observer(capability, sink, geometry):
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_refinement_states import RefStateCapture
    if (type(capability) is not larger.LargerObserverCapability
            or capability._issuer is not _ACTIVATION_ISSUER or capability._policy != POLICY
            or type(capability._context) is not ActivationContext
            or type(capability._witness) is not _ActivationWitness
            or capability._witness.issuer is not _ACTIVATION_ISSUER
            or type(sink) is not RefStateCapture or sink is not capability._sink
            or capability._witness.sink is not sink
            or geometry != larger.inference.PROFILE
            or type(capability._raw) is not bytes
            or hashlib.sha256(capability._raw).hexdigest() != capability._sha256
            or capability._witness.plan_sha256 != capability._sha256
            or capability._witness.request_sha256 != capability._context.sha256):
        raise ValueError("invalid Kaggle activation observer capability/sink/profile")
    current = observer_preflight(capability._raw, capability._sha256, sink,
        context=capability._context)
    witness = capability._witness
    if (current.variant != witness.variant or current.branch_root != witness.branch_root
            or current.capture_root != witness.capture_root
            or current.resource_callback is not witness.resource_callback
            or witness.resource_callback is not sink.resource_check
            or current.physical_sha256 != witness.physical_sha256
            or current.native_k != witness.native_k
            or current.row_identities != witness.row_identities
            or current.input_bindings_sha256 != witness.input_bindings_sha256):
        raise ValueError("Kaggle activation observer witness changed after issuance")


def _detector_record(recorded, observed, bank_sha256):
    from .quality_adaptation_larger_parity import _detector_evidence
    if recorded is None and observed is None:
        return None
    detector = _detector_evidence(recorded, observed)
    if not isinstance(detector, dict) or not {"scores", "selected_proposal", "reason", "bank_sha256"} <= set(detector):
        raise ActivationError("actual observed association lacks scores, selected proposal, or failure reason")
    if detector["bank_sha256"] != bank_sha256:
        raise ActivationError("observed association is not bound to the shared CNOS bank")
    import numpy as np
    scores = np.asarray(detector["scores"])
    if scores.ndim != 1 or scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
        raise ActivationError("actual detector scores are malformed or nonfinite")
    selected = detector["selected_proposal"]
    if selected is not None and (type(selected) is not int or not 0 <= selected < len(scores)):
        raise ActivationError("actual detector selected-proposal index is out of range")
    if detector["reason"] is not None and not isinstance(detector["reason"], str):
        raise ActivationError("actual detector reason must be a string or null")
    return detector


def _reconstruct_captured_crops(crops, geometry):
    from .quality_adaptation_prepare import crops_from_retained_capture
    return crops_from_retained_capture(dict(
        query_rgb_crop=crops["query_rgb"].transpose(2, 0, 1)[None],
        template_rgb=crops["template_rgb"].transpose(2, 0, 1)[None],
        rendered_depth_mm=geometry["rendered_depth_mm"],
        rendered_mask=geometry["rendered_mask"],
        crop_k=geometry["crop_k"],
        current_crop_camera_from_object_mm=geometry["current_crop_camera_from_object_mm"]),
        preserve_float_rgb=True)


def _collect_snapshot(context: ActivationContext, variant: str, *, hardware, fatal_error):
    """Reopen and authenticate the real branch artifacts into the frozen parity schema."""
    import numpy as np
    if variant not in BRANCHES or not isinstance(hardware, dict) or hardware.get("device") != "cuda":
        raise ActivationError("a physical CUDA identity is required for an activation snapshot")
    doc, activation, rows = _frozen(context)
    root = Path(doc["output_root"])
    branch_root = root / "branches" / variant
    if branch_root.is_symlink() or not branch_root.is_dir():
        raise ActivationError("branch artifact root is missing or aliased")
    verify_sources(context, upstream=True)
    render_index = _read_render_index(context)
    banks_receipt = _read_banks(context)
    mask_path = branch_root / "mask-index.json"
    mask_doc = _read_json(mask_path, "activation branch mask index")
    if (set(mask_doc) != {"schema", "variant", "request_sha256", "rows", "test_payload_reads", "annotation_inputs"}
            or mask_doc["schema"] != BRANCH_SCHEMA or mask_doc["variant"] != variant
            or mask_doc["request_sha256"] != context.sha256 or mask_doc["test_payload_reads"] != 0
            or mask_doc["annotation_inputs"] != [] or not isinstance(mask_doc["rows"], list)
            or len(mask_doc["rows"]) != 2):
        raise ActivationError("branch mask inventory differs from exact two-view activation")
    mask_rows = []
    for physical, entry in zip(rows, mask_doc["rows"]):
        if not isinstance(entry, dict) or entry.get("physical_id") != physical["id"]:
            raise ActivationError("branch mask row identity/order differs")
        value = {key: child for key, child in entry.items() if key != "physical_id"}
        if set(value) != {"status", "failure_reason", "mask", "mask_index", "detector", "prompt_box_xyxy"}:
            raise ActivationError("branch mask row has unknown or missing evidence fields")
        if value["status"] == "available":
            if value["failure_reason"] is not None:
                raise ActivationError("available branch mask carries a failure")
            verify_file_pin(value["mask"], under=branch_root, maximum=16 * 1024**2)
            verify_file_pin(value["mask_index"], under=branch_root, maximum=MAX_METADATA_BYTES)
            mask_record = _read_json(Path(value["mask_index"]["path"]), "activation observed mask record")
            expected_source = MASK_PRODUCER
            expected_fields = {"schema", "status", "request_sha256", "physical_id", "variant",
                "input", "video", "mesh", "cnos_bank", "pose_bank", "mask", "detector",
                "prompt_box_xyxy", "producer_source", "producer_source_sha256",
                "proposal_intermediate_sha256", "automatic", "mask_origin",
                "test_payload_reads", "annotation_inputs"}
            if (set(mask_record) != expected_fields or mask_record.get("schema") != "kaggle_activation_observed_mask_v1"
                    or mask_record.get("status") != "available" or mask_record.get("request_sha256") != context.sha256
                    or mask_record.get("physical_id") != physical["id"] or mask_record.get("variant") != variant
                    or mask_record.get("mask") != value["mask"] or mask_record.get("producer_source") != expected_source
                    or mask_record.get("producer_source_sha256") != doc["source_files"].get(expected_source)
                    or mask_record.get("automatic") is not True
                    or mask_record.get("mask_origin") != "observed_CNOS_box_then_SAM2_image_predictor"
                    or mask_record.get("test_payload_reads") != 0 or mask_record.get("annotation_inputs") != []):
                raise ActivationError("observed mask producer/source/request binding changed")
            proposal = mask_record["proposal_intermediate_sha256"]
            if variant == "legacy-v1":
                if proposal is not None:
                    raise ActivationError("legacy branch cannot claim an amortized proposal intermediate")
            else:
                _digest(proposal, "proposal intermediate sha256")
            if (mask_record["input"] != render_index["rows"][len(mask_rows)]["input"]
                    or mask_record["video"] != render_index["rows"][len(mask_rows)]["video"]
                    or mask_record["mesh"] != render_index["rows"][len(mask_rows)]["mesh"]
                    or mask_record["cnos_bank"] != banks_receipt["cnos_bank"]
                    or mask_record["pose_bank"] != banks_receipt["pose_bank"]):
                raise ActivationError("observed mask is detached from its actual shared inference inputs")
            detector = mask_record["detector"]
            if (value["detector"] != detector or value["prompt_box_xyxy"] != mask_record["prompt_box_xyxy"]
                    or not isinstance(detector, dict) or not isinstance(value["prompt_box_xyxy"], list)
                    or len(value["prompt_box_xyxy"]) != 4
                    or any(type(number) is not int for number in value["prompt_box_xyxy"])):
                raise ActivationError("branch mask index differs from authentic detector/prompt record")
            x0, y0, x1, y1 = value["prompt_box_xyxy"]
            if not (0 <= x0 <= x1 < 640 and 0 <= y0 <= y1 < 480):
                raise ActivationError("observed mask prompt box is outside the normalized frame")
            if detector.get("bank_sha256") != banks_receipt["cnos_bank"]["sha256"]:
                raise ActivationError("observed mask detector differs from the authenticated CNOS bank")
            from .vision import cv2
            image = cv2.imread(str(value["mask"]["path"]), cv2.IMREAD_GRAYSCALE)
            if (image is None or image.shape != (480, 640) or image.dtype.name != "uint8"
                    or not image.any() or not set(map(int, np.unique(image))).issubset({0, 255})):
                raise ActivationError("observed SAM mask PNG is empty, nonbinary, or wrong-sized")
        elif value["status"] != "unavailable" or not isinstance(value["failure_reason"], str) or not value["failure_reason"]:
            raise ActivationError("unavailable branch mask must retain its failure without artifacts")
        elif value["mask"] is not None or value["mask_index"] is not None:
            raise ActivationError("unavailable branch mask cannot substitute files")
        mask_rows.append(value)
    branch_rows = _branch_entries(context, variant, rows, render_index, banks_receipt, mask_rows)
    raw_plan = _observer_plan_bytes(context, variant, branch_rows)
    plan_path = branch_root / "capture-plan.json"
    if plan_path.is_symlink() or read_metadata(plan_path, "activation capture plan")[0] != raw_plan:
        raise ActivationError("branch capture plan differs from authenticated physical mask/input inventory")
    capture_root = branch_root / "capture"
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise ActivationError("branch capture directory is missing or aliased")
    capture_index = _read_json(capture_root / "capture-index.json", "activation capture index")
    if (set(capture_index) != {"rows", "test_payload_reads", "training_ready"}
            or capture_index["test_payload_reads"] != 0 or capture_index["training_ready"] is not False
            or len(capture_index["rows"]) != 2):
        raise ActivationError("activation capture index is incomplete or has foreign fields")
    expected_states = _physical_state_rows(branch_rows)
    from .quality_adaptation_refinement_states import _base
    for actual, expected in zip(capture_index["rows"], expected_states):
        if _row_identity(actual) != _row_identity(expected) or _base(actual) != _base(expected):
            raise ActivationError("capture state identity/order differs from both frozen TRAIN views")
        if actual.get("status") not in ("available", "unavailable"):
            raise ActivationError("capture state is not terminal")
        expected_keys = set(expected)
        if actual["status"] == "unavailable":
            if (set(actual) != expected_keys or not actual.get("failure_reason")
                    or actual["query_rgb_sha256"] is not None):
                raise ActivationError("unavailable capture state lost failure or contains substituted artifacts")
        else:
            allowed = (expected_keys - {"failure_reason", "unavailable_provenance"}) | {
                "crops", "capture", "sampling_map", "seed_selection", "pnp_state",
                "native_calibration", "pnp_failure"}
            required = (expected_keys - {"failure_reason", "unavailable_provenance"}) | {
                "crops", "capture", "sampling_map", "seed_selection", "pnp_state",
                "native_calibration"}
            if not required <= set(actual) or not set(actual) <= allowed:
                raise ActivationError("available capture state has unknown or missing retained evidence")
    observations = _read_json(branch_root / "observations.json", "activation runtime observations")
    if (set(observations) != {"detections", "candidates", "test_payload_reads", "annotation_inputs"}
            or observations["test_payload_reads"] != 0 or observations["annotation_inputs"] != []):
        raise ActivationError("runtime observer contains TEST/annotation or foreign fields")
    detections, candidate_rows = {}, {}
    for item in observations["detections"]:
        if (not isinstance(item, dict) or set(item) != {"input_sha256", "detector"}
                or item["input_sha256"] in detections):
            raise ActivationError("runtime observer has duplicate/malformed detection identities")
        detections[item["input_sha256"]] = item["detector"]
    for item in observations["candidates"]:
        if (not isinstance(item, dict) or set(item) != {"rgb_sha256", "poses"}
                or item["rgb_sha256"] in candidate_rows):
            raise ActivationError("runtime observer has duplicate/malformed recovery identities")
        candidate_rows[item["rgb_sha256"]] = item["poses"]
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_larger_parity import SCHEMA as PARITY_SCHEMA, _pixel_pin
    from .quality_adaptation_data import descriptor, CROP_KEYS, load_npz, validate_crops
    from .quality_adaptation_expanded import validate_retained_geometry
    from .quality_adaptation_larger import GEOMETRY_KEYS
    from .quality_runner import read_rgb
    from .vision import cv2
    snapshot_rows = []
    for ordinal, (branch_row, state, render_entry, mask_entry) in enumerate(
            zip(branch_rows, capture_index["rows"], render_index["rows"], mask_rows)):
        physical = branch_row["physical"]
        input_pins = {}
        for name in ("input", "video", "mesh", "bank"):
            pin = branch_row[name]
            if pin is None:
                input_pins[name] = None
            else:
                verify_file_pin(pin, under=root, maximum=MAX_CACHE_BYTES if name == "bank" else 64 * 1024**2)
                input_pins[name] = {"sha256": pin["sha256"], "bytes": pin["bytes"]}
        mask_value = None
        mask_pin = branch_row["mask"]
        if mask_pin is not None:
            verified_mask = verify_file_pin(mask_pin, under=branch_root, maximum=16 * 1024**2)
            image = cv2.imread(str(verified_mask), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (480, 640) or image.dtype.name != "uint8" or not image.any():
                raise ActivationError("activation mask could not be independently decoded")
            mask_value = _pixel_pin(image)
        observed_detector = detections.get(branch_row["input"]["sha256"]) if branch_row["input"] else None
        detector = _detector_record(mask_entry["detector"], observed_detector,
            banks_receipt["cnos_bank"]["sha256"] if banks_receipt["status"] == "available" else "")
        prompt_box = mask_entry["prompt_box_xyxy"]
        poses = None
        native_k = None
        if render_entry["status"] == "available":
            bundle, manifest, native_k = _manifest_for(context, physical, render_entry)
            rgb = read_rgb(bundle, manifest, 0)
            if rgb.shape != (480, 640, 3) or rgb.dtype.name != "uint8":
                raise ActivationError("authenticated normalized input RGB has wrong shape/dtype")
            rgb_sha = _pixel_pin(rgb)["sha256"]
            poses = candidate_rows.get(rgb_sha)
        if poses not in (None, []):
            candidate_array = np.asarray(poses)
            if (candidate_array.ndim != 3 or candidate_array.shape[1:] != (4, 4)
                    or candidate_array.shape[0] > 5 or candidate_array.dtype.kind not in "fiu"
                    or not np.isfinite(candidate_array).all()):
                raise ActivationError("observed FoundPose candidates are malformed/nonfinite")
        crops = geometry = None
        failure_reason = state.get("failure_reason")
        if state["status"] == "available":
            if failure_reason is not None or poses is None or mask_value is None or detector is None:
                raise ActivationError("available refinement state lacks authentic mask/detector/candidate evidence")
            crop_path = descriptor(capture_root, state["crops"], verify=True)
            geometry_path = descriptor(capture_root, state["capture"], verify=True)
            for path in (crop_path, geometry_path):
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise ActivationError("captured array artifact is not a regular nonsymlink file")
            crop_arrays = load_npz(crop_path, CROP_KEYS)
            validate_crops(crop_arrays)
            geometry_arrays = load_npz(geometry_path, GEOMETRY_KEYS)
            validate_retained_geometry(geometry_arrays["crop_from_orig"])
            if (not np.array_equal(crop_arrays["query_K"], geometry_arrays["crop_k"])
                    or not np.array_equal(geometry_arrays["native_k"], state["native_calibration"])
                    or native_k is None or not np.array_equal(native_k, state["native_calibration"])):
                raise ActivationError("captured crop/native geometry differs from its retained state")
            from .quality_adaptation_data import crop_rgb_hash
            if state["query_rgb_sha256"] != crop_rgb_hash(crop_arrays["query_rgb"]):
                raise ActivationError("captured query crop digest differs from its state receipt")
            reconstructed = _reconstruct_captured_crops(crop_arrays, geometry_arrays)
            if any(not np.array_equal(reconstructed[key], crop_arrays[key]) for key in CROP_KEYS):
                raise ActivationError("captured crops do not reconstruct from retained geometry")
            crops = {"pin": {"path": crop_path.as_posix(), "bytes": crop_path.stat().st_size,
                    "sha256": sha256_file(crop_path)}, "keys": sorted(CROP_KEYS)}
            geometry = {"pin": {"path": geometry_path.as_posix(), "bytes": geometry_path.stat().st_size,
                    "sha256": sha256_file(geometry_path)}, "keys": sorted(GEOMETRY_KEYS)}
        snapshot_rows.append({"id": state["id"], "input_pins": input_pins,
            "status": state["status"], "failure_reason": failure_reason,
            "mask": mask_value, "detector": detector, "prompt_box_xyxy": prompt_box,
            "candidates": poses, "crops": crops, "geometry": geometry})
    return {"schema": PARITY_SCHEMA, "device": "cuda", "rows": snapshot_rows,
        "requested_ids": [row["id"] for row in snapshot_rows], "fatal_error": fatal_error,
        "physical_frames_requested": 2, "refinement_states_requested": 2,
        "test_payload_reads": 0, "annotation_inputs": []}


def snapshot_activation(context: ActivationContext, variant: str) -> dict[str, Any]:
    """Reopen a branch snapshot and bind it to its real branch/stage receipts."""
    if variant not in BRANCHES:
        raise ValueError("activation snapshot branch must be exact")
    doc, activation, _ = _frozen(context)
    root = Path(doc["output_root"])
    branch_root = root / "branches" / variant
    receipt_path = branch_root / "branch-receipt.json"
    receipt = _read_json(receipt_path, "activation branch receipt")
    expected_fields = {"schema", "status", "request_sha256", "variant", "stage_attempt_id",
        "activation_plan_sha256", "branch_root", "capture_plan", "mask_index", "observations",
        "capture_index", "snapshot", "hardware", "fatal_error", "cleanup_errors", "cleanup_success",
        "requested_physical_frames", "requested_refinement_states", "elapsed_seconds",
        "test_payload_reads", "annotation_inputs"}
    if (set(receipt) != expected_fields or receipt["schema"] != BRANCH_SCHEMA
            or receipt["request_sha256"] != context.sha256 or receipt["variant"] != variant
            or receipt["branch_root"] != branch_root.as_posix()
            or receipt["activation_plan_sha256"] != sha256_file(root / "activation-plan.json")
            or type(receipt["stage_attempt_id"]) is not str or not receipt["stage_attempt_id"]
            or receipt["requested_physical_frames"] != 2 or receipt["requested_refinement_states"] != 2
            or receipt["test_payload_reads"] != 0 or receipt["annotation_inputs"] != []):
        raise ActivationError("branch receipt differs from the frozen activation identity")
    marker, _ = _stage_paths(root, "legacy" if variant == "legacy-v1" else "amortized")
    marker_doc = _read_json(marker, "activation branch stage marker")
    if marker_doc.get("attempt_id") != receipt["stage_attempt_id"] or marker_doc.get("request_sha256") != context.sha256:
        raise ActivationError("branch receipt is detached from its exclusive stage attempt")
    for name in ("capture_plan", "mask_index", "observations", "capture_index"):
        verify_file_pin(receipt[name], under=branch_root, maximum=MAX_METADATA_BYTES)
    snapshot_path = branch_root / "snapshot.json"
    if receipt["snapshot"] is None:
        raise ActivationError("branch execution has no authentic CUDA parity snapshot")
    verify_file_pin(receipt["snapshot"], under=branch_root, maximum=MAX_METADATA_BYTES)
    if Path(receipt["snapshot"]["path"]) != snapshot_path:
        raise ActivationError("branch snapshot escaped its unique branch directory")
    fresh = _collect_snapshot(context, variant, hardware=receipt["hardware"],
        fatal_error=receipt["fatal_error"])
    actual = _read_json(snapshot_path, "activation parity snapshot")
    if actual != fresh:
        raise ActivationError("branch snapshot differs from independently reopened actual artifacts")
    sidecar_path = branch_root / "snapshot-receipt.json"
    sidecar = {"schema": SNAPSHOT_RECEIPT_SCHEMA, "status": "completed" if receipt["status"] == "completed" else "failed",
        "request_sha256": context.sha256, "activation_plan_sha256": receipt["activation_plan_sha256"],
        "variant": variant, "stage_attempt_id": receipt["stage_attempt_id"],
        "branch_root": branch_root.as_posix(), "branch_receipt": file_pin(receipt_path),
        "snapshot": file_pin(snapshot_path), "hardware": receipt["hardware"],
        "fatal_error": receipt["fatal_error"], "test_payload_reads": 0, "annotation_inputs": []}
    if sidecar_path.exists():
        if _read_json(sidecar_path, "activation snapshot receipt") != sidecar:
            raise ActivationError("branch snapshot receipt changed")
    else:
        _atomic_write_json(sidecar_path, sidecar)
    return actual


def _verified_branch_for_compare(context: ActivationContext, variant: str):
    stage_name = "legacy" if variant == "legacy-v1" else "amortized"
    stage_receipt = _read_completed_stage(context, stage_name)
    snapshot = snapshot_activation(context, variant)
    doc = context.document()
    branch_root = Path(doc["output_root"]) / "branches" / variant
    receipt = _read_json(branch_root / "branch-receipt.json", "activation branch receipt")
    sidecar = _read_json(branch_root / "snapshot-receipt.json", "activation snapshot receipt")
    if (receipt["status"] != "completed" or receipt["fatal_error"] is not None
            or receipt["cleanup_success"] is not True or receipt["cleanup_errors"] != []
            or receipt["stage_attempt_id"] != stage_receipt["attempt_id"]
            or sidecar["status"] != "completed" or sidecar["stage_attempt_id"] != stage_receipt["attempt_id"]
            or snapshot.get("fatal_error") is not None or snapshot.get("device") != "cuda"):
        raise ActivationError("branch/stage did not complete a clean CUDA capture")
    if len(snapshot["rows"]) != 2 or sum(row["status"] == "available" for row in snapshot["rows"]) != 2:
        raise ActivationError("activation parity requires both exact requested views to be available")
    hardware = receipt["hardware"]
    if not isinstance(hardware, dict) or hardware.get("device") != "cuda" or not hardware.get("uuid"):
        raise ActivationError("branch lacks an actual physical CUDA UUID")
    return snapshot, receipt, sidecar


def compare(context: ActivationContext) -> dict[str, Any]:
    """Compare distinct authenticated legacy/amortized CUDA snapshots unchanged."""
    _require_stage(context, "compare")
    doc, activation, rows = _frozen(context)
    from .quality_adaptation_larger_parity import compare_runs
    reference, reference_receipt, reference_sidecar = _verified_branch_for_compare(context, "legacy-v1")
    candidate, candidate_receipt, candidate_sidecar = _verified_branch_for_compare(context, "amortized-v1")
    if (reference_receipt["branch_root"] == candidate_receipt["branch_root"]
            or reference_sidecar["snapshot"]["path"] == candidate_sidecar["snapshot"]["path"]
            or reference_receipt["stage_attempt_id"] == candidate_receipt["stage_attempt_id"]):
        raise ActivationError("activation branches must have independent directories and stage attempts")
    if reference_receipt["hardware"]["uuid"] != candidate_receipt["hardware"]["uuid"]:
        raise ActivationError("activation branches ran on different physical CUDA devices")
    parity = compare_runs(reference, candidate)
    if parity.get("passed") is not True or parity.get("requested") != 2 or parity.get("available") != 2:
        raise ActivationError("unchanged snapshot comparator did not establish exact two-view parity")
    root = Path(doc["output_root"])
    path = root / "comparison.json"
    value = {"schema": COMPARISON_SCHEMA, "status": "completed", "request_sha256": context.sha256,
        "activation_plan_sha256": sha256_file(root / "activation-plan.json"),
        "source_files_sha256": hashlib.sha256(canonical_bytes(doc["source_files"])).hexdigest(),
        "reference": reference_sidecar["snapshot"], "candidate": candidate_sidecar["snapshot"],
        "reference_branch_receipt": reference_sidecar["branch_receipt"],
        "candidate_branch_receipt": candidate_sidecar["branch_receipt"],
        "hardware": reference_receipt["hardware"],
        "parity": {"reference": reference_sidecar["snapshot"],
            "candidate": candidate_sidecar["snapshot"]},
        "parity_result": parity,
        "test_payload_reads": 0, "annotation_inputs": []}
    pin = _atomic_write_json(path, value)
    result = dict(value, comparison=pin)
    return result
