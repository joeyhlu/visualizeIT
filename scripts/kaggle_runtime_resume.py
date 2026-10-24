#!/usr/bin/env python3
"""Resume the one authenticated Kaggle runtime attempt without rerunning uv pip."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile
from typing import Any, BinaryIO


SCHEMA = "kaggle_isolated_runtime_resume_v1"
COMPAT_SCHEMA = "kaggle_isolated_runtime_v1"
UV_VERSION = "0.12.22"
UV_VERSION_OUTPUTS = frozenset((
    f"uv {UV_VERSION}",
    f"uv {UV_VERSION}\n",
    f"uv {UV_VERSION} (x86_64-unknown-linux-gnu)",
    f"uv {UV_VERSION} (x86_64-unknown-linux-gnu)\n",
))
UV_WHEEL_BYTES = 20_587_257
UV_WHEEL_SHA256 = "680769ef701922dd93426b73476073411941f51ac46991a6e5c5a6efbf68a713"
UV_BINARY_BYTES = 47_991_416
UV_BINARY_SHA256 = "96e1603cb62aebb1a804fe9866a5708ee8bba4c39d07202ad29cf256a67366c3"
FAILED_ATTEMPT_BYTES = 828
FAILED_ATTEMPT_SHA256 = "3fae23f2c39dfefd544776a02a47df1e30cf9c73f39397ce7ed02ac7a2ccd084"
LAYOUT_EVIDENCE_SHA256 = "68f8cb41d77fe361557fb1295aedd6a6f04d1fde4aaf08eaef59467b2ee656f"
BOOTSTRAP_SHA256 = "cbe624d3432b30e575141b2c00f34fca3f690ce812d97db7cc7c315fb075a9d3"
LAUNCHER_SHA256 = "1823fc93ef95057a3f2e4928c3d43123f130ce719683260fdce4e5f58212b388"
PYTHON_REQUEST = "3.11.13"
GIB = 1024**3
MAX_JSON_BYTES = 16 * 1024**2
MAX_PARTIAL_UV_BYTES = 128 * 1024**2
MAX_INSTALLED_RECORD_BYTES = 1024**2
MAX_WALL_SECONDS = 1200
MAX_RUNTIME_BYTES = 18 * GIB
MIN_WORKING_FREE_BYTES = 2 * GIB
POLL_SECONDS = 0.1
PROCESS_GROUP_TERM_GRACE_SECONDS = 5.0
PROCESS_GROUP_KILL_GRACE_SECONDS = 5.0
PROCESS_GROUP_SIGTERM = getattr(signal, "SIGTERM", 15)
PROCESS_GROUP_SIGKILL = getattr(signal, "SIGKILL", 9)

EXPECTED_SOURCE_INSTALL = Path(
    "/kaggle/working/visualizeit-reviewed-source-installer-smoke-20261018-v1"
)
EXPECTED_RUNTIME_ROOT = Path("/kaggle/working/visualizeit-pinned-runtime-v2")
EXPECTED_LAUNCHER = Path(
    "/kaggle/working/visualizeit-prepared-source-v3-20261018/scripts/kaggle_adaptation.py"
)
EXPECTED_BOOTSTRAP = EXPECTED_LAUNCHER.with_name("kaggle_runtime_bootstrap.sh")
EXPECTED_WHEEL = Path(
    "/kaggle/working/visualizeit-bootstrap-downloads-v1/"
    "uv-0.12.22-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
TORCH_CONSTRAINTS = b"torch==2.5.1+cu124\ntorchvision==0.20.1+cu124\n"
REQUIREMENT_FILES = (
    "bench/runtime/requirements-ranker.txt",
    "bench/runtime/requirements-quality.txt",
)


class ResumeError(RuntimeError):
    """An authentication, resource, or command contract failed."""


def verify_uv_version_output(output: str) -> None:
    """Accept only the pinned release's known Linux output forms."""
    if output not in UV_VERSION_OUTPUTS:
        raise ResumeError("verified uv binary did not report the pinned 0.12.22 Linux version")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def strict_json_bytes(data: bytes, label: str) -> Any:
    if len(data) > MAX_JSON_BYTES:
        raise ResumeError(f"metadata exceeds {MAX_JSON_BYTES} bytes: {label}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ResumeError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ResumeError(f"non-finite JSON value in {label}: {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResumeError(f"invalid JSON evidence: {label}") from error


def strict_json_file(path: Path) -> Any:
    with path.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    return strict_json_bytes(data, str(path))


def no_symlink_components(path: Path) -> Path:
    if not path.is_absolute():
        raise ResumeError(f"absolute path required: {path}")
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        try:
            if component.is_symlink():
                raise ResumeError(f"symlink path component: {component}")
        except OSError as error:
            raise ResumeError(f"cannot inspect path component: {component}") from error
    return absolute


def require_inside(path: Path, root: Path, *, resolve: bool = False) -> Path:
    path = no_symlink_components(path) if not resolve else path.absolute()
    root = no_symlink_components(root) if not resolve else root.absolute()
    candidate = path.resolve(strict=True) if resolve else path
    canonical_root = root.resolve(strict=True) if resolve else root
    if not candidate.is_relative_to(canonical_root):
        raise ResumeError(f"path escapes authenticated root {canonical_root}: {path}")
    return candidate


def _record_digest(raw: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode("ascii").rstrip("=")


def authenticate_uv_wheel(
    wheel: Path,
    *,
    expected_wheel_sha256: str = UV_WHEEL_SHA256,
    expected_wheel_bytes: int = UV_WHEEL_BYTES,
    expected_binary_sha256: str = UV_BINARY_SHA256,
    expected_binary_bytes: int = UV_BINARY_BYTES,
) -> dict[str, Any]:
    """Authenticate uv metadata, RECORD, and the unique wheel-installed script."""
    no_symlink_components(wheel)
    if not wheel.is_file() or wheel.stat().st_size != expected_wheel_bytes:
        raise ResumeError("uv wheel size differs from the pinned official wheel")
    if sha256_file(wheel) != expected_wheel_sha256:
        raise ResumeError("uv wheel SHA-256 differs from the pinned official wheel")
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = [item.filename for item in archive.infolist()]
            if len(names) != len(set(names)):
                raise ResumeError("uv wheel contains duplicate member names")
            for info in archive.infolist():
                name = info.filename
                member_name = name[:-1] if info.is_dir() else name
                member = PurePosixPath(member_name)
                mode = info.external_attr >> 16
                if ("\\" in name or member.is_absolute() or ":" in name
                        or any(part in ("", ".", "..") for part in member_name.split("/"))
                        or stat.S_ISLNK(mode)
                        or (not info.is_dir() and stat.S_IFMT(mode) not in (0, stat.S_IFREG))):
                    raise ResumeError(f"unsafe uv wheel member: {name!r}")
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
            script_names = [name for name in names if name.endswith(".data/scripts/uv")]
            if len(metadata_names) != 1 or len(record_names) != 1 or len(script_names) != 1:
                raise ResumeError("uv wheel METADATA, RECORD, or installed uv script is ambiguous")
            metadata = archive.read(metadata_names[0]).decode("utf-8")
            fields: dict[str, list[str]] = {}
            for line in metadata.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields.setdefault(key.lower(), []).append(value.strip())
            if fields.get("name") != ["uv"] or fields.get("version") != [UV_VERSION]:
                raise ResumeError("official wheel metadata must identify exactly uv==0.12.22")
            script_name = script_names[0]
            metadata_bytes = metadata.encode("utf-8")
            record_rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
            matching = [row for row in record_rows if row and row[0] == script_name]
            if len(matching) != 1 or len(matching[0]) != 3:
                raise ResumeError("uv wheel RECORD does not uniquely bind the installed script")
            script_info = archive.getinfo(script_name)
            if matching[0][1] != _record_digest(archive.read(script_name)):
                raise ResumeError("uv wheel RECORD hash does not authenticate the script")
            if matching[0][2] != str(script_info.file_size):
                raise ResumeError("uv wheel RECORD size does not authenticate the script")
            if script_info.file_size != expected_binary_bytes:
                raise ResumeError("uv wheel script size differs from the pinned uv binary")
            script_bytes = archive.read(script_name)
            script_hash = sha256_bytes(script_bytes)
            if script_hash != expected_binary_sha256:
                raise ResumeError("uv wheel script hash differs from the pinned uv binary")
            record_digest = _record_digest(script_bytes)
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as error:
        raise ResumeError("cannot inspect the authenticated uv wheel") from error
    return {
        "wheel": descriptor(wheel),
        "metadata_member": metadata_names[0],
        "metadata_bytes": len(metadata_bytes),
        "metadata_sha256": sha256_bytes(metadata_bytes),
        "record_member": record_names[0],
        "script_member": script_name,
        "script_bytes": script_info.file_size,
        "script_sha256": script_hash,
        "record_sha256": record_digest,
        "layout_evidence_sha256": LAYOUT_EVIDENCE_SHA256,
    }


def authenticate_installed_uv(
    runtime_root: Path,
    wheel_evidence: dict[str, Any],
    *,
    expected_binary_sha256: str = UV_BINARY_SHA256,
    expected_binary_bytes: int = UV_BINARY_BYTES,
) -> dict[str, Any]:
    binary = runtime_root / "uv-tool" / "bin" / "uv"
    no_symlink_components(binary)
    if not binary.is_file() or binary.stat().st_size != expected_binary_bytes:
        raise ResumeError("installed uv binary is missing or has an unexpected size")
    digest = sha256_file(binary)
    if digest != expected_binary_sha256 or digest != wheel_evidence["script_sha256"]:
        raise ResumeError("installed uv binary is not the RECORD-authenticated official script")
    metadata_name = wheel_evidence["metadata_member"].split("/", 1)[0]
    metadata_path = runtime_root / "uv-tool" / metadata_name / "METADATA"
    record_path = runtime_root / "uv-tool" / metadata_name / "RECORD"
    no_symlink_components(metadata_path)
    no_symlink_components(record_path)
    if (not metadata_path.is_file() or metadata_path.stat().st_size != wheel_evidence["metadata_bytes"]
            or sha256_file(metadata_path) != wheel_evidence["metadata_sha256"]
            or not record_path.is_file() or record_path.stat().st_size > MAX_INSTALLED_RECORD_BYTES):
        raise ResumeError("installed uv target metadata is missing or differs from the authenticated wheel")
    try:
        rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ResumeError("installed uv target RECORD is unreadable") from error
    binary_rows = [row for row in rows if row and PurePosixPath(row[0]).name == "uv"
                   and "bin" in PurePosixPath(row[0]).parts]
    expected_record_hash = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    if (len(binary_rows) != 1 or len(binary_rows[0]) != 3
            or binary_rows[0][1] != expected_record_hash
            or binary_rows[0][2] != str(binary.stat().st_size)):
        raise ResumeError("installed uv target RECORD does not bind the actual bin/uv")
    return {
        "path": str(binary), "bytes": binary.stat().st_size, "sha256": digest,
        "metadata": descriptor(metadata_path), "record": descriptor(record_path),
    }


def inspect_partial_runtime(
    runtime_root: Path,
    *,
    expected_binary_sha256: str = UV_BINARY_SHA256,
    expected_binary_bytes: int = UV_BINARY_BYTES,
) -> dict[str, Any]:
    """Accept only the original uv-target partial tree; reject later install state."""
    no_symlink_components(runtime_root)
    if not runtime_root.is_dir():
        raise ResumeError("the pinned partial runtime root must already exist")
    top = {entry.name for entry in runtime_root.iterdir()}
    if top != {"uv-tool"}:
        raise ResumeError("partial runtime must contain only the original uv-tool installation")
    files: list[Path] = []
    total_bytes = 0
    for parent, directories, filenames in os.walk(runtime_root, followlinks=False):
        parent_path = Path(parent)
        for name in directories + filenames:
            path = parent_path / name
            if path.is_symlink():
                raise ResumeError(f"partial runtime cannot contain symlinks: {path}")
        for name in filenames:
            path = parent_path / name
            mode = path.stat().st_mode
            if not stat.S_ISREG(mode):
                raise ResumeError(f"partial uv target contains a non-regular file: {path}")
            files.append(path)
            total_bytes += path.stat().st_size
            if total_bytes > MAX_PARTIAL_UV_BYTES:
                raise ResumeError("original partial uv target exceeds its 128 MiB admission ceiling")
    binary = runtime_root / "uv-tool" / "bin" / "uv"
    if binary not in files or binary.stat().st_size != expected_binary_bytes:
        raise ResumeError("actual installed uv binary does not match the observed target layout")
    digest = sha256_file(binary)
    if digest != expected_binary_sha256:
        raise ResumeError("actual installed uv binary hash differs from observed target layout")
    forbidden = {
        "python", "venv", "runtime-observed.json", "resolved-requirements.txt",
        "runtime-receipt.json", "runtime-resume-receipt.json", "torch-constraints.txt",
    }
    if forbidden.intersection(top) or any(path.name in forbidden for path in files):
        raise ResumeError("runtime already contains later installation state")
    return {
        "runtime_root": str(runtime_root),
        "uv_binary": {"path": str(binary), "bytes": binary.stat().st_size, "sha256": digest},
        "files": [descriptor(path) for path in sorted(files)],
    }


def verify_failed_attempt(path: Path, expected_sha256: str = FAILED_ATTEMPT_SHA256) -> dict[str, Any]:
    no_symlink_components(path)
    if not path.is_file() or path.stat().st_size != FAILED_ATTEMPT_BYTES:
        raise ResumeError("parent-observed failed runtime attempt is missing or changed in size")
    digest = sha256_file(path)
    if digest != expected_sha256 or digest != FAILED_ATTEMPT_SHA256:
        raise ResumeError("parent-observed failed runtime attempt SHA-256 mismatch")
    evidence = strict_json_file(path)
    if (evidence.get("schema") != "visualizeit_kaggle_runtime_status_v1"
            or evidence.get("pid") != 129 or evidence.get("exit_code") != 2
            or evidence.get("training_started") is not False
            or evidence.get("new_optimizer_updates") != 0
            or evidence.get("actual_process_live") is not False
            or evidence.get("runtime_receipt_exists") is not False):
        raise ResumeError("failed attempt evidence does not match the stopped uv-layout failure")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest,
            "schema": evidence["schema"], "pid": evidence["pid"],
            "exit_code": evidence["exit_code"], "training_started": False,
            "new_optimizer_updates": 0, "actual_process_live": False,
            "runtime_receipt_exists": False}


def clean_environment(base: dict[str, str] | None, runtime_root: Path) -> dict[str, str]:
    source = os.environ if base is None else base
    env = {
        key: value for key, value in source.items()
        if not key.startswith(("PYTHON", "PIP_", "UV_"))
        and key not in {"VIRTUAL_ENV", "CONDA_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH"}
    }
    env.update(
        VISUALIZEIT_CLOUD="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PIP_NO_CACHE_DIR="1",
        UV_PYTHON_INSTALL_DIR=str(runtime_root / "python"),
        UV_CACHE_DIR=str(runtime_root / "uv-cache"),
        UV_NO_PROGRESS="1",
        UV_NO_TELEMETRY="1",
        TMPDIR=str(runtime_root / "tmp"),
    )
    return env


def runtime_links(runtime_root: Path) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for parent, directories, filenames in os.walk(runtime_root, followlinks=False):
        for name in directories + filenames:
            path = Path(parent) / name
            if not path.is_symlink():
                continue
            try:
                target = os.readlink(path)
                lexical = Path(os.path.abspath(path.parent / target))
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise ResumeError(f"broken or cyclic runtime symlink: {path}") from error
            if (not lexical.is_relative_to(runtime_root)
                    or not resolved.is_relative_to(runtime_root)
                    or (resolved.is_dir() and path.parent.is_relative_to(resolved))):
                raise ResumeError(f"runtime symlink escapes or cycles: {path}")
            links.append({"path": path.relative_to(runtime_root).as_posix(), "target": target})
    return sorted(links, key=lambda entry: entry["path"])


def tree_bytes(root: Path, wheel: Path) -> int:
    """Count owned inodes without following links during concurrent extraction.

    uv creates links before their targets and atomically renames temporary trees.
    A live storage sample must tolerate disappearing names; completed installs
    still undergo the separate strict runtime_links validation before admission.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for base in (root, wheel):
        candidates: list[Path] = []
        if base.is_file():
            candidates.append(base)
        elif base.is_dir():
            def walk_error(error: OSError) -> None:
                if not isinstance(error, FileNotFoundError):
                    raise ResumeError(f"cannot traverse runtime storage: {error.filename}") from error
            for parent, directories, filenames in os.walk(base, followlinks=False, onerror=walk_error):
                candidates.extend(Path(parent) / name for name in directories + filenames)
        for path in candidates:
            try:
                info = path.lstat()
            except FileNotFoundError:
                # An extraction rename can invalidate a previously listed name.
                continue
            except OSError as error:
                raise ResumeError(f"cannot account for runtime storage: {path}") from error
            if stat.S_ISDIR(info.st_mode):
                continue
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                raise ResumeError(f"runtime storage contains a non-regular file: {path}")
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += info.st_size
    return total


def check_resources(runtime_root: Path, wheel: Path, working: Path) -> None:
    free = shutil.disk_usage(working).free
    if free < MIN_WORKING_FREE_BYTES:
        raise ResumeError("working filesystem free reserve fell below 2 GiB")
    if tree_bytes(runtime_root, wheel) > MAX_RUNTIME_BYTES:
        raise ResumeError("runtime and pinned wheel exceeded the 18 GiB byte ceiling")


def _process_group_exists(pgid: int) -> bool:
    """Probe only the new-session process group created for this child."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise ResumeError(f"cannot inspect owned process group {pgid}: {error}") from error
    return True


def _signal_process_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        # A child may exit between the liveness probe and the signal.
        return
    except PermissionError as error:
        raise ResumeError(f"cannot signal owned process group {pgid}: {error}") from error


def _wait_process_group_exit(pgid: int, deadline: float, process: subprocess.Popen[bytes],
                             *, poll_interval: float = 0.05) -> bool:
    while _process_group_exists(pgid):
        # poll() also reaps the direct child when it has exited. Descendants
        # are tracked independently through the group id, not the leader.
        process.poll()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(poll_interval, remaining))
    return True


def _kill_process_group(process: subprocess.Popen[bytes], *, term_grace: float | None = None,
                        kill_grace: float | None = None) -> None:
    """Terminate and verify the dedicated process group, even after leader exit.

    Popen uses ``start_new_session=True``; therefore the child's PID is the
    process-group ID owned by this invocation. Group liveness is checked
    separately from ``Popen.poll`` so a completed leader cannot hide a still
    running pip/uv descendant. Any inability to reap the child or clear the
    group is surfaced to the caller and retained in its terminal failure receipt.
    """
    pgid = process.pid
    if type(pgid) is not int or pgid <= 0:
        raise ResumeError("cannot clean a subprocess without its owned process-group id")
    term_grace = PROCESS_GROUP_TERM_GRACE_SECONDS if term_grace is None else term_grace
    kill_grace = PROCESS_GROUP_KILL_GRACE_SECONDS if kill_grace is None else kill_grace
    if term_grace < 0 or kill_grace < 0:
        raise ResumeError("process-group cleanup grace periods must be nonnegative")

    # Do not signal by leader state alone. A leader that exited after spawning
    # a child still has a live group that must be stopped before failure returns.
    if _process_group_exists(pgid):
        _signal_process_group(pgid, PROCESS_GROUP_SIGTERM)
        term_deadline = time.monotonic() + term_grace
        _wait_process_group_exit(pgid, term_deadline, process)

    if _process_group_exists(pgid):
        _signal_process_group(pgid, PROCESS_GROUP_SIGKILL)
        try:
            process.wait(timeout=kill_grace)
        except subprocess.TimeoutExpired as error:
            raise ResumeError("direct child was not reaped after process-group SIGKILL") from error
        kill_deadline = time.monotonic() + kill_grace
        if not _wait_process_group_exit(pgid, kill_deadline, process):
            raise ResumeError("owned process group survived SIGKILL cleanup")
    else:
        # Even if the group is already gone, explicitly reap a finished direct
        # child. It should be immediate when poll() or the leader exit caused
        # the group probe to become empty; retain any unexpected wait failure.
        try:
            process.wait(timeout=kill_grace)
        except subprocess.TimeoutExpired as error:
            raise ResumeError("direct child was not reaped after its process group exited") from error


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    stdout_path: Path | None,
    deadline: float,
    runtime_root: Path,
    wheel: Path,
    working: Path,
    popen: Any = subprocess.Popen,
    poll_interval: float = POLL_SECONDS,
) -> dict[str, Any]:
    if not argv or not all(isinstance(arg, str) for arg in argv):
        raise ResumeError("subprocess command must be a non-empty string argument list")
    no_symlink_components(cwd)
    no_symlink_components(log_path)
    if stdout_path is not None:
        no_symlink_components(stdout_path)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ResumeError("cumulative runtime wall limit exhausted")
    check_resources(runtime_root, wheel, working)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_stream: BinaryIO | None = None
    stderr_stream: BinaryIO | None = None
    try:
        log_path.open("xb").close()
        stderr_stream = log_path.open("ab")
        if stdout_path is None:
            stdout_stream = log_path.open("ab")
        else:
            stdout_stream = stdout_path.open("xb")
        process = popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            shell=False,
            start_new_session=True,
        )
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResumeError("cumulative runtime wall limit exhausted; process-group cleanup required")
            check_resources(runtime_root, wheel, working)
            time.sleep(min(poll_interval, remaining))
        stdout_stream.flush()
        stderr_stream.flush()
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
        returncode = process.returncode
        if returncode != 0:
            raise ResumeError(f"command exited {returncode}: {argv[0]}")
        check_resources(runtime_root, wheel, working)
        result = {"argv": argv, "returncode": returncode, "log": str(log_path)}
        if stdout_path is not None:
            result["stdout"] = str(stdout_path)
            result["stdout_sha256"] = sha256_file(stdout_path)
        return result
    except BaseException as command_error:
        process_obj = locals().get("process")
        if process_obj is not None:
            try:
                _kill_process_group(process_obj)
            except BaseException as cleanup_error:
                raise ResumeError(
                    f"command failed and owned process-group cleanup failed: {cleanup_error}"
                ) from command_error
        raise
    finally:
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()


def build_commands(
    uv: Path,
    managed_python: Path,
    training_python: Path,
    runtime_root: Path,
    root18: Path,
    launcher: Path,
) -> list[tuple[str, list[str], str | None]]:
    """Return remaining frozen commands; output names are relative to runtime root."""
    return [
        ("uv-version", [str(uv), "--version"], "uv-version.txt"),
        ("python-install", [str(uv), "python", "install", PYTHON_REQUEST], None),
        ("python-find", [str(uv), "python", "find", "--managed-python", PYTHON_REQUEST], "managed-python.txt"),
        ("managed-python-check", [str(managed_python), "-I", "-B", "-c",
                                  "import platform;print(platform.python_version())"], "managed-version.txt"),
        ("venv", [str(uv), "venv", "--python", str(managed_python), str(runtime_root / "venv")], None),
        ("bootstrap-packages", [str(uv), "pip", "install", "--python", str(training_python),
                                "pip==24.3.1", "setuptools==75.3.0", "wheel==0.44.0"], None),
        ("torch", [str(training_python), "-I", "-B", "-m", "pip", "install", "--isolated",
                   "--no-cache-dir", "torch==2.5.1+cu124", "torchvision==0.20.1+cu124",
                   "--index-url", "https://download.pytorch.org/whl/cu124"], None),
        ("requirements", [str(training_python), "-I", "-B", "-m", "pip", "install", "--isolated",
                           "--no-cache-dir", "-c", str(runtime_root / "torch-constraints.txt"),
                           "-r", str(root18 / "bench/runtime/requirements-ranker.txt"),
                           "--index-url", "https://pypi.org/simple"], None),
        ("pip-check", [str(training_python), "-I", "-B", "-m", "pip", "check"], None),
        ("pip-freeze", [str(training_python), "-I", "-B", "-m", "pip", "freeze", "--all"],
         "resolved-requirements.txt"),
        ("runtime-probe", [sys.executable, "-I", "-B", str(launcher), "runtime-probe",
                            "--python", str(training_python)], "runtime-observed.json"),
    ]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}-{os.getpid()}.tmp")
    no_symlink_components(path)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    no_symlink_components(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_runtime_probe(observed: Any, training_python: Path, runtime_root: Path) -> dict[str, Any]:
    if not isinstance(observed, dict):
        raise ResumeError("runtime-probe output must be a JSON object")
    packages = observed.get("packages")
    expected = {
        "torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124", "numpy": "1.26.4",
        "onnxruntime": "1.20.1", "pyrender": "0.1.45",
    }
    if (observed.get("python") != PYTHON_REQUEST or not isinstance(packages, dict)
            or any(packages.get(name) != version for name, version in expected.items())):
        raise ResumeError("runtime-probe did not satisfy the pinned Python/package contract")
    if observed.get("prefix") == observed.get("base_prefix"):
        raise ResumeError("training interpreter is not an isolated virtual environment")
    try:
        executable = Path(observed["executable"])
        expected_executable = training_python.absolute()
        if executable != expected_executable:
            raise ResumeError("runtime-probe executable path differs from the isolated training Python")
        resolved = training_python.resolve(strict=True)
        if not resolved.is_relative_to(runtime_root.resolve(strict=True)):
            raise ResumeError("training interpreter escaped the authenticated runtime root")
        if observed.get("executable_sha256") != sha256_file(training_python):
            raise ResumeError("runtime-probe executable binary SHA-256 mismatch")
        if not Path(observed["prefix"]).resolve(strict=True).is_relative_to(runtime_root.resolve(strict=True)):
            raise ResumeError("runtime-probe venv prefix escaped the authenticated runtime root")
        if not Path(observed["base_prefix"]).resolve(strict=True).is_relative_to(runtime_root.resolve(strict=True)):
            raise ResumeError("runtime-probe managed base Python escaped the authenticated runtime root")
    except (KeyError, OSError, RuntimeError) as error:
        raise ResumeError("runtime-probe interpreter identity is incomplete") from error
    return observed


def _validate_arguments(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, dict[str, Any], dict[str, Any]]:
    if not args.cloud_execution:
        raise ResumeError("explicit --cloud-execution required")
    if platform.system() != "Linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ResumeError("runtime resume requires an actual Kaggle Linux session")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise ResumeError("actual Kaggle input and working roots are required")

    launcher = no_symlink_components(args.launcher)
    source_install = no_symlink_components(args.source_install)
    runtime_root = no_symlink_components(args.runtime_root)
    wheel = no_symlink_components(args.uv_wheel)
    failed_attempt = no_symlink_components(args.failed_attempt)
    if (launcher != EXPECTED_LAUNCHER or source_install != EXPECTED_SOURCE_INSTALL
            or runtime_root != EXPECTED_RUNTIME_ROOT or wheel != EXPECTED_WHEEL):
        raise ResumeError("resume paths differ from parent-observed Kaggle partial attempt")
    if args.uv_wheel_sha256 != UV_WHEEL_SHA256:
        raise ResumeError("explicit official uv wheel pin does not match the admitted wheel")
    if not (failed_attempt.is_relative_to(Path("/kaggle/input"))
            or failed_attempt.is_relative_to(Path("/kaggle/working"))):
        raise ResumeError("failed attempt evidence must stay inside Kaggle input or working roots")
    bootstrap = no_symlink_components(launcher.with_name("kaggle_runtime_bootstrap.sh"))
    if sha256_file(launcher) != LAUNCHER_SHA256 or sha256_file(bootstrap) != BOOTSTRAP_SHA256:
        raise ResumeError("frozen runtime launcher/bootstrap source pin changed")
    if not source_install.is_dir() or not runtime_root.is_dir() or not wheel.is_file():
        raise ResumeError("source install, partial runtime, or official uv wheel is missing")
    if source_install.is_relative_to(runtime_root) or runtime_root.is_relative_to(source_install):
        raise ResumeError("source install and isolated runtime roots overlap")
    if runtime_root.parent != Path("/kaggle/working"):
        raise ResumeError("runtime root must remain the exact admitted working directory")

    failure_evidence = verify_failed_attempt(failed_attempt, args.failed_attempt_sha256)
    partial = inspect_partial_runtime(runtime_root)
    wheel_evidence = authenticate_uv_wheel(wheel)
    installed_uv = authenticate_installed_uv(runtime_root, wheel_evidence)
    if not os.access(installed_uv["path"], os.X_OK):
        raise ResumeError("the authenticated installed uv binary is not executable")
    return launcher, source_install, runtime_root, wheel, failed_attempt, {
        "failure": failure_evidence, "partial": partial, "wheel": wheel_evidence,
        "installed_uv": installed_uv, "bootstrap": descriptor(bootstrap),
        "launcher": descriptor(launcher),
    }, {"path": str(source_install)}


class RuntimeResume:
    def __init__(self, args: argparse.Namespace, *, command_runner: Any = run_command,
                 clock: Any = time.monotonic) -> None:
        (self.launcher, self.source_install, self.runtime_root, self.wheel,
         self.failed_attempt, self.evidence, self.source) = _validate_arguments(args)
        self.args = args
        self.command_runner = command_runner
        self.clock = clock
        self.working = Path("/kaggle/working")
        self.root18 = self.source_install / "visualizeit-cloud-v18"
        self.host_python = Path(sys.executable).absolute()
        self.env = clean_environment(None, self.runtime_root)
        self.uv = self.runtime_root / "uv-tool" / "bin" / "uv"
        self.managed_python = self.runtime_root / "python" / "cpython-3.11.13" / "bin" / "python3.11"
        self.training_python = self.runtime_root / "venv" / "bin" / "python"
        self.history: list[dict[str, Any]] = []
        self.requirement_hashes: dict[str, str] = {}
        self.marker = self.runtime_root / "resume-started.json"
        self.resume_receipt = self.runtime_root / "runtime-resume-receipt.json"
        self.generic_receipt = self.runtime_root / "runtime-receipt.json"
        self.deadline = 0.0

    def _source_verify(self) -> None:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise ResumeError("cumulative runtime wall limit exhausted during source verification")
        result = subprocess.run(
            [str(self.host_python), "-I", "-B", str(self.launcher),
             "verify-source", "--destination", str(self.source_install)],
            cwd=str(self.working), env=self.env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=min(30, remaining), check=False, shell=False,
        )
        if result.returncode != 0:
            raise ResumeError(f"existing launcher rejected the installed source: {result.stderr[-4000:]}")
        output = strict_json_bytes(result.stdout.encode("utf-8"), "verify-source output")
        if (output.get("status") != "completed"
                or output.get("root18") != str(self.root18)):
            raise ResumeError("existing launcher did not verify the exact ROOT18 installation")

    def _record(self, stage: str, status: str, **details: Any) -> None:
        row: dict[str, Any] = {"stage": stage, "status": status,
                               "elapsed_seconds": round(self.clock() - self.started_at, 3)}
        row.update(details)
        self.history.append(row)
        atomic_json(self.marker, self._marker_document("running"))

    def _marker_document(self, status: str) -> dict[str, Any]:
        return {
            "schema": "kaggle_isolated_runtime_resume_attempt_v1",
            "status": status,
            "started_at_unix": self.started_at_unix,
            "wall_limit_seconds": MAX_WALL_SECONDS,
            "failed_attempt": self.evidence["failure"],
            "bootstrap": self.evidence["bootstrap"],
            "launcher": self.evidence["launcher"],
            "wheel": self.evidence["wheel"],
            "installed_uv": self.evidence["installed_uv"],
            "partial_runtime": self.evidence["partial"],
            "stages": self.history,
        }

    def _stage(self, name: str, argv: list[str], stdout_name: str | None = None) -> dict[str, Any]:
        stdout_path = self.runtime_root / stdout_name if stdout_name else None
        log_path = self.runtime_root / "resume-logs" / f"{len(self.history):02d}-{name}.log"
        self._record(name, "started", argv=argv, stdout=str(stdout_path) if stdout_path else None)
        result = self.command_runner(
            argv, cwd=self.working, env=self.env, log_path=log_path,
            stdout_path=stdout_path, deadline=self.deadline,
            runtime_root=self.runtime_root, wheel=self.wheel, working=self.working,
        )
        self._record(name, "completed", argv=argv, returncode=result["returncode"],
                     log=result["log"], stdout=result.get("stdout"),
                     stdout_sha256=result.get("stdout_sha256"))
        return result

    def _write_failure(self, error: BaseException) -> None:
        receipt = self._marker_document("failed")
        receipt.update({
            "schema": SCHEMA,
            "failure": {"type": type(error).__name__, "message": str(error)[:4000]},
            "compatibility_receipt_written": self.generic_receipt.exists(),
            "assets_acquired": False,
            "training_executed": False,
            "new_optimizer_updates": 0,
        })
        atomic_json(self.marker, self._marker_document("failed"))
        if not self.resume_receipt.exists():
            write_exclusive_json(self.resume_receipt, receipt)
        else:
            prior = strict_json_file(self.resume_receipt)
            if prior.get("status") != "completed":
                atomic_json(self.resume_receipt, receipt)

    def _final_receipts(self, runtime: dict[str, Any]) -> dict[str, Any]:
        if (sha256_file(self.launcher) != LAUNCHER_SHA256
                or sha256_file(self.launcher.with_name("kaggle_runtime_bootstrap.sh")) != BOOTSTRAP_SHA256):
            raise ResumeError("frozen runtime launcher/bootstrap changed during resume")
        current_failure = verify_failed_attempt(self.failed_attempt, self.args.failed_attempt_sha256)
        if current_failure != self.evidence["failure"]:
            raise ResumeError("parent-observed failed attempt evidence changed during resume")
        if authenticate_uv_wheel(self.wheel) != self.evidence["wheel"]:
            raise ResumeError("authenticated official uv wheel changed during resume")
        for relative, expected in self.requirement_hashes.items():
            if sha256_file(self.root18 / relative) != expected:
                raise ResumeError(f"authenticated source requirement changed during resume: {relative}")
        links = runtime_links(self.runtime_root)
        compatibility = {
            "schema": COMPAT_SCHEMA,
            "uv_version": UV_VERSION,
            "uv_wheel": self.evidence["wheel"]["wheel"],
            "layout_evidence_sha256": LAYOUT_EVIDENCE_SHA256,
            "runtime_root": str(self.runtime_root),
            "runtime_links": links,
            "python_request": PYTHON_REQUEST,
            "managed_python": str(self.managed_python),
            "python_install_provenance": (
                "https://github.com/astral-sh/uv/blob/0.12.22/docs/concepts/python-versions.md"
            ),
            "runtime": runtime,
            "resolved_requirements": descriptor(self.runtime_root / "resolved-requirements.txt"),
            "requirements": {
                relative: self.requirement_hashes[relative] for relative in REQUIREMENT_FILES
            },
            "assets_acquired": False,
            "training_executed": False,
            "producer": {
                "name": "scripts/kaggle_runtime_resume.py",
                "sha256": sha256_file(Path(__file__).resolve(strict=True)),
                "failed_original_attempt": self.evidence["failure"],
                "original_bootstrap_completed": False,
            },
        }
        encoded = (json.dumps(compatibility, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        compatibility_sha = sha256_bytes(encoded)
        completion = self._marker_document("completed")
        resume = {
            **completion,
            "schema": SCHEMA,
            "compatibility_receipt": {
                "path": str(self.generic_receipt),
                "bytes": len(encoded),
                "sha256": compatibility_sha,
            },
            "runtime_observed": descriptor(self.runtime_root / "runtime-observed.json"),
            "resolved_requirements": descriptor(self.runtime_root / "resolved-requirements.txt"),
            "uv_wheel_evidence": self.evidence["wheel"],
            "actual_uv_binary": authenticate_installed_uv(self.runtime_root, self.evidence["wheel"]),
            "requirements": compatibility["requirements"],
            "assets_acquired": False,
            "training_executed": False,
            "new_optimizer_updates": 0,
        }
        resume["status"] = "publishing"
        write_exclusive_json(self.resume_receipt, resume)
        write_exclusive_json(self.generic_receipt, compatibility)
        resume["status"] = "completed"
        atomic_json(self.resume_receipt, resume)
        atomic_json(self.marker, completion)
        return resume

    def run(self) -> dict[str, Any]:
        if (self.marker.exists() or self.resume_receipt.exists() or self.generic_receipt.exists()
                or any((self.runtime_root / name).exists() for name in (
                    "python", "venv", "runtime-observed.json", "resolved-requirements.txt",
                    "torch-constraints.txt"))):
            raise ResumeError("runtime attempt already started or contains later install state; preserve it")
        self.started_at = self.clock()
        self.started_at_unix = time.time()
        self.deadline = self.started_at + MAX_WALL_SECONDS
        self._source_verify()
        for relative in REQUIREMENT_FILES:
            path = self.root18 / relative
            no_symlink_components(path)
            if not path.is_file():
                raise ResumeError(f"authenticated source requirement missing: {relative}")
            self.requirement_hashes[relative] = sha256_file(path)
        check_resources(self.runtime_root, self.wheel, self.working)
        write_exclusive_json(self.marker, self._marker_document("running"))
        try:
            temporary = self.runtime_root / "tmp"
            no_symlink_components(temporary)
            temporary.mkdir(mode=0o700, exist_ok=False)
            no_symlink_components(temporary)
            if (not temporary.is_dir() or temporary.resolve(strict=True) != temporary
                    or not os.access(temporary, os.W_OK | os.X_OK)):
                raise ResumeError("configured TMPDIR failed post-marker authentication")
            (self.runtime_root / "resume-logs").mkdir()
            commands = build_commands(
                self.uv, self.managed_python, self.training_python,
                self.runtime_root, self.root18, self.launcher,
            )
            for name, argv, stdout_name in commands[:3]:
                result = self._stage(name, argv, stdout_name)
                if name == "uv-version":
                    version_text = Path(result["stdout"]).read_text(encoding="utf-8")
                    verify_uv_version_output(version_text)
                elif name == "python-find":
                    discovered = Path(Path(result["stdout"]).read_text(encoding="utf-8").strip())
                    require_inside(discovered, self.runtime_root / "python", resolve=True)
                    self.managed_python = discovered
                    self.training_python = self.runtime_root / "venv" / "bin" / "python"
            commands = build_commands(
                self.uv, self.managed_python, self.training_python,
                self.runtime_root, self.root18, self.launcher,
            )
            for name, argv, stdout_name in commands[3:]:
                if name == "requirements":
                    for relative, expected in self.requirement_hashes.items():
                        if sha256_file(self.root18 / relative) != expected:
                            raise ResumeError(f"authenticated source requirement changed before install: {relative}")
                    constraints = self.runtime_root / "torch-constraints.txt"
                    with constraints.open("xb") as stream:
                        stream.write(TORCH_CONSTRAINTS)
                        stream.flush()
                        os.fsync(stream.fileno())
                    self._record("torch-constraints", "completed", descriptor=descriptor(constraints))
                result = self._stage(name, argv, stdout_name)
                if name == "managed-python-check":
                    if Path(result["stdout"]).read_text(encoding="utf-8").strip() != PYTHON_REQUEST:
                        raise ResumeError("uv-managed Python patch version mismatch")
                elif name == "pip-freeze":
                    if Path(result["stdout"]).stat().st_size == 0:
                        raise ResumeError("pip freeze returned an empty requirements file")
                elif name == "runtime-probe":
                    observed = strict_json_file(self.runtime_root / "runtime-observed.json")
                    runtime = verify_runtime_probe(observed, self.training_python, self.runtime_root)
            if "runtime" not in locals():
                raise ResumeError("runtime-probe did not complete")
            resume = self._final_receipts(runtime)
            return resume
        except BaseException as error:
            self._write_failure(error)
            if isinstance(error, KeyboardInterrupt):
                raise
            if isinstance(error, ResumeError):
                raise
            raise ResumeError(f"runtime resume failed: {type(error).__name__}: {error}") from error


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--source-install", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--uv-wheel", type=Path, required=True)
    parser.add_argument("--uv-wheel-sha256", required=True)
    parser.add_argument("--failed-attempt", type=Path, required=True)
    parser.add_argument("--failed-attempt-sha256", required=True)
    parser.add_argument("--cloud-execution", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = RuntimeResume(args).run()
    except (ResumeError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "receipt": str(
        Path(result["compatibility_receipt"]["path"]))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
